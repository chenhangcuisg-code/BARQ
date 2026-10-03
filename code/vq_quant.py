# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import torch
from torch import nn
import time

from uniform_quantizers import SymmetricUniformQuantizer


def get_assignments(X, centroids, chunk_size=None, H_inv_diag=None):
    """
    X: G x N x D
    centroids: G x K x D
    """
    if H_inv_diag is None:
        H_inv_diag = torch.ones(X.shape[-1]).to(X.device)
    elif H_inv_diag.ndim > 2:  # should then be 1 x N x D
        assert (
            H_inv_diag.shape[0] == 1
            and H_inv_diag.shape[1] == X.shape[1]
            and H_inv_diag.shape[2] == X.shape[2]
        ), f"{H_inv_diag.shape, X.shape}"
        H_inv_diag = H_inv_diag.unsqueeze(2)  # 1 x N x 1 x D

    if chunk_size is None:
        X_chunks = [X]
        H_inv_diag_chunks = [H_inv_diag]
    else:
        X_chunks = torch.split(X, chunk_size, dim=1)
        if H_inv_diag.ndim > 1:
            H_inv_diag_chunks = torch.split(H_inv_diag, chunk_size, dim=1)
        else:
            H_inv_diag_chunks = [H_inv_diag] * len(X_chunks)

    centroids = centroids.unsqueeze(1)  # G x 1 x K x D

    assignments = []
    for X, H_inv_diag in zip(X_chunks, H_inv_diag_chunks):
        X = X.unsqueeze(2)  # G x N' x 1 x D

        dist = ((X - centroids).pow(2) * H_inv_diag).sum(-1)

        assignments.append(dist.argmin(-1))  # G x N'
    assignments = torch.concat(assignments, dim=1)

    return assignments  # G x N


def vq_quantize(X, quantizer, H_inv_diag=None, centroids=None):
    assert len(X.shape) == 2
    orig_shape = X.shape

    vq_dim = quantizer.vq_dim

    X = X.reshape(quantizer.groups_per_column, -1, vq_dim)  # G x N x D
    if centroids is None:
        centroids = quantizer.all_centroids[-1]  # G x K x D
    idx = get_assignments(
        X, centroids, chunk_size=quantizer.assignment_chunk_size, H_inv_diag=H_inv_diag
    )  # G x N

    # below, idx expanded to G x N x D
    values = torch.gather(centroids, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, vq_dim))

    # return shapes: G x N x D, G x N
    return values.view(orig_shape), idx


def kmeans_m_step_3(
    centroids: torch.Tensor,
    n_centroids: int,
    assignments: torch.LongTensor,
    X: torch.Tensor,
    H_inv_diag=None,
):
    """
    X: G x N x D
    centroids: G x K x D
    assignments: G x N
    H_inv_diag: 1 x N x D
    """
    crange = torch.arange(0, n_centroids).to(centroids.device)

    # G x N x 1 == 1 x 1 x K --> G x N x K
    assignments_expanded = (assignments.unsqueeze(-1) == crange.view(1, 1, -1)).to(X.dtype)

    if H_inv_diag is None:
        norm = 1.0 / torch.clip(assignments_expanded.sum(1), min=1)  # G x K
        clusters_for_centroid = torch.einsum("gnd,gnk,gk->gkd", X, assignments_expanded, norm)
    else:
        norm = 1.0 / torch.clip(
            torch.einsum("gnk,nd->gkd", assignments_expanded, H_inv_diag[0]), min=1e-10
        )
        clusters_for_centroid = torch.einsum(
            "gnd,nd,gnk,gkd->gkd", X, H_inv_diag[0], assignments_expanded, norm
        )

    centroids.copy_(clusters_for_centroid)


def kmeans_vq(
    X,
    centroids,
    iters=10,
    assignment_chunk_size=None,
    H_inv_diag=None,
    codebook_bitwidth=None,
    per_codebook=False,
):
    n_centroids = centroids.shape[1]
    for iter in range(iters):
        # E-step
        assignments = get_assignments(
            X, centroids, chunk_size=assignment_chunk_size, H_inv_diag=H_inv_diag
        )

        # M-step: gather all values for each centroid and compute means
        # Centroids is shape G x D x K; assignments is shape G x N
        kmeans_m_step_3(centroids, n_centroids, assignments, X, H_inv_diag=H_inv_diag)

        if codebook_bitwidth is not None:
            quantize_centroids(centroids, codebook_bitwidth, per_codebook=per_codebook)


def kpp_parallel_sampled(data: torch.Tensor, k: int):
    G, N, D = data.shape

    if N * D < 32768 * 2:
        split_data = data.split(16)
    elif N * D * k < 32768 * 2 * 16:
        split_data = data.split(4)
    else:
        split_data = data.split(1)

    all_init = []

    for data in split_data:
        init = torch.zeros((data.shape[0], k, data.shape[-1]), dtype=torch.float16).to(
            data.device
        )  # G, K, D
        all_dists = torch.cdist(data.float(), data.float(), p=2)  # G, N, N
        init[:, 0] = data[:, 0]

        D2 = torch.zeros(data.shape[0], k, N).to(data.device)
        D2[:, 0] = all_dists[:, 0]

        for i in range(1, k):
            dists = D2[:, :i].amin(dim=1)  # G, N
            dists = (dists / dists.sum(-1, keepdims=True)).cumsum(-1)  # G, N

            v = torch.rand_like(dists[:, :1])  # G, 1

            idx = torch.clip(torch.searchsorted(dists, v).unsqueeze(-1), 0, N - 1)  # G, 1, 1

            D2[:, i : i + 1] = torch.gather(all_dists, dim=1, index=idx.expand(-1, 1, N))
            init[:, i : i + 1] = torch.gather(data, dim=1, index=idx.expand(-1, 1, D))
        all_init.append(init)
    return torch.concatenate(all_init)


def mahalanobis_init(X, n_centroids):
    """
    X: G x N x D
    centroids: G x K x D
    """
    vq_dim = X.shape[-1]
    mu = X.mean(1).unsqueeze(1)
    Xcentered = X - mu

    Sigma = torch.bmm(Xcentered.transpose(1, 2), Xcentered)  # G x D x D
    Lambda = torch.linalg.inv(Sigma)

    dists = (torch.bmm(Xcentered, Lambda) * Xcentered).sum(-1)  # G x N
    sorted_dists = torch.argsort(dists, dim=1)  # G x N
    idx = torch.round(torch.linspace(0, Xcentered.shape[1] - 1, n_centroids)).long()  # K
    idx = (
        sorted_dists[:, idx].unsqueeze(-1).expand(-1, -1, vq_dim)
    )  # G x K --> G x K x 1 --> G x K x D

    return torch.gather(X, dim=1, index=idx)


def ot_equal_mass_init_1d(X, n_centroids, H_inv_diag=None):
    """
    Hessian-weighted equal-mass codebook initialization for 1D VQ.
    X: G x N x 1
    H_inv_diag: optional [1 x N x 1], [1 x N], or [N]
    """
    assert X.ndim == 3 and X.shape[-1] == 1
    G, N, _ = X.shape
    device = X.device
    dtype = X.dtype

    if H_inv_diag is None:
        h_weights = torch.ones(N, device=device, dtype=dtype)
    else:
        if H_inv_diag.ndim == 3:
            h_weights = H_inv_diag[0, :, 0]
        elif H_inv_diag.ndim == 2:
            h_weights = H_inv_diag[0]
        else:
            h_weights = H_inv_diag
        h_weights = torch.clamp(h_weights, min=1e-12)
        # H_inv diagonal is inversely related to curvature importance.
        h_weights = 1.0 / h_weights

    centroids = torch.empty((G, n_centroids, 1), device=device, dtype=dtype)
    targets = torch.linspace(0, 1, n_centroids + 1, device=device, dtype=dtype)

    for g in range(G):
        w = X[g, :, 0]
        sort_idx = torch.argsort(w)
        w_sorted = w[sort_idx]
        h_sorted = h_weights[sort_idx]

        cdf = torch.cumsum(h_sorted, dim=0)
        cdf = cdf / torch.clamp(cdf[-1], min=1e-12)

        edges = torch.empty(n_centroids + 1, device=device, dtype=dtype)
        edges[0] = w_sorted[0] - 1e-8
        edges[-1] = w_sorted[-1] + 1e-8
        for k in range(1, n_centroids):
            idx = torch.searchsorted(cdf, targets[k], right=False)
            idx = torch.clamp(idx, 0, N - 1)
            edges[k] = w_sorted[idx]

        prev_centroid = None
        for k in range(n_centroids):
            left_edge = edges[k]
            right_edge = edges[k + 1]
            mask = (w_sorted > left_edge) & (w_sorted <= right_edge)
            if torch.any(mask):
                w_bin = w_sorted[mask]
                h_bin = h_sorted[mask]
                centroid = torch.sum(h_bin * w_bin) / torch.clamp(torch.sum(h_bin), min=1e-12)
            else:
                mid = 0.5 * (left_edge + right_edge)
                centroid = mid if prev_centroid is None else 0.5 * (prev_centroid + mid)
            centroids[g, k, 0] = centroid
            prev_centroid = centroid

    centroids, _ = torch.sort(centroids, dim=1)
    return centroids


def quantize_centroids(centroids, bitwidth, per_codebook=True):
    orig_shape = centroids.shape
    if not per_codebook:
        centroids_ = centroids.view(1, -1)
    else:
        centroids_ = centroids.flatten(start_dim=1)

    imin, imax = -(2 ** (bitwidth - 1)), 2 ** (bitwidth - 1) - 1
    qmin, qmax = centroids_.min(dim=1)[0].abs(), centroids_.max(dim=1)[0]

    qmax = torch.max(qmin, qmax).unsqueeze(1)

    scale = qmax / imax

    qcentroids = torch.clip(torch.round(centroids_ / scale), imin, imax) * scale
    centroids.copy_(qcentroids.view(orig_shape))
    return centroids


class VQQuantizer(nn.Module):

    def __init__(
        self,
        vq_dim=2,
        n_subsample=100000,
        columns_per_group=None,
        kmeans_init_method="mahalanobis",
        assignment_chunk_size=None,
        kmeans_iters=10,
        codebook_bitwidth=None,
        quantize_per_codebook=True,
        vq_scaling_blocksize=-1,
        vq_scaling_norm="max",
        vq_scaling_n_bits=4,
        vq_scaling_domain="log",
        quantize_during_kmeans=False,
        use_ot_transfer=False,
        use_ot_refine=False,
    ):
        super().__init__()
        self.vq_dim = vq_dim
        self.n_centroids = None
        self.scale = self.maxq = self.zero = None
        self.all_centroids = []
        self.columns_per_group = columns_per_group
        self.rows_per_group = None
        self.kpp_subsamples = n_subsample
        self.kmeans_init_method = kmeans_init_method
        self.assignment_chunk_size = assignment_chunk_size
        self.kmeans_iters = kmeans_iters
        self.codebook_bitwidth = codebook_bitwidth
        self.quantize_per_codebook = quantize_per_codebook
        self.quantize_during_kmeans = quantize_during_kmeans
        self.vq_scaling_blocksize = vq_scaling_blocksize
        self.vq_scaling_norm = vq_scaling_norm
        self.vq_scaling_n_bits = vq_scaling_n_bits
        self.vq_scaling_domain = vq_scaling_domain
        self.use_ot_transfer = use_ot_transfer
        self.use_ot_refine = use_ot_refine
        self._prev_X = None
        self._prev_centroids = None

    def get_groupsize(self, X, groupsize):
        if self.columns_per_group is not None:
            if groupsize < self.columns_per_group:
                assert self.columns_per_group % groupsize == 0
                self.columns_per_group = groupsize

            assert groupsize % self.columns_per_group == 0
            assert X.shape[1] % self.columns_per_group == 0

            self.rows_per_group = groupsize // self.columns_per_group
            assert X.shape[0] % self.rows_per_group == 0

            self.groups_per_column = X.shape[0] // self.rows_per_group

            return self.columns_per_group

        if groupsize < X.shape[1]:
            assert X.shape[1] % groupsize == 0
            self.groups_per_column = X.shape[0]
            return groupsize

        if groupsize % X.shape[1] != 0:
            print(
                f"Requested groupsize {groupsize} doesn't fit tensor shape[0] {X.shape[0]}. "
                f"Upscaling to {int(np.ceil(groupsize / X.shape[0]) * X.shape[0])}"
            )

        rows_per_group = int(np.ceil(groupsize / X.shape[1]))
        self.groups_per_column = X.shape[0] // rows_per_group
        return X.shape[1]

    def ready(self):
        return self.n_centroids != None

    def configure(self, wbits, **_):
        self.wbits = int(wbits * self.vq_dim)
        self.n_centroids = int(2**self.wbits)

    def find_params(self, X: torch.Tensor, weight=True, H_inv_diag=None):
        assert weight
        assert len(X.shape) == 2

        X = X.reshape(self.groups_per_column, -1, self.vq_dim)  # G x N x D
        if H_inv_diag is not None:
            H_inv_diag = H_inv_diag.reshape(1, -1, self.vq_dim)  # 1 x N x D
            if self.rows_per_group > 1:
                H_inv_diag = H_inv_diag.tile(1, self.rows_per_group, 1)

        # OT Codebook Transfer: use previous group codebook via 1D OT map
        use_transfer = (
            self.use_ot_transfer
            and self.vq_dim == 1
            and self._prev_X is not None
            and self._prev_centroids is not None
            and self._prev_X.shape == X.shape
        )

        if use_transfer:
            centroids = ot_transfer_codebook_1d(self._prev_X, X, self._prev_centroids)
        elif self.kmeans_init_method == "cdf":
            assert self.vq_dim == 1
            centroids = ot_equal_mass_init_1d(X, self.n_centroids, H_inv_diag=H_inv_diag)
        elif self.kmeans_init_method == "kpp":
            centroids = kpp_parallel_sampled(X, self.n_centroids)
        elif self.kmeans_init_method == "mahalanobis":
            centroids = mahalanobis_init(X, self.n_centroids)
        elif self.kmeans_init_method == "wasserstein":
            assert self.vq_dim == 1
            centroids = ot_wasserstein_init_1d(X, self.n_centroids)
        elif self.kmeans_init_method == "dp":
            assert self.vq_dim == 1
            centroids = dp_optimal_1d_quantizer(X, self.n_centroids, H_inv_diag=H_inv_diag)
        elif self.kmeans_init_method == "dp_wass":
            assert self.vq_dim == 1
            centroids = dp_wasserstein_optimal_1d(X, self.n_centroids, H_inv_diag=H_inv_diag)
        elif self.kmeans_init_method == "dp_bary":
            assert self.vq_dim == 1
            centroids = dp_bary_ot_init_1d(X, self.n_centroids, alpha=0.1, H_inv_diag=H_inv_diag)
        elif self.kmeans_init_method == "dp_bary_02":
            assert self.vq_dim == 1
            centroids = dp_bary_ot_init_1d(X, self.n_centroids, alpha=0.02, H_inv_diag=H_inv_diag)
        elif self.kmeans_init_method == "dp_bary_05":
            assert self.vq_dim == 1
            centroids = dp_bary_ot_init_1d(X, self.n_centroids, alpha=0.05, H_inv_diag=H_inv_diag)
        elif self.kmeans_init_method == "dp_mccann":
            assert self.vq_dim == 1
            centroids = dp_mccann_ot_init_1d(X, self.n_centroids, t=0.05, H_inv_diag=H_inv_diag)
        elif self.kmeans_init_method == "dp_mccann_02":
            assert self.vq_dim == 1
            centroids = dp_mccann_ot_init_1d(X, self.n_centroids, t=0.02, H_inv_diag=H_inv_diag)
        elif self.kmeans_init_method == "dp_mccann_10":
            assert self.vq_dim == 1
            centroids = dp_mccann_ot_init_1d(X, self.n_centroids, t=0.10, H_inv_diag=H_inv_diag)
        elif self.kmeans_init_method == "dp_bary_full":
            assert self.vq_dim == 1
            centroids = dp_bary_ot_full_1d(X, self.n_centroids, H_inv_diag=H_inv_diag)
        else:
            raise ValueError(f"Unkown k-means init method: {self.kmeans_init_method}")

        # At this point, centroids should be shape G x K x D
        extra_args = {}
        if self.quantize_during_kmeans and self.codebook_bitwidth is not None:
            extra_args = dict(
                codebook_bitwidth=self.codebook_bitwidth, per_codebook=self.quantize_per_codebook
            )

        kmeans_vq(
            X,
            centroids,
            iters=self.kmeans_iters,
            assignment_chunk_size=self.assignment_chunk_size,
            H_inv_diag=H_inv_diag,
            **extra_args,
        )

        # OT Dead Codeword Fix: ensure full codebook utilization
        if self.use_ot_refine and self.vq_dim == 1:
            centroids, n_fixed = ot_fix_dead_codewords_1d(X, centroids)
            if n_fixed > 0:
                # Run a few more K-means iters to stabilize after fixing
                kmeans_vq(
                    X,
                    centroids,
                    iters=3,
                    assignment_chunk_size=self.assignment_chunk_size,
                    H_inv_diag=H_inv_diag,
                    **extra_args,
                )

        if self.codebook_bitwidth is not None and not self.quantize_during_kmeans:
            quantize_centroids(
                centroids, self.codebook_bitwidth, per_codebook=self.quantize_per_codebook
            )

        self.all_centroids.append(centroids)

        # Store for OT transfer to next group
        if self.use_ot_transfer and self.vq_dim == 1:
            self._prev_X = X.detach().clone()
            self._prev_centroids = centroids.detach().clone()

    def blockwise_normalize_data(
        self,
        x_float,
        vq_scaling_blocksize,
        vq_scaling_norm="max",
        n_bits_scales=4,
        vq_scaling_domain="log",
    ):
        self.vq_scaling_blocksize = vq_scaling_blocksize
        orig_shape = x_float.shape
        if self.vq_scaling_blocksize > 0:
            x_float = x_float.view(
                x_float.shape[0],
                int(x_float.shape[1] // self.vq_scaling_blocksize),
                self.vq_scaling_blocksize,
            )
            if vq_scaling_norm == "L2":
                self.scales = torch.sqrt((torch.sum(x_float**2, dim=2)))
            elif vq_scaling_norm == "L1":
                self.scales = torch.sum(torch.abs(x_float), dim=2)
            elif vq_scaling_norm == "max":
                self.scales = torch.abs(x_float).max(dim=2).values
            else:
                raise NotImplementedError("This type of norm is not supported")

            if vq_scaling_domain == "log":
                self.log_scales = torch.log10(self.scales)
                self.log_scales[torch.abs(self.scales) < 1.0e-8] = (
                    0.0  # don't scale zeros, keep them as it is
                )

                self.min_log_scale, _ = torch.min(self.log_scales, dim=0, keepdim=True)
                self.log_scales -= self.min_log_scale

                if n_bits_scales < 16:
                    quant = SymmetricUniformQuantizer(n_bits=n_bits_scales, per_channel=True)
                    quant_range_min, _ = torch.min(self.log_scales, dim=0, keepdim=True)
                    quant_range_max, _ = torch.max(self.log_scales, dim=0, keepdim=True)
                    quant.set_quant_range(quant_range_min, quant_range_max)

                    self.log_scales = quant.forward(self.log_scales)

                log_scales = (self.log_scales + self.min_log_scale).unsqueeze(2)
                self.scales = torch.pow(10.0, log_scales)
            elif vq_scaling_domain == "linear":
                self.scales[torch.abs(self.scales) < 1.0e-8] = 1.0

                if n_bits_scales < 16:
                    quant = SymmetricUniformQuantizer(n_bits=n_bits_scales, per_channel=True)
                    self.min_scale, _ = torch.min(self.scales, dim=0, keepdim=True)
                    self.scales -= self.min_scale

                    quant_range_min, _ = torch.min(self.scales, dim=0, keepdim=True)
                    quant_range_max, _ = torch.max(self.scales, dim=0, keepdim=True)
                    quant.set_quant_range(quant_range_min, quant_range_max)

                    self.scales = quant.forward(self.scales)
                    self.scales += self.min_scale
                self.scales = self.scales.unsqueeze(2)
            else:
                raise NotImplementedError

            scales_repeated = self.scales.squeeze(-1).repeat_interleave(vq_scaling_blocksize, dim=1)
            x_float = torch.div(x_float, self.scales)
            x_float = x_float.view(orig_shape)

        return x_float, scales_repeated



# ===================== DP Optimal 1D Quantizer (Core OT) =====================


def dp_optimal_1d_quantizer(X, n_centroids, H_inv_diag=None):
    """
    Solve the 1D Wasserstein-2 optimal quantization problem exactly via DP.
    
    Finds K codewords that minimize (importance-weighted) quantization error.
    Unlike K-means which finds local optima, this gives the GLOBAL optimum.
    
    For 1D data, optimal quantization = optimal transport: finding the K-point
    discrete distribution nu* = argmin W_2(mu_data, nu) is equivalent to
    finding the optimal K-interval partition of sorted data.
    
    Solved via DP in O(N^2 * K) per group, where N = groupsize (typically 32-128).
    
    X: G x N x 1
    n_centroids: K (number of codewords, e.g. 8 for 3-bit)
    H_inv_diag: optional 1 x N x 1 importance weights (matches K-means convention)
    Returns: G x K x 1 (globally optimal centroids)
    """
    G, N, _ = X.shape
    K = n_centroids
    device = X.device
    dtype = X.dtype
    
    # Sort data within each group
    x_sorted, sort_idx = X[:, :, 0].sort(dim=1)  # G x N
    
    # Importance weights (matching K-means get_assignments convention)
    if H_inv_diag is not None:
        if H_inv_diag.ndim == 3:
            h = H_inv_diag[0, :, 0]
        elif H_inv_diag.ndim == 2:
            h = H_inv_diag[0]
        else:
            h = H_inv_diag
        h = h.clamp(min=1e-12)
        h = h.unsqueeze(0).expand(G, -1)  # G x N
        h_sorted = torch.gather(h, 1, sort_idx)  # G x N
    else:
        h_sorted = torch.ones(G, N, device=device, dtype=dtype)
    
    x_f = x_sorted.float()
    h_f = h_sorted.float()
    
    # Prefix sums for O(1) interval cost queries
    S_h = torch.zeros(G, N + 1, device=device, dtype=torch.float32)
    S_hx = torch.zeros(G, N + 1, device=device, dtype=torch.float32)
    S_hx2 = torch.zeros(G, N + 1, device=device, dtype=torch.float32)
    S_h[:, 1:] = torch.cumsum(h_f, dim=1)
    S_hx[:, 1:] = torch.cumsum(h_f * x_f, dim=1)
    S_hx2[:, 1:] = torch.cumsum(h_f * x_f * x_f, dim=1)
    
    # Precompute cost matrix: cost[g, i, j] = weighted MSE of interval [i, j]
    sh_end = S_h[:, 1:]
    sh_start = S_h[:, :N]
    shx_end = S_hx[:, 1:]
    shx_start = S_hx[:, :N]
    shx2_end = S_hx2[:, 1:]
    shx2_start = S_hx2[:, :N]
    
    w = sh_end.unsqueeze(1) - sh_start.unsqueeze(2)
    s_hx = shx_end.unsqueeze(1) - shx_start.unsqueeze(2)
    s_hx2 = shx2_end.unsqueeze(1) - shx2_start.unsqueeze(2)
    
    cost = s_hx2 - s_hx ** 2 / w.clamp(min=1e-12)
    cost = cost.clamp(min=0)
    
    ii = torch.arange(N, device=device)
    mask = ii.unsqueeze(1) > ii.unsqueeze(0)
    cost[:, mask] = float("inf")
    
    # DP: dp[g, k, j] = min cost to partition x[0..j] into k clusters
    INF = float("inf")
    dp = torch.full((G, K + 1, N), INF, device=device, dtype=torch.float32)
    split = torch.zeros((G, K + 1, N), device=device, dtype=torch.long)
    
    dp[:, 1, :] = cost[:, 0, :]
    split[:, 1, :] = 0
    
    for k in range(2, K + 1):
        for j in range(k - 1, N):
            prev = dp[:, k - 1, k - 2:j]
            cur_cost = cost[:, k - 1:j + 1, j]
            total = prev + cur_cost
            best_val, best_idx = total.min(dim=1)
            dp[:, k, j] = best_val
            split[:, k, j] = best_idx + (k - 1)
    
    # Vectorized backtracking
    g_idx = torch.arange(G, device=device)
    boundaries_i = torch.zeros(G, K, device=device, dtype=torch.long)
    boundaries_j = torch.zeros(G, K, device=device, dtype=torch.long)
    
    j_vec = torch.full((G,), N - 1, device=device, dtype=torch.long)
    for k in range(K, 0, -1):
        i_vec = split[:, k, :].gather(1, j_vec.unsqueeze(1)).squeeze(1)
        boundaries_i[:, k - 1] = i_vec
        boundaries_j[:, k - 1] = j_vec
        j_vec = i_vec - 1
    
    # Compute centroids (weighted means of each interval)
    centroids = torch.zeros(G, K, device=device, dtype=dtype)
    for k_idx in range(K):
        i_idx = boundaries_i[:, k_idx]
        j_idx = boundaries_j[:, k_idx]
        w_val = S_h[g_idx, j_idx + 1] - S_h[g_idx, i_idx]
        s_val = S_hx[g_idx, j_idx + 1] - S_hx[g_idx, i_idx]
        valid = w_val > 1e-12
        centroids[:, k_idx] = torch.where(
            valid,
            (s_val / w_val).to(dtype),
            x_f[g_idx, i_idx].to(dtype)
        )
    
    return centroids.unsqueeze(-1)


def dp_wasserstein_optimal_1d(X, n_centroids, H_inv_diag=None):
    """
    Wasserstein Barycenter + DP Optimal Quantization.
    
    1. Compute Wasserstein barycenter across all groups
    2. Solve optimal quantization on barycenter via DP (reference codebook)  
    3. OT-transfer reference codebook to each group
    4. Per-group DP refinement (globally optimal per group)
    
    Combines cross-group OT (barycenter) with per-group OT (DP quantizer).
    """
    G, N, D = X.shape
    K = n_centroids
    
    # Per-group DP already gives global optimum, barycenter not needed
    # But we keep the function for API compatibility
    centroids = dp_optimal_1d_quantizer(X, K, H_inv_diag=H_inv_diag)
    return centroids



# ===================== OT Methods =====================


def ot_wasserstein_init_1d(X, n_centroids):
    """
    OT-based codebook initialization via Wasserstein barycenter.
    
    1. Compute the 1D Wasserstein barycenter of all G row-groups
       (= pointwise average of sorted distributions)
    2. Design a reference codebook on the barycenter via kpp + short K-means
    3. OT-transfer the reference codebook to each row-group
    
    This shares distributional information across row-groups while respecting
    each group's specific structure through the OT map.
    
    X: G x N x 1
    Returns: G x K x 1
    """
    G, N, D = X.shape
    K = n_centroids
    device = X.device
    dtype = X.dtype
    
    # Step 1: Wasserstein barycenter = average of sorted 1D distributions
    x_sorted = X[:, :, 0].sort(dim=1)[0]  # G x N
    barycenter = x_sorted.mean(dim=0)  # N
    
    # Step 2: Design reference codebook on barycenter
    bary_3d = barycenter.unsqueeze(0).unsqueeze(-1)  # 1 x N x 1
    ref_centroids = kpp_parallel_sampled(bary_3d, K)  # 1 x K x 1
    kmeans_vq(bary_3d, ref_centroids, iters=20)  # quick refinement
    
    # Step 3: OT-transfer reference codebook to each row-group
    bary_expanded = bary_3d.expand(G, -1, -1).contiguous()  # G x N x 1
    ref_expanded = ref_centroids.expand(G, -1, -1).contiguous()  # G x K x 1
    
    centroids = ot_transfer_codebook_1d(bary_expanded, X, ref_expanded)
    
    return centroids

def ot_transfer_codebook_1d(X_prev, X_curr, prev_centroids):
    """
    Transfer codebook from previous group to current group via 1D OT map.
    Uses quantile matching: F_curr^{-1}(F_prev(centroid)).
    
    X_prev: G x N x 1 (previous group weights, reshaped)
    X_curr: G x N x 1 (current group weights, reshaped)
    prev_centroids: G x K x 1
    Returns: G x K x 1 (transferred codebook)
    """
    G, N, _ = X_prev.shape
    K = prev_centroids.shape[1]
    
    x_p_sorted = X_prev[:, :, 0].sort(dim=1)[0]  # G x N
    x_c_sorted = X_curr[:, :, 0].sort(dim=1)[0]  # G x N
    
    c = prev_centroids[:, :, 0].contiguous()  # G x K
    
    # Find quantile position of each centroid in prev distribution
    pos = torch.searchsorted(x_p_sorted.contiguous(), c)  # G x K
    pos = pos.clamp(0, N - 1)
    
    # Map to quantile [0, 1] then to curr distribution index
    quantiles = pos.float() / max(N - 1, 1)
    curr_pos = (quantiles * (N - 1)).long().clamp(0, N - 1)  # G x K
    
    new_c = torch.gather(x_c_sorted, 1, curr_pos)  # G x K
    return new_c.unsqueeze(-1)  # G x K x 1


def ot_fix_dead_codewords_1d(X, centroids, min_usage_frac=0.02):
    """
    Fix dead/underutilized codewords via OT-based cluster splitting.
    Ensures codebook measure covers the data measure (Wasserstein coverage).
    
    X: G x N x 1
    centroids: G x K x 1
    min_usage_frac: minimum fraction of points per cluster
    Returns: (centroids, n_fixed)
    """
    G, N, _ = X.shape
    K = centroids.shape[1]
    min_size = max(1, int(min_usage_frac * N))
    
    assignments = get_assignments(X, centroids)  # G x N
    
    # Count cluster sizes: G x K
    sizes = torch.zeros(G, K, device=X.device, dtype=torch.long)
    for k in range(K):
        sizes[:, k] = (assignments == k).sum(dim=1)
    
    has_dead = (sizes < min_size).any(dim=1)  # G
    if not has_dead.any():
        return centroids, 0
    
    n_fixed = 0
    dead_groups = has_dead.nonzero().flatten()
    
    for g in dead_groups.tolist():
        x = X[g, :, 0]
        a = assignments[g]
        s = sizes[g].clone()
        
        for k in range(K):
            if s[k] >= min_size:
                continue
            
            largest = s.argmax().item()
            if s[largest] < 2 * min_size:
                break
            
            mask = (a == largest)
            pts = x[mask].sort()[0]
            mid = len(pts) // 2
            
            centroids[g, largest, 0] = pts[:mid].mean()
            centroids[g, k, 0] = pts[mid:].mean()
            s[largest] = mid
            s[k] = len(pts) - mid
            n_fixed += 1
    
    return centroids, n_fixed



# ===================== OT + DP Hybrid Methods =====================


def dp_bary_ot_init_1d(X, n_centroids, alpha=0.1, H_inv_diag=None):
    """
    DP-Barycenter OT Regularized Codebook Initialization.
    
    Combines per-group DP optimal with cross-group OT regularization:
    1. Per-group DP optimal codebook (global optimum per group)
    2. Wasserstein barycenter across all groups (cross-group consensus)
    3. DP optimal codebook on the barycenter (optimal reference)
    4. OT-transfer reference codebook to each group (quantile matching)
    5. Shrink per-group DP toward OT-transferred reference
    
    alpha: shrinkage strength (0 = pure DP, 1 = pure OT-transferred barycenter)
    """
    G, N, _ = X.shape
    K = n_centroids
    device = X.device
    dtype = X.dtype
    
    # Step 1: Per-group DP optimal codebook
    dp_centroids = dp_optimal_1d_quantizer(X, K, H_inv_diag=H_inv_diag)  # G x K x 1
    
    # Step 2: Wasserstein barycenter (= pointwise avg of sorted distributions)
    x_sorted = X[:, :, 0].sort(dim=1)[0]  # G x N
    barycenter = x_sorted.mean(dim=0)      # N
    bary_sorted = barycenter.sort()[0]      # N (already sorted from sorted inputs)
    
    # Step 3: DP optimal codebook on the barycenter
    bary_3d = barycenter.unsqueeze(0).unsqueeze(-1)  # 1 x N x 1
    bary_dp_centroids = dp_optimal_1d_quantizer(bary_3d, K)  # 1 x K x 1
    bc = bary_dp_centroids[0, :, 0]  # K
    
    # Step 4: OT-transfer barycenter codebook to each group via quantile matching
    bary_pos = torch.searchsorted(bary_sorted.contiguous(), bc.contiguous())
    quantiles = bary_pos.float().clamp(0, N - 1) / max(N - 1, 1)  # K, in [0,1]
    curr_pos = (quantiles * (N - 1)).long().clamp(0, N - 1)  # K
    
    transferred = x_sorted[:, curr_pos]  # G x K (gather from each group sorted data)
    
    # Step 5: Shrink DP toward OT-transferred barycenter
    dp_c = dp_centroids[:, :, 0]  # G x K
    regularized = (1 - alpha) * dp_c + alpha * transferred  # G x K
    
    return regularized.unsqueeze(-1)  # G x K x 1


def dp_mccann_ot_init_1d(X, n_centroids, t=0.05, H_inv_diag=None):
    """
    DP + McCann Displacement Interpolation along W2 Geodesic.
    
    Uses the MEDIAN barycenter (robust to outlier groups) as regularization
    target, and applies McCann displacement interpolation (the W2 geodesic)
    rather than naive linear blending.
    
    For 1D: McCann interpolation of centroid c at parameter t is:
        c_new = (1-t)*c + t*T(c)
    where T = F_target^{-1} o F_group is the OT map from group to target.
    
    t: interpolation parameter (0 = pure DP, 1 = pure target)
    """
    G, N, _ = X.shape
    K = n_centroids
    device = X.device
    dtype = X.dtype
    
    # Step 1: Per-group DP optimal
    dp_centroids = dp_optimal_1d_quantizer(X, K, H_inv_diag=H_inv_diag)  # G x K x 1
    
    # Step 2: Robust target = MEDIAN of sorted distributions
    x_sorted = X[:, :, 0].sort(dim=1)[0]  # G x N
    target = x_sorted.median(dim=0)[0]     # N (robust to outlier groups)
    target_sorted = target.sort()[0]       # N
    
    # Step 3: DP optimal on target (optimal reference codebook)
    target_3d = target.unsqueeze(0).unsqueeze(-1)  # 1 x N x 1
    target_dp = dp_optimal_1d_quantizer(target_3d, K)  # 1 x K x 1
    
    # Step 4: McCann displacement interpolation for each group
    dp_c = dp_centroids[:, :, 0]  # G x K
    new_centroids = torch.zeros_like(dp_c)
    
    for g in range(G):
        c_g = dp_c[g]  # K
        xs_g = x_sorted[g]  # N (sorted)
        
        # Find quantile of each centroid in group g distribution
        pos = torch.searchsorted(xs_g.contiguous(), c_g.contiguous())
        quantiles = pos.float().clamp(0, N - 1) / max(N - 1, 1)  # K, [0,1]
        
        # Map to target distribution at same quantiles
        target_pos = (quantiles * (N - 1)).long().clamp(0, N - 1)
        target_vals = target_sorted[target_pos]  # K
        
        # McCann displacement interpolation along W2 geodesic
        new_centroids[g] = (1 - t) * c_g + t * target_vals
    
    return new_centroids.unsqueeze(-1)  # G x K x 1


def dp_bary_ot_full_1d(X, n_centroids, H_inv_diag=None):
    """
    Full OT Pipeline: Wasserstein Barycenter + DP Reference + OT Transfer.
    
    No per-group DP, purely OT-driven:
    1. Wasserstein barycenter across all groups
    2. DP optimal codebook on the barycenter (reference)
    3. OT-transfer to each group
    
    This tests whether OT cross-group sharing + DP reference can match
    per-group DP alone.
    """
    G, N, _ = X.shape
    K = n_centroids
    device = X.device
    dtype = X.dtype
    
    # Step 1: Wasserstein barycenter
    x_sorted = X[:, :, 0].sort(dim=1)[0]  # G x N
    barycenter = x_sorted.mean(dim=0)      # N
    bary_sorted = barycenter.sort()[0]
    
    # Step 2: DP optimal on barycenter (better than kpp used in wasserstein)
    bary_3d = barycenter.unsqueeze(0).unsqueeze(-1)  # 1 x N x 1
    ref_centroids = dp_optimal_1d_quantizer(bary_3d, K)  # 1 x K x 1
    bc = ref_centroids[0, :, 0]  # K
    
    # Step 3: OT-transfer to each group via quantile matching
    bary_pos = torch.searchsorted(bary_sorted.contiguous(), bc.contiguous())
    quantiles = bary_pos.float().clamp(0, N - 1) / max(N - 1, 1)
    curr_pos = (quantiles * (N - 1)).long().clamp(0, N - 1)
    
    transferred = x_sorted[:, curr_pos]  # G x K
    
    return transferred.unsqueeze(-1)  # G x K x 1


# ===================== OT Adaptive Codebook (GPTQ Error Compensation) =====================


def ot_adapt_codebook_1d(X_remaining, centroids, X_original_sorted, alpha=0.2):
    """
    OT-based online codebook adaptation during GPTQ error propagation.

    As GPTQ propagates quantization errors to subsequent columns, the weight
    distribution shifts away from what the codebook was designed for. This
    function uses the 1D optimal transport map to track the distribution shift
    and adapt the codebook accordingly.

    The OT map T = F_current^{-1} o F_original maps each centroid from its
    position in the original distribution to the corresponding position in
    the current (error-shifted) distribution.

    Args:
        X_remaining: G x N_remain x 1 - current weights for remaining columns
        centroids: G x K x 1 - current codebook
        X_original_sorted: G x N_orig - sorted original weights (saved at group start)
        alpha: blending strength (0 = no adaptation, 1 = full OT transport)

    Returns: G x K x 1 (adapted codebook)
    """
    import torch
    G, N_remain, _ = X_remaining.shape
    N_orig = X_original_sorted.shape[1]
    K = centroids.shape[1]

    if N_remain < max(2, K // 2):
        return centroids  # not enough data to adapt

    # Sort current remaining weights per group
    x_curr_sorted = X_remaining[:, :, 0].sort(dim=1)[0]  # G x N_remain
    c = centroids[:, :, 0].contiguous()  # G x K

    # OT map: find quantile of each centroid in original distribution
    pos = torch.searchsorted(X_original_sorted.contiguous(), c)  # G x K
    quantiles = pos.float().clamp(0, N_orig - 1) / max(N_orig - 1, 1)  # [0, 1]

    # Map to current distribution at same quantile (F_current^{-1})
    curr_pos = (quantiles * (N_remain - 1)).long().clamp(0, N_remain - 1)  # G x K
    c_transported = torch.gather(x_curr_sorted, 1, curr_pos)  # G x K

    # Blend: McCann-style interpolation along OT geodesic
    c_adapted = (1 - alpha) * c + alpha * c_transported

    # Ensure centroids remain sorted (important for searchsorted in next adapt)
    c_adapted = c_adapted.sort(dim=1)[0]

    return c_adapted.unsqueeze(-1)  # G x K x 1


def ot_adapt_codebook_progressive_1d(X_remaining, centroids, X_original_sorted,
                                      col_in_group, groupsize, alpha_max=0.3):
    """
    Progressive OT adaptation: strength increases as we move further from group start.

    Rationale: error propagation accumulates, so distribution drift grows with
    distance from the group start. Early columns need little adaptation,
    later columns need more.

    alpha = alpha_max * (col_in_group / groupsize)
    """
    alpha = alpha_max * (col_in_group / groupsize)
    return ot_adapt_codebook_1d(X_remaining, centroids, X_original_sorted, alpha=alpha)


# ===================== OT-Adapt N-Dimensional (per-axis 1D OT) =====================


def ot_adapt_codebook_nd(X_remaining, centroids, X_original_sorted_per_dim, alpha=0.2):
    """
    Multi-dim OT adapt: apply 1D OT independently on each coordinate axis.
    Equivalent to marginal-OT: each axis matches current to original quantile.

    Args:
        X_remaining: G x N_remain x D
        centroids: G x K x D
        X_original_sorted_per_dim: G x N_orig x D (sorted along dim=1 per axis independently)
        alpha: blending strength

    Returns: G x K x D (adapted codebook)
    """
    import torch
    G, N_remain, D = X_remaining.shape
    K = centroids.shape[1]
    N_orig = X_original_sorted_per_dim.shape[1]

    if N_remain < max(2, K // 2):
        return centroids

    c_adapted = centroids.clone()
    for d in range(D):
        # Per-axis sort
        x_curr_sorted_d = X_remaining[:, :, d].sort(dim=1)[0]  # G x N_remain
        x_orig_sorted_d = X_original_sorted_per_dim[:, :, d].contiguous()  # G x N_orig
        c_d = centroids[:, :, d].contiguous()  # G x K

        pos = torch.searchsorted(x_orig_sorted_d, c_d)  # G x K
        quantiles = pos.float().clamp(0, N_orig - 1) / max(N_orig - 1, 1)
        curr_pos = (quantiles * (N_remain - 1)).long().clamp(0, N_remain - 1)
        c_transported = torch.gather(x_curr_sorted_d, 1, curr_pos)

        c_adapted[:, :, d] = (1 - alpha) * c_d + alpha * c_transported

    # Note: per-axis sorting across K not enforced (because centroids are
    # D-dim vectors; sorting each axis independently would break vector
    # identity). We leave them in their existing order.
    return c_adapted


def ot_adapt_codebook_progressive_nd(X_remaining, centroids, X_original_sorted_per_dim,
                                     col_in_group, groupsize, alpha_max=0.3):
    alpha = alpha_max * (col_in_group / groupsize)
    return ot_adapt_codebook_nd(X_remaining, centroids, X_original_sorted_per_dim, alpha=alpha)


# ===================== Post-MStep OT Refinement =====================


def ot_refine_post_mstep_1d(centroids, W_group_scaled, assignments=None, alpha=0.1):
    """
    After m-step has globally optimized centroids, do ONE pass of OT-based refinement
    per group.  For each centroid c_k, compute the OT quantile of its currently-assigned
    data points' distribution, then move centroid toward data median (1D).

    Args:
        centroids: G x K x 1  (post-mstep centroids)
        W_group_scaled: G x N x 1  (per-group data vectors)
        assignments: G x N  (each data point's cluster index)
        alpha: blending strength (0 = no refine, 1 = pure OT target)

    Returns: G x K x 1 refined centroids
    """
    import torch
    G, N, D = W_group_scaled.shape
    K = centroids.shape[1]
    assert D == 1

    # For each group g and centroid k, find weights assigned to k and compute
    # OT-adjusted target as their weighted quantile mapping
    # OT approach: sort centroids, sort data, interpolate centroids toward
    # rank-matched quantile positions in data distribution
    refined = centroids.clone()
    for g in range(G):
        data_g = W_group_scaled[g, :, 0].contiguous()  # N
        sorted_data, _ = torch.sort(data_g)
        sorted_c, sort_idx = torch.sort(centroids[g, :, 0])
        # Map each centroid rank to same-quantile in data
        ranks = torch.linspace(0, N-1, K, device=centroids.device).long()
        targets_sorted = sorted_data[ranks]  # K target positions
        # Blend
        refined_sorted = (1 - alpha) * sorted_c + alpha * targets_sorted
        # Undo the sort
        unsort_idx = torch.argsort(sort_idx)
        refined[g, :, 0] = refined_sorted[unsort_idx]
    return refined


def ot_refine_post_mstep_nd(centroids, W_group_scaled, assignments=None, alpha=0.1):
    """
    Multi-dim version: for each centroid c_k, move toward the per-axis median of its
    currently-assigned data points.
    
    Args:
        centroids: G x K x D
        W_group_scaled: G x N x D
        assignments: G x N
    """
    import torch
    G, N, D = W_group_scaled.shape
    K = centroids.shape[1]
    
    # For nd: apply 1D OT per axis independently (marginal OT)
    refined = centroids.clone()
    for g in range(G):
        for d in range(D):
            data_gd = W_group_scaled[g, :, d].contiguous()
            sorted_data, _ = torch.sort(data_gd)
            sorted_c, sort_idx = torch.sort(centroids[g, :, d])
            ranks = torch.linspace(0, N-1, K, device=centroids.device).long()
            targets_sorted = sorted_data[ranks]
            refined_sorted = (1 - alpha) * sorted_c + alpha * targets_sorted
            unsort_idx = torch.argsort(sort_idx)
            refined[g, :, d] = refined_sorted[unsort_idx]
    return refined


# ===================== Sinkhorn-EM with H-weighted cost (Idea I) =====================


def sinkhorn_em_hessian_weighted(W_group_scaled, centroids, H_inv_diag=None,
                                  eps=0.05, sinkhorn_iters=30,
                                  marginal_mode='uniform',
                                  prior_alpha=4.0,
                                  mix_lambda=1.0,
                                  entropy_gate=None,
                                  stats_out=None):
    """
    Hessian-weighted Sinkhorn-EM for centroid refinement (centroid-only; assignment
    is recomputed by GPTVQ Hessian-nearest lookup downstream).

    marginal_mode controls the centroid (K) marginal a:
      'uniform' — a_i = 1/K  (original I_eps001)
      'prior'   — a_i = (n_i + prior_alpha) / (N + prior_alpha*K)
                  with n_i = nearest-assignment counts in H-weighted metric
      'mix'     — a = (1-mix_lambda)*a_prior(prior_alpha) + mix_lambda*a_uniform

    entropy_gate: float in [0,1] or None.  If set, group-level normalized usage
      entropy H(p)/log(K) above the gate => skip Sinkhorn for that group
      (centroid kept as-is).  Use to spare already-balanced groups.

    stats_out: optional dict to receive {'skipped','total','mean_ent'} aggregates.

    Returns: (G, K, D) refined centroids.
    """
    import torch
    import math
    G, N, D = W_group_scaled.shape
    K = centroids.shape[1]
    device = W_group_scaled.device
    dtype = W_group_scaled.dtype

    if H_inv_diag is None:
        h = torch.ones(D, device=device, dtype=dtype)
    else:
        if H_inv_diag.ndim == 3:
            h = H_inv_diag[0].mean(dim=0)
        elif H_inv_diag.ndim == 2:
            h = H_inv_diag.mean(dim=0)
        else:
            h = H_inv_diag
        h = h.clamp(min=1e-10)
    w_dim = (1.0 / h).to(dtype)

    log_K_const = math.log(float(K))
    refined = centroids.clone()

    # Process G in chunks to bound memory.
    # Largest intermediate is diff = (Gb, K, N, D) for cost computation, plus M (Gb, K, N) and P (Gb, K, N).
    # Conservative budget: 64MB per chunk after observing OOM at 256MB on vd=4 cpg=256 (large G, K).
    bytes_budget = 64_000_000
    per_g_bytes = (
        K * N * D * 4   # diff tensor (Gb, K, N, D) — largest
        + K * N * 4 * 3 # M, log_K_mat, log_P, P — multiple (Gb, K, N) buffers
        + K * D * 4     # centroids slab Cb (Gb, K, D)
        + N * D * 4     # data slab Xb (Gb, N, D)
    )
    chunk_g = max(1, bytes_budget // max(per_g_bytes, 1))

    skipped = 0
    ent_sum = 0.0

    for g_start in range(0, G, chunk_g):
        g_end = min(G, g_start + chunk_g)
        Xb = W_group_scaled[g_start:g_end]   # (Gb, N, D)
        Cb = centroids[g_start:g_end]         # (Gb, K, D)
        Gb = Xb.shape[0]

        # Cost (Gb, K, N) = sum_d w_d (C - X)^2 in H-weighted metric
        diff = Cb.unsqueeze(2) - Xb.unsqueeze(1)               # (Gb, K, N, D)
        M = (diff.pow(2) * w_dim).sum(dim=-1)                  # (Gb, K, N)
        # Per-group robust scale: divide by median over (K,N) per group
        med = M.flatten(1).median(dim=-1, keepdim=True).values  # (Gb, 1)
        M = M / med.unsqueeze(-1).clamp(min=1e-8)               # (Gb, K, N)

        # Nearest assignment counts (Gb, K) for prior/entropy
        assign0 = M.argmin(dim=1)                               # (Gb, N)
        # one-hot bincount: (Gb, K)
        counts = torch.zeros(Gb, K, device=device, dtype=torch.float32)
        counts.scatter_add_(1, assign0, torch.ones_like(assign0, dtype=torch.float32))
        cs = counts.sum(dim=-1, keepdim=True).clamp(min=1.0)    # (Gb, 1)
        p_use = (counts / cs).clamp(min=1e-12)
        ent_per = -(p_use * p_use.log()).sum(dim=-1)            # (Gb,)
        ent_norm = ent_per / max(log_K_const, 1e-12)             # (Gb,)
        ent_sum += float(ent_norm.sum().item())

        # Marginal a: (Gb, K)
        if marginal_mode == 'uniform':
            a = torch.full((Gb, K), 1.0 / K, device=device, dtype=dtype)
        elif marginal_mode in ('prior', 'mix'):
            a_prior = (counts + prior_alpha) / (cs + prior_alpha * K)
            a_prior = a_prior.to(dtype)
            if marginal_mode == 'prior':
                a = a_prior
            else:
                a_uniform = torch.full_like(a_prior, 1.0 / K)
                a = (1.0 - mix_lambda) * a_prior + mix_lambda * a_uniform
        else:
            raise ValueError(f"unknown marginal_mode {marginal_mode}")

        log_a = torch.log(a.clamp(min=1e-12))                   # (Gb, K)
        log_b = torch.full((Gb, N), -math.log(float(N)), device=device, dtype=dtype)

        log_K_mat = -M / eps                                     # (Gb, K, N)
        log_u = torch.zeros(Gb, K, device=device, dtype=dtype)
        log_v = torch.zeros(Gb, N, device=device, dtype=dtype)
        for _ in range(sinkhorn_iters):
            log_u = log_a - torch.logsumexp(log_K_mat + log_v.unsqueeze(1), dim=2)
            log_v = log_b - torch.logsumexp(log_K_mat + log_u.unsqueeze(2), dim=1)
        log_P = log_K_mat + log_u.unsqueeze(2) + log_v.unsqueeze(1)
        P = torch.exp(log_P)                                     # (Gb, K, N)

        denom = P.sum(dim=2, keepdim=True).clamp(min=1e-12)      # (Gb, K, 1)
        C_new = torch.bmm(P, Xb) / denom                         # (Gb, K, D)

        if entropy_gate is not None:
            keep_mask = (ent_norm > entropy_gate)                # True = SKIP (already balanced)
            skipped += int(keep_mask.sum().item())
            update_mask = ~keep_mask                              # (Gb,)
            refined[g_start:g_end] = torch.where(
                update_mask.view(-1, 1, 1), C_new, Cb
            )
        else:
            refined[g_start:g_end] = C_new

        # Free chunk-local tensors to reduce peak memory before next chunk
        del Xb, Cb, diff, M, log_K_mat, log_u, log_v, log_P, P, C_new
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if stats_out is not None:
        stats_out['skipped'] = skipped
        stats_out['total'] = G
        stats_out['mean_ent'] = ent_sum / max(G, 1)
        stats_out['marginal_mode'] = marginal_mode

    return refined
# ============================================================================
# G-dim chunked get_assignments — incremental addon (does NOT modify existing code)
# Activated via env var GPTVQ_G_CHUNK_SIZE (set by --g-chunk-size CLI flag).
# Monkey-patches module-level `get_assignments` so all callsites benefit
# without touching their code.
# ============================================================================
import os as _gptvq_os
_GPTVQ_G_CHUNK_ORIG = get_assignments  # save original


def _gptvq_get_assignments_g_chunked(X, centroids, chunk_size=None, H_inv_diag=None):
    """Drop-in for get_assignments that chunks G dim if GPTVQ_G_CHUNK_SIZE is set.

    Falls back to original when env var unset / G already small.
    Args:
      X: (G, N, D)
      centroids: (G, K, D)
      chunk_size: forwarded to original (chunks N dim)
      H_inv_diag: forwarded
    Returns: (G, N) assignments (same as original)
    """
    g_size = _gptvq_os.environ.get("GPTVQ_G_CHUNK_SIZE")
    try:
        g_size = int(g_size) if g_size else None
    except ValueError:
        g_size = None
    if g_size is None or X.shape[0] <= g_size:
        return _GPTVQ_G_CHUNK_ORIG(X, centroids, chunk_size=chunk_size, H_inv_diag=H_inv_diag)
    out = []
    G_total = X.shape[0]
    for gs in range(0, G_total, g_size):
        ge = min(G_total, gs + g_size)
        Xb = X[gs:ge]
        Cb = centroids[gs:ge]
        # H_inv_diag is shape (1, N, D) or (N, D) or 1d; not G-indexed
        Hb = H_inv_diag
        ab = _GPTVQ_G_CHUNK_ORIG(Xb, Cb, chunk_size=chunk_size, H_inv_diag=Hb)
        out.append(ab)
        del Xb, Cb, ab
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(out, dim=0)


# Replace module-level binding so all callsites pick up the wrapper.
# Existing code path is preserved when env var unset.
get_assignments = _gptvq_get_assignments_g_chunked
# ============================================================================
# H_inv_diag reshape fix — incremental addon
# Bug: find_params at line ~354 does H_inv_diag.reshape(1, -1, vq_dim)
#      which fails when H_inv_diag.numel() == 1 (cpg=1 scalar) and vq_dim > 1.
# Fix: monkey-patch find_params to pre-tile H_inv_diag to multiple of vq_dim
#      before reshape.
# Activated via env var GPTVQ_HINV_FIX=1.
# ============================================================================
import os as _hinv_os

_HINV_FIX_ENABLED = _hinv_os.environ.get("GPTVQ_HINV_FIX", "") == "1"

if _HINV_FIX_ENABLED:
    _orig_find_params = VQQuantizer.find_params

    def _patched_find_params(self, X, weight=True, H_inv_diag=None):
        # If H_inv_diag doesn't divide evenly into vq_dim, tile it
        if H_inv_diag is not None and self.vq_dim > 1:
            numel = H_inv_diag.numel()
            if numel % self.vq_dim != 0:
                # Pad or tile to make it divisible
                target = ((numel + self.vq_dim - 1) // self.vq_dim) * self.vq_dim
                if numel == 1:
                    # Scalar -> repeat to match X.shape[1]*vq_dim (at least vq_dim)
                    import torch as _t
                    H_inv_diag = H_inv_diag.reshape(-1).expand(X.shape[1]).contiguous()
                else:
                    # Pad by repeating last value
                    import torch as _t
                    pad = target - numel
                    last = H_inv_diag.reshape(-1)[-1:]
                    H_inv_diag = _t.cat([H_inv_diag.reshape(-1), last.repeat(pad)], 0)
        return _orig_find_params(self, X, weight=weight, H_inv_diag=H_inv_diag)

    VQQuantizer.find_params = _patched_find_params
    print("[hinv-fix] enabled: H_inv_diag auto-tile before reshape")


# ============================================================
# DUAL-GPU PATCH v2 (2026-05-14): full m-step on cuda:1
# - GPU0 keeps model+Hessian (~70GB), no m-step intermediates allocated on it
# - GPU1 (143GB free) does ALL m-step compute in chunks
# - Result centroids copied back to GPU0 slice-by-slice
# Env: GPTVQ_DUAL_GPU_M_STEP=1, GPTVQ_DGPU_CHUNK=64 (default)
# ============================================================
import os as _os_dgpu
_DUAL_GPU_M_STEP = _os_dgpu.getenv('GPTVQ_DUAL_GPU_M_STEP', '0') == '1'
_DGPU_CHUNK = int(_os_dgpu.getenv('GPTVQ_DGPU_CHUNK', '64'))

_orig_kmeans_m_step_3 = kmeans_m_step_3

def _kmeans_m_step_3_dual_gpu(centroids, n_centroids, assignments, X, H_inv_diag=None):
    """Route entire m-step compute to cuda:1 in chunks. cuda:0 free of m-step intermediates."""
    if (not _DUAL_GPU_M_STEP) or torch.cuda.device_count() < 2 or X.shape[0] < 2:
        return _orig_kmeans_m_step_3(centroids, n_centroids, assignments, X, H_inv_diag)
    dev = torch.device('cuda:1')
    G = X.shape[0]
    chunk = max(1, _DGPU_CHUNK)
    h1 = H_inv_diag.to(dev, non_blocking=True) if H_inv_diag is not None else None
    for g0 in range(0, G, chunk):
        g1 = min(g0 + chunk, G)
        Xc = X[g0:g1].to(dev, non_blocking=True)
        ac = assignments[g0:g1].to(dev, non_blocking=True)
        cc = centroids[g0:g1].detach().clone().to(dev, non_blocking=True)
        _orig_kmeans_m_step_3(cc, n_centroids, ac, Xc, h1)
        torch.cuda.synchronize(dev)
        _tmp = cc.to('cuda:0')
        torch.cuda.synchronize('cuda:0')
        centroids[g0:g1].copy_(_tmp)
        del Xc, ac, cc, _tmp
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
    del h1
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()

kmeans_m_step_3 = _kmeans_m_step_3_dual_gpu
print('[GPTVQ_DUAL_GPU_M_STEP] patch v2 loaded, enabled=' + str(_DUAL_GPU_M_STEP) + ' chunk=' + str(_DGPU_CHUNK), flush=True)
