# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/IST-DASLab/gptq
# Copyright 2022 IST-DASLab, Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution


import torch
from torch import nn
import numpy as np

import math
import time

import transformers

from quant import *
from vq_quant import vq_quantize, quantize_centroids, ot_adapt_codebook_1d
from ot_gptq import design_codebook, sinkhorn_assignment, hard_assignment_fallback

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def quad_loss(w_q, G, v, offset):
    """
    A generic function for computing the quadratic loss:
    L = 1/2 (G w_q, w_q) + (v, w_q) + offset

    Parameters
    ----------
    w_q : (c_out, m) or (m, 1)
        Quantized weights to be optimized.
    G : (m, m)
        Matrix part.
    v : shape(w_q)
        Linear part.
    offset : ()
        Scalar part.
    """
    # Quadratic loss: 1/2 wGw^T
    loss = 0.5 * (w_q.mm(G) * w_q).sum()
    # Add linear term and offset
    loss += (v * w_q).sum()
    loss += offset
    return loss


def quad_loss_2(W, Q, G):
    Werr = W - Q
    return (Werr.mm(G) * Werr).sum()


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def lut_m_step(self, Q_orig, groupsize, quantizer, scale=None, svd_rank=None):
        with torch.enable_grad():
            W = self.layer.weight.data.clone().float()
            G = self.G
            del self.G
            if scale is not None:
                scale.detach()

            offset = (W.mm(G) * W).sum()

            all_centroids = quantizer.all_centroids
            all_assignments = self.assignments
            vq_dim = quantizer.vq_dim

            if svd_rank is not None:
                assert vq_dim == 1, "In this implementation, SVD only works on 1D VQ"
                r = int(all_centroids[0].shape[1] * svd_rank)
                print(f"Effective SVD rank: {r}")
                Groups = all_centroids[0].shape[0]

                C = torch.concat(all_centroids, dim=0).squeeze()  # G x K
                C, new_idx = torch.sort(C, dim=1)
                new_idx = torch.argsort(new_idx, dim=1).split(Groups)

                U, S, V = torch.linalg.svd(C, full_matrices=False)
                all_centroids, V = (U * S[None])[:, :r].split(Groups), V[:r]

                new_assignments = []
                for idx, a in zip(new_idx, all_assignments):
                    new_assignments.append([])
                    for a_ in a:
                        remapped_a = torch.gather(idx, dim=1, index=a_)
                        new_assignments[-1].append(remapped_a)
                all_assignments = new_assignments

            def make_quantized_weight(centroids, assignments, scale=None):
                all_values = []
                for c, a in zip(centroids, assignments):
                    if svd_rank is not None:
                        c = (c @ V).unsqueeze(-1)
                    for a_ in a:
                        values = torch.gather(
                            c, dim=1, index=a_.unsqueeze(-1).expand(-1, -1, vq_dim)
                        )
                        all_values.append(values.view(W.shape[0], -1))
                Q = torch.concat(all_values, dim=1)
                if scale is not None:
                    Q = torch.mul(Q, scale)
                return Q

            with torch.no_grad():
                Q = make_quantized_weight(all_centroids, all_assignments, scale)

                orig_loss = quad_loss_2(W, Q, G)
                snr_before = 10 * np.log10(offset.item() / orig_loss.item())

            # FAST_B_PATCH_V2: inner=25 (vanilla), outer cap=3.
            # If still diverging after 3 restarts, accept current loss (or keep orig if worse).
            must_restart = True
            lr = 1e-3
            _fast_b_restart_count = 0
            _fast_b_max_restart = 3
            _fast_b_inner_iters = 25
            while must_restart:
                orig_centroids = [c.data.clone() for c in all_centroids]
                [c.requires_grad_() for c in all_centroids]
                param_list = list(all_centroids) + ([] if svd_rank is None else [V])
                o = torch.optim.Adam(param_list, lr=lr)
                for _ in range(_fast_b_inner_iters):
                    must_restart = False
                    o.zero_grad()
                    Q = make_quantized_weight(all_centroids, all_assignments, scale)
                    loss = quad_loss_2(W, Q, G)
                    if loss > orig_loss or torch.isnan(loss):
                        lr *= 1e-1
                        _fast_b_restart_count += 1
                        if _fast_b_restart_count >= _fast_b_max_restart:
                            # Give up: revert to orig centroids, exit loop
                            all_centroids = orig_centroids
                            must_restart = False
                            print(f"FAST_B_PATCH: giving up after {_fast_b_restart_count} restarts, keeping orig")
                            break
                        print(f"Inner loop: Restarting M-step with lr={lr:.2e} (restart {_fast_b_restart_count}/{_fast_b_max_restart})")
                        must_restart = True
                        all_centroids = orig_centroids
                        break
                    loss.backward()
                    o.step()

                if not must_restart:
                    if quantizer.codebook_bitwidth is not None:
                        new_all_centroids = [
                            quantize_centroids(
                                c.requires_grad_(False),
                                quantizer.codebook_bitwidth,
                                per_codebook=quantizer.quantize_per_codebook,
                            )
                            for c in all_centroids
                        ]
                    else:
                        new_all_centroids = all_centroids
                    Q = make_quantized_weight(new_all_centroids, all_assignments, scale)
                    loss = quad_loss_2(W, Q, G)
                    if torch.isnan(loss):
                        lr *= 1e-1
                        print(f"Outer loop: Restarting M-step with lr={lr:.2e}")
                        must_restart = True
                        all_centroids = orig_centroids
                        continue

                    del orig_centroids
                    print(
                        f"time M-step SGD {(time.time() - self.tick):.2f}; final loss: {loss.item():.4f}"
                    )
                    orig_loss = quad_loss_2(W, Q, G)
                    snr_after = 10 * np.log10(offset.item() / orig_loss.item())

                    print(f"improvement: {snr_before:.2f} -> {snr_after:.2f}")

            # ==== POST-MSTEP OT REFINEMENT ====
            if getattr(self, 'post_mstep_ot_enable', False):
                from vq_quant import ot_refine_post_mstep_1d, ot_refine_post_mstep_nd
                alpha_refine = getattr(self, 'post_mstep_ot_alpha', 0.1)
                vq_dim_ = quantizer.vq_dim
                # For each group's codebook, refine by OT-shift toward assigned data median
                new_all_centroids = []
                for cb_idx, c in enumerate(all_centroids):
                    # c shape: G x K x D
                    # W for this group (float, already dequant = m-step Q ≈ original weight)
                    col_start = cb_idx * groupsize
                    col_end = col_start + groupsize
                    W_grp = W[:, col_start:col_end]  # R x gs (float)
                    X = W_grp.reshape(quantizer.groups_per_column, -1, vq_dim_)
                    if vq_dim_ == 1:
                        c_refined = ot_refine_post_mstep_1d(c.detach(), X, alpha=alpha_refine)
                    else:
                        c_refined = ot_refine_post_mstep_nd(c.detach(), X, alpha=alpha_refine)
                    new_all_centroids.append(c_refined)
                # Regenerate Q with refined centroids
                Q_refined = make_quantized_weight(new_all_centroids, all_assignments, scale)
                loss_refined = quad_loss_2(W, Q_refined, G)
                snr_refined = 10 * np.log10(offset.item() / loss_refined.item())
                print(f"post-mstep OT refine: snr {snr_after:.2f} -> {snr_refined:.2f} (alpha={alpha_refine})")
                if loss_refined < orig_loss:
                    Q = Q_refined
                    print('  -> refined Q accepted')
                else:
                    print('  -> refined Q rejected (loss worse), keeping m-step Q')
            # ==== END POST-MSTEP OT ====

        return Q

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        include_m_step=False,
        use_vq=False,
        svd_rank=None,
        hessian_weighted_lookups=False,
        only_init_kmeans=False,
        use_ot_sinkhorn=False,
        sinkhorn_eps=0.05,
        ot_soft_assign=False,
        use_ot_adapt=False,
        ot_adapt_interval=8,
        ot_adapt_strength=0.2,
        ot_adapt_progressive=False,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        self.tick = time.time()

        if not self.quantizer.ready() and not use_vq:
            self.quantizer.find_params(W, weight=True)

        H = self.H
        self.G = self.H.clone()
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            raise NotImplementedError("Static groups are not supported in this repo")

        if actorder:
            raise NotImplementedError("Activation (re)-ordering is not supported in this repo")

        vq_dim = self.assignments = None
        S = vq_scaling_blocksize = vq_scaling_n_bits = None
        if use_vq:
            vq_dim = self.quantizer.vq_dim
            groupsize = self.quantizer.get_groupsize(W, groupsize)
            self.assignments = []
            assert blocksize % vq_dim == 0

            vq_scaling_blocksize = self.quantizer.vq_scaling_blocksize
            vq_scaling_n_bits = self.quantizer.vq_scaling_n_bits
            if vq_scaling_blocksize > 0:
                assert vq_scaling_blocksize % vq_dim == 0
                S = torch.ones_like(W)

            print(W.shape)
            print(
                f"VQ scaling BS {vq_scaling_blocksize} @ {vq_scaling_n_bits}b "
                f"({self.quantizer.vq_scaling_domain} domain)"
            )
            print(f"Using Hessian-aware K-means {hessian_weighted_lookups}")
            if use_ot_adapt:
                _adapt_mode = "progressive" if ot_adapt_progressive else f"fixed alpha={ot_adapt_strength}"
                print(f"OT-Adaptive Codebook: interval={ot_adapt_interval}, {_adapt_mode}")
            if use_ot_sinkhorn:
                assert vq_dim == 1, "use_ot_sinkhorn 仅支持 vq_dim=1"
                decode_mode = "soft decode (Gamma @ C)" if ot_soft_assign else "hard decode (argmax)"
                print(f"Using OT-GPTQ: Sinkhorn 软分配 (eps={sinkhorn_eps}, {decode_mode})")

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        h_bar = None
        if use_vq and use_ot_sinkhorn:
            h_diag = torch.diag(self.G)
            h_bar = h_diag / h_diag.sum()

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            if use_vq and vq_scaling_blocksize > 0:
                W1_scaled, S1 = self.quantizer.blockwise_normalize_data(
                    W1,
                    vq_scaling_blocksize,
                    self.quantizer.vq_scaling_norm,
                    vq_scaling_n_bits,
                    self.quantizer.vq_scaling_domain,
                )
                S[:, i1:i2] = S1
            else:
                W1_scaled = W1
                S1 = torch.ones_like(W1)

            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        extra_args = {}
                        if use_vq and hessian_weighted_lookups:
                            H_inv_diag = torch.diag(Hinv)[i1 + i : i1 + i + groupsize]
                            extra_args["H_inv_diag"] = H_inv_diag

                        W_group = W[:, (i1 + i) : (i1 + i + groupsize)]

                        W_group_scaled = W_group
                        if use_vq:
                            self.assignments.append([])
                            if vq_scaling_blocksize > 0:
                                assert vq_scaling_blocksize % vq_dim == 0
                                W_group_scaled, S_group = self.quantizer.blockwise_normalize_data(
                                    W_group,
                                    vq_scaling_blocksize,
                                    self.quantizer.vq_scaling_norm,
                                    self.quantizer.vq_scaling_n_bits,
                                    self.quantizer.vq_scaling_domain,
                                )

                        # OT-GPTQ 修正：用 find_params 做每组独立码本设计
                        # 原 design_codebook 创建全局共享码本（所有行组同码本），
                        # 导致行分布差异大的层（k_proj、q_proj 等）误差暴增。
                        # find_params + kmeans_init=cdf 已实现 per-group OT 等质量码本，
                        # 与 design_codebook 数学等价但在正确的粒度（每行组）上运行。
                        self.quantizer.find_params(W_group_scaled, weight=True, **extra_args)

                        # ==== Sinkhorn-EM Hessian-weighted centroid refinement (Idea I) ====
                        if getattr(self, 'sinkhorn_em_enable', False) and vq_dim > 0:
                            from vq_quant import sinkhorn_em_hessian_weighted
                            _eps = getattr(self, 'sinkhorn_em_eps', 0.05)
                            _iters = getattr(self, 'sinkhorn_em_iters', 30)
                            _marg = getattr(self, 'sinkhorn_em_marginal', 'uniform')
                            _alpha = getattr(self, 'sinkhorn_em_prior_alpha', 4.0)
                            _lambda = getattr(self, 'sinkhorn_em_mix_lambda', 1.0)
                            _gate = getattr(self, 'sinkhorn_em_entropy_gate', None)
                            _X = W_group_scaled.reshape(self.quantizer.groups_per_column, -1, vq_dim)
                            _H = extra_args.get('H_inv_diag', None)
                            if _H is not None:
                                _H = _H.reshape(1, -1, vq_dim)
                            _stats = {}
                            self.quantizer.all_centroids[-1] = sinkhorn_em_hessian_weighted(
                                _X, self.quantizer.all_centroids[-1],
                                H_inv_diag=_H, eps=_eps, sinkhorn_iters=_iters,
                                marginal_mode=_marg, prior_alpha=_alpha,
                                mix_lambda=_lambda, entropy_gate=_gate,
                                stats_out=_stats,
                            )
                            if not getattr(self, '_sinkhorn_em_logged', False):
                                print(f"[sinkhorn-em] mode={_marg} alpha={_alpha} lambda={_lambda} eps={_eps} iters={_iters} gate={_gate}")
                                self._sinkhorn_em_logged = True
                        # ==== END Sinkhorn-EM ====

                        # OT-Adaptive: save sorted original data for this group
                        if use_ot_adapt and use_vq:
                            _grp_3d = W_group_scaled.reshape(
                                self.quantizer.groups_per_column, -1, vq_dim
                            )
                            if vq_dim == 1:
                                _ot_adapt_orig_sorted = _grp_3d[:, :, 0].sort(dim=1)[0]  # G x N
                            else:
                                # vd >= 2: per-axis sorted, shape G x N x D
                                _ot_adapt_orig_sorted = _grp_3d.sort(dim=1)[0]

                if not use_vq:
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q = quantize(
                        w.unsqueeze(1),
                        self.quantizer.scale,
                        self.quantizer.zero,
                        self.quantizer.maxq,
                    ).flatten()

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d**2

                    err1 = (w - q) / d
                    # (R x 1).matmul(1 x C') --> R x C' (C': remaining (unquantized) columns)
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                elif i % vq_dim == 0:
                    w = W1[:, i : i + vq_dim]  # R x D
                    d = torch.diag(Hinv1)[i : i + vq_dim].unsqueeze(0)  # 1 x D
                    w_scaled = W1_scaled[:, i : i + vq_dim]  # R x D
                    s = S1[:, i : i + vq_dim]

                    if use_ot_sinkhorn and vq_dim == 1:
                        # OT-GPTQ：使用 per-group 码本 + 最近邻分配（+ 可选软解码）
                        # vq_quantize 正确处理每行组独立码本，避免全局共享码本的覆盖问题
                        if ot_soft_assign:
                            # 软解码：先找 NN，再用 softmax(-d/T) 做组内加权重建
                            G_cnt = self.quantizer.groups_per_column
                            rpg = getattr(self.quantizer, "rows_per_group", None) or (w_scaled.shape[0] // G_cnt)
                            centroids = self.quantizer.all_centroids[-1]  # [G, K, 1]
                            w_grouped = w_scaled.reshape(G_cnt, rpg, 1)   # [G, rpg, 1]
                            dists2 = (w_grouped.unsqueeze(2) - centroids.unsqueeze(1)) ** 2  # [G, rpg, K, 1]
                            dists2 = dists2.squeeze(-1)                     # [G, rpg, K]
                            k_indices = dists2.argmin(dim=2).reshape(-1)    # [R]
                            # 归一化到 [0,1] 后用 softmax：d_min=0 → softmax=1，d_max=1 → softmax≈0
                            d_min = dists2.min(dim=2, keepdim=True)[0]     # [G, rpg, 1]
                            d_max = dists2.max(dim=2, keepdim=True)[0]
                            d_scale = (d_max - d_min).clamp(min=1e-8)
                            dists_norm = (dists2 - d_min) / d_scale        # [0, 1]
                            # sinkhorn_eps=0.05: exp(-1/0.05)=exp(-20)≈0 → 近硬分配
                            soft_w = torch.softmax(-dists_norm / sinkhorn_eps, dim=2)  # [G, rpg, K]
                            q_grouped = (soft_w.unsqueeze(-1) * centroids.unsqueeze(1)).sum(dim=2)  # [G, rpg, 1]
                            q = q_grouped.reshape(w_scaled.shape)           # [R, 1]
                        else:
                            q, assmt_t = vq_quantize(w_scaled, self.quantizer)
                            G_cnt = self.quantizer.groups_per_column
                            rpg = getattr(self.quantizer, "rows_per_group", None) or (w_scaled.shape[0] // G_cnt)
                            k_indices = assmt_t.reshape(-1)
                        q = torch.mul(q, s)
                        gpc = self.quantizer.groups_per_column
                        rpg = getattr(
                            self.quantizer, "rows_per_group", None
                        ) or (w_scaled.shape[0] // gpc)
                        assmt = k_indices.view(gpc, rpg)
                        self.assignments[-1].append(assmt)
                    else:
                        H_inv_diag = None
                        if vq_dim > 1 and hessian_weighted_lookups:
                            H_inv_diag = 1.0 / d.to(w.device)
                        q, assmt = vq_quantize(
                            w_scaled, self.quantizer, H_inv_diag=H_inv_diag
                        )
                        q = torch.mul(q, s)
                        self.assignments[-1].append(assmt)

                    Q1[:, i : i + vq_dim] = q
                    Losses1[:, i : i + vq_dim] = (w - q) ** 2 / d**2  # R x D / 1 x D

                    err1 = (w - q) / d  # R x D
                    # batch matmul solution: (D x R x 1).matmul(D x 1 x C').sum(0) --> R x C'
                    if not only_init_kmeans:
                        update = torch.bmm(
                            err1.transpose(0, 1).unsqueeze(-1),
                            Hinv1[i : i + vq_dim, i + vq_dim :].unsqueeze(1),
                        ).sum(0)
                        W1[:, i + vq_dim :] -= update
                        Err1[:, i : i + vq_dim] = err1

                    # OT-Adaptive Codebook: adapt centroids to track distribution shift
                    if use_ot_adapt and not only_init_kmeans:
                        _col_in_grp = (i1 + i) % groupsize
                        if _col_in_grp > 0 and _col_in_grp % ot_adapt_interval == 0:
                            _grp_end_abs = (((i1 + i) // groupsize) + 1) * groupsize
                            _rem_start = i + vq_dim
                            _rem_end = min(count, _grp_end_abs - i1)
                            if _rem_end > _rem_start and (_rem_end - _rem_start) >= max(2, self.quantizer.n_centroids // 2):
                                _W_rem = W1[:, _rem_start:_rem_end].clone()
                                if vq_scaling_blocksize > 0 and S1 is not None:
                                    _S_rem = S1[:, _rem_start:_rem_end].clamp(min=1e-8)
                                    _W_rem = _W_rem / _S_rem
                                _W_rem_3d = _W_rem.reshape(
                                    self.quantizer.groups_per_column, -1, vq_dim
                                )
                                if vq_dim == 1:
                                    if ot_adapt_progressive:
                                        from vq_quant import ot_adapt_codebook_progressive_1d
                                        self.quantizer.all_centroids[-1] = ot_adapt_codebook_progressive_1d(
                                            _W_rem_3d,
                                            self.quantizer.all_centroids[-1],
                                            _ot_adapt_orig_sorted,
                                            col_in_group=_col_in_grp,
                                            groupsize=groupsize,
                                            alpha_max=ot_adapt_strength,
                                        )
                                    else:
                                        self.quantizer.all_centroids[-1] = ot_adapt_codebook_1d(
                                            _W_rem_3d,
                                            self.quantizer.all_centroids[-1],
                                            _ot_adapt_orig_sorted,
                                            alpha=ot_adapt_strength,
                                        )
                                else:
                                    # vd >= 2: per-axis OT
                                    if ot_adapt_progressive:
                                        from vq_quant import ot_adapt_codebook_progressive_nd
                                        self.quantizer.all_centroids[-1] = ot_adapt_codebook_progressive_nd(
                                            _W_rem_3d,
                                            self.quantizer.all_centroids[-1],
                                            _ot_adapt_orig_sorted,
                                            col_in_group=_col_in_grp,
                                            groupsize=groupsize,
                                            alpha_max=ot_adapt_strength,
                                        )
                                    else:
                                        from vq_quant import ot_adapt_codebook_nd
                                        self.quantizer.all_centroids[-1] = ot_adapt_codebook_nd(
                                            _W_rem_3d,
                                            self.quantizer.all_centroids[-1],
                                            _ot_adapt_orig_sorted,
                                            alpha=ot_adapt_strength,
                                        )

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            if not only_init_kmeans:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - self.tick))
        print("error", torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        if include_m_step:
            Q = self.lut_m_step(Q, groupsize, self.quantizer, scale=S, svd_rank=svd_rank)

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
