# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
#
# OT-GPTQ: Hessian 加权码本 + Sinkhorn 软分配 + OBS 误差补偿
# 参考 OT-GPTQ 算法 Cursor 工程实现技术文档 完整实现 Phase 1-4
#

import torch
from typing import Tuple, Optional


def compute_hessian(X: torch.Tensor, lam: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Phase 1: Hessian 预处理
    输入: X [d_in, N] 校准激活
    输出: H, H_inv, U, h_bar
    """
    d_in, N = X.shape
    device = X.device
    dtype = X.dtype

    H = (2.0 / N) * torch.mm(X.float(), X.float().t()) + lam * torch.eye(d_in, device=device, dtype=torch.float32)
    H = H.to(dtype)

    try:
        L = torch.linalg.cholesky(H)
        I_matrix = torch.eye(d_in, device=device, dtype=H.dtype)
        L_inv = torch.linalg.solve_triangular(L, I_matrix, upper=False)
        H_inv = torch.mm(L_inv.t(), L_inv)
        U = torch.linalg.cholesky(H_inv, upper=True)
    except RuntimeError as e:
        print(f"Cholesky 分解失败，增大阻尼系数: {e}")
        return compute_hessian(X, lam * 10)

    h_diag = torch.diag(H)
    h_bar = h_diag / h_diag.sum()
    return H, H_inv, U, h_bar


def design_codebook(W: torch.Tensor, h_bar: torch.Tensor, K: int) -> torch.Tensor:
    """
    Phase 2: Hessian 加权 OT 码本设计（等质量分箱）
    输入: W [d_out, d_in], h_bar [d_in], K
    输出: C [K] 升序码本
    """
    device = W.device
    dtype = W.dtype
    d_out, d_in = W.shape

    w_flat = W.flatten()
    h_expanded = h_bar.repeat(d_out)

    sorted_indices = torch.argsort(w_flat)
    w_sorted = w_flat[sorted_indices]
    h_sorted = h_expanded[sorted_indices]

    h_cumsum = torch.cumsum(h_sorted, dim=0)
    h_total = h_cumsum[-1].clamp(min=1e-12)
    h_cdf = h_cumsum / h_total

    bin_edges = []
    for k in range(K + 1):
        target_mass = k / K
        if target_mass == 0.0:
            bin_edges.append(w_sorted[0].item() - 1e-8)
        elif target_mass >= 1.0:
            bin_edges.append(w_sorted[-1].item() + 1e-8)
        else:
            idx = torch.searchsorted(h_cdf, target_mass, right=False)
            idx = torch.clamp(idx, 0, len(w_sorted) - 1)
            bin_edges.append(w_sorted[idx].item())

    codebook = []
    prev_centroid = None
    for k in range(K):
        left_edge = bin_edges[k]
        right_edge = bin_edges[k + 1]
        mask = (w_sorted > left_edge) & (w_sorted <= right_edge)
        if mask.sum() == 0:
            mid = 0.5 * (left_edge + right_edge)
            centroid = mid if prev_centroid is None else 0.5 * (prev_centroid + mid)
        else:
            w_bin = w_sorted[mask]
            h_bin = h_sorted[mask]
            weighted_sum = torch.sum(h_bin * w_bin)
            weight_sum = torch.sum(h_bin).clamp(min=1e-12)
            centroid = (weighted_sum / weight_sum).item()
        codebook.append(centroid)
        prev_centroid = centroid

    C = torch.tensor(codebook, device=device, dtype=dtype)
    C, _ = torch.sort(C)
    return C


def sinkhorn_assignment(
    w_col: torch.Tensor,
    C: torch.Tensor,
    U_jj: float,
    eps: float = 0.05,
    max_iter: int = 50,
    tol: float = 1e-6,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Phase 3: Sinkhorn 熵正则化软分配
    输入: w_col [d_out], C [K], U_jj
    输出: Gamma [d_out, K] 软分配矩阵
    """
    d_out = w_col.shape[0]
    K = C.shape[0]
    device = w_col.device
    dtype = w_col.dtype

    if torch.isnan(w_col).any() or torch.isinf(w_col).any():
        raise ValueError("输入权重包含 NaN 或 Inf 值")
    if U_jj <= 0:
        raise ValueError(f"Cholesky 对角值必须为正数，当前值: {U_jj}")
    if eps <= 0:
        raise ValueError(f"正则化参数必须为正数，当前值: {eps}")

    w_expanded = w_col.unsqueeze(1)
    C_expanded = C.unsqueeze(0)

    # 代价矩阵：纯 L2，归一化到 [0,1]
    # 注：Sinkhorn 等质量约束（每个码本点被等量使用）在 GPTQ 逐列量化中
    # 会迫使大量权重分配到非最近码本点，导致误差暴增。
    # 在 gptq.py 集成路径中已改为最近邻分配。
    # 此函数保留为独立工具，使用归一化 L2 代价。
    M_raw = (w_expanded - C_expanded) ** 2  # [R, K]
    M_scale = M_raw.max().clamp(min=1e-8)
    M = M_raw / M_scale  # M in [0, 1]

    # eps 自适应（相对于归一化代价）
    w_range = (w_col.max() - w_col.min()).item()
    if w_range > 1e-12 and verbose:
        print(f"sinkhorn: w_range={w_range:.4f}, U_jj={U_jj:.6f}")

    p = torch.ones(d_out, device=device, dtype=dtype) / d_out
    q = torch.ones(K, device=device, dtype=dtype) / K

    # Log-domain Sinkhorn updates:
    # u <- p / (K v), v <- q / (K^T u)
    # where K = exp(-M / eps), implemented in log-space for stability.
    log_K = -M / eps
    log_u = torch.zeros(d_out, device=device, dtype=dtype)
    log_v = torch.zeros(K, device=device, dtype=dtype)
    log_p = torch.log(p.clamp(min=1e-12))
    log_q = torch.log(q.clamp(min=1e-12))

    for iteration in range(max_iter):
        log_u_old = log_u.clone()
        log_u = log_p - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        log_v = log_q - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        if torch.max(torch.abs(log_u - log_u_old)) < tol:
            if verbose:
                print(f"Sinkhorn 收敛于第 {iteration + 1} 次迭代")
            break

    log_Gamma = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    Gamma = torch.exp(log_Gamma)
    if torch.isnan(Gamma).any() or torch.isinf(Gamma).any():
        raise ValueError("Sinkhorn 输出包含 NaN/Inf")
    # 仅作为数值防护；argmax 对缩放不敏感。
    Gamma = Gamma / Gamma.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return Gamma


def hard_assignment_fallback(w_col: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """硬最近邻分配作为 Sinkhorn 失败时的回退"""
    dists = (w_col.unsqueeze(1) - C.unsqueeze(0)) ** 2
    k_indices = dists.argmin(dim=1)
    Gamma = torch.zeros(w_col.shape[0], C.shape[0], device=w_col.device, dtype=w_col.dtype)
    Gamma.scatter_(1, k_indices.unsqueeze(1), 1.0)
    return Gamma


def quantize_columns_ot(
    W: torch.Tensor,
    C: torch.Tensor,
    U: torch.Tensor,
    eps: float = 0.05,
    block_size: int = 128,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Phase 4: 逐列量化 + OBS 误差补偿（Sinkhorn 软分配 + 硬分配取码）
    输入: W [d_out, d_in], C [K], U [d_in, d_in]
    输出: W_hat [d_out, d_in]
    """
    d_out, d_in = W.shape
    device = W.device
    dtype = W.dtype
    W = W.clone()

    W_hat = torch.zeros_like(W)

    for j in range(d_in):
        if verbose and j % 100 == 0:
            print(f"量化进度: {j}/{d_in} ({100 * j / d_in:.1f}%)")

        w_col = W[:, j].clone()
        U_jj = U[j, j].item()

        if U_jj <= 1e-8:
            if verbose:
                print(f"警告：列 {j} 的 Cholesky 对角值过小 ({U_jj})，跳过量化")
            W_hat[:, j] = w_col
            continue

        try:
            Gamma = sinkhorn_assignment(w_col, C, U_jj, eps=eps, verbose=False)
        except Exception as e:
            if verbose:
                print(f"列 {j} Sinkhorn 失败: {e}，使用硬分配")
            Gamma = hard_assignment_fallback(w_col, C)

        k_indices = torch.argmax(Gamma, dim=1)
        W_hat[:, j] = C[k_indices]

        error = w_col - W_hat[:, j]
        if j < d_in - 1:
            compensation_weights = U[j, j + 1 :] / U_jj
            W[:, j + 1 :] -= torch.outer(error, compensation_weights)

    return W_hat


def ot_gptq_complete(
    W: torch.Tensor,
    X: torch.Tensor,
    bits: int = 4,
    lam: float = 0.01,
    eps: float = 0.05,
    block_size: int = 128,
    verbose: bool = True,
) -> dict:
    """
    OT-GPTQ 完整流程（Phase 1-4，不含可选 Phase 5/6）
    W: [d_out, d_in], X: [d_in, N]
    """
    if verbose:
        print("=" * 60)
        print("OT-GPTQ 量化开始")
        print("=" * 60)

    H, H_inv, U, h_bar = compute_hessian(X, lam)
    if verbose:
        print(f"Phase 1: Hessian 预处理完成，H shape {H.shape}")

    K = 2 ** bits
    C = design_codebook(W, h_bar, K)
    if verbose:
        print(f"Phase 2: OT 码本设计完成，K={K}, 范围 [{C.min():.4f}, {C.max():.4f}]")

    W_original = W.clone()
    W_hat = quantize_columns_ot(W, C, U, eps=eps, block_size=block_size, verbose=verbose)
    if verbose:
        mse = torch.mean((W_original - W_hat) ** 2).item()
        print(f"Phase 3&4: 逐列量化完成，量化 MSE: {mse:.2e}")

    distances = torch.abs(W_hat.unsqueeze(-1) - C.unsqueeze(0).unsqueeze(0))
    indices = torch.argmin(distances, dim=-1)

    results = {
        "quantized_weight": W_hat,
        "indices": indices,
        "codebook": C,
        "bits": bits,
        "h_bar": h_bar,
        "U": U,
        "compression_ratio": (W.numel() * 32) / (W.numel() * bits + K * 32),
    }
    if verbose:
        print(f"压缩比: {results['compression_ratio']:.2f}x")
        print("=" * 60)
    return results
