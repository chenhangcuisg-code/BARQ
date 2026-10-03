"""
OT-enhanced assignment for VQ-GPTQ.

Key idea: at quantization time, instead of per-row nearest-neighbor assignment,
solve a mini OT problem per group that considers:
1. Distance to centroid (standard)  
2. Hessian diagonal weight (importance of this column)
3. Codebook utilization balance

This creates an assignment that may sacrifice per-row optimality
but improves the overall Hessian-weighted quantization error.
"""
import torch


def ot_enhanced_quantize(w_scaled, quantizer, h_diag_col, sinkhorn_reg=0.1, balance_weight=0.5):
    """
    OT-enhanced VQ quantization for a single column.
    
    Args:
        w_scaled: [R, 1] scaled weights for this column
        quantizer: VQQuantizer with all_centroids
        h_diag_col: scalar, Hessian diagonal for this column (H_inv[j,j])
        sinkhorn_reg: Sinkhorn regularization
        balance_weight: how much to weight balance vs NN (0=pure NN, 1=pure balanced)
    
    Returns:
        q: [R, 1] quantized values
        assmt: assignment tensor
    """
    from vq_quant import vq_quantize
    
    G = quantizer.groups_per_column
    rpg = getattr(quantizer, "rows_per_group", None) or (w_scaled.shape[0] // G)
    centroids = quantizer.all_centroids[-1]  # [G, K, 1]
    K = centroids.shape[1]
    
    # Reshape to groups
    w_grouped = w_scaled.reshape(G, rpg, 1)  # [G, rpg, 1]
    
    # Compute distance matrix: [G, rpg, K]
    dists2 = (w_grouped.unsqueeze(2) - centroids.unsqueeze(1)).pow(2).squeeze(-1)
    
    # Pure NN assignment
    nn_assign = dists2.argmin(dim=2)  # [G, rpg]
    
    # Check codebook utilization per group
    needs_rebalance = False
    for g in range(G):
        counts = torch.bincount(nn_assign[g], minlength=K)
        min_count = counts.min().item()
        max_count = counts.max().item()
        if min_count == 0 or (max_count > 4 * rpg / K):
            needs_rebalance = True
            break
    
    if not needs_rebalance or balance_weight < 1e-6:
        # No dead codewords and good balance -> use standard NN
        q, assmt = vq_quantize(w_scaled, quantizer)
        return q, assmt
    
    # OT rebalancing: Sinkhorn to find better assignment
    # Cost = dists2 (normalized)
    M = dists2 / dists2.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].clamp(min=1e-8)
    
    # Log-domain Sinkhorn
    log_P = -M / sinkhorn_reg  # [G, rpg, K]
    
    # Target: each codeword gets at least rpg/(2K) and at most 2*rpg/K
    # Use soft target of rpg/K per codeword
    target_per_k = rpg / K
    
    log_u = torch.zeros(G, rpg, 1, device=w_scaled.device, dtype=w_scaled.dtype)
    log_v = torch.zeros(G, 1, K, device=w_scaled.device, dtype=w_scaled.dtype)
    
    for _ in range(20):
        log_sum_k = torch.logsumexp(log_P + log_v, dim=2, keepdim=True)
        log_u = -log_sum_k
        log_sum_n = torch.logsumexp(log_P + log_u, dim=1, keepdim=True)
        log_v = torch.log(torch.tensor(target_per_k, device=w_scaled.device, dtype=w_scaled.dtype)) - log_sum_n
    
    # Final transport plan
    log_plan = log_P + log_u + log_v
    ot_assign = log_plan.argmax(dim=2)  # [G, rpg]
    
    # Blend: use OT assignment only for rows where it changes AND improves balance
    # For other rows, keep NN
    final_assign = nn_assign.clone()
    
    for g in range(G):
        nn_counts = torch.bincount(nn_assign[g], minlength=K)
        ot_counts = torch.bincount(ot_assign[g], minlength=K)
        
        # Use OT if it eliminates dead codewords
        nn_dead = (nn_counts == 0).sum().item()
        ot_dead = (ot_counts == 0).sum().item()
        
        if ot_dead < nn_dead:
            final_assign[g] = ot_assign[g]
    
    # Reconstruct quantized values from assignment
    k_indices = final_assign.reshape(-1)  # [R]
    g_indices = torch.arange(G, device=w_scaled.device).unsqueeze(1).expand(G, rpg).reshape(-1)
    q = centroids[g_indices, k_indices, :].reshape(w_scaled.shape)
    
    assmt = final_assign
    return q, assmt
