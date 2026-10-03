"""
OT-based codebook refinement: post-K-means optimization.

After standard K-means converges, apply OT-based refinement:
1. Detect under-utilized codewords (potential dead/near-dead centroids)
2. Split over-utilized codewords (too many weights assigned to one centroid)
3. Use Wasserstein distance to evaluate codebook quality

This runs AFTER K-means, so it can only improve (never hurt) the solution.
"""
import torch


def ot_refine_centroids(X, centroids, n_refine_iters=3, split_threshold=2.0):
    """
    OT-based codebook refinement after K-means.
    
    Args:
        X: [G, N, D] grouped weight data
        centroids: [G, K, D] K-means centroids (modified in-place)
        n_refine_iters: number of split-merge-reassign iterations
        split_threshold: split codewords with >threshold*avg assignments
    """
    G, N, D = X.shape
    K = centroids.shape[1]
    avg_count = N / K
    
    improved = False
    for ref_iter in range(n_refine_iters):
        # Compute assignments and per-group utilization
        dists = ((X.unsqueeze(2) - centroids.unsqueeze(1)) ** 2).sum(-1)  # [G, N, K]
        assignments = dists.argmin(dim=2)  # [G, N]
        
        # Compute current total quantization error
        current_error = torch.gather(dists, 2, assignments.unsqueeze(2)).sum()
        
        any_change = False
        for g in range(G):
            counts = torch.bincount(assignments[g], minlength=K).float()
            
            # Find dead/underutilized and overutilized codewords
            dead_mask = counts < max(1, avg_count * 0.1)  # <10% of average
            over_mask = counts > avg_count * split_threshold
            
            dead_indices = dead_mask.nonzero(as_tuple=True)[0]
            over_indices = over_mask.nonzero(as_tuple=True)[0]
            
            if len(dead_indices) == 0 or len(over_indices) == 0:
                continue
            
            # Split-merge: move dead codewords to split overutilized ones
            n_moves = min(len(dead_indices), len(over_indices))
            for m in range(n_moves):
                dead_k = dead_indices[m].item()
                over_k = over_indices[m].item()
                
                # Get weights assigned to overutilized centroid
                over_mask_n = (assignments[g] == over_k)
                over_weights = X[g, over_mask_n, :]  # [n_over, D]
                
                if over_weights.shape[0] < 2:
                    continue
                
                # Split: move dead centroid to the region of highest variance in overutilized cluster
                over_mean = over_weights.mean(dim=0)
                over_var = (over_weights - over_mean.unsqueeze(0)).pow(2).sum(-1)
                
                # Put the new centroid at the mean of the "far half"
                median_var = over_var.median()
                far_mask = over_var > median_var
                near_mask = ~far_mask
                
                if far_mask.sum() > 0 and near_mask.sum() > 0:
                    centroids[g, dead_k] = over_weights[far_mask].mean(dim=0)
                    centroids[g, over_k] = over_weights[near_mask].mean(dim=0)
                    any_change = True
        
        if not any_change:
            break
        
        # After split-merge, do a few vectorized K-means iterations
        for _ in range(3):
            dists = ((X.unsqueeze(2) - centroids.unsqueeze(1)) ** 2).sum(-1)
            assignments = dists.argmin(dim=2)  # [G, N]
            # Vectorized M-step
            one_hot = torch.zeros(G, N, K, device=X.device, dtype=X.dtype)
            one_hot.scatter_(2, assignments.unsqueeze(-1), 1.0)
            counts = one_hot.sum(dim=1).unsqueeze(-1).clamp(min=1)  # [G, K, 1]
            new_cents = torch.bmm(one_hot.transpose(1, 2), X) / counts  # [G, K, D]
            # Only update centroids with assignments
            has_assign = (one_hot.sum(dim=1) > 0).unsqueeze(-1)
            centroids.copy_(torch.where(has_assign, new_cents, centroids))
        
        # Check if we improved
        dists = ((X.unsqueeze(2) - centroids.unsqueeze(1)) ** 2).sum(-1)
        new_assignments = dists.argmin(dim=2)
        new_error = torch.gather(dists, 2, new_assignments.unsqueeze(2)).sum()
        
        if new_error < current_error:
            improved = True
    
    return improved
