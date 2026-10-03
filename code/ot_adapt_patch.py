#!/usr/bin/env python3
"""
Patch vq_quant.py, gptq.py, llama.py to add OT-Adaptive Codebook method.

New method: During GPTQ error propagation, use OT transport map to track
the weight distribution shift and adapt the codebook online.

T = F_current^{-1} o F_original  -->  C_new = (1-alpha)*C + alpha*T(C)
"""
import re
import sys

BASE = "/data/chenhang/codes/gptvq-main"

# ============================================================
# 1. Add ot_adapt_codebook_1d() to vq_quant.py
# ============================================================
vq_path = f"{BASE}/vq_quant.py"
with open(vq_path, "r") as f:
    vq_code = f.read()

if "ot_adapt_codebook_1d" in vq_code:
    print("vq_quant.py already has ot_adapt_codebook_1d, skipping")
else:
    new_func = r'''

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

    if N_remain < K:
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
'''
    vq_code += new_func
    with open(vq_path, "w") as f:
        f.write(vq_code)
    print("vq_quant.py: added ot_adapt_codebook_1d and progressive variant")


# ============================================================
# 2. Modify gptq.py to add OT adaptation in the GPTQ loop
# ============================================================
gptq_path = f"{BASE}/gptq.py"
with open(gptq_path, "r") as f:
    gptq_code = f.read()

if "ot_adapt_codebook_1d" in gptq_code:
    print("gptq.py already patched, skipping")
else:
    # 2a. Add import
    old_import = "from vq_quant import vq_quantize, quantize_centroids"
    new_import = "from vq_quant import vq_quantize, quantize_centroids, ot_adapt_codebook_1d"
    gptq_code = gptq_code.replace(old_import, new_import)

    # 2b. Add parameters to fasterquant signature
    old_sig = "ot_soft_assign=False,"
    new_sig = (
        "ot_soft_assign=False,\n"
        "        use_ot_adapt=False,\n"
        "        ot_adapt_interval=8,\n"
        "        ot_adapt_strength=0.2,\n"
        "        ot_adapt_progressive=False,"
    )
    gptq_code = gptq_code.replace(old_sig, new_sig)

    # 2c. After VQ info print, add OT adapt info
    old_print = 'print(f"Using Hessian-aware K-means {hessian_weighted_lookups}")'
    new_print = (
        'print(f"Using Hessian-aware K-means {hessian_weighted_lookups}")\n'
        '            if use_ot_adapt and vq_dim == 1:\n'
        '                _adapt_mode = "progressive" if ot_adapt_progressive else f"fixed alpha={ot_adapt_strength}"\n'
        '                print(f"OT-Adaptive Codebook: interval={ot_adapt_interval}, {_adapt_mode}")'
    )
    gptq_code = gptq_code.replace(old_print, new_print)

    # 2d. After find_params, save sorted original data for OT adapt
    old_find_params = "self.quantizer.find_params(W_group_scaled, weight=True, **extra_args)"
    new_find_params = (
        "self.quantizer.find_params(W_group_scaled, weight=True, **extra_args)\n"
        "\n"
        "                        # OT-Adaptive: save sorted original data for this group\n"
        "                        if use_ot_adapt and use_vq and vq_dim == 1:\n"
        "                            _grp_3d = W_group_scaled.reshape(\n"
        "                                self.quantizer.groups_per_column, -1, vq_dim\n"
        "                            )\n"
        "                            _ot_adapt_orig_sorted = _grp_3d[:, :, 0].sort(dim=1)[0]  # G x N"
    )
    gptq_code = gptq_code.replace(old_find_params, new_find_params)

    # 2e. After VQ error propagation, add OT adaptation
    old_err_prop = "                        W1[:, i + vq_dim :] -= update\n                        Err1[:, i : i + vq_dim] = err1"
    new_err_prop = (
        "                        W1[:, i + vq_dim :] -= update\n"
        "                        Err1[:, i : i + vq_dim] = err1\n"
        "\n"
        "                    # OT-Adaptive Codebook: adapt centroids to track distribution shift\n"
        "                    if use_ot_adapt and vq_dim == 1 and not only_init_kmeans:\n"
        "                        _col_in_grp = (i1 + i) % groupsize\n"
        "                        if _col_in_grp > 0 and _col_in_grp % ot_adapt_interval == 0:\n"
        "                            _grp_end_abs = (((i1 + i) // groupsize) + 1) * groupsize\n"
        "                            _rem_start = i + vq_dim\n"
        "                            _rem_end = min(count, _grp_end_abs - i1)\n"
        "                            if _rem_end > _rem_start and (_rem_end - _rem_start) >= self.quantizer.n_centroids:\n"
        "                                _W_rem = W1[:, _rem_start:_rem_end].clone()\n"
        "                                if vq_scaling_blocksize > 0 and S1 is not None:\n"
        "                                    _S_rem = S1[:, _rem_start:_rem_end].clamp(min=1e-8)\n"
        "                                    _W_rem = _W_rem / _S_rem\n"
        "                                _W_rem_3d = _W_rem.reshape(\n"
        "                                    self.quantizer.groups_per_column, -1, vq_dim\n"
        "                                )\n"
        "                                if ot_adapt_progressive:\n"
        "                                    from vq_quant import ot_adapt_codebook_progressive_1d\n"
        "                                    self.quantizer.all_centroids[-1] = ot_adapt_codebook_progressive_1d(\n"
        "                                        _W_rem_3d,\n"
        "                                        self.quantizer.all_centroids[-1],\n"
        "                                        _ot_adapt_orig_sorted,\n"
        "                                        col_in_group=_col_in_grp,\n"
        "                                        groupsize=groupsize,\n"
        "                                        alpha_max=ot_adapt_strength,\n"
        "                                    )\n"
        "                                else:\n"
        "                                    self.quantizer.all_centroids[-1] = ot_adapt_codebook_1d(\n"
        "                                        _W_rem_3d,\n"
        "                                        self.quantizer.all_centroids[-1],\n"
        "                                        _ot_adapt_orig_sorted,\n"
        "                                        alpha=ot_adapt_strength,\n"
        "                                    )"
    )
    gptq_code = gptq_code.replace(old_err_prop, new_err_prop)

    with open(gptq_path, "w") as f:
        f.write(gptq_code)
    print("gptq.py: added OT-Adaptive Codebook logic")


# ============================================================
# 3. Modify llama.py to add CLI args
# ============================================================
llama_path = f"{BASE}/llama.py"
with open(llama_path, "r") as f:
    llama_code = f.read()

if "ot-adapt" in llama_code:
    print("llama.py already has --ot-adapt, skipping")
else:
    # 3a. Add CLI args after --ot-soft-assign
    if "--ot-soft-assign" in llama_code:
        old_arg = "parser.add_argument('--ot-soft-assign', action='store_true')"
        new_arg = (
            "parser.add_argument('--ot-soft-assign', action='store_true')\n"
            "    parser.add_argument('--ot-adapt', action='store_true', help='Enable OT-adaptive codebook during GPTQ')\n"
            "    parser.add_argument('--ot-adapt-interval', type=int, default=8, help='Adapt codebook every N columns')\n"
            "    parser.add_argument('--ot-adapt-strength', type=float, default=0.2, help='OT adaptation blending strength')\n"
            "    parser.add_argument('--ot-adapt-progressive', action='store_true', help='Progressive adaptation strength')"
        )
        llama_code = llama_code.replace(old_arg, new_arg)
    else:
        print("WARNING: Could not find --ot-soft-assign in llama.py, trying alternative insertion")
        # Insert before args = parser.parse_args()
        old_parse = "args = parser.parse_args()"
        new_parse = (
            "parser.add_argument('--ot-adapt', action='store_true', help='Enable OT-adaptive codebook during GPTQ')\n"
            "    parser.add_argument('--ot-adapt-interval', type=int, default=8, help='Adapt codebook every N columns')\n"
            "    parser.add_argument('--ot-adapt-strength', type=float, default=0.2, help='OT adaptation blending strength')\n"
            "    parser.add_argument('--ot-adapt-progressive', action='store_true', help='Progressive adaptation strength')\n"
            "\n"
            "    args = parser.parse_args()"
        )
        llama_code = llama_code.replace(old_parse, new_parse)

    # 3b. Pass args to fasterquant call
    if "ot_soft_assign=args.ot_soft_assign," in llama_code:
        old_call = "ot_soft_assign=args.ot_soft_assign,"
        new_call = (
            "ot_soft_assign=args.ot_soft_assign,\n"
            "                use_ot_adapt=args.ot_adapt,\n"
            "                ot_adapt_interval=args.ot_adapt_interval,\n"
            "                ot_adapt_strength=args.ot_adapt_strength,\n"
            "                ot_adapt_progressive=args.ot_adapt_progressive,"
        )
        llama_code = llama_code.replace(old_call, new_call)
    else:
        # Try finding the fasterquant call with a regex
        match = re.search(r'(use_ot_sinkhorn=[^,]+,)', llama_code)
        if match:
            old_call = match.group(1)
            new_call = (
                old_call + "\n"
                "                use_ot_adapt=args.ot_adapt,\n"
                "                ot_adapt_interval=args.ot_adapt_interval,\n"
                "                ot_adapt_strength=args.ot_adapt_strength,\n"
                "                ot_adapt_progressive=args.ot_adapt_progressive,"
            )
            llama_code = llama_code.replace(old_call, new_call)
        else:
            print("ERROR: Could not find fasterquant call to patch in llama.py")
            sys.exit(1)

    with open(llama_path, "w") as f:
        f.write(llama_code)
    print("llama.py: added --ot-adapt CLI args and fasterquant passthrough")

print("\n=== All patches applied successfully! ===")
print("\nUsage:")
print("  python llama.py ... --kmeans-init-method dp --ot-adapt --ot-adapt-interval 8 --ot-adapt-strength 0.2")
print("  python llama.py ... --kmeans-init-method dp --ot-adapt --ot-adapt-progressive --ot-adapt-strength 0.3")
