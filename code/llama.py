# ===== BEGIN save_patch_v3 (DO NOT DUPLICATE) =====
import importlib.util as _ds_impu
_ds_orig_find_spec = _ds_impu.find_spec  # bind once
def _ds_patched_find_spec(name, *a, **kw):
    if name == "deepspeed" or (isinstance(name, str) and name.startswith("deepspeed.")):
        return None
    return _ds_orig_find_spec(name, *a, **kw)
# Guard: only patch once even if re-imported
if not getattr(_ds_impu.find_spec, "_is_ds_patch_v3", False):
    _ds_patched_find_spec._is_ds_patch_v3 = True
    _ds_impu.find_spec = _ds_patched_find_spec
# ===== END save_patch_v3 =====

# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from https://github.com/IST-DASLab/gptq
# Copyright 2022 IST-DASLab, Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, Not a Contribution

import os
import time
import json
import gc

import torch
import torch.nn as nn

import transformers

from gptq import *
from modelutils import *
from quant import *
from vq_quant import *

HF_TOKEN = os.environ.get("HF_TOKEN")  # Optional Hugging Face access token

def get_llama(model, model_type):
    import torch

    def skip(*_, **__):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    token_auth_kwargs = {}
    if HF_TOKEN is not None:
        token_auth_kwargs["use_auth_token"] = HF_TOKEN

    local_kwargs = {}
    if os.path.isdir(model):
        local_kwargs["local_files_only"] = True
    token_auth_kwargs.update(local_kwargs)

    if model_type == "mistral":
        from transformers import MistralForCausalLM

        model = MistralForCausalLM.from_pretrained(model, torch_dtype="auto", **token_auth_kwargs)
    elif model_type == "mixtral":
        from transformers import MixtralForCausalLM

        model = MixtralForCausalLM.from_pretrained(model, torch_dtype="auto", **token_auth_kwargs)
    elif model_type == "auto":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", **token_auth_kwargs)
    else:
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto", **token_auth_kwargs)
    model.seqlen = 2048
    return model

@torch.no_grad()
def llama_sequential(model, dataloader, dev, args):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache.get("position_ids", None)

    if args.use_vq:
        QClass = lambda: VQQuantizer(
            vq_dim=args.vq_dim,
            columns_per_group=args.columns_per_group,
            vq_scaling_blocksize=args.vq_scaling_blocksize,
            vq_scaling_norm=args.vq_scaling_norm,
            vq_scaling_n_bits=args.vq_scaling_n_bits,
            vq_scaling_domain=args.vq_scaling_domain,
            kmeans_init_method=args.kmeans_init_method,
            assignment_chunk_size=args.assignment_chunk_size,
            kmeans_iters=args.kmeans_iters,
            codebook_bitwidth=args.codebook_bitwidth,
            quantize_per_codebook=args.quantize_per_codebook,
            quantize_during_kmeans=args.quantize_during_kmeans,
            n_subsample=args.kpp_n_subsample,
            use_ot_transfer=args.use_ot_transfer,
            use_ot_refine=args.use_ot_refine,
        )
    else:
        QClass = Quantizer

    print("Ready.")

    nlayers = 1 if getattr(args, "debug_one_layer", False) else len(layers)
    if getattr(args, "debug_one_layer", False):
        print(f"[DEBUG] 只量化第 0 层 (共 {len(layers)} 层)")

    quantizers = {}
    for i in range(nlayers):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            # rpg_min: rows-per-row-subgroup floor used to filter layers whose out_features
            # can't be cleanly split. Original formula gs//cpg returns 0 when cpg > gs (which
            # is paper's vd=4 g=128 cpg=1024 setting). Clamp to 1 in that regime so the
            # divisibility check is a no-op (no spurious filtering).
            rpg_min = max(1, args.groupsize // args.columns_per_group) if (args.use_vq and args.columns_per_group is not None) else 1
            _skipped = [k for k, v in full.items() if v.weight.shape[0] % rpg_min != 0]
            if _skipped:
                print(f"[SKIP] layers with out_features not divisible by rpg={rpg_min}: {_skipped}")
            sequential = [[k for k, v in full.items() if "block_sparse_moe.gate" not in k and v.weight.shape[0] % rpg_min == 0]]

        for names in sequential:

            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = QClass()
                gptq[name].post_mstep_ot_enable = args.post_mstep_ot_adapt
                gptq[name].post_mstep_ot_alpha = args.post_mstep_ot_alpha
                gptq[name].sinkhorn_em_enable = args.sinkhorn_em
                gptq[name].sinkhorn_em_eps = args.sinkhorn_em_eps
                gptq[name].sinkhorn_em_iters = args.sinkhorn_em_iters
                gptq[name].sinkhorn_em_marginal = args.sinkhorn_em_marginal
                gptq[name].sinkhorn_em_prior_alpha = args.sinkhorn_em_prior_alpha
                gptq[name].sinkhorn_em_mix_lambda = args.sinkhorn_em_mix_lambda
                gptq[name].sinkhorn_em_entropy_gate = args.sinkhorn_em_entropy_gate
                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                seq = inps[j].unsqueeze(0)
                seq_len = seq.shape[1]
                device = seq.device
                cache_position = torch.arange(seq_len, device=device)
                pos_ids = position_ids
                if pos_ids is None:
                    pos_ids = cache_position.unsqueeze(0)
                position_embeddings = model.model.rotary_emb(seq, pos_ids)
                outs[j] = layer(
                    seq,
                    attention_mask=attention_mask,
                    position_ids=pos_ids,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Quantizing ...")
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    static_groups=args.static_groups,
                    include_m_step=args.include_m_step,
                    use_vq=args.use_vq,
                    svd_rank=args.svd_rank,
                    hessian_weighted_lookups=args.hessian_weighted_lookups,
                    only_init_kmeans=args.only_init_kmeans,
                    use_ot_sinkhorn=args.use_ot_sinkhorn,
                    sinkhorn_eps=args.sinkhorn_eps,
                    ot_soft_assign=args.ot_soft_assign,
                use_ot_adapt=args.ot_adapt,
                ot_adapt_interval=args.ot_adapt_interval,
                ot_adapt_strength=args.ot_adapt_strength,
                ot_adapt_progressive=args.ot_adapt_progressive,
                )
                _q = gptq[name].quantizer
                def _to_cpu_recursive(_obj):
                    if isinstance(_obj, torch.Tensor):
                        return _obj.detach().cpu() if _obj.is_cuda else _obj
                    elif isinstance(_obj, list):
                        return [_to_cpu_recursive(_x) for _x in _obj]
                    elif isinstance(_obj, dict):
                        return {_k: _to_cpu_recursive(_v) for _k, _v in _obj.items()}
                    elif isinstance(_obj, tuple):
                        return tuple(_to_cpu_recursive(_x) for _x in _obj)
                    return _obj
                for _a in list(vars(_q).keys()):
                    _v = getattr(_q, _a, None)
                    if _v is None: continue
                    setattr(_q, _a, _to_cpu_recursive(_v))
                quantizers["model.layers.%d.%s" % (i, name)] = _q
                gptq[name].free()
                del _q

        for j in range(args.nsamples):
            seq = inps[j].unsqueeze(0)
            seq_len = seq.shape[1]
            device = seq.device
            cache_position = torch.arange(seq_len, device=device)
            pos_ids = position_ids
            if pos_ids is None:
                pos_ids = cache_position.unsqueeze(0)
            position_embeddings = model.model.rotary_emb(seq, pos_ids)
            outs[j] = layer(
                seq,
                attention_mask=attention_mask,
                position_ids=pos_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]

        layers[i] = layer.cpu()
        del layer
        del gptq, subset
        for _ in range(2):
            gc.collect()
        for _di in range(torch.cuda.device_count()):
            with torch.cuda.device(_di):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        _mem = [torch.cuda.memory_allocated(_di)/1e9 for _di in range(torch.cuda.device_count())]
        print(f"[mem] after layer {i}: GPU0={_mem[0]:.2f}GB GPU1={_mem[1] if len(_mem)>1 else 0:.2f}GB", flush=True)

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev, no_quant):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache.get("position_ids", None)

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest and not no_quant:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                orig_shape = W.shape
                if args.groupsize > -1:
                    W = W.view(-1, args.groupsize)

                quantizer.find_params(W, weight=True)
                subset[name].weight.data = (
                    quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq)
                    .to(next(iter(layer.parameters())).dtype)
                    .view(orig_shape)
                )

        for j in range(nsamples):
            seq = inps[j].unsqueeze(0)
            seq_len = seq.shape[1]
            device = seq.device
            cache_position = torch.arange(seq_len, device=device)
            pos_ids = position_ids
            if pos_ids is None:
                pos_ids = cache_position.unsqueeze(0)
            position_embeddings = model.model.rotary_emb(seq, pos_ids)
            outs[j] = layer(
                seq,
                attention_mask=attention_mask,
                position_ids=pos_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=str,
        help="LlaMa model to load; pass location of hugginface converted checkpoint.",
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--nearest", action="store_true", help="Whether to run the RTN baseline.")
    parser.add_argument(
        "--wbits",
        type=float,
        default=16,
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--sym", action="store_true", help="Whether to perform symmetric quantization."
    )
    parser.add_argument(
        "--save", type=str, default="", help="Save quantized checkpoint under this name."
    )
    parser.add_argument(
        "--new-eval", action="store_true", help="Whether to use the new PTB and C4 eval."
    )
    parser.add_argument(
        "--no-quant", action="store_true", help="If set, run FP16 model without quantization"
    )
    parser.add_argument(
        "--act-order",
        action="store_true",
        help="Whether to apply the activation order GPTQ heuristic",
    )
    parser.add_argument(
        "--true-sequential", action="store_true", help="Whether to run in true sequential model."
    )
    parser.add_argument(
        "--static-groups",
        action="store_true",
        help="Whether to use static groups; recommended when using `--actorder` for more efficient inference.",
    )
    parser.add_argument(
        "--use-vq", action="store_true", help="If set, use VQ (multi-dim non-uniform) quantization"
    )
    parser.add_argument("--vq-dim", type=int, default=2, help="Dimensionality of VQ (if using)")
    parser.add_argument(
        "--vq-scaling-blocksize", type=int, default=-1, help="VQ scaling block size"
    )

    parser.add_argument("--vq-scaling-n-bits", type=int, default=4, help="VQ scaling bit-width")

    parser.add_argument("--vq-scaling-norm", type=str, default="max", help="VQ scaling norm")
    parser.add_argument(
        "--vq-scaling-domain",
        type=str,
        default="log",
        choices=["log", "linear"],
        help="VQ scaling domain",
    )

    parser.add_argument(
        "--include-m-step",
        action="store_true",
        help="If set, perform an M-step (centroid updating) after GPTQ with VQ",
    )
    parser.add_argument(
        "--columns-per-group",
        type=int,
        default=None,
        help="For group-/blockwise quant: force number of columns each group spans (rest is absorbed in rows)",
    )
    parser.add_argument(
        "--kmeans-init-method",
        type=str,
        default="cdf",
        choices=["cdf", "kpp", "mahalanobis", "wasserstein", "dp", "dp_wass", "dp_bary", "dp_bary_02", "dp_bary_05", "dp_mccann", "dp_mccann_02", "dp_mccann_10", "dp_bary_full"],
        help="init method for Kmeans",
    )
    parser.add_argument(
        "--assignment-chunk-size",
        type=int,
        default=None,
        help="Chunk assignment step for better memory management",
    )
    parser.add_argument("--kmeans-iters", type=int, default=10)
    parser.add_argument(
        "--codebook-bitwidth", type=int, default=None, help="Bitwidth for codebook quantization"
    )
    parser.add_argument(
        "--quantize-per-codebook",
        action="store_true",
        default=False,
        help="Quantize codebooks individually (more overhead) or per column block",
    )
    parser.add_argument(
        "--quantize-during-kmeans",
        action="store_true",
        default=False,
        help="Quantize codebooks after every M-step. If not set: only quantize after k-means",
    )
    parser.add_argument(
        "--model-type",
        choices=["llama", "mistral", "mixtral", "auto"],
        default="llama",
        help="In case this is a Mistral model (GPTQ layerwise remains the same)",
    )
    parser.add_argument("--kpp-n-subsample", type=int, default=10000)
    parser.add_argument("--svd-rank", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save model in")
    parser.add_argument("--hessian-weighted-lookups", action="store_true", default=False)
    parser.add_argument("--only-init-kmeans", action="store_true", default=False)
    parser.add_argument(
        "--use-ot-sinkhorn",
        action="store_true",
        default=False,
        help="OT-GPTQ: 使用 Sinkhorn 软分配替代硬最近邻（需 vq-dim=1）",
    )
    parser.add_argument("--sinkhorn-eps", type=float, default=0.05, help="Sinkhorn 熵正则化强度")
    parser.add_argument(
        "--ot-soft-assign",
        action="store_true",
        default=False,
        help="OT-GPTQ: 用软解码 q=Gamma@C 替代 argmax 硬取码（更稳定，但不再严格离散码本）",
    )
    parser.add_argument(
        "--use-ot-transfer",
        action="store_true",
        default=False,
        help="OT codebook transfer: warm-start each group from previous via 1D OT map",
    )
    parser.add_argument(
        "--use-ot-refine",
        action="store_true",
        default=False,
        help="OT dead codeword fix: ensure full codebook utilization via OT splitting",
    )
    parser.add_argument(
        "--debug-one-layer",
        action="store_true",
        default=False,
        help="测试模式：只量化第 0 层，快速走通流程",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="测试模式：等价于 --debug-one-layer --nsamples 4",
    )

    parser.add_argument('--ot-adapt', action='store_true', default=False,
        help='Enable OT-adaptive codebook during GPTQ error propagation')
    parser.add_argument('--ot-adapt-interval', type=int, default=8,
        help='Adapt codebook every N columns within a group')
    parser.add_argument('--ot-adapt-strength', type=float, default=0.2,
        help='OT adaptation blending strength alpha')
    parser.add_argument('--ot-adapt-progressive', action='store_true', default=False,
        help='Progressive adaptation: alpha grows with distance from group start')
    parser.add_argument('--post-mstep-ot-adapt', action='store_true', default=False,
        help='After m-step, do one pass of OT refinement on each group codebook')
    parser.add_argument('--post-mstep-ot-alpha', type=float, default=0.1,
        help='Blending strength for post-m-step OT refine (0 = no refine)')
    parser.add_argument('--sinkhorn-em', action='store_true', default=False,
        help='Use Sinkhorn-EM with Hessian-weighted cost to refine centroids after k-means')
    parser.add_argument('--sinkhorn-em-eps', type=float, default=0.05,
        help='Sinkhorn regularization (smaller = sharper coupling)')
    parser.add_argument('--sinkhorn-em-iters', type=int, default=30,
        help='Sinkhorn iterations')
    parser.add_argument('--sinkhorn-em-marginal', type=str, default='uniform',
        choices=['uniform', 'prior', 'mix'],
        help='Sinkhorn centroid marginal: uniform (orig), prior (usage-aware), mix (interpolated)')
    parser.add_argument('--sinkhorn-em-prior-alpha', type=float, default=4.0,
        help='Smoothing alpha for prior/mix marginal: a_prior = (n + alpha)/(N + alpha*K)')
    parser.add_argument('--sinkhorn-em-mix-lambda', type=float, default=1.0,
        help='Mix weight on uniform: a = (1-lambda)*a_prior + lambda*a_uniform (used when marginal=mix)')
    parser.add_argument('--sinkhorn-em-entropy-gate', type=float, default=None,
        help='Skip Sinkhorn for groups whose normalized usage entropy > this gate (e.g. 0.9). None = always run.')
    args = parser.parse_args()

    if args.debug:
        args.debug_one_layer = True
        args.nsamples = 4

    if not args.use_vq:
        args.wbits = int(args.wbits)

    model = get_llama(args.model, args.model_type)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest and not args.no_quant:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV, args)
        print(time.time() - tick)

    datasets = ["wikitext2"]
    if args.new_eval:
        datasets = ["wikitext2", "ptb-new", "c4-new"]
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama_eval(model, testloader, DEV, no_quant=args.no_quant)

    if args.output_dir is not None:
        from datautils import _load_tokenizer

        os.makedirs(args.output_dir, exist_ok=True)
        tokenizer = _load_tokenizer(args.model)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        quant_meta = {
            "use_vq": bool(args.use_vq),
            "use_ot_sinkhorn": bool(args.use_ot_sinkhorn),
            "ot_soft_assign": bool(args.ot_soft_assign),
            "sinkhorn_eps": float(args.sinkhorn_eps),
            "wbits": float(args.wbits),
            "vq_dim": int(args.vq_dim),
            "groupsize": int(args.groupsize),
            "columns_per_group": args.columns_per_group,
            "kmeans_init_method": args.kmeans_init_method,
            "only_init_kmeans": bool(args.only_init_kmeans),
        }
        with open(os.path.join(args.output_dir, "quant_config.json"), "w", encoding="utf-8") as f:
            json.dump(quant_meta, f, ensure_ascii=False, indent=2)
