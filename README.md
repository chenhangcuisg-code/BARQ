# BARQ: Balanced Codebook Refinement for Low-Bit LLM Quantization

Code release for **BARQ**, by Chenhang Cui, Xu Xie, Linrui Xu, Xiaohao Liu, Xingyu Zhu, Fei Shen, and Tat-Seng Chua.

BARQ refines a vector quantization codebook with curvature-weighted balanced entropic assignments and barycentric updates. The refined codebook is then used with hard nearest-codeword encoding. The implementation builds on GPTVQ and GPTQ; their notices and licenses are preserved.

## Installation

The recorded experiment environment uses Python 3.10, PyTorch 2.4.1 with CUDA 12.1, Transformers 4.53.2, and datasets 2.20.0. Full LLM quantization requires an NVIDIA GPU and access to the selected model checkpoint.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

For gated Hugging Face models, authenticate with Hugging Face or set `HF_TOKEN` in your environment. Do not put tokens in source files. `MODEL` may be a Hugging Face model identifier or a local checkpoint directory.

## Quantize a model

```bash
bash scripts/quantize.sh Qwen/Qwen3-1.7B outputs/qwen3-1.7b-barq
```

The script invokes the original quantization entry point with uniform balanced marginals, epsilon 0.001, and 30 Sinkhorn iterations. It exposes the vector dimension, group size, columns per group, and codebook bit width through environment variables. For example:

```bash
VQ_DIM=4 GROUPSIZE=64 COLUMNS_PER_GROUP=256 CODEBOOK_BITS=8 \
  bash scripts/quantize.sh Qwen/Qwen3-1.7B outputs/qwen3-1.7b-barq
```

Vector dimension 4 and an 8-bit codebook index correspond to a nominal 2-bit index payload per weight. Actual storage also includes codebooks, scales, and metadata; nominal index bits are not the total stored bits per weight. The saved Hugging Face checkpoint contains reconstructed weights for evaluation and is not a packed low-bit deployment format.

For the paired GPTVQ baseline, run the same command with `METHOD=baseline`. The default WikiText-2 loader samples calibration sequences from the training split and evaluates perplexity on the test split. The defaults are 128 calibration sequences of 2048 tokens; consult the source arguments when matching a particular experiment configuration.

## Evaluate a saved checkpoint

```bash
bash scripts/evaluate.sh outputs/qwen3-1.7b-barq outputs/qwen3-1.7b-eval
```

This runs zero-shot ARC-Easy, ARC-Challenge, HellaSwag, PIQA, and WinoGrande using lm-evaluation-harness. The quantization entry point also computes WikiText-2 perplexity. The archived environment notes report that some lm-eval 0.4.11 installations with Transformers 4.53.2 need `dtype=get_dtype(dtype)` changed to `torch_dtype=get_dtype(dtype)` in the Hugging Face backend; check your installed backend if model loading rejects `dtype`.

## CPU smoke check

```bash
python tests/smoke_refinement.py
```

This checks the refinement on small synthetic tensors: finite outputs, correct dimensions, exact recovery of identical input vectors, and valid hard nearest-codeword encoding. Full GPU quantization and the paper's benchmark matrix were not rerun as part of this release.

## Repository layout

| Path | Contents |
|---|---|
| `code/vq_quant.py` | Codebook fitting and `sinkhorn_em_hessian_weighted` |
| `code/gptq.py` | GPTQ weight compensation and vector quantization integration |
| `code/llama.py` | Model loading, quantization, perplexity, and checkpoint saving |
| `code/datautils.py` | Calibration and perplexity dataset loaders |
| `code/ot_*.py` | Supporting transport routines and optional experimental variants |
| `scripts/` | Portable quantization and evaluation commands |
| `results/results_table.md` | Archived experiment summary table |
| `tests/` | Small CPU validation |

The original development repository calls the BARQ path "Sinkhorn-EM". Its CLI flags are retained for compatibility. Optional prior/mixed marginals and transport adaptation flags are experimental alternatives; the paper configuration uses `--sinkhorn-em --sinkhorn-em-marginal uniform`.

The release is based on development snapshot `cee9e4e`. It includes the implementation and the archived summary table; machine-specific scheduler scripts, development history, model weights, raw per-run logs, and private working files are excluded. The summary table contains additional development settings beyond the final paper, and is not a claim that every row was rerun or that every launch script is available.

## Citation

```bibtex
@misc{cui2026barq,
  title={BARQ: Balanced Codebook Refinement for Low-Bit LLM Quantization},
  author={Chenhang Cui and Xu Xie and Linrui Xu and Xiaohao Liu and Xingyu Zhu and Fei Shen and Tat-Seng Chua},
  year={2026},
  howpublished={\url{https://github.com/chenhangcuisg-code/BARQ}}
}
```

## License and acknowledgments

BARQ additions use BSD-3-Clause-Clear (`LICENSE-BARQ`). The upstream GPTVQ license is in `LICENSE`, and GPTQ's Apache-2.0 license is in `licenses/`. See `THIRD_PARTY_NOTICES.md` for attribution. Models and datasets retain their own licenses.
