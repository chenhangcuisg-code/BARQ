#!/usr/bin/env bash
set -euo pipefail
checkpoint="${1:?Usage: bash scripts/evaluate.sh CHECKPOINT OUTPUT_DIR}"
output="${2:?Provide OUTPUT_DIR}"
lm_eval --model hf --model_args "pretrained=$checkpoint,dtype=bfloat16" \
  --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande --num_fewshot 0 \
  --device cuda:0 --batch_size auto --output_path "$output"
