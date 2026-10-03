#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model="${1:?Usage: bash scripts/quantize.sh MODEL OUTPUT_DIR}"
output="${2:?Provide OUTPUT_DIR}"
extra=()
case "${METHOD:-barq}" in
  barq) extra+=(--sinkhorn-em --sinkhorn-em-eps "${EPSILON:-0.001}" --sinkhorn-em-iters "${SINKHORN_ITERS:-30}" --sinkhorn-em-marginal uniform) ;;
  baseline) ;;
  *) echo "METHOD must be barq or baseline" >&2; exit 2 ;;
esac
python code/llama.py "$model" wikitext2 \
  --model-type auto --use-vq --wbits 2 \
  --vq-dim "${VQ_DIM:-4}" --groupsize "${GROUPSIZE:-64}" \
  --columns-per-group "${COLUMNS_PER_GROUP:-256}" \
  --codebook-bitwidth "${CODEBOOK_BITS:-8}" \
  --quantize-per-codebook --kmeans-init-method mahalanobis --kmeans-iters "${KMEANS_ITERS:-100}" \
  --hessian-weighted-lookups "${extra[@]}" --output-dir "$output"
