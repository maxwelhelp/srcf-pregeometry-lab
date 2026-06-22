#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -u srcf_realdata_benchmark_v1.py \
  --device cuda --amp fp16 \
  --dataset all \
  --pretrain-steps 200 \
  --batch 16 \
  --eval-batch 32 \
  --dim 48 --ops 8 --iters 6 \
  --results-csv results/realdata_summary.csv \
  | tee srcf_realdata_v1.log
