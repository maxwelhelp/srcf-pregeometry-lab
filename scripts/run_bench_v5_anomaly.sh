#!/usr/bin/env bash
set -euo pipefail
python -u srcf_benchmark_v5_hard.py \
  --device cuda --amp fp16 \
  --task anomaly --pretrain-steps 150 \
  --batch 8 --eval-batch 8 \
  --n 32 --dim 48 --ops 8 --iters 6 \
  --results-csv results/summary.csv \
  | tee srcf_benchmark_v5_anomaly.log
