#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -u closure_basin_memory_benchmark_v1.py \
  --device cuda --amp fp16 \
  --K-list 2,4,8 \
  --seeds 0,1,2 \
  --steps 180 \
  --batch 32 \
  --eval-batch 64 \
  --eval-batches 3 \
  --n 32 \
  --dim 48 \
  --hidden 96 \
  --deep-hidden 256 \
  --deep-depth 4 \
  --iters 5 \
  --memory-slots 16 \
  --memory-beta 1.5 \
  --memory-strength 0.12 \
  --late-memory \
  --eval-every 60 \
  --results-csv results/closure_basin_memory_v1_3seed.csv \
  | tee closure_basin_memory_v1_3seed.log
