#!/usr/bin/env bash
set -euo pipefail
python -u closure_graph_real_benchmark_v4_memory.py \
  --device cuda --amp fp16 \
  --experiment per_graph \
  --seeds 0 \
  --steps 120 \
  --batch 32 --eval-batch 32 --eval-batches 2 \
  --n 32 --dim 48 --hidden 64 \
  --big-hidden 128 --big-depth 4 \
  --iters 5 --eval-iters 1,2,3,5 \
  --use-memory --ablate-memory --memory-slots 16 --memory-beta 2.0 --memory-strength 0.20 \
  --eval-every 30 \
  --results-csv results/closure_graph_real_v4_memory_quick.csv \
  | tee closure_graph_real_v4_memory_quick.log
