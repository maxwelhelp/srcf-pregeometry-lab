#!/usr/bin/env bash
set -euo pipefail
python -u closure_graph_real_benchmark_v3_triangle.py \
  --device cuda --amp fp16 \
  --experiment per_graph \
  --seeds 0 \
  --steps 120 \
  --batch 32 --eval-batch 32 --eval-batches 2 \
  --n 32 --dim 48 --hidden 64 \
  --big-hidden 128 --big-depth 4 \
  --iters 5 --eval-iters 1,2,3,5 \
  --rel-mode raw --ablate-tri \
  --eval-every 30 \
  --results-csv results/closure_graph_real_v3_raw_ablate.csv \
  | tee closure_graph_real_v3_raw_ablate.log
