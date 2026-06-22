#!/usr/bin/env bash
set -euo pipefail
python -u closure_relation_diagnostics_v1.py \
  --mode nongraph --device cuda --amp fp16 \
  --K-list 2,4 --seeds 0 --steps 120 \
  --batch 24 --eval-batch 48 --eval-batches 2 \
  --n 32 --dim 48 --hidden 96 --iters 5 \
  --rel-mode raw --text-root . --eval-every 30 \
  --results-csv results/closure_relation_nongraph_v1.csv \
  | tee closure_relation_nongraph_v1.log
