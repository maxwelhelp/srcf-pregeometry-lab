#!/usr/bin/env bash
set -euo pipefail
python -u self_referential_closure_field_v5_basin_dna.py \
  --device cuda --amp fp16 \
  --steps 300 --batch-size 8 --eval-batch-size 8 \
  --data synthetic --n 32 --ood-n 48 \
  --dim 48 --ops 8 --iters 6 --eval-every 25 \
  --metrics-csv results/srcf_v5_synth_metrics.csv \
  --save-path ./srcf_v5_synth.pt | tee srcf_v5_synth.log
