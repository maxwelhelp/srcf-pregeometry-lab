#!/usr/bin/env bash
set -euo pipefail
python -u self_referential_closure_field_v6_basin_dna.py \
  --device cuda --amp fp16 \
  --steps 300 --batch-size 4 --eval-batch-size 4 \
  --data dna --dna-bases 200000 --dna-window 2048 --kmer 3 \
  --dim 48 --ops 8 --iters 6 --eval-every 25 \
  --metrics-csv results/srcf_v6_dna_metrics.csv \
  --save-path ./srcf_v6_dna.pt \
  | tee srcf_v6_dna.log
