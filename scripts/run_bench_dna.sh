#!/usr/bin/env bash
set -euo pipefail
python -u srcf_benchmark_v4_hard.py \
  --device cuda --amp fp16 \
  --task dna --pretrain-steps 150 \
  --batch 4 --eval-batch 4 \
  --n 64 --dim 48 --ops 8 --iters 6 \
  --dna-bases 200000 --dna-window 2048 --kmer 3 \
  --results-csv results/summary.csv | tee srcf_benchmark_v4_dna.log
