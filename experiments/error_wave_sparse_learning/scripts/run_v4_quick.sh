#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
mkdir -p experiments/error_wave_sparse_learning/results experiments/error_wave_sparse_learning/logs
python -u experiments/error_wave_sparse_learning/error_wave_sparse_learning_v4_weightlevel.py \
  --device cuda \
  --amp fp16 \
  --seeds 0,1,2 \
  --pretrain-steps 400 \
  --patch-steps 200 \
  --batch 128 \
  --mask-batch 128 \
  --eval-batch 512 \
  --eval-every 100 \
  --contexts 4 \
  --block-dim 12 \
  --hidden 96 \
  --layers 4 \
  --lr 2e-3 \
  --wave-frac 0.25 \
  --patch-wave-frac 0.08 \
  --patch-wave-frac-end 0.04 \
  --random-patch-density 0.014 \
  --input-frac 0.35 \
  --context-blame \
  --retain-penalty 0.5 \
  --wave-ema 0.90 \
  --selectivity-power 1.0 \
  --activity-threshold 0.05 \
  --weight-level-first \
  --frac-decay \
  --fisher-guard 0.0 \
  --results-csv experiments/error_wave_sparse_learning/results/error_wave_sparse_learning_v4_weightlevel.csv \
  | tee experiments/error_wave_sparse_learning/logs/error_wave_sparse_learning_v4_weightlevel.log
