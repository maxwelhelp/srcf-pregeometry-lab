# SRCF Pregeometry Lab

Self-Referential Closure Fields (SRCF): coordinate-free relation tensors, recurrent self-application, and basin-contraction / closure diagnostics.

## Key files

- `self_referential_closure_field_v5_basin_dna.py` — main SRCF training file. Synthetic + DNA k-mer relation states. Writes CSV metrics.
- `srcf_benchmark_v4_hard.py` — hard calibrated anomaly/DNA benchmarks with calibrated closure scores and baselines.
- `pregeometric_self_query_field_v2.py` — supervised coordinate-free self-query prototype.
- `field_archs_v2_diagnostic.py` — diagnostic demos/baselines.

## Current rules for judging runs

Training is good only if these hold together:

- `contract < 1.0`
- `h_contract < 1.0`
- `far_keep >= 0.75-0.80`
- `move > 0.12`
- curve decays over iterations
- `state_var` does not collapse, roughly `>= 0.20`
- `eff_ops` does not collapse into 1-2 ops
- `perm ~ 1e-6`

Benchmark is interesting only if calibrated closure beats simple baselines:

- `calibrated_closure > embedding_dist`
- `calibrated_closure > raw_summary_dist`

If `embedding_dist == 1.0`, the anomaly task is still too easy.

## Commands

### DNA training

```bash
python -u self_referential_closure_field_v5_basin_dna.py \
  --device cuda --amp fp16 \
  --steps 300 --batch-size 4 --eval-batch-size 4 \
  --data dna --dna-bases 200000 --dna-window 2048 --kmer 3 \
  --dim 48 --ops 8 --iters 6 --eval-every 25 \
  --metrics-csv results/srcf_v5_dna_metrics.csv \
  --save-path ./srcf_v5_dna.pt | tee srcf_v5_dna.log
```

### Synthetic training

```bash
python -u self_referential_closure_field_v5_basin_dna.py \
  --device cuda --amp fp16 \
  --steps 300 --batch-size 8 --eval-batch-size 8 \
  --data synthetic --n 32 --ood-n 48 \
  --dim 48 --ops 8 --iters 6 --eval-every 25 \
  --metrics-csv results/srcf_v5_synth_metrics.csv \
  --save-path ./srcf_v5_synth.pt | tee srcf_v5_synth.log
```

### Hard anomaly benchmark

```bash
python -u srcf_benchmark_v4_hard.py \
  --device cuda --amp fp16 \
  --task anomaly --pretrain-steps 150 \
  --batch 8 --eval-batch 8 \
  --n 32 --dim 48 --ops 8 --iters 6 \
  --results-csv results/summary.csv | tee srcf_benchmark_v4_anomaly.log
```

### DNA benchmark

```bash
python -u srcf_benchmark_v4_hard.py \
  --device cuda --amp fp16 \
  --task dna --pretrain-steps 150 \
  --batch 4 --eval-batch 4 \
  --n 64 --dim 48 --ops 8 --iters 6 \
  --dna-bases 200000 --dna-window 2048 --kmer 3 \
  --results-csv results/summary.csv | tee srcf_benchmark_v4_dna.log
```
