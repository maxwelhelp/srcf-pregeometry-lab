# SRCF Pregeometry Lab

Self-Referential Closure Fields (SRCF) and coordinate-free relation learning experiments.

Core idea:

- coordinates are not inputs; relation tensors are the substrate;
- the same learned operator is applied to its own state repeatedly;
- near perturbations should converge into the same basin;
- unrelated states should stay distinct;
- geometry/topology is treated as an output/hypothesis, not as a required prior.

## Main files

- `self_referential_closure_field_v5_basin_dna.py` — main SRCF model/trainer, synthetic + DNA k-mer relation states, CSV metrics.
- `srcf_benchmark_v4_hard.py` — hard/calibrated anomaly and DNA benchmarks.
- `pregeometric_self_query_field_v2.py` — supervised coordinate-free self-query prototype.
- `field_archs_v2_diagnostic.py` — diagnostic demos/baselines.

## Current status

See `docs/STATUS.md` and `results/summary_seed.md`.

Main conclusion so far:

- PG-SQF supervised toy benchmark works, but is too easy.
- SRCF-v4/v5 DNA training shows real basin contraction (`contract < 1`, `h_contract < 1`) without labels.
- Old anomaly benchmarks were too easy or incorrectly scored; `embedding_dist=1.0` means the synthetic anomaly set was not hard enough.
- New benchmark uses calibrated closure metrics and harder near-normal anomalies.

## Quick commands

Synthetic SRCF training:

```bash
python -u self_referential_closure_field_v5_basin_dna.py \
  --device cuda --amp fp16 \
  --steps 300 --batch-size 8 --eval-batch-size 8 \
  --data synthetic --n 32 --ood-n 48 \
  --dim 48 --ops 8 --iters 6 --eval-every 25 \
  --metrics-csv results/srcf_v5_synth_metrics.csv \
  --save-path ./srcf_v5_synth.pt | tee srcf_v5_synth.log
```

DNA SRCF training:

```bash
python -u self_referential_closure_field_v5_basin_dna.py \
  --device cuda --amp fp16 \
  --steps 300 --batch-size 4 --eval-batch-size 4 \
  --data dna --dna-bases 200000 --dna-window 2048 --kmer 3 \
  --dim 48 --ops 8 --iters 6 --eval-every 25 \
  --metrics-csv results/srcf_v5_dna_metrics.csv \
  --save-path ./srcf_v5_dna.pt | tee srcf_v5_dna.log
```

Hard anomaly benchmark:

```bash
python -u srcf_benchmark_v4_hard.py \
  --device cuda --amp fp16 \
  --task anomaly --pretrain-steps 150 \
  --batch 8 --eval-batch 8 \
  --n 32 --dim 48 --ops 8 --iters 6 \
  --results-csv results/summary.csv | tee srcf_benchmark_v4_anomaly.log
```

DNA benchmark:

```bash
python -u srcf_benchmark_v4_hard.py \
  --device cuda --amp fp16 \
  --task dna --pretrain-steps 150 \
  --batch 4 --eval-batch 4 \
  --n 64 --dim 48 --ops 8 --iters 6 \
  --dna-bases 200000 --dna-window 2048 --kmer 3 \
  --results-csv results/summary.csv | tee srcf_benchmark_v4_dna.log
```
