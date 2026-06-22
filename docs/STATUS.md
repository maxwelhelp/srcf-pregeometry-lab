# SRCF status

Current fix: v5.2 model + v5 hard benchmark.

## Why this patch exists

Previous v5 model crashed in DNA mode with `NameError: csv_file is not defined`. Fixed.

Previous hard anomaly benchmark reported very low calibrated AUROC. This exposed two issues:

1. v4 benchmark mixed batch-level intrinsic metrics into per-sample anomaly features.
2. Some anomalies are not high-instability; they are over-stable / over-contractive. Therefore the benchmark now reports high-tail, low-tail and best separability AUROC.

## What matters

Training:
- `contract < 1`
- `h_contract < 1`
- `far_keep >= 0.75`
- `move > 0.12`
- `curve` decays
- `state_var` should not collapse

Benchmark:
- `high` means anomaly score high = anomaly.
- `low` means inverted direction; anomaly score low = anomaly.
- `best` is separability only, useful for research but not a deployable unsupervised anomaly score by itself.

Goal: make `calibrated_closure_high` beat `embedding_dist_high` and `raw_summary_dist_high` on hard anomalies. If only `low/best` is high, closure dynamics separates the data but the anomaly scoring direction is not solved yet.
