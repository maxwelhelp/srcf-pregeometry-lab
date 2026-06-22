# Seed results from chat logs

These are manual seed results used to guide code changes. New runs should append machine-readable data to `results/summary.csv` or per-run metrics CSVs.

## SRCF benchmark v2 anomaly

```text
pretrain 300/300: loss=0.1583 contract=1.43 move=0.176
AUROC:
  combo          : 0.9392
  instability    : 0.9951
  recovery       : 0.9901
  contract_fail  : 0.6556
  move_inv       : 0.2313
  embedding_dist : 1.0000
  random         : 0.5179
```

Conclusion: useful closure signal exists, but benchmark too easy because `embedding_dist=1.0`.

## SRCF benchmark v3 basin anomaly

```text
pretrain 150/150: loss=0.3376 contract=0.73 h_contract=0.51 move=0.496
AUROC:
  combo          : 0.5555
  instability    : 0.5730
  recovery       : 0.5680
  contract_fail  : 0.5213
  move_inv       : 0.5554
  embedding_dist : 1.0000
```

Conclusion: basin dynamics works, but simple high-instability score fails because some anomalies are abnormally over-stable/collapsed.

## SRCF v4 DNA training

```text
step 25:  contract=0.65 h_contract=0.69 far_keep=1.05 move=0.368
step 50:  contract=0.55 h_contract=0.69 far_keep=0.92 move=0.341
step 100: contract=0.54 h_contract=0.72 far_keep=0.63 move=0.356
step 125: contract=0.52 h_contract=0.74 far_keep=0.69 move=0.359
step 150: contract=0.71 h_contract=0.73 far_keep=0.83 move=0.383
step 175: contract=0.66 h_contract=0.76 far_keep=0.62 move=0.377
```

Conclusion: real DNA k-mer relation training is promising; v5 strengthens far preservation.
