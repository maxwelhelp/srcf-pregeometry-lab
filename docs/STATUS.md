# SRCF status

## Observed so far

- PG-SQF supervised toy tasks solve quickly, but the benchmark is too easy.
- Older SRCF anomaly benchmark showed high instability AUROC, but embedding distance was also perfect, so it did not prove unique value.
- SRCF v4 DNA training showed real basin contraction: contract and h_contract below 1 with decaying curves.
- v5 fixes CSV logging, version printing, DNA k-mer indexing, far-preservation and variance guards.

## Main risk

A model can learn damping or identity-like stability instead of true basin formation. Therefore every run must track identity/random baseline, contract/h_contract, far_keep, move, curve, state_var, and calibrated benchmarks against embedding/raw baselines.
