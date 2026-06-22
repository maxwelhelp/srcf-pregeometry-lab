# SRCF research status

## Why this exists

The project tests whether a coordinate-free relation system can learn useful self-referential closure dynamics:

```text
R0 relation tensor -> encoder -> repeated self-application -> H*
```

No supervised labels are used in SRCF training.

## Metrics

Good SRCF training should show:

- `contract < 1.0`: near perturbations are closer after self-application than before.
- `h_contract < 1.0`: hidden states also contract, not only descriptors.
- `far_keep >= 0.7-0.8`: unrelated states do not collapse together.
- `move > 0.12`: not a lazy identity map.
- `curve` decays: early motion is larger than late motion.
- `state_var` and `desc_var` do not collapse.
- `perm ~ 1e-6`: no hidden coordinate/order leakage.

## Observed results so far

### Old benchmark v2

Old synthetic anomaly benchmark produced strong AUROC:

```text
instability    : 0.9951
recovery       : 0.9901
combo          : 0.9392
embedding_dist : 1.0000
```

Interpretation: closure instability is a strong signal, but the dataset was too easy because ordinary embedding distance solved it perfectly.

### Basin benchmark v3

Training metrics improved:

```text
pretrain 150/150: contract=0.73 h_contract=0.51 move=0.496
```

But anomaly AUROC became weak because some anomalies are too stable/collapsed rather than unstable:

```text
combo=0.5555 instability=0.5730 embedding_dist=1.0000
rank1 instab=0.0000
broken_blocks instab=0.0000
```

Interpretation: anomaly scoring must be two-sided/calibrated, not simply “high instability = anomaly”.

### DNA v4 training

Real E. coli 3-mer relation states showed promising self-supervised basin contraction:

```text
step 25:  contract=0.65 h_contract=0.69 move=0.368 far_keep=1.05
step 50:  contract=0.55 h_contract=0.69 move=0.341 far_keep=0.92
step 100: contract=0.54 h_contract=0.72 move=0.356 far_keep=0.63
step 125: contract=0.52 h_contract=0.74 move=0.359 far_keep=0.69
step 150: contract=0.71 h_contract=0.73 move=0.383 far_keep=0.83
step 175: contract=0.66 h_contract=0.76 move=0.377 far_keep=0.62
```

Interpretation: basin contraction works on real DNA relation states. Main risk: `far_keep` sometimes drops below 0.7, so v5 strengthens far preservation and variance guards.

## Current next checks

1. Run `self_referential_closure_field_v5_basin_dna.py` on DNA and synthetic.
2. Run `srcf_benchmark_v4_hard.py` to test calibrated closure scores against embedding and raw-summary baselines.
3. The key target is not just high AUROC; the key target is:

```text
calibrated_closure > embedding_dist and raw_summary_dist
```

on hard near-normal anomalies.
