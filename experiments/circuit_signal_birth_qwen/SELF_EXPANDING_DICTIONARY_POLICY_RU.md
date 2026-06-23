# Self-Expanding Dictionary Policy для Exact Program Transplant

Этот документ уточняет главный конфликт:

```text
В проекте уже есть residual mining / target-mined atoms / self-expanding primitive dictionary.
Но Exact Program Transplant Doctrine запрещает learned dictionary as base dictionary.
```

Решение: разделить словари на уровни.

```text
Level 0: analytic_base_dictionary
Level 1: verified_extension_dictionary
Level 2: universal_promoted_dictionary
```

Саморасширяющийся словарь не запрещён. Запрещено только подсовывать его как базовый аналитический словарь и выдавать target-mined atoms за universal primitives без проверки.

---

## 1. Что у нас уже было

В текущих экспериментах Qwen circuit signal birth используется схема:

```text
base readable operator dictionary
-> decode M_qk/C_vo
-> residual
-> mine BirthOp candidates from residual by PCA/SVD
-> accept only if train/heldout target or functional metrics improve
```

То есть это уже не чисто аналитический словарь. Это residual-driven extension.

Для ParT full-model decoder в планах также явно записано:

```text
MatrixOpLibrary + SparseProgramDecoder + OMP
residual mining with target_mined atoms
shared primitive candidates with train/heldout gate
```

И отдельно было сформулировано важное правило:

```text
Target-mined atoms should be marked as target-specific, not universal.
```

Именно это правило надо сделать формальным.

---

## 2. Три уровня словаря

### 2.1 Level 0 — analytic_base_dictionary

Это единственный словарь, который можно использовать для первичного exact decode без риска hidden fitting.

Источник:

```text
архитектурные формулы
identity / shift / block average / bias column
RoPE-derived analytic templates
fixed DCT/Hadamard templates
ручные typed primitives с явной формулой
```

Запрещено:

```text
PCA по M_qk/C_vo текущей модели
SVD residual текущей модели
clustering heads текущего checkpoint
learning dictionary from checkpoints
```

Manifest:

```json
{
  "dictionary_level": "analytic_base_dictionary",
  "dictionary_source": "analytic_only",
  "depends_on_checkpoint_data": false,
  "depends_on_training_data": false,
  "eligible_for_exact_base_decode": true
}
```

### 2.2 Level 1 — verified_extension_dictionary

Это саморасширяющийся словарь.

Источник:

```text
residual после Level 0 decode
PCA/SVD/OMP/mined atoms
target_mined atoms
BirthOps
```

Но каждый атом обязан быть помечен:

```json
{
  "dictionary_level": "verified_extension_dictionary",
  "source": "target_mined_residual",
  "depends_on_checkpoint_data": true,
  "universal": false,
  "transferable": false,
  "eligible_for_exact_base_decode": false
}
```

Level 1 можно использовать для:

```text
residual explanation
same-checkpoint extension
same-family candidate transfer
diagnostics
minimal repair plan
```

Level 1 нельзя использовать для:

```text
заявления universal primitive
cross-model exact transplant без structural promotion
base dictionary в synthetic calibration
```

### 2.3 Level 2 — universal_promoted_dictionary

Level 1 atom может стать универсальным primitive только после promotion gate.

Promotion требует:

```text
1. synthetic calibration passed;
2. typed signature stable;
3. heldout heads/layers passed;
4. at least two checkpoints or model states passed;
5. no raw passthrough;
6. causal/closure evidence exists;
7. exact formula or canonical typed generator exists.
```

Manifest:

```json
{
  "dictionary_level": "universal_promoted_dictionary",
  "source": "promoted_verified_extension",
  "depends_on_checkpoint_data": false,
  "promotion_evidence": {
    "synthetic_passed": true,
    "heldout_heads_passed": true,
    "multi_checkpoint_passed": true,
    "causal_closure_passed": true
  },
  "eligible_for_exact_base_decode": true
}
```

Только Level 2 можно добавлять в будущий base dictionary.

---

## 3. Как self-expanding dictionary совместим с exact transplant

Exact transplant требует:

```text
не скрывать fitting внутри base dictionary.
```

Но residual mining полезен и нужен. Поэтому порядок такой:

```text
1. Decode with Level 0 analytic dictionary.
2. Compute residual.
3. Mine candidates from residual.
4. Mark candidates as Level 1 target-specific typed_birth.
5. Verify them on heldout heads/prompts.
6. Use them only as extension/diagnostic/repair candidates.
7. Promote to Level 2 only after separate promotion gate.
```

То есть:

```text
self-expanding dictionary = разрешён
self-expanding base dictionary = запрещён
```

---

## 4. Candidate mining formula

Пусть:

```python
M        # target circuit matrix
D0       # Level 0 analytic dictionary
M0_hat   # reconstruction from D0
R0 = M - M0_hat
```

Candidate from residual:

```python
u = mine_residual_atom(R0)
u = u / ||u||
```

Best closed-form coefficient:

```python
c = <R0, u> / (<u, u> + eps)
```

Candidate reconstruction:

```python
M1_hat = M0_hat + c * u
```

Marginal gain:

```python
gain = rel_err(M0_hat, M) - rel_err(M1_hat, M)
```

Energy fraction explained:

```python
explained_energy_frac = (||R0||^2 - ||R0 - c*u||^2) / (||M||^2 + eps)
```

Acceptance into Level 1 requires:

```text
gain >= MIN_EXTENSION_GAIN
explained_energy_frac >= MIN_EXTENSION_ENERGY_FRAC
heldout_gain >= MIN_HELDOUT_GAIN
dup_corr <= MAX_DUP_CORR
retain/function safety passed
```

But even if accepted, it remains:

```text
source = target_mined_residual
universal = false
eligible_for_exact_base_decode = false
```

---

## 5. Typed extension, not raw vector

A mined atom cannot stay as anonymous PCA/SVD vector if we want exact program transplant.

It must become one of:

```text
QK_BirthTypedRoute
VO_BirthTypedWrite
MLP_BirthTypedTransform
DIAGNOSTIC_ONLY_UNTYPED_ATOM
```

Typing requires signature:

```json
{
  "rank": 4,
  "diagonal_mass": 0.72,
  "local_band_mass": 0.81,
  "bias_col_mass": 0.02,
  "symmetry": 0.93,
  "shift_peak": 1,
  "block_avg_score": 0.14,
  "role_guess": "local_route"
}
```

If typing fails:

```text
DIAGNOSTIC_ONLY_UNTYPED_ATOM
```

and the atom cannot be transferred.

---

## 6. Verification gates for Level 1

A target-mined atom is accepted into `verified_extension_dictionary` only if all selected gates pass.

### 6.1 Target heldout gate

For QK:

```text
train heads/deltas -> mine atom
heldout heads/deltas -> verify gain
```

For VO:

```text
same head -> train prompts / heldout prompts
```

This matches current finding:

```text
QK = shared routing language
VO = private/head-specific write language
```

### 6.2 Functional gate

For QK:

```text
A_rel improves
KL improves
top1 improves or does not degrade
```

For VO:

```text
Y_all_rel improves
H_after_rel improves
retain damage is bounded
```

### 6.3 Duplicate gate

```python
dup_corr = max(abs(cosine(candidate, existing_op)) for existing_op in dictionary)
accept if dup_corr <= MAX_DUP_CORR
```

### 6.4 Safety gate

```text
retain_loss_delta <= MAX_RETAIN_DELTA
retain_logprob_damage <= MAX_RETAIN_DAMAGE
```

### 6.5 Report gate

Every accepted Level 1 atom must be written to:

```text
verified_extension_dictionary.jsonl
```

with:

```json
{
  "op_id": "QK_BirthTypedRoute_L23_H0_PCA0",
  "dictionary_level": "verified_extension_dictionary",
  "source": "target_mined_residual",
  "universal": false,
  "transferable": false,
  "train_gain": 0.71,
  "heldout_gain": 0.67,
  "functional_gain": {"A_rel_delta": -0.75},
  "dup_corr": 0.12,
  "typing_status": "typed",
  "promotion_status": "not_promoted"
}
```

---

## 7. Promotion gate Level 1 -> Level 2

A verified extension can become universal only after promotion.

Required evidence:

```text
synthetic calibration passed
same-checkpoint roundtrip passed
heldout heads/layers passed
multi-seed stable
multi-checkpoint stable
causal closure or direct replay evidence
no excessive retain damage
```

Promotion metrics:

```text
op_type_precision >= 0.98
op_type_recall >= 0.95
false_birth_rate <= 0.02
heldout_gain_mean > threshold
heldout_gain_std bounded
causal_closure_rate >= 0.8
```

Promotion manifest:

```json
{
  "old_level": "verified_extension_dictionary",
  "new_level": "universal_promoted_dictionary",
  "promotion_passed": true,
  "promotion_evidence_files": [
    "synthetic_calibration_report.json",
    "heldout_gate_report.json",
    "multi_checkpoint_report.json",
    "causal_closure_report.json"
  ]
}
```

Until promoted:

```text
not universal
not base dictionary
not exact cross-model primitive
```

---

## 8. Threshold calibration freeze

The remaining hidden tuning risk is thresholds.

Therefore:

```text
After synthetic calibration, thresholds must be frozen into calibrated_thresholds.json.
Real Qwen/same-checkpoint/cross-model runs must read this file and must not recalculate thresholds silently.
```

Required file:

```json
{
  "version": "calibrated_thresholds_v1",
  "source": "synthetic_program_roundtrip_v1",
  "frozen": true,
  "MIN_COEFF_ABS": 1e-6,
  "MIN_MARGINAL_REL_DROP": 1e-4,
  "MIN_EXPLAINED_ENERGY_FRAC": 1e-4,
  "MIN_EXTENSION_GAIN": 1e-3,
  "MIN_EXTENSION_ENERGY_FRAC": 1e-4,
  "MIN_HELDOUT_GAIN": 1e-3,
  "MAX_DUP_CORR": 0.985,
  "MAX_RETAIN_DELTA": 0.003,
  "closure_tol_multiplier": 1.5,
  "created_by": "synthetic_program_roundtrip_v1.py",
  "locked_before_real_runs": true
}
```

Real run manifest must contain:

```json
{
  "thresholds_loaded_from": "calibrated_thresholds.json",
  "thresholds_recomputed_in_real_run": false,
  "thresholds_frozen": true
}
```

If thresholds are changed after seeing real Qwen result:

```text
THRESHOLD_TUNING_CONTAMINATION
```

and the run is not valid as exact transplant evidence.

---

## 9. How to use this in implementation

### Step 1: analytic base only

```python
base_dict = build_analytic_base_dictionary(config)
program0 = decode_with_analytic_base(M, base_dict)
```

### Step 2: residual extension

```python
residual = M - encode(program0)
ext_candidates = mine_residual_candidates(residual)
verified_ext = verify_extension_candidates(ext_candidates, heldout, functional, retain)
```

### Step 3: decode with explicit levels

```python
program = Program(
    ops=program0.ops + verified_ext.ops,
    dictionary_levels={
        "base": "analytic_base_dictionary",
        "extension": "verified_extension_dictionary",
    },
)
```

### Step 4: closure status

If only Level 0 closes:

```text
ANALYTIC_PROGRAM_CLOSED
```

If Level 0 + Level 1 closes:

```text
EXTENDED_TARGET_SPECIFIC_PROGRAM_CLOSED
```

If Level 2 universal primitives close:

```text
UNIVERSAL_PROGRAM_CLOSED
```

Do not call Level 1 closure universal.

---

## 10. Updated statuses

Add statuses:

```text
ANALYTIC_PROGRAM_CLOSED
EXTENDED_TARGET_SPECIFIC_PROGRAM_CLOSED
UNIVERSAL_PROGRAM_CLOSED
THRESHOLD_TUNING_CONTAMINATION
EXTENSION_DICTIONARY_USED
UNPROMOTED_EXTENSION_USED
PROMOTED_EXTENSION_USED
```

Example manifest:

```json
{
  "status": "EXTENDED_TARGET_SPECIFIC_PROGRAM_CLOSED",
  "closure_level": "circuit_target",
  "base_dictionary_level": "analytic_base_dictionary",
  "extension_dictionary_level": "verified_extension_dictionary",
  "extension_dictionary_used": true,
  "unpromoted_extension_used": true,
  "universal_claim_allowed": false,
  "thresholds_loaded_from": "calibrated_thresholds.json",
  "thresholds_frozen": true
}
```

---

## 11. Practical decision for our next implementation

For `exact_program_transplant_v1.py` we should implement in this order:

```text
1. analytic_base_dictionary only
2. synthetic roundtrip and threshold freeze
3. same-checkpoint roundtrip with Level 0 only
4. if Level 0 residual remains, enable Level 1 verified extensions
5. report status as EXTENDED_TARGET_SPECIFIC_PROGRAM_CLOSED, not universal
6. only after multi-checkpoint promotion can extension become Level 2
```

This keeps the unique self-expanding dictionary idea while preserving exact transplant discipline.

---

## 12. Short AGENT rule

```text
Self-expanding dictionary is allowed only as verified_extension_dictionary.
It must be target-specific, typed, heldout-verified, threshold-frozen, and never silently promoted to base dictionary.
Base dictionary is analytic-only.
Universal dictionary requires explicit promotion evidence.
```
