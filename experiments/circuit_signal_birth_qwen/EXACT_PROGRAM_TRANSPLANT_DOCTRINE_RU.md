# Exact Program Transplant Doctrine

Жёсткая спецификация для переноса знаний почти без обучения через матричный декодер, typed program DSL, structural diff, direct encode и closure-check.

Цель документа — зафиксировать **exact program transplant**, а не ещё один distillation/fine-tune pipeline.

```text
teacher weights
-> exact matrix/program decoder
-> canonical typed symbolic program
-> structural diff with student program
-> structural translation into student basis
-> direct encode into student circuit/weights
-> closure check
-> optional minimal residual repair only after closure attempt
```

Главный запрет:

```text
Не превращать перенос знаний в KL/L2/alpha-sweep/LoRA-fitting под новым названием.
```

Главное уточнение про словарь:

```text
self-expanding dictionary разрешён, но только как verified_extension_dictionary.
analytic_base_dictionary остаётся analytic-only.
Нельзя подсовывать target-mined atoms как base primitives.
```

---

## 1. Главная доктрина

### 1.1 Основной путь

```text
decode -> typed program -> structural diff -> structural translation -> encode -> closure
```

Где:

- `decode` не возвращает просто коэффициенты;
- `program` состоит из typed ops с полями read/write/condition;
- `diff` делается на уровне структуры программы;
- `translate` переписывает программу в student basis только если basis compatible;
- `encode` строит circuit target или weights напрямую из программы;
- `closure` проверяет воспроизведение forward/circuit с численным допуском.

### 1.2 Запрещено как основной метод

Запрещено использовать как основной механизм переноса:

```text
KL(student, teacher)
L2(coeff_student, coeff_teacher)
alpha/beta sweep как поиск патча
LoRA/SVD patch как основной перенос
activation patch как success criterion без program closure
raw weight tensor as program
learned dictionary from target checkpoint as base dictionary
fitting/regression inside encode()
soft structural matching via arbitrary similarity score
```

Разрешено только как:

```text
diagnostics
negative baseline
minimal residual repair after exact transplant attempt
```

Если любой из этих методов использован как main path, manifest обязан писать:

```json
{
  "kl_distillation_used_as_main_method": true,
  "alpha_sweep_used_as_main_method": true,
  "gradient_used_as_main_method": true,
  "status": "APPROXIMATE_METHOD_REJECTED_AS_MAIN"
}
```

---

## 2. Математическая база Qwen circuit targets

### 2.1 QK target

```python
M_qk_aug[h, delta] = Wq_aug[h].T @ R_delta @ Wk_aug[kv] / sqrt(D)
```

Где:

```text
Wq_aug[h]  : [D, H+1]
Wk_aug[kv] : [D, H+1]
R_delta    : [D, D]
M_qk_aug   : [H+1, H+1]
x_aug      : [H+1]
```

Score:

```python
score_ij = x_aug_i @ M_qk_aug[h, i-j] @ x_aug_j
```

### 2.2 VO target

```python
C_vo_aug[h] = Wo[h] @ Wv_aug[kv]
```

Где:

```text
Wo[h]      : [H, D]
Wv_aug[kv] : [D, H+1]
C_vo_aug   : [H, H+1]
```

Payload/write:

```python
payload_j = x_aug_j @ C_vo_aug[h].T
Y_h_i = sum_j A_h[i,j] * payload_j
Y_all = sum_h Y_h
H_after = H_before + Y_all
```

### 2.3 Round-trip

```python
P = decode(M)
M_hat = encode(P)
round_trip_error = ||M_hat - M|| / ||M||
```

Closure tolerance:

```python
closure_tol = max(
    1.5 * teacher_self_roundtrip_error,
    1.5 * student_self_roundtrip_error,
    1e-5,
)
```

PASS только если:

```text
transplant_forward_error <= closure_tol
program_closure_rate >= required_program_closure_rate
raw_weight_passthrough_used == false
approximate_main_method == false
```

---

## 3. DSL: typed symbolic program

### 3.1 Program schema

```python
@dataclass
class Program:
    program_id: str
    model_family: str
    model_name: str
    layer: int
    head: int | None
    kv_head: int | None
    block: str              # qk | vo | mlp | full_block
    input_basis: str
    output_basis: str
    head_config: dict
    ops: list[Op]
    residual: ResidualReport
    roundtrip: RoundTripReport
    decode_status: str      # typed_program | partial_program | rejected
```

### 3.2 Op schema

```python
@dataclass
class Op:
    op_id: str
    op_type: str
    source: str             # analytic_primitive | typed_birth | diagnostic_only
    dictionary_level: str   # analytic_base_dictionary | verified_extension_dictionary | universal_promoted_dictionary
    read_fields: tuple[str, ...]
    write_fields: tuple[str, ...]
    condition_type: str     # enum, exact match only
    condition: dict
    params: dict
    shape: tuple[int, ...]
    energy: float
    marginal_error_drop: float
    encode_status: str      # exact | closed_form | unavailable
    decode_error_after: float
    typing_status: str      # typed | untyped_rejected | diagnostic_only
    depends_on_checkpoint_data: bool
    universal: bool
    transferable: bool
```

### 3.3 Residual report

```python
@dataclass
class ResidualReport:
    residual_rel: float
    residual_energy_ratio: float
    typed_coverage: float
    raw_residual_used_as_program: bool
    unreplayed_ops_count: int
```

### 3.4 Program diff schema

```python
@dataclass
class ProgramDiff:
    matched_ops: list[OpMatch]
    teacher_only_ops: list[Op]
    student_only_ops: list[Op]
    basis_mismatches: list[dict]
    routing_mismatches: list[dict]
    write_mismatches: list[dict]
    missing_primitives: list[dict]
    transfer_candidates: list[Op]
    non_transferable_ops: list[Op]
    structural_match_rate: float
    transfer_candidate_rate: float
```

---

## 4. Словарь примитивов: analytic base + self-expanding extension

Это главный раздел. Он специально разводит **аналитический базовый словарь** и **саморасширяющийся проверенный словарь**, чтобы не было конфликта между exact-transplant doctrine и signal_birth/residual-mining технологией.

### 4.1 Три уровня словаря

```text
Level 0: analytic_base_dictionary
Level 1: verified_extension_dictionary
Level 2: universal_promoted_dictionary
```

Коротко:

```text
self-expanding dictionary = разрешён
self-expanding base dictionary = запрещён
```

### 4.2 Level 0 — analytic_base_dictionary

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
PCA/clustering по M_qk/C_vo текущей модели для построения базового словаря
обученный autoencoder/dictionary learning как dictionary.qk_primitives
подгонка primitive.matrix под target checkpoint
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

### 4.3 QK analytic primitives v1

Каждый primitive строится функцией от `(H, delta, rope, config)`, а не от данных модели.

```python
def qk_identity(H):
    return eye(H + 1)


def qk_bias_column(H):
    M = zeros(H + 1, H + 1)
    M[:-1, -1] = 1 / sqrt(H)
    return M


def qk_self_route(H):
    M = zeros(H + 1, H + 1)
    M[:H, :H] = eye(H)
    return M


def qk_shift_feature(H, shift):
    M = zeros(H + 1, H + 1)
    for i in range(H):
        j = i - shift
        if 0 <= j < H:
            M[i, j] = 1.0
    return M


def qk_block_avg(H, block):
    M = zeros(H + 1, H + 1)
    for s in range(0, H, block):
        e = min(H, s + block)
        M[s:e, s:e] = 1.0 / (e - s)
    return M


def qk_rope_relative_template(H, rope_delta_id):
    # Допустимо только если строится аналитически из RoPE/R_delta,
    # а не fitted по checkpoint.
    return analytic_rope_template(H, rope_delta_id)
```

Типы:

```text
QK_IdentityRoute
QK_SelfRoute
QK_ShiftFeatureRoute
QK_BlockAverageRoute
QK_BiasRoute
QK_RoPERelativeRoute
```

### 4.4 VO analytic primitives v1

```python
def vo_identity_rect(H):
    C = zeros(H, H + 1)
    C[:, :H] = eye(H)
    return C


def vo_bias_write(H):
    C = zeros(H, H + 1)
    C[:, -1] = 1.0 / sqrt(H)
    return C


def vo_shift_write(H, shift):
    C = zeros(H, H + 1)
    for i in range(H):
        j = i - shift
        if 0 <= j < H:
            C[i, j] = 1.0
    return C


def vo_block_avg_write(H, block):
    C = zeros(H, H + 1)
    for s in range(0, H, block):
        e = min(H, s + block)
        C[s:e, s:e] = 1.0 / (e - s)
    return C
```

Типы:

```text
VO_IdentityWrite
VO_BiasWrite
VO_ShiftFeatureWrite
VO_BlockAverageWrite
```

### 4.5 Primitive manifest

Каждый Level 0 primitive обязан иметь:

```json
{
  "op_type": "QK_SelfRoute",
  "dictionary_level": "analytic_base_dictionary",
  "source": "analytic_primitive",
  "formula": "diag(I_H, 0_bias)",
  "depends_on_checkpoint_data": false,
  "depends_on_training_data": false,
  "shape_family": "H+1,H+1",
  "read_fields": ["residual_content"],
  "write_fields": ["attention_score"],
  "condition_type": "none"
}
```

Если `depends_on_checkpoint_data=true`, primitive не может входить в Level 0. Он может быть только Level 1 verified extension.

### 4.6 Level 1 — verified_extension_dictionary

Это саморасширяющийся словарь, который рождается из residual после Level 0 decode.

Источник:

```text
residual после Level 0 decode
PCA/SVD/OMP/mined atoms
target_mined atoms
BirthOps
```

Каждый атом обязан быть помечен:

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

### 4.7 Candidate mining formula для Level 1

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

Даже после acceptance кандидат остаётся:

```text
source = target_mined_residual
universal = false
eligible_for_exact_base_decode = false
```

### 4.8 Typed extension, not raw vector

Mined atom не может оставаться анонимным PCA/SVD-вектором, если он участвует в exact program transplant.

Он должен стать одним из:

```text
QK_BirthTypedRoute
VO_BirthTypedWrite
MLP_BirthTypedTransform
DIAGNOSTIC_ONLY_UNTYPED_ATOM
```

Typing signature example:

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

### 4.9 Verification gates for Level 1

A target-mined atom is accepted into `verified_extension_dictionary` only if selected gates pass.

QK gate:

```text
train heads/deltas -> mine atom
heldout heads/deltas -> verify gain
```

VO gate:

```text
same head -> train prompts / heldout prompts
```

This matches current finding:

```text
QK = shared routing language
VO = private/head-specific write language
```

Functional gate:

```text
QK: A_rel improves, KL improves, top1 does not degrade
VO: Y_all_rel improves, H_after_rel improves, retain damage bounded
```

Duplicate gate:

```python
dup_corr = max(abs(cosine(candidate, existing_op)) for existing_op in dictionary)
accept if dup_corr <= MAX_DUP_CORR
```

Safety gate:

```text
retain_loss_delta <= MAX_RETAIN_DELTA
retain_logprob_damage <= MAX_RETAIN_DAMAGE
```

Every accepted Level 1 atom must be written to:

```text
verified_extension_dictionary.jsonl
```

Example:

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

### 4.10 Level 2 — universal_promoted_dictionary

Level 1 atom может стать universal primitive только после promotion.

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

### 4.11 Updated closure statuses for dictionary levels

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

Example manifest:

```json
{
  "status": "EXTENDED_TARGET_SPECIFIC_PROGRAM_CLOSED",
  "closure_level": "circuit_target",
  "base_dictionary_level": "analytic_base_dictionary",
  "extension_dictionary_level": "verified_extension_dictionary",
  "extension_dictionary_used": true,
  "unpromoted_extension_used": true,
  "universal_claim_allowed": false
}
```

---

## 5. Acceptance criteria для op

### 5.1 Не принимать микроскопические op ради typed_coverage

`accept_structural_op()` обязан быть жёстким.

```python
def accept_structural_op(
    op,
    coeff,
    err_before,
    err_after,
    energy_before,
    residual_energy_after,
    total_energy,
):
    marginal_drop = err_before - err_after
    explained_energy = energy_before - residual_energy_after

    return (
        abs(coeff) >= MIN_COEFF_ABS
        and marginal_drop >= MIN_MARGINAL_REL_DROP
        and explained_energy / (total_energy + eps) >= MIN_EXPLAINED_ENERGY_FRAC
        and op.source == "analytic_primitive"
        and op.depends_on_checkpoint_data is False
    )
```

Initial values only for synthetic calibration:

```text
MIN_COEFF_ABS = 1e-6
MIN_MARGINAL_REL_DROP = 1e-4
MIN_EXPLAINED_ENERGY_FRAC = 1e-4
MAX_OPS_PER_PROGRAM = 64
```

These numbers are not valid for real runs until frozen in `calibrated_thresholds.json`.

### 5.2 Sequential marginal accept

Op принимается только по sequential residual:

```python
residual = M.clone()
accepted = []

for primitive in ordered_primitives:
    err_before = rel_err(encode(accepted), M)
    coeff = project_closed_form(residual, primitive.matrix)
    candidate = coeff * primitive.matrix
    residual_new = residual - candidate
    err_after = rel_err(M - residual_new, M)

    if accept_structural_op(...):
        accepted.append(op)
        residual = residual_new
```

Запрещено принять 50 op одновременно по одной global least-squares fit без marginal report.

### 5.3 Global solve допустим только после marginal screen

Allowed:

```text
refit coefficients for accepted typed ops only
```

Forbidden:

```text
use least-squares to discover arbitrary ops
```

Manifest:

```json
{
  "global_refit_used": true,
  "global_refit_scope": "accepted_typed_ops_only"
}
```

---

## 6. BirthOps: типизация, calibration и threshold freeze

### 6.1 BirthOp не является программой сам по себе

PCA/SVD residual vector не считается валидным op, пока не прошёл typing.

```text
PCA vector -> candidate_birth
candidate_birth -> type_birth_op()
if typing failed -> diagnostic_only, not transferable
```

### 6.2 Type signatures

QK:

```python
def qk_birth_signature(M):
    return {
        "diagonal_mass": diag_energy(M) / total_energy(M),
        "local_band_mass": band_energy(M, radius=8) / total_energy(M),
        "bias_col_mass": col_energy(M, -1) / total_energy(M),
        "bias_row_mass": row_energy(M, -1) / total_energy(M),
        "symmetry": 1.0 - rel_err(M, M.T),
        "effective_rank": effective_rank(M),
        "shift_peak": best_shift_correlation(M),
        "block_avg_score": best_block_avg_score(M),
    }
```

VO:

```python
def vo_birth_signature(C):
    return {
        "bias_col_mass": col_energy(C, -1) / total_energy(C),
        "diagonal_mass": diag_rect_energy(C) / total_energy(C),
        "row_sparsity": row_sparsity(C),
        "col_sparsity": col_sparsity(C),
        "effective_rank": effective_rank(C),
        "shift_peak": best_rect_shift_correlation(C),
        "block_avg_score": best_rect_block_avg_score(C),
    }
```

### 6.3 Synthetic calibration обязательна

Before real Qwen:

```text
synthetic program -> encode -> decode -> compare true ops
```

Example:

```python
true_program = Program(ops=[
    Op("QK_SelfRoute", coeff=0.7),
    Op("QK_ShiftFeatureRoute", coeff=-0.2, params={"shift": 1}),
    Op("QK_BlockAverageRoute", coeff=0.1, params={"block": 8}),
])

M = encode(true_program)
P_hat = decode(M)
assert recovered_op_types(P_hat) == recovered_op_types(true_program)
assert rel_err(encode(P_hat), M) <= 1e-5
```

Synthetic calibration metrics:

```text
op_type_precision
op_type_recall
coeff_rel_error
roundtrip_error
false_birth_rate
trivial_decode_reject_rate
```

Acceptance before real model:

```text
op_type_precision >= 0.98
op_type_recall >= 0.95
false_birth_rate <= 0.02
roundtrip_error <= 1e-5 for synthetic exact programs
```

### 6.4 Threshold calibration freeze

After synthetic calibration, thresholds must be frozen into `calibrated_thresholds.json`.

Real Qwen / same-checkpoint / cross-model runs must read this file and must not recalculate thresholds silently.

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

## 7. Structural matching без fuzzy matching

### 7.1 Fields compare exact enum/string

Запрещено:

```text
compatible_fields через cosine/similarity threshold
semantic fuzzy matching
learned field alignment
```

Same-checkpoint / same-family identity:

```python
def compatible_fields(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    return tuple(a) == tuple(b)
```

Cross-model допускает только ручной alias table:

```python
FIELD_ALIASES = {
    "residual_content": {"residual_content"},
    "position_delta": {"position_delta"},
    "bias_aug": {"bias_aug"},
}

def compatible_fields_cross_model(a, b):
    if len(a) != len(b):
        return False
    return all(bi in FIELD_ALIASES.get(ai, {ai}) for ai, bi in zip(a, b))
```

Alias table пишется вручную в config и сохраняется в manifest.

### 7.2 Conditions exact enum

```python
def compatible_condition(c1, c2):
    return (
        c1["condition_type"] == c2["condition_type"]
        and c1.get("delta") == c2.get("delta")
        and c1.get("window") == c2.get("window")
        and c1.get("mask_type") == c2.get("mask_type")
    )
```

Запрещено:

```text
condition similarity > threshold
fuzzy matching by score
Procrustes/CCA for condition match
```

### 7.3 Op structural match

```python
def structural_match(op_t, op_s, mode):
    if op_t.op_type != op_s.op_type:
        return False
    if mode in ("same_checkpoint", "same_family_identity"):
        fields_ok = op_t.read_fields == op_s.read_fields and op_t.write_fields == op_s.write_fields
    else:
        fields_ok = compatible_fields_cross_model(op_t.read_fields, op_s.read_fields) \
            and compatible_fields_cross_model(op_t.write_fields, op_s.write_fields)
    return fields_ok and compatible_condition(op_t.condition, op_s.condition)
```

---

## 8. Closure level: circuit target vs weights

### 8.1 Обязательное поле

Manifest обязан иметь:

```json
{
  "closure_level": "circuit_target"
}
```

Допустимые значения:

```text
circuit_target
weights
full_forward_logits
```

### 8.2 Нельзя смешивать уровни

If only circuit target is closed:

```json
{
  "status": "CIRCUIT_TARGET_PROGRAM_CLOSED",
  "closure_level": "circuit_target",
  "weights_rewritten": false,
  "full_forward_closed": false
}
```

Нельзя писать `EXACT_PROGRAM_TRANSPLANT_CLOSED`, если закрыт только `M_qk/C_vo`, но не было weight rewrite/full forward replay.

### 8.3 Status list

```text
SYNTHETIC_PROGRAM_CLOSED
ANALYTIC_PROGRAM_CLOSED
EXTENDED_TARGET_SPECIFIC_PROGRAM_CLOSED
UNIVERSAL_PROGRAM_CLOSED
CIRCUIT_TARGET_PROGRAM_CLOSED
WEIGHT_LEVEL_PROGRAM_CLOSED
FULL_FORWARD_PROGRAM_CLOSED
PARTIAL_PROGRAM_CLOSED
STRUCTURAL_TRANSFER_BLOCKED
TRIVIAL_DECODE_REJECTED
APPROXIMATE_METHOD_REJECTED_AS_MAIN
THRESHOLD_TUNING_CONTAMINATION
```

---

## 9. PASS per-head/per-layer

### 9.1 Per-head metrics

Every head gets its own report:

```json
{
  "layer": 23,
  "head": 4,
  "block": "qk",
  "roundtrip_error": 0.00022,
  "typed_coverage": 0.971,
  "program_closure_rate": 1.0,
  "status": "CIRCUIT_TARGET_PROGRAM_CLOSED"
}
```

### 9.2 Общий PASS

Run-level PASS cannot be only an average.

```python
head_pass_rate = passed_heads / total_heads
max_head_error = max(head.roundtrip_error for head in heads)
```

Default:

```text
REQUIRED_HEAD_PASS_RATE = 1.0 for same-checkpoint
REQUIRED_HEAD_PASS_RATE = 0.8 for early cross-model structural diff
```

Same-checkpoint PASS:

```text
all heads closed
max_head_error <= closure_tol
no hidden failed heads
```

If 5 of 7 heads closed:

```text
PARTIAL_PROGRAM_CLOSED
```

not PASS.

### 9.3 Per-layer aggregation

```python
layer_pass = all(head.status.endswith("CLOSED") for head in layer.head_reports)
run_pass = all(layer.pass for layer in layers)
```

No averaging-based hiding.

---

## 10. Cross-model head/KV mismatch rules

### 10.1 Head config check

```python
def head_config(model):
    return {
        "num_heads": cfg.num_attention_heads,
        "num_kv_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "kv_groups": cfg.num_attention_heads // cfg.num_key_value_heads,
        "hidden_size": cfg.hidden_size,
    }
```

### 10.2 Same structure required for direct transplant

Direct transplant allowed only if:

```text
num_heads equal
num_kv_heads equal
head_dim equal
kv_groups equal
hidden_size equal
RoPE config compatible
```

If not:

```json
{
  "status": "NO_STRUCTURAL_ANALOG_FOUND",
  "reason": "head_or_kv_config_mismatch",
  "direct_transplant_allowed": false
}
```

### 10.3 No reshape/interpolation trick

Запрещено:

```text
reshape teacher head into student head
interpolate head_dim
average teacher heads into student heads
split one head into many
Procrustes fit hidden basis
```

This can be an approximate baseline, but not exact transplant.

---

## 11. Encode rules

### 11.1 Encode to circuit target first

First encode target:

```text
Program -> M_qk_aug_hat
Program -> C_vo_aug_hat
```

Not weights first.

### 11.2 QK circuit encode

```python
def encode_qk_program(program, basis):
    M = zeros(H + 1, H + 1)
    for op in program.ops:
        if op.encode_status not in ("exact", "closed_form"):
            continue
        primitive = build_analytic_or_verified_primitive(op, basis)
        M += op.params["coeff"] * primitive
    return M
```

`build_analytic_or_verified_primitive` must obey dictionary level rules:

```text
Level 0: analytic formula only
Level 1: target-specific verified atom only, not universal
Level 2: promoted universal primitive
```

### 11.3 VO circuit encode

```python
def encode_vo_program(program, basis):
    C = zeros(H, H + 1)
    for op in program.ops:
        primitive = build_analytic_or_verified_primitive(op, basis)
        C += op.params["coeff"] * primitive
    return C
```

### 11.4 VO weight encode closed-form

If rewriting weights:

```python
# Fixed Wo, solve Wv_aug
Wv_aug_new = torch.linalg.pinv(Wo) @ C_target
C_recon = Wo @ Wv_aug_new
err = rel_err(C_recon, C_target)
```

Manifest:

```json
{
  "weight_encode_method": "closed_form_pinv_fixed_Wo",
  "gradient_used": false,
  "weight_encode_error": 0.00012
}
```

### 11.5 QK weight encode not v1 main path

QK weight encode is harder:

```text
M = Wq_aug.T @ R @ Wk_aug
```

v1 closes QK at circuit-target/replay level first.

Weight-level QK transplant is v2.

---

## 12. Repair policy

Repair is allowed only after exact transplant attempt.

Repair order:

```text
1. closed-form repair
2. least-squares on explicitly missing primitive only
3. gradient repair only if 1-2 impossible
```

Repair manifest:

```json
{
  "repair_used": true,
  "repair_method": "closed_form" | "least_squares" | "gradient",
  "repair_scope": "unreplayed_ops_only",
  "closed_ops_frozen": true,
  "repair_steps": 0,
  "repair_steps_needed": 0
}
```

Gradient caps:

```text
same_checkpoint: repair_steps <= 20
cross_model: repair_steps <= 50
```

If more:

```text
REPAIR_TOO_LARGE_NOT_EXACT_TRANSFER
```

---

## 13. Qwen replay closure inspired by PART replay

The PART replay file gives the discipline:

```text
capture real forward
manually replay primitives/MHA/container/root
compare final logits
emit FULL_FORWARD_CLOSED only if decoded logits replay passes tolerance
```

Qwen version must implement:

```text
QwenForwardCapture
replay_qwen_rmsnorm
replay_qwen_attention_from_circuit_targets
replay_qwen_mlp
replay_qwen_decoder_layer
replay_qwen_model_to_logits
```

The status `FULL_FORWARD_PROGRAM_CLOSED` is emitted only when final logits replay is closed.

---

## 14. exact_program_transplant_v1.py

### 14.1 Modes

```text
--mode synthetic_ground_truth_roundtrip
--mode same_checkpoint_roundtrip
--mode same_family_transplant
--mode cross_model_structural_diff
--mode cross_model_transplant_if_structural_match
```

### 14.2 CLI

```bash
python exact_program_transplant_v1.py \
  --mode same_checkpoint_roundtrip \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layers 23 \
  --heads 0,1,2,3,4,5,6 \
  --blocks qk,vo \
  --out runs/exact_program_transplant_v1/same_checkpoint
```

### 14.3 Outputs

```text
transplant_runs/<run_id>/
  primitive_dictionary_manifest.json
  verified_extension_dictionary.jsonl
  synthetic_calibration_report.json
  calibrated_thresholds.json
  teacher_program.json
  student_program_before.json
  program_diff.json
  translated_program.json
  encoded_targets.pt
  student_program_after.json
  per_head_closure.csv
  closure_report.json
  unreplayed_ops.json
  repair_plan.json
  manifest.json
  README.md
```

---

## 15. Required manifest

```json
{
  "mode": "same_checkpoint_roundtrip",
  "closure_level": "circuit_target",
  "status": "EXTENDED_TARGET_SPECIFIC_PROGRAM_CLOSED",

  "base_dictionary_level": "analytic_base_dictionary",
  "extension_dictionary_level": "verified_extension_dictionary",
  "extension_dictionary_used": true,
  "unpromoted_extension_used": true,
  "universal_claim_allowed": false,

  "dictionary_source": "analytic_base_plus_verified_extension",
  "dictionary_learned_from_checkpoint_as_base": false,
  "dictionary_learned_from_training_data_as_base": false,

  "thresholds_loaded_from": "calibrated_thresholds.json",
  "thresholds_recomputed_in_real_run": false,
  "thresholds_frozen": true,

  "decode_status": "typed_program",
  "raw_weight_passthrough_used": false,
  "trivial_decode_rejected": false,

  "alpha_sweep_used_as_main_method": false,
  "kl_distillation_used_as_main_method": false,
  "coefficient_l2_used_as_main_method": false,
  "gradient_used_as_main_method": false,

  "same_checkpoint_roundtrip_closed": true,
  "program_diff_exists": true,
  "direct_encode_attempted": true,
  "closure_report_exists": true,

  "program_closure_rate": 1.0,
  "op_closure_rate": 1.0,
  "structural_match_rate": 1.0,
  "head_pass_rate": 1.0,
  "max_head_error": 0.00031,

  "teacher_roundtrip_error": 0.00028,
  "student_roundtrip_error": 0.00029,
  "transplant_forward_error": 0.00031,
  "closure_tol": 0.00045,

  "unreplayed_ops_count": 0,
  "repair_used": false,
  "repair_steps_needed": 0
}
```

---

## 16. Acceptance criteria

### 16.1 Synthetic ground truth

PASS if:

```text
op_type_precision >= 0.98
op_type_recall >= 0.95
false_birth_rate <= 0.02
roundtrip_error <= 1e-5
raw_passthrough_used = false
thresholds_written_to_calibrated_thresholds_json = true
```

### 16.2 Same-checkpoint

PASS if:

```text
synthetic_calibration_passed = true
thresholds_loaded_from calibrated_thresholds.json
head_pass_rate = 1.0
typed_coverage >= 0.95 per head
max_head_error <= closure_tol
raw_passthrough_used = false
closure_level reported
```

### 16.3 Same-family

PASS if:

```text
same-checkpoint passed first
basis_status in [SAME_INIT_FINETUNE_IDENTITY, EXACT_SAME_CHECKPOINT]
structural_match_rate reported
transfer candidates reported
closure_level reported
```

### 16.4 Cross-model

PASS for structural diff if:

```text
head_config_checked = true
structural_match_rate reported
non_transferable_ops reported
missing_primitives reported
NO_STRUCTURAL_ANALOG_FOUND allowed and explicit
```

Direct transplant cross-model allowed only if:

```text
head_config compatible
basis compatibility explicit
structural_match_rate >= threshold
no fuzzy matching
no unpromoted Level 1 extension as universal primitive
```

---

## 17. Development order

Do not write a giant transfer training script first.

Order:

```text
1. program_dsl_v1.py
2. analytic_primitives_v1.py
3. synthetic_program_roundtrip_v1.py
4. freeze calibrated_thresholds.json
5. qwen_program_decode_v1.py with Level 0 analytic base
6. qwen_program_encode_v1.py to circuit target
7. exact_program_transplant_v1.py same_checkpoint_roundtrip Level 0
8. enable Level 1 verified_extension_dictionary if residual remains
9. qwen_full_replay_closure_v1.py
10. same_family_transplant
11. cross_model_structural_diff
12. direct transplant only if structural match exists
13. minimal residual repair
```

---

## 18. AGENT RULES

```text
Exact Program Transplant Doctrine:

Approximate ML methods are diagnostics/fallback only.
The main path is:
decode -> typed program -> structural diff -> structural translation -> encode -> closure.

Base dictionary is analytic-only.
Self-expanding dictionary is allowed only as verified_extension_dictionary.
It must be target-specific, typed, heldout-verified, threshold-frozen, and never silently promoted to base dictionary.
Universal dictionary requires explicit promotion evidence.

Do not use KL, coefficient L2, alpha sweep, LoRA/SVD fitting, learned base dictionaries,
Procrustes/CCA, or gradient descent as the main transfer mechanism.

Primitive dictionaries must report their level:
analytic_base_dictionary | verified_extension_dictionary | universal_promoted_dictionary.

Fields and conditions must match by exact enum/string equality or explicit alias table.
Raw weight passthrough is rejected.
PASS is per-head/per-layer; averages cannot hide failed heads.
Manifest must report closure_level: circuit_target | weights | full_forward_logits.
Cross-model head/KV mismatch blocks direct transplant.
If exact closure fails, report the missing primitive/residual. Do not hide failure behind approximate fitting.
```

---

## 19. Why this matters

The value is not just safer fine-tuning.

The value is:

```text
neural network -> exact matrix program -> portable typed operators -> direct structural transplant
```

If this works, knowledge transfer becomes:

```text
not logits imitation
not weight interpolation
not long fine-tune
but program-level transplant with closure proof
```

This is the core research direction.
