# Exact Program Transplant Doctrine v1.1

Жёсткая спецификация для переноса знаний почти без обучения через матричный декодер, typed program DSL, structural diff, direct encode и closure-check.

Главный pipeline:

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

## 1. Scope v1

v1 покрывает только:

```text
QK circuit target: M_qk_aug[h, delta]
VO circuit target: C_vo_aug[h]
synthetic Level-0 roundtrip
same-checkpoint QK/VO circuit-target roundtrip
same-family QK/VO structural transplant
```

MLP/SwiGLU пока **не входит в v1 exact transplant scope**.

Manifest обязан писать:

```json
{
  "mlp_in_v1_scope": false,
  "mlp_decode_attempted": false,
  "mlp_status": "OUT_OF_SCOPE_V1"
}
```

Если агент пытается делать MLP без отдельного Level-0 словаря и calibration:

```text
MLP_OUT_OF_SCOPE_V1_REJECTED
```

MLP будет v2 после отдельного DSL/analytic primitive spec для SwiGLU/Jacobian/product-step programs.

---

## 2. Запрещено как основной метод

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
Procrustes/CCA as structural match
```

Разрешено только как:

```text
diagnostics
negative baseline
minimal residual repair after exact transplant attempt
```

Manifest flags:

```json
{
  "alpha_sweep_used_as_main_method": false,
  "kl_distillation_used_as_main_method": false,
  "coefficient_l2_used_as_main_method": false,
  "gradient_used_as_main_method": false,
  "raw_weight_passthrough_used": false
}
```

---

## 3. Qwen circuit formulas

### 3.1 QK

```python
M_qk_aug[h, delta] = Wq_aug[h].T @ R_delta @ Wk_aug[kv] / sqrt(D)
score_ij = x_aug_i @ M_qk_aug[h, i-j] @ x_aug_j
```

Where:

```text
Wq_aug[h]  : [D, H+1]
Wk_aug[kv] : [D, H+1]
R_delta    : [D, D]
M_qk_aug   : [H+1, H+1]
x_aug      : [H+1]
```

### 3.2 VO

```python
C_vo_aug[h] = Wo[h] @ Wv_aug[kv]
payload_j = x_aug_j @ C_vo_aug[h].T
Y_h_i = sum_j A_h[i,j] * payload_j
Y_all = sum_h Y_h
H_after = H_before + Y_all
```

Where:

```text
Wo[h]      : [H, D]
Wv_aug[kv] : [D, H+1]
C_vo_aug   : [H, H+1]
```

---

## 4. DSL: typed symbolic program

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
    decode_status: str
```

```python
@dataclass
class Op:
    op_id: str
    op_type: str
    source: str             # analytic_primitive | typed_birth | diagnostic_only
    dictionary_level: str   # analytic_base_dictionary | verified_extension_dictionary | universal_promoted_dictionary
    read_fields: tuple[str, ...]
    write_fields: tuple[str, ...]
    condition_type: str
    condition: dict
    params: dict
    shape: tuple[int, ...]
    energy: float
    marginal_error_drop: float
    encode_status: str      # exact | closed_form | unavailable
    decode_error_after: float
    typing_status: str
    depends_on_checkpoint_data: bool
    universal: bool
    transferable: bool
```

Raw tensor as op is forbidden:

```text
RAW_WEIGHT -> TRIVIAL_DECODE_REJECTED
RAW_RESIDUAL_AS_PROGRAM -> TRIVIAL_DECODE_REJECTED
```

---

## 5. Dictionary levels

```text
Level 0: analytic_base_dictionary
Level 1: verified_extension_dictionary
Level 2: universal_promoted_dictionary
```

Short rule:

```text
self-expanding dictionary = разрешён
self-expanding base dictionary = запрещён
```

### 5.1 Level 0 — analytic_base_dictionary

Only analytic formulas:

```text
identity
bias column
feature shift
block average
fixed DCT/Hadamard-like templates
RoPE analytic templates
manual typed primitive with explicit formula
```

Forbidden for Level 0:

```text
PCA/clustering over target M_qk/C_vo
SVD residual of current checkpoint
autoencoder/dictionary learning
checkpoint-fitted primitive.matrix
```

Manifest:

```json
{
  "dictionary_level": "analytic_base_dictionary",
  "dictionary_source": "analytic_only",
  "depends_on_checkpoint_data": false,
  "eligible_for_exact_base_decode": true
}
```

### 5.2 QK analytic primitives v1

Every primitive is built from `(H, delta, rope, config)`, not from checkpoint data.

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
```

Allowed QK op types v1:

```text
QK_IdentityRoute
QK_SelfRoute
QK_ShiftFeatureRoute
QK_BlockAverageRoute
QK_BiasRoute
QK_RoPERelativeRoute
QK_BirthTypedRoute
```

### 5.3 VO analytic primitives v1

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

Allowed VO op types v1:

```text
VO_IdentityWrite
VO_BiasWrite
VO_ShiftFeatureWrite
VO_BlockAverageWrite
VO_BirthTypedWrite
```

### 5.4 Level 1 — verified_extension_dictionary

Level 1 is the self-expanding dictionary. It is allowed only after Level 0 residual.

```python
M0_hat = encode(level0_program)
R0 = M - M0_hat
u = mine_residual_atom(R0)
u = u / ||u||
c = <R0, u> / (<u, u> + eps)
M1_hat = M0_hat + c * u
gain = rel_err(M0_hat, M) - rel_err(M1_hat, M)
explained_energy_frac = (||R0||^2 - ||R0 - c*u||^2) / (||M||^2 + eps)
```

Acceptance into Level 1 requires:

```text
gain >= MIN_EXTENSION_GAIN
explained_energy_frac >= MIN_EXTENSION_ENERGY_FRAC
heldout_gain >= MIN_HELDOUT_GAIN
dup_corr <= MAX_DUP_CORR
retain/function safety passed
typed signature assigned
thresholds loaded from calibrated_thresholds.json
```

Even if accepted, Level 1 remains:

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

### 5.5 Level 2 — universal_promoted_dictionary

Level 2 promotion is **not part of v1 experiments**.

Manifest must write:

```json
{
  "level2_promotion_in_v1_scope": false,
  "level2_promotion_attempted": false
}
```

Level 1 -> Level 2 requires a separate future promotion run with:

```text
synthetic calibration passed
same-checkpoint roundtrip passed
heldout heads/layers passed
multi-seed stable
multi-checkpoint stable
causal closure/direct replay evidence
no excessive retain damage
```

Until then:

```text
not universal
not base dictionary
not cross-model exact primitive
```

---

## 6. Op acceptance and threshold freeze

### 6.1 Sequential marginal accept

Op accepted only by sequential residual contribution:

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

Forbidden:

```text
global least-squares discovery over arbitrary ops
accepting microscopic ops only to inflate typed_coverage
```

Allowed:

```text
global refit over already accepted typed ops only
```

### 6.2 Thresholds

Initial calibration values:

```text
MIN_COEFF_ABS = 1e-6
MIN_MARGINAL_REL_DROP = 1e-4
MIN_EXPLAINED_ENERGY_FRAC = 1e-4
MIN_EXTENSION_GAIN = 1e-3
MIN_EXTENSION_ENERGY_FRAC = 1e-4
MIN_HELDOUT_GAIN = 1e-3
MAX_DUP_CORR = 0.985
MAX_RETAIN_DELTA = 0.003
closure_tol_multiplier = 1.5
```

These values are **not valid for real runs** until frozen by `synthetic_program_roundtrip_v1.py`.

Required file:

```text
calibrated_thresholds.json
```

Real run manifest:

```json
{
  "thresholds_loaded_from": "calibrated_thresholds.json",
  "thresholds_recomputed_in_real_run": false,
  "thresholds_frozen": true
}
```

If thresholds are changed after looking at real Qwen result:

```text
THRESHOLD_TUNING_CONTAMINATION
```

---

## 7. Synthetic calibration before real Qwen

Before real Qwen:

```text
synthetic Program -> encode -> decode -> compare true ops
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

Required metrics:

```text
op_type_precision >= 0.98
op_type_recall >= 0.95
false_birth_rate <= 0.02
roundtrip_error <= 1e-5 for synthetic exact programs
trivial_decode_reject_rate reported
calibrated_thresholds.json written
```

If synthetic calibration fails, exact transplant on real Qwen is blocked.

---

## 8. Train/heldout split protocol for Level 1 verification

This section is mandatory before enabling verified_extension_dictionary.

### 8.1 QK split

For each selected layer:

```text
heads sorted ascending
train_heads = first 70% heads
heldout_heads = remaining 30% heads
```

For 7 heads `0,1,2,3,4,5,6`:

```text
train_heads = 0,1,2,3,4
heldout_heads = 5,6
```

Deltas are split deterministically by parity unless explicitly overridden:

```text
train_deltas = deltas where delta % 2 == 0
heldout_deltas = deltas where delta % 2 == 1
```

No overlap allowed:

```python
assert set(train_heads).isdisjoint(heldout_heads)
assert set(train_deltas).isdisjoint(heldout_deltas)
```

QK candidate accepted only if it improves heldout heads/deltas not used for mining.

### 8.2 VO split

VO is private/per-head.

Do not split VO by heads for transfer validation.

For each head:

```text
train_prompts = deterministic first 70% of prompt ids
heldout_prompts = remaining 30%
```

No overlap allowed:

```python
assert set(train_prompts).isdisjoint(heldout_prompts)
```

VO candidate accepted only if same-head heldout prompts improve `Y_all/H_after` and retain damage is bounded.

### 8.3 Split manifest

Every Level 1 run must save:

```json
{
  "split_protocol_version": "split_v1_fixed_70_30_parity_delta",
  "qk_train_heads": [0, 1, 2, 3, 4],
  "qk_heldout_heads": [5, 6],
  "qk_train_deltas_rule": "delta % 2 == 0",
  "qk_heldout_deltas_rule": "delta % 2 == 1",
  "vo_split_rule": "same_head_train_70_heldout_30_prompts",
  "no_overlap_verified": true
}
```

If no split manifest exists:

```text
LEVEL1_VERIFICATION_INVALID_NO_SPLIT_PROTOCOL
```

---

## 9. Structural matching without fuzzy matching

Same-checkpoint/same-family identity:

```python
def compatible_fields(a, b):
    return tuple(a) == tuple(b)
```

Cross-model only manual alias table:

```python
FIELD_ALIASES = {
    "residual_content": {"residual_content"},
    "position_delta": {"position_delta"},
    "bias_aug": {"bias_aug"},
}
```

Condition exact enum:

```python
def compatible_condition(c1, c2):
    return (
        c1["condition_type"] == c2["condition_type"]
        and c1.get("delta") == c2.get("delta")
        and c1.get("window") == c2.get("window")
        and c1.get("mask_type") == c2.get("mask_type")
    )
```

Forbidden:

```text
cosine/fuzzy field matching
semantic matching
Procrustes/CCA
similarity-threshold structural match
```

---

## 10. Closure levels

Manifest field required:

```json
{
  "closure_level": "circuit_target"
}
```

Allowed:

```text
circuit_target
weights
full_forward_logits
```

If only circuit target is closed:

```json
{
  "status": "CIRCUIT_TARGET_PROGRAM_CLOSED",
  "closure_level": "circuit_target",
  "weights_rewritten": false,
  "full_forward_closed": false
}
```

Do not call it full transplant until `full_forward_logits` is closed.

---

## 11. PASS per-head/per-layer

Every head gets report:

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

Run PASS cannot be an average:

```python
head_pass_rate = passed_heads / total_heads
max_head_error = max(head.roundtrip_error for head in heads)
```

Default:

```text
same-checkpoint: REQUIRED_HEAD_PASS_RATE = 1.0
cross-model structural diff: REQUIRED_HEAD_PASS_RATE = 0.8
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

---

## 12. Cross-model head/KV mismatch rules

Before matching:

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

Forbidden:

```text
reshape teacher head into student head
interpolate head_dim
average/split heads
Procrustes fit hidden basis
```

---

## 13. Encode rules

### 13.1 Circuit target first

First encode target:

```text
Program -> M_qk_aug_hat
Program -> C_vo_aug_hat
```

### 13.2 QK circuit encode

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

`build_analytic_or_verified_primitive` obeys dictionary level rules.

### 13.3 VO circuit encode

```python
def encode_vo_program(program, basis):
    C = zeros(H, H + 1)
    for op in program.ops:
        primitive = build_analytic_or_verified_primitive(op, basis)
        C += op.params["coeff"] * primitive
    return C
```

### 13.4 VO weight encode closed-form

```python
Wv_aug_new = torch.linalg.pinv(Wo) @ C_target
C_recon = Wo @ Wv_aug_new
err = rel_err(C_recon, C_target)
```

No gradient.

### 13.5 QK weight encode not v1 main path

QK weight encode is v2 because:

```text
M = Wq_aug.T @ R @ Wk_aug
```

v1 closes QK at circuit-target/replay level first.

---

## 14. Repair policy

Repair only after exact transplant attempt.

Order:

```text
1. closed-form repair
2. least-squares on explicitly missing primitive only
3. gradient repair only if 1-2 impossible
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

## 15. Status list

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
MLP_OUT_OF_SCOPE_V1_REJECTED
LEVEL1_VERIFICATION_INVALID_NO_SPLIT_PROTOCOL
NO_STRUCTURAL_ANALOG_FOUND
```

---

## 16. exact_program_transplant_v1.py modes

```text
--mode synthetic_ground_truth_roundtrip
--mode same_checkpoint_roundtrip
--mode same_family_transplant
--mode cross_model_structural_diff
--mode cross_model_transplant_if_structural_match
```

Outputs:

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
  per_head_closure.csv
  closure_report.json
  unreplayed_ops.json
  repair_plan.json
  manifest.json
  README.md
```

---

## 17. Development order

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
Approximate ML methods are diagnostics/fallback only.
The main path is:
decode -> typed program -> structural diff -> structural translation -> encode -> closure.

Base dictionary is analytic-only.
Self-expanding dictionary is allowed only as verified_extension_dictionary.
It must be target-specific, typed, heldout-verified, threshold-frozen, and never silently promoted to base dictionary.
Universal dictionary requires explicit promotion evidence and is not part of v1 experiments.

MLP is out of v1 scope.
Level2 promotion is out of v1 scope.
QK/VO Level1 verification must use fixed split protocol.

Do not use KL, coefficient L2, alpha sweep, LoRA/SVD fitting, learned base dictionaries,
Procrustes/CCA, or gradient descent as the main transfer mechanism.

Fields and conditions must match by exact enum/string equality or explicit alias table.
Raw weight passthrough is rejected.
PASS is per-head/per-layer; averages cannot hide failed heads.
Manifest must report closure_level: circuit_target | weights | full_forward_logits.
Cross-model head/KV mismatch blocks direct transplant.
If exact closure fails, report the missing primitive/residual. Do not hide failure behind approximate fitting.
```
