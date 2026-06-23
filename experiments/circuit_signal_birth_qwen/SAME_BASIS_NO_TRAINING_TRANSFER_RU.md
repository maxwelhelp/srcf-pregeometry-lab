# Same-Basis No-Training Transfer

## Что проверяем сейчас

Текущий быстрый proof — не cross-model universal transfer, а same-basis / same-architecture transfer.

Цель:

```text
Qwen circuit target -> autoexpanded program -> saved bundle -> replay без обучения
```

Если replay закрывается, значит знание/операторный патч можно хранить как программу и переносить внутри того же базиса без gradient training.

## Главное разделение

### Same-basis / same-architecture

Пример:

```text
Qwen checkpoint -> same Qwen architecture / same residual basis
```

В этом режиме private exact atoms разрешены.

Причина простая: если базис тот же, private residual atom уже находится в правильной системе координат. Ему не нужно быть universal primitive.

Для same-basis режима достаточно доказать:

```text
saved program bundle -> replay -> circuit target closed
```

### Cross-model / different-basis

Пример:

```text
Qwen-Coder -> Qwen-base
Qwen larger -> Qwen smaller
другая модель / другой residual basis
```

В этом режиме private atoms сами по себе не гарантируют перенос. Нужны:

```text
basis alignment
structural match
rotation / translation of atoms
student closure check
```

Universal / transferable metadata нужна для этого будущего этапа, но не должна блокировать текущий same-basis proof.

## Текущие результаты

### Level0 analytic dictionary

```text
QK error ~= 1.000
```

Базовый аналитический словарь почти не объясняет real Qwen QK.

### Joint all-head shared program

```text
QK heldout error: 1.000 -> 0.142
gain ~= 85.8%
```

Вывод: QK routing language распределён по всем головам слоя. Читать heads отдельно неправильно.

### Autoexpanded same-basis program

```text
shared atoms:        8
head-specific atoms: 16
delta-bucket atoms:  12
private exact atoms: 210
total atoms:         246

final_all_err_mean ~= 1e-9
final_max_err      ~= 4e-9
```

Это доказывает near-zero same-basis QK circuit closure без обучения.

## Почему 210 private atoms не ломают proof

Для proof-of-possibility нам важно сначала проверить:

```text
можно ли закрыть QK circuit target программой без gradient training
```

Ответ: да.

Private atoms маркируются так:

```json
{
  "universal": false,
  "cross_model_transferable": false,
  "allowed_for_same_arch_transplant": true
}
```

То есть они разрешены для same-basis transfer, но не являются claim для cross-model transfer.

## Следующий обязательный gate

Файл:

```text
qwen_same_basis_program_replay_v1.py
```

Он проверяет:

```text
saved qk_autoexpand_atoms.pt
+ saved gates from per_matrix_autoexpand_closure.jsonl
+ re-extracted Qwen M_qk targets
-> replay closure
```

PASS:

```text
SAME_BASIS_PROGRAM_REPLAY_CLOSED
program_err_mean < 1e-3
missing_atom_count = 0
```

## После replay PASS

1. QK program intervention replay:

```text
M_program -> score -> A
```

2. QK+VO functional replay:

```text
QK program + VO program -> Y_all/H_after
```

3. Compression:

```text
210 private exact atoms -> private family dictionary / SVD / clusters / grammar
```

4. Cross-model basis alignment:

```text
x_teacher ~= x_student @ A
M_student = A @ M_teacher @ A.T
```

## Главное правило

Не требовать universal primitive для same-basis proof.

Universal / transferable metadata — это будущий cross-model этап. Текущий этап должен доказать saved program replay без обучения в том же базисе.
