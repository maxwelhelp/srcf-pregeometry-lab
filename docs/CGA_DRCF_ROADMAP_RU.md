# Error-Wave / CGA / DRCF: текущий статус и план проверок

## 1. Где мы сейчас

Проект ушёл от общей идеи “самозамыкающегося поля” к более конкретной и проверяемой задаче:

> как дообучать модель на новом конфликтном контексте так, чтобы не ломать старое поведение.

Практически это сейчас выглядит как sparse gradient update:

1. считаем patch-gradient;
2. считаем retain-gradient;
3. выбираем малую часть весов;
4. применяем обновление только там, где оно полезно для patch и безопасно для retain.

Главный рабочий механизм на текущий момент:

```text
error_wave_v4:
score = |g_patch| - retain_penalty * |g_retain|
```

Он отвечает на вопрос:

> какие веса сильнее виновны в ошибке patch-контекста, но меньше нужны retain-контексту?

## 2. Что уже показали тесты

### 2.1 Transformer benchmark

На реальном `distilgpt2` в конфликтном тесте error-wave показал важный результат:

- почти full-backprop patch learning;
- sparse update около 1.9%;
- меньше retain degradation, чем full_bp/LoRA;
- конкурентен magnitude_sparse.

Честная формулировка:

> Error-Wave не “рвёт всех”, а находится на Pareto-фронте: почти тот же patch-effect, но с меньшим retain-risk.

### 2.2 Magnitude sparse оказался сильным baseline

`magnitude_sparse` делает очень простую вещь:

```text
score = |g_patch|
```

То есть обновляет веса с самым большим patch-градиентом.

Это сильный baseline, потому что он часто почти догоняет error-wave по patch. Поэтому главный вопрос:

> зачем нужен error-wave, если top-|grad| почти так же хорош?

Ответ после диагностики:

- маски error-wave и magnitude совпадают далеко не полностью;
- error-wave выбирает веса с меньшим retain-risk;
- error-wave может быть не сильнее по patch, но безопаснее по retain.

## 3. CGA — Causal Gradient Arbitration

### 3.1 Идея

Error-wave отвечает на вопрос:

> какой вес важен для patch и не слишком важен для retain?

Но он не отвечает на другой вопрос:

> безопасно ли само направление обновления этого веса?

CGA добавляет арбитраж направления.

Для веса `W[j,i]` есть:

```text
g_patch[j,i]  — градиент patch-loss
g_retain[j,i] — градиент retain-loss
```

Обычное обновление:

```text
W[j,i] -= lr * g_patch[j,i]
```

Локальная аппроксимация изменения retain-loss:

```text
retain_delta ≈ -lr * g_retain[j,i] * g_patch[j,i]
```

Отсюда:

```text
если g_patch * g_retain > 0:
    update -g_patch снижает retain-loss или согласован с retain
    направление безопасно

если g_patch * g_retain < 0:
    update -g_patch повышает retain-loss
    направление конфликтное
```

### 3.2 CGA-soft

Берём v4-кандидатов и масштабируем их:

```python
align = g_patch * g_retain / (abs(g_patch) * abs(g_retain) + eps)
scale = sigmoid(beta * align)
final_grad = g_patch * wave_mask * scale
```

Смысл:

> не выкидывать вес полностью, а уменьшать опасные направления.

### 3.3 CGA-hard-refill

Берём только безопасные направления:

```python
safe = (g_patch * g_retain >= 0)
score = wave_score * safe
candidate_mask = topk(score)
final_grad = g_patch * candidate_mask
```

Смысл:

> если направление конфликтует с retain, не применяем его вообще; добираем top-k из безопасных кандидатов.

## 4. Результат CGA v8

На toy-context benchmark:

```text
full_bp:
  patch_gain  = +0.4630
  retain_drop = +0.3004

magnitude_sparse:
  patch_gain  = +0.4530
  retain_drop = +0.2210

error_wave_v4:
  patch_gain  = +0.4592
  retain_drop = +0.1766

cga_soft:
  patch_gain  = +0.4285
  retain_drop = +0.1059

cga_hard_refill:
  patch_gain  = +0.4309
  retain_drop = +0.0798
```

Вывод:

```text
error_wave_v4 = aggressive mode
  почти full_bp/magnitude по patch
  меньше retain damage

cga_hard_refill = safe mode
  patch немного ниже
  retain damage сильно ниже
```

CGA-hard-refill потерял около 6% patch-силы относительно v4, но снизил retain_drop примерно в 2.2 раза.

Это не замена v4, а второй режим:

```text
v4  — когда нужно сильнее выучить patch
CGA — когда нужно сильнее сохранить retain
```

## 5. DRCF — Dynamic Relation Credit Field

### 5.1 Идея

Сейчас error-wave/CGA каждый шаг решает заново:

```text
какие веса обновлять сейчас?
```

Но модель может запоминать историю конфликтов.

DRCF хранит EMA-поле конфликтов:

```python
conflict = (g_patch * g_retain < 0).float()

conflict_history[layer] = (
    0.95 * conflict_history[layer]
    + 0.05 * conflict * candidate_mask.float()
)
```

Потом это используется как prior:

```python
safe_prior = 1.0 - conflict_history[layer]
score = wave_score * safe_prior
```

Смысл:

> если вес раньше часто ломал retain, в будущем он получает штраф ещё до арбитража.

Это “память о конфликтных весах”, не память о данных.

## 6. Orthogonal Gradient Update

Ещё один режим — не просто выбрать/не выбрать вес, а изменить направление градиента.

На уровне слоя:

```python
G_p = grad_patch
G_r = grad_retain

proj = (G_p * G_r).sum() / (G_r.norm()**2 + eps) * G_r
G_orth = G_p - proj
```

Обновляем только компоненту patch-gradient, ортогональную retain-gradient:

```python
W -= lr * G_orth * candidate_mask
```

Смысл:

> не запрещать обновление, а удалить из него часть, которая пересекается с retain-направлением.

Это ближе к gradient surgery, но в комбинации с error-wave prefilter.

## 7. Что тестировать дальше

### Тест 1 — CGA на transformer benchmark

Перенести `cga_hard_refill` из toy-кода в transformer publish benchmark.

Сравнить:

```text
full_bp
random_sparse
magnitude_sparse
DARE
LoRA
error_wave_v4
CGA-hard-refill
```

Главная проверка:

```text
CGA-hard-refill должен иметь:
  patch_improve немного ниже v4
  forget_loss ниже v4/full_bp/LoRA
```

Если это подтвердится на `distilgpt2`, CGA становится важной частью paper.

### Тест 2 — Qwen 0.5B conflict benchmark

То же самое на Qwen:

```text
Qwen/Qwen2.5-0.5B
patch_mode=random_vocab
retain=WikiText
methods: full_bp, magnitude, error_wave_v4, CGA
```

Даже seed=0 уже полезен, потому что Qwen сильно дороже.

### Тест 3 — DRCF

Добавить `conflict_history`:

```text
error_wave_v4
error_wave_v4 + DRCF
CGA-hard-refill
CGA-hard-refill + DRCF
```

Проверить:

```text
уменьшается ли retain_drop без сильной потери patch_gain?
```

### Тест 4 — Orthogonal update

Добавить режим:

```text
orthogonal_gradient
wave_prefilter + orthogonal_gradient
```

Ожидание:

```text
patch ниже v4
retain лучше v4
```

Это может стать третьим режимом:

```text
aggressive: v4
safe: CGA-hard
surgery: orthogonal
```

## 8. Самая сильная текущая формулировка

```text
Error-Wave is a blame-directed sparse gradient update method.
It selects a small subset of parameters that are locally responsible for a patch objective while avoiding retain-critical directions.

CGA extends Error-Wave with per-weight causal arbitration:
before applying an update, it checks whether the patch-gradient direction is locally aligned or conflicting with the retain-gradient.

Together, Error-Wave and CGA form a controllable Pareto trade-off:
aggressive patch learning with v4, or safer retain-preserving updates with CGA.
```

По-русски:

> Error-Wave выбирает виновные веса, а CGA решает, безопасно ли обновлять их в текущем направлении.

## 9. Текущий статус

```text
Готово:
  error_wave_v4 toy
  error_wave_v4 transformer
  magnitude baseline
  LoRA baseline
  CGA toy

Нужно:
  CGA transformer
  DRCF toy
  DRCF transformer
  orthogonal gradient toy
  Qwen conflict benchmark
```

## 10. Публикационный claim

Не завышать:

```text
не “мы решили catastrophic forgetting”
не “мы лучше LoRA во всём”
не “новый backprop”
```

Честный claim:

```text
Blame-directed sparse updates can match most of full fine-tuning's patch improvement while updating only a small fraction of trainable parameters.

Compared to magnitude sparse updates, Error-Wave selects masks with lower retain-risk.

CGA further shifts the method toward safer retain-preserving updates by arbitrating gradient directions per weight.
```

Это уже нормальная основа для workshop/short paper.
