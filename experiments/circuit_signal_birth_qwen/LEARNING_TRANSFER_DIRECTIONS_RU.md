# Обучение, перенос знаний и новые направления на базе Circuit-Target Signal Birth

Этот документ фиксирует расширенный план развития идеи **circuit-target signal birth** после первых успешных экспериментов на Qwen.

Ключевая мысль:

```text
Если мы умеем читать точные circuit targets модели, то можем работать не только с loss/gradient,
а с внутренними структурными операторами модели: routing, write, residual, MLP/Jacobian circuits.
```

Текущий статус:

```text
1. QK signal birth доказан на реальном Qwen.
2. VO signal birth работает как per-head/private operator при prompt-heldout validation.
3. Circuit-birth protection в LoRA fine-tune снижает drift защищённых QK/VO BirthOps на 97–99% при сохранении edit success.
```

Теперь вопрос: можно ли из этого сделать не только безопасное обучение, но и улучшение обучения, перенос знаний между моделями, model merging и точечную сборку способностей?

---

## 1. Базовая концепция

Обычное обучение смотрит на модель так:

```text
input -> model -> loss -> gradient -> update weights
```

Проблема: gradient говорит, как уменьшить loss, но почти не объясняет, какие внутренние механизмы модели меняются.

Наша схема добавляет структурный слой:

```text
weights / activations
-> exact circuit targets
-> operator decode
-> residual
-> signal birth
-> structural signals
-> protect / route / transfer / merge
```

Для внимания:

```python
M_qk_aug[h, delta] = Wq_aug[h].T @ R_delta @ Wk_aug[kv] / sqrt(head_dim)
C_vo_aug[h]        = Wo[h] @ Wv_aug[kv]
```

Для MLP позже нужно добавить:

```text
MLP effective Jacobian
activation-conditioned Jacobian
product-step decoder
residual operator mining
```

Главное отличие от обычной интерпретации:

```text
Мы не просто смотрим на attention maps.
Мы строим точные матрицы circuit-эффекта и майним из них новые operators/BirthOps.
```

---

## 2. Безопасное обучение: что уже есть

Сейчас уже есть proof-of-concept:

```text
baseline LoRA fine-tune
vs
LoRA + circuit-birth protected-subspace regularizer
```

Результат:

```text
edit accuracy сохранился 100%
QK drift reduction примерно 97–99%
VO drift reduction примерно 97–99%
retain accuracy чаще лучше
retain loss пока смешанный
```

То есть доказано:

```text
BirthOps можно использовать как protected subspace во время fine-tune.
```

Формула защиты:

```text
до обучения:
  c0 = <CircuitTarget, BirthOp>

во время обучения:
  c = <CircuitTarget_current, BirthOp>
  penalty = |c - c0| / (|c0| + eps)
```

Смысл:

```text
не запрещать обучение вообще,
а запрещать ломать конкретные важные circuit-направления.
```

---

## 3. Как перейти от safe training к улучшению обучения

Safe training отвечает на вопрос:

```text
как не сломать важные circuits?
```

Улучшение обучения отвечает на другой вопрос:

```text
как учиться быстрее, точнее и с меньшим лишним drift?
```

Для этого нужны не только protect penalties, а активное направление update.

### 3.1 Birth-guided LoRA initialization

Сейчас LoRA обычно стартует случайно.

Идея:

```text
найти residual/BirthOp, который нужен для edit/task,
разложить его low-rank,
инициализировать LoRA_A/LoRA_B из этого направления.
```

Для VO это проще:

```text
C_vo residual -> SVD -> LoRA init for V/O branch
```

Для QK сложнее, потому что:

```text
M_qk = Wq.T @ R @ Wk
```

Но можно искать факторизацию BirthOp в Q/K space:

```text
BirthOp_M ≈ ΔWq.T @ R @ Wk + Wq.T @ R @ ΔWk
```

Ожидаемый эффект:

```text
меньше steps_to_edit_success
меньше random drift
лучше retain
меньше нужный LoRA rank
```

### 3.2 Birth-routed gradient update

Идея:

```text
gradient показывает направление снижения loss,
а circuit birth показывает структурное подпространство,
где update должен происходить.
```

Можно проектировать gradient:

```text
grad_projected = project(grad, allowed_birth_subspace)
```

Или наоборот запрещать вредное:

```text
grad_safe = grad - project(grad, protected_birth_subspace)
```

Три режима:

```text
protect-only: не ломать важные BirthOps
route-only: обновлять только нужные BirthOps
protect+route: обновлять нужные, не ломать важные
```

### 3.3 Closed-loop update accept

После каждого N шагов:

```text
1. сделать update
2. извлечь circuit targets
3. измерить:
   - edit circuit improved?
   - protected circuit drift?
   - retain loss?
4. если update плохой:
   - откатить
   - уменьшить lr
   - усилить penalty
   - сменить update mask
```

Это превращает fine-tune в замкнутый контур:

```text
loss -> update -> circuit check -> accept/reject/regularize
```

### 3.4 Residual-targeted learning

Если после edit остаётся residual:

```text
residual = target_effect - explained_effect
```

то signal birth создаёт новый operator, а обучение целится в него:

```text
minimize residual coefficient error
instead of only CE loss
```

Это может быть полезно для model editing:

```text
не просто заставить модель отвечать правильно,
а встроить нужный внутренний operator.
```

---

## 4. Перенос знаний между моделями

Это одно из самых интересных направлений.

Обычная distillation:

```text
teacher logits -> student logits
teacher hidden -> student hidden
```

Circuit transfer:

```text
teacher circuits -> operator program -> student circuits
```

То есть переносим не только ответы, а внутренние структурные причины поведения:

```text
куда смотреть
что писать
какой residual operator нужен
какой MLP/Jacobian step делает вычисление
```

### 4.1 Что именно переносить

Не raw weights, а:

```text
QK BirthOps
VO BirthOps
operator coefficients
circuit deltas before/after edit
MLP Jacobian BirthOps
product-step programs
protected subspaces
```

Пример:

```text
Teacher после code fine-tune имеет новые QK routing operators.
Student получает loss на появление похожих QK operator coefficients.
```

### 4.2 Same-model transfer: первый минимальный тест

Самый простой тест:

```text
Teacher = Qwen2.5-0.5B после edit/fine-tune
Student = чистый Qwen2.5-0.5B
```

План:

```text
1. Fine-tune teacher на edit facts.
2. Извлечь circuit delta:
   ΔM_qk = M_qk_teacher_after - M_qk_teacher_before
   ΔC_vo = C_vo_teacher_after - C_vo_teacher_before
3. Decode/birth этих deltas.
4. Обучить student не только CE loss, а circuit-delta loss:
   <M_student_current, BirthOp_delta> -> target coefficient
5. Проверить:
   - edit success
   - steps_to_success
   - retain
   - circuit drift
```

Так как архитектура одинаковая, alignment почти не нужен.

### 4.3 Same-family transfer: Qwen bigger -> Qwen smaller

Следующий уровень:

```text
Teacher = Qwen2.5-1.5B или 3B
Student = Qwen2.5-0.5B
```

Здесь уже размеры разные, нужен alignment.

Варианты alignment:

```text
activation PCA
Procrustes map
CCA / SVCCA
learned linear bridge
token-matched hidden states
operator coefficient matching instead of matrix matching
```

Лучший первый вариант:

```text
не пытаться перенести всю матрицу M_qk,
а перенести коэффициенты по общему operator dictionary.
```

То есть:

```text
teacher M_qk -> coefficients over operator bank
student M_qk -> coefficients over same/similar operator bank
loss = ||coeff_student - coeff_teacher||
```

Это легче, чем matching full matrices.

### 4.4 Cross-family transfer

Пример:

```text
Teacher = code-specialized model
Student = обычная instruct model
```

Цель:

```text
перенести не весь code model,
а конкретные code-routing/code-write operators.
```

Проблема:

```text
разные tokenizer, hidden basis, layer count, head count
```

Решение:

```text
переносить не raw circuits, а abstract operator bank + coefficients + functional validation.
```

Схема:

```text
Teacher:
  extract code-task BirthOps
  group into operator families

Student:
  find analogous BirthOps/residuals
  train to reproduce teacher operator coefficients on code prompts
  validate by code task loss
```

### 4.5 Можно ли переносить знания за один проход?

Честная оценка:

```text
полное знание модели за один проход перенести нельзя.
```

Но возможно:

```text
перенести точечный circuit delta за 1-несколько проходов,
если модели близки и alignment известен.
```

Примеры возможного fast transfer:

```text
один fact/edit
один routing pattern
один code operator
один safety refusal pattern
один memory/write operator
```

То есть не:

```text
вся модель -> вся модель
```

а:

```text
конкретная способность / circuit patch -> student
```

---

## 5. Комбинирование моделей и model merging

Обычный model merging смешивает веса:

```text
W_merge = alpha W_A + (1-alpha) W_B
```

Проблема:

```text
raw weights могут быть в разных базисах;
полезные circuits смешиваются и ломаются.
```

Circuit-level merging:

```text
model A -> BirthOps/coefs
model B -> BirthOps/coefs
merge -> выбрать совместимые operators/coefs
```

Пример:

```text
Model A: сильные code operators
Model B: сильные math operators
Merged model: сохраняем code BirthOps A + math BirthOps B
```

План merge:

```text
1. Extract circuits from both models.
2. Decode into common operator dictionary.
3. Find overlapping operators.
4. Find conflicting operators.
5. Merge coefficients with validation.
6. Apply as LoRA/circuit patch to base model.
```

Метрики:

```text
code task
math task
retain task
circuit drift
operator conflict score
```

---

## 6. Model diff: понять, что изменилось после обучения

Сравнить:

```text
base model vs instruct model
before fine-tune vs after fine-tune
teacher vs student
successful edit vs failed edit
```

Не по raw weights, а по:

```text
ΔM_qk BirthOps
ΔC_vo BirthOps
ΔMLP Jacobian BirthOps
operator coefficient drift
newborn operators
dead operators
```

Отчёт может быть таким:

```text
Layer 23:
  QK_Birth_0 increased +0.32
  VO_Head_4_Write_Operator decreased -0.12
  protected operator drift 0.00008
  new residual operator appeared in heads 2,5,6
```

Это гораздо информативнее, чем:

```text
loss changed
weight norm changed
```

---

## 7. Compression через circuit programs

Обычное сжатие:

```text
SVD/quantization/pruning raw weights
```

Наш вариант:

```text
M_qk/C_vo -> operator dictionary + BirthOps + coefficients
```

Вместо хранения полной матрицы:

```text
M ≈ Σ c_i Op_i
```

Преимущества:

```text
интерпретируемо
можно защищать/редактировать отдельные operators
можно переносить operators между моделями
можно искать общие operator bank
```

Недостаток:

```text
пока QK лучше, VO требует более компактный словарь
```

---

## 8. Architecture search и operator bank

Если signal birth постоянно рождает похожий operator:

```text
в разных слоях
в разных головах
после разных fine-tune
в разных моделях
```

то этот operator надо добавить в базовый словарь или даже в архитектуру.

Пример:

```text
QK_BirthPCA0 появляется в L2/L6/L23.
Значит это общий routing primitive.
```

Дальше:

```text
1. собрать банк BirthOps по моделям;
2. кластеризовать;
3. выделить shared primitives;
4. добавить их в decoder;
5. проверить, падает ли residual без нового birth.
```

Это превращает signal birth в NAS не для слоёв, а для внутренних операторов.

---

## 9. Интерпретируемые отчёты обучения

Вместо обычного лога:

```text
step 100 loss=...
```

можно писать:

```text
step 100:
  edit_loss=...
  retain_loss=...
  QK protected drift=...
  VO protected drift=...
  newborn QK operator detected in L23/H4
  old routing operator preserved
  write operator shifted in H2
```

Это даёт разработчику карту изменений модели.

---

## 10. Возможные продукты/инструменты

### 10.1 Circuit Inspector

CLI:

```bash
python circuit_inspector.py --model ... --layers 2,6,23
```

Вывод:

```text
operator inventory
BirthOps
residual maps
important heads
protected subspaces
```

### 10.2 Circuit-Aware FineTune

CLI:

```bash
python circuit_finetune.py --protect-source birth --protect-part both
```

Смысл:

```text
fine-tune с защитой важных circuits.
```

### 10.3 Circuit Transfer

CLI:

```bash
python circuit_transfer.py --teacher ... --student ... --task code
```

Смысл:

```text
перенести operator coefficients / circuit deltas.
```

### 10.4 Circuit Merge

CLI:

```bash
python circuit_merge.py --model-a code --model-b math --base qwen
```

Смысл:

```text
слияние способностей через operator bank.
```

---

## 11. Главные риски

### 11.1 Proxy не равен downstream

Circuit metrics могут улучшаться, а downstream task — нет.

Решение:

```text
всегда проверять task loss / retain / edit / functional validation.
```

### 11.2 BirthOp может быть слишком private

Особенно VO.

Решение:

```text
QK: shared birth
VO: per-head/per-cluster birth
MLP: activation-conditioned birth
```

### 11.3 Alignment между моделями сложный

Решение:

```text
начинать с same-model transfer,
потом same-family,
потом cross-family.
```

### 11.4 Random regularizer может выглядеть сильным

Решение:

```text
обязательный random/base-dict ablation.
```

---

## 12. Ближайший кодовый план

### v1.4: experiment runner

Добавить:

```text
--protect-source birth/random/base-dict
--protect-part qk/vo/both
--summary-csv
--summary-json
--lambda-sweep
--retain-size
```

### v1.5: Birth-guided LoRA init

Добавить:

```text
--lora-init random/birth-vo/birth-qk
```

Метрики:

```text
steps_to_edit_success
edit_loss
retain_loss
circuit_drift
```

### v1.6: Circuit delta transfer same-model

Добавить:

```text
teacher after edit -> extract circuit delta
student clean -> train on circuit delta loss
```

### v1.7: Model diff report

Добавить:

```text
before/after circuit diff
operator coefficient changes
new/dead BirthOps
```

---

## 13. Главный итог

Circuit-target signal birth открывает не один метод, а целое семейство методов:

```text
1. Interpretability:
   понять, какие operators есть в модели.

2. Safe training:
   защищать важные circuits.

3. Better training:
   направлять update в правильные structural subspaces.

4. Transfer:
   переносить не logits, а circuit operators.

5. Merge:
   комбинировать модели по operator bank.

6. Compression:
   хранить circuit programs вместо raw matrices.

7. Architecture search:
   превращать часто рождающиеся BirthOps в primitives.
```

Текущий доказанный результат:

```text
Circuit-birth protection уже снижает drift защищённых QK/VO компонентов на 97–99% при сохранении edit success.
```

Следующая цель:

```text
доказать, что birth-protect лучше random/base-dict protect,
и что birth-guided update/LoRA-init ускоряет обучение или улучшает retain.
```
