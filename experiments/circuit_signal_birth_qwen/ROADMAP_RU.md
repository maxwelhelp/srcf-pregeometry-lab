# Roadmap: Circuit-Target Signal Birth дальше

## 1. Что уже есть

У нас есть три уровня результата:

```text
1. Interpretation:
   circuit targets можно читать и майнить BirthOps.

2. Structural signal:
   QK/VO BirthOps реально влияют на functional behavior.

3. Training control:
   BirthOps можно защищать во время LoRA fine-tune,
   и drift этих компонентов падает на 97–99% при сохранении edit success.
```

Это означает, что circuit-target signal birth уже работает как structural signal channel.

---

## 2. Ближайший технический план

### 2.1 v1.4: нормальный runner + CSV/JSON

Нужно перестать читать результаты руками из терминала.

Добавить:

```text
runs/circuit_protect_v1_4/
  summary.csv
  summary.json
  per_seed.jsonl
```

Колонки:

```text
seed
mode
protect_source
lambda_protect
edit_loss_before
edit_loss_after
edit_acc_after
retain_loss_after
retain_acc_after
qk_drift
vo_drift
qk_drift_reduction
vo_drift_reduction
```

### 2.2 Random/base-dict ablation

Главный критический вопрос:

```text
BirthOps особенные или любой random regularizer тоже снижает drift?
```

Нужны режимы:

```text
baseline
birth_protect
random_protect
base_dictionary_protect
```

Если `birth_protect` даёт лучший tradeoff, чем `random_protect`, это уже сильное доказательство, что signal birth несёт полезную структуру.

### 2.3 QK-only / VO-only / QK+VO

Нужны режимы:

```text
QK-only protect
VO-only protect
QK+VO protect
```

Гипотеза:

```text
QK защищает routing.
VO защищает write/residual stream.
QK+VO должен давать лучший общий drift-control.
```

### 2.4 Lambda sweep

Нужно построить Pareto curve:

```text
lambda_protect = 0, 1, 3, 10, 30, 50
```

Метрики:

```text
edit_loss
retain_loss
retain_acc
qk_drift
vo_drift
```

Ищем sweet spot:

```text
минимальный drift
без ухудшения edit
без ухудшения retain
```

### 2.5 Более честный retain

Текущий retain маленький. Нужно расширить до 100–300 задач:

```text
math
facts
translation
short reasoning
code snippets
common knowledge
```

Только после этого можно говорить про forgetting серьёзнее.

### 2.6 Несколько слоёв

Сейчас основной fine-tune protection был на `L23`.

Проверить:

```text
layers 2
layers 6
layers 23
layers 2,6,23
```

---

## 3. Идея: перенос знаний между моделями через circuit targets

Это потенциально очень интересное направление.

Обычный перенос знаний:

```text
teacher logits -> student logits
или teacher hidden states -> student hidden states
```

Наш вариант:

```text
teacher weights/activations -> exact circuit targets -> BirthOps/operators -> student circuit targets
```

То есть переносить не просто ответы, а структурные операции:

```text
QK routing operator
VO write operator
MLP/Jacobian product operator
layer residual operator
```

### 3.1 Почему это может быть сильнее обычной distillation

Обычная distillation учит student имитировать поведение teacher на данных.

Circuit transfer может учить student внутренним причинам поведения:

```text
куда смотреть
что писать в residual stream
какие circuit components защищать
какие residual operators добавить
```

Это может быть особенно полезно для:

```text
маленькая модель <- большая модель
обычная модель <- code model
другая архитектура <- Qwen-like teacher
модель после fine-tune <- чистая base model
```

### 3.2 Реалистичный метод переноса

Нельзя просто скопировать матрицы, если размеры/базисы разные.

Нужно делать так:

```text
1. Teacher:
   extract M_qk_teacher, C_vo_teacher
   decode/operator birth
   получить набор операторов O_teacher

2. Alignment:
   найти map между hidden spaces teacher/student:
   A_in, A_out через activation PCA / Procrustes / CCA / learned linear map

3. Student:
   extract M_qk_student, C_vo_student
   добавить loss:
     student operator coefficients должны приблизиться к teacher operator coefficients
   или:
     student residual должен получить teacher BirthOp direction

4. Проверка:
   task loss
   teacher agreement
   retain
   circuit drift
```

### 3.3 Минимальный первый тест

Самый простой первый тест:

```text
Teacher = Qwen2.5-0.5B-Instruct fine-tuned/edited
Student = чистый Qwen2.5-0.5B-Instruct
```

То есть архитектура одинаковая. Тогда alignment почти не нужен.

Проверка:

```text
1. Сделать edit на teacher.
2. Извлечь changed BirthOps / changed coefficients.
3. Обучить student не по всем logits, а по circuit coefficient loss.
4. Проверить, получил ли student тот же edit при меньшем количестве шагов.
```

Если сработает, следующий шаг:

```text
Teacher = Qwen2.5-1.5B
Student = Qwen2.5-0.5B
```

Там уже нужен alignment.

### 3.4 Может ли быть перенос “за 1 проход”?

Честно: полный перенос знания за один проход почти точно не получится для сложных знаний.

Но возможен быстрый targeted transfer:

```text
за 1-несколько проходов перенести не всю модель,
а конкретный circuit delta / operator coefficient / edit subspace.
```

Особенно если:

```text
модели одной семьи;
есть alignment hidden spaces;
переносим не всё знание, а конкретный QK/VO/MLP operator;
проверяем functional accept.
```

---

## 4. Можно ли не только безопасно учить, но и улучшать обучение?

Да, но это отдельная ветка. Сейчас protection доказывает safe fine-tune: не ломать важные circuits.

Для улучшения обучения нужно не только защищать, но и направлять update.

### 4.1 Update routing

Идея:

```text
loss gradient говорит, что менять;
circuit birth говорит, в каком structural subspace менять.
```

Вместо:

```text
обновлять все LoRA directions
```

делать:

```text
обновлять только directions, которые уменьшают нужный BirthOp residual
и не ломают protected BirthOps.
```

### 4.2 Birth-guided LoRA initialization

Сейчас LoRA стартует случайно.

Можно инициализировать LoRA из BirthOps:

```text
LoRA_A/B стартуют из low-rank разложения нужного QK/VO residual operator.
```

Ожидание:

```text
меньше шагов до edit success
меньше лишнего drift
лучше retain
```

### 4.3 Residual-targeted learning

При fine-tune:

```text
измеряем circuit residual до/после;
если residual движется в полезную сторону — update разрешён;
если ломает protected circuits — penalty/rollback.
```

Это превращает обучение в closed-loop:

```text
loss -> update -> circuit check -> accept/reject/regularize
```

### 4.4 Что считать улучшением обучения

Нельзя мерить только circuit drift.

Нужны метрики:

```text
steps_to_edit_success
final_edit_loss
retain_loss
retain_acc
circuit_drift
gradient/update efficiency
```

Метод реально улучшает обучение, если:

```text
same edit success
fewer steps или lower retain damage или lower drift
```

---

## 5. Другие применения

### 5.1 Model editing / patching

Найти circuit residual, который отвечает за ошибку, и обновлять только его.

### 5.2 Anti-forgetting

Перед fine-tune сохранить important BirthOps и penalize drift.

### 5.3 Model merging

Сливать не веса напрямую, а operator coefficients:

```text
model A: code operators
model B: math operators
merge: choose compatible BirthOps / coefficients
```

### 5.4 Model diff

Сравнивать две модели:

```text
base vs instruct
before vs after fine-tune
teacher vs student
```

Не по raw weights, а по circuit operator changes.

### 5.5 Architecture search

Если residual постоянно рождает один и тот же operator, значит он должен быть явным модулем/словарным элементом.

### 5.6 Compression

Сжимать не raw weights, а circuit programs:

```text
M_qk/C_vo -> operator dictionary + BirthOps + coefficients
```

### 5.7 Interpretability reports

Для каждого слоя:

```text
какие QK routing operators есть
какие VO write operators есть
что изменилось после обучения
что защищено
что сломалось
```

---

## 6. Главный следующий эксперимент

Самый важный следующий код:

```text
circuit_birth_finetune_protect_v1_4.py
```

Функции:

```text
--protect-source birth/random/base-dict
--protect-part qk/vo/both
--lambda-sweep
--summary-csv
--summary-json
--retain-size 100+
```

Цель:

```text
доказать, что birth-protection лучше random/base protection,
и что это не просто любой penalty, а именно structural signal.
```
