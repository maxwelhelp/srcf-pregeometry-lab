# Результаты: Circuit-Target Signal Birth

## 1. Controlled hard compose test

Цель: проверить, может ли signal birth не просто выбрать готовый сигнал, а собрать формулу из примитивов.

Вывод:

- compose miner работает;
- sequential residual refit нужен, иначе возможны слабые ложноположительные вторые birth;
- accept должен быть residual-aware: train + heldout + retain + non-duplicate.

Это была синтетика/контроль, не финальное доказательство.

---

## 2. Реальный Qwen: circuit-target signal birth

Файл: `real_qwen_circuit_signal_birth_v1_2.py`

Модель: `Qwen/Qwen2.5-0.5B-Instruct`

Слои: `L2`, `L6`, `L23`

Проверялись реальные:

- `M_qk_aug[h, delta]`
- `C_vo_aug[h]`
- real prompts
- heldout heads / heldout prompts
- functional метрики `score_rel`, `A_rel`, `KL`, `top1`, `Y_all_rel`, `H_after_rel`

### QK BirthOps

QK birth делался shared across heads/deltas. Результат по heldout `M_qk`:

| Layer | base heldM | birth heldM | улучшение ошибки |
|---:|---:|---:|---:|
| L2 | 0.9190 | 0.2428 | ~73.6% |
| L6 | 0.9194 | 0.2552 | ~72.2% |
| L23 | 0.9191 | 0.2734 | ~70.3% |

Вывод: QK circuit birth доказан. В Qwen есть повторяющийся routing language, который базовый словарь не покрывал, а PCA BirthOps нашли.

### Functional routing

| Layer | A_rel base | A_rel birth | top1 base | top1 birth |
|---:|---:|---:|---:|---:|
| L2 | 1.4040 | 0.6510 | 0.233 | 0.497 |
| L6 | 0.9765 | 0.7779 | 0.335 | 0.481 |
| L23 | 1.1974 | 0.4259 | 0.264 | 0.753 |

Это real proxy-улучшение routing, но не downstream task accuracy.

---

## 3. VO: что решено

Сначала VO проверялся как shared между головами. Это почти не работало.

Вывод: VO не должен переноситься между heads так же, как QK.

Правильная политика:

```text
QK = shared routing language
VO = private/head-specific write language
```

В `v1.2` VO birth делается per-head и валидируется по train/heldout prompts через `Y_all/H_after`.

С включённым `--vo-include-full-residual` получили сильный upper-bound/private-operator test:

| Layer | qkonly Y_all | birth Y_all | qkonly H_after | birth H_after |
|---:|---:|---:|---:|---:|
| L2 | 1.0028 | 0.6968 | 0.2900 | 0.2015 |
| L6 | 1.0162 | 0.9760 | 0.0075 | 0.0072 |
| L23 | 0.9998 | 0.4700 | 0.2156 | 0.1014 |

Вывод: VO functional usefulness доказана, но compact VO dictionary ещё не закрыт. `full residual` — это верхняя граница, не финальный компактный оператор.

---

## 4. Fine-tune protection test

Файл: `circuit_birth_finetune_protect_v1_3.py`

Тест: baseline LoRA fine-tune vs LoRA + circuit-birth protected-subspace regularizer.

Настройка стресс-теста:

- layer `23`
- heads `0..6`
- deltas `0..2`
- LoRA rank `8`
- steps `160`
- `lambda_protect=10.0`
- `protect-loss=l1`

Результат:

| Метрика | Baseline | Protect | Вывод |
|---|---:|---:|---|
| edit loss | 0.0016 | 0.0021 | почти одинаково |
| edit acc | 1.000 | 1.000 | edit выучен полностью |
| retain loss | 2.4559 | 2.4603 | почти одинаково |
| retain acc | 0.577 | 0.615 | protect лучше на +3.8 п.п. |
| QK coeff drift | 0.002819 | 0.000039 | drift меньше на 98.60% |
| VO coeff drift | 0.014717 | 0.000213 | drift меньше на 98.55% |

Главный вывод:

```text
Circuit-birth protection сохранил edit accuracy 100%, почти не ухудшил edit loss, и снизил drift защищённых QK/VO BirthOps примерно на 98.5%.
```

Это уже доказывает практическую применимость как circuit-aware regularizer / protected subspace.

---

## 5. Что можно говорить честно

Можно говорить:

- circuit-target signal birth доказан как structural signal channel на реальном Qwen;
- QK BirthOps обобщаются между головами;
- VO требует per-head prompt validation;
- protected-subspace regularizer резко снижает drift защищённых circuit-компонентов при сохранении edit success;
- в нашем stress test drift снизился примерно на 98.5%.

Нельзя пока говорить:

- “обучение стало лучше на 98%”;
- “метод гарантированно снижает forgetting”;
- “VO полностью закрыт компактным словарём”.

Корректная формулировка:

```text
Метод уменьшает circuit drift на 98.5% при сохранении edit success в первом LoRA stress test. Это сильный proof-of-concept для circuit-aware training, но нужен multi-seed и real downstream benchmark.
```
