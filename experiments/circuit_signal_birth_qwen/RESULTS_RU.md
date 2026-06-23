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

## 4. Fine-tune protection test v1.3

Файл: `circuit_birth_finetune_protect_v1_3.py`

Тест: baseline LoRA fine-tune vs LoRA + circuit-birth protected-subspace regularizer.

Цель: проверить не только reconstruction/interp, а можно ли использовать найденные BirthOps как training-control signal.

### 4.1 Stress run: сильная защита

Настройка:

- layer `23`
- heads `0..6`
- deltas `0..2`
- LoRA rank `8`
- steps `200`
- lr `3e-4`
- `lambda_protect=50.0`
- `protect-loss=l1`

Результат:

| Метрика | Baseline | Protect | Вывод |
|---|---:|---:|---|
| edit loss | 0.0011 | 0.0022 | protect чуть хуже, но разница мала |
| edit acc | 1.000 | 1.000 | edit выучен полностью |
| retain loss | 2.4598 | 2.4364 | protect лучше |
| retain acc | 0.577 | 0.577 | одинаково |
| QK coeff drift | 0.002154 | 0.000060 | drift меньше на 97.22% |
| VO coeff drift | 0.017149 | 0.000079 | drift меньше на 99.54% |

Вывод stress run:

```text
Protect сохранил edit accuracy 100%, улучшил retain loss на 0.0234 и резко снизил drift защищённых QK/VO components: QK -97.22%, VO -99.54%.
```

### 4.2 Multi-seed check: seeds 123/124/125

Настройка:

- layer `23`
- heads `0..6`
- deltas `0..2`
- LoRA rank `8`
- steps `160`
- lr `2e-4`
- `lambda_protect=10.0`
- `protect-loss=l1`

Сводка по трём seed:

| Метрика | Baseline avg | Protect avg | Вывод |
|---|---:|---:|---|
| edit acc | 1.000 | 1.000 | edit success сохранён |
| edit loss | ~0.00163 | ~0.00207 | protect чуть хуже на ~0.00043 |
| retain loss | ~2.4908 | ~2.5020 | protect чуть хуже на ~0.0112 |
| retain acc | ~0.5767 | ~0.6023 | protect лучше на ~+2.6 п.п. |
| QK drift reduction | — | ~98.49% | стабильно большой эффект |
| VO drift reduction | — | ~98.12% | стабильно большой эффект |

По seed:

| Seed | QK drift reduction | VO drift reduction | edit acc | retain acc baseline → protect |
|---:|---:|---:|---:|---:|
| 123 | 98.60% | 98.55% | 1.000 | 0.577 → 0.615 |
| 124 | 99.01% | 98.14% | 1.000 | 0.615 → 0.615 |
| 125 | 97.87% | 97.66% | 1.000 | 0.538 → 0.577 |

Главный вывод multi-seed:

```text
Circuit-birth protection стабильно сохраняет edit success и снижает drift защищённых QK/VO circuit-компонентов примерно на 97–99%. Retain accuracy чаще лучше, retain loss пока смешанный/почти нейтральный.
```

---

## 5. Что можно говорить честно

Можно говорить:

- circuit-target signal birth доказан как structural signal channel на реальном Qwen;
- QK BirthOps обобщаются между головами;
- VO требует per-head prompt validation;
- protected-subspace regularizer резко снижает drift защищённых circuit-компонентов при сохранении edit success;
- в stress run drift снизился на 97.22% по QK и 99.54% по VO;
- в multi-seed check drift reduction стабилен: примерно 98.49% QK и 98.12% VO.

Нельзя пока говорить:

- “обучение стало лучше на 98%”;
- “метод гарантированно снижает forgetting”;
- “VO полностью закрыт компактным словарём”;
- “метод доказан на больших downstream benchmark”.

Корректная формулировка:

```text
Метод уменьшает drift защищённых circuit-компонентов примерно на 97–99% при сохранении edit success в LoRA fine-tune. Это сильный proof-of-concept для circuit-aware training. Следующий обязательный шаг — ablation против random/base-dict protection и более честный downstream/retain benchmark.
```

---

## 6. Текущая оценка

```text
Circuit signal birth как structural signal channel: 9/10
Fine-tune protection как proof-of-concept: 8/10
Anti-forgetting доказательство: 5.5/10 пока
Готовый training method: 6/10 пока
```

Почему сильно:

- real Qwen;
- real LoRA fine-tune;
- edit task выучен полностью;
- protected circuit drift стабильно падает на 97–99%;
- эффект повторяется на нескольких seed.

Почему не финал:

- edit/retain набор маленький;
- retain loss не всегда лучше;
- нет random-protect ablation;
- нет QK-only / VO-only сравнения;
- нет lambda sweep;
- нет настоящего downstream benchmark.
