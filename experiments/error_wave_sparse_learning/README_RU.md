# Error-Wave Sparse Learning

Экспериментальная ветка про локальное обучение по волне ошибки.

## Идея

Обычный backprop обновляет почти всю сеть от одного глобального loss. Здесь проверяется другой принцип:

```text
ошибка на конкретном контексте
→ blame / error-wave
→ локализованные виновные пути
→ sparse update только части весов
```

Цель: быстро патчить новое правило/поведение и меньше ломать старые контексты.

## Что тестируется

Сеть сначала учит несколько контекстных правил. Потом меняется только `context=0`. Нужно:

- выучить новый патч для `context=0`;
- сохранить старые правила для `context=1..3`;
- обновлять мало параметров.

## Метрики

- `patch` — точность нового правила для `context=0` после патча. Больше лучше.
- `retain` — сохранение старых правил для `context=1..3`. Больше лучше.
- `forget` — насколько старые контексты просели после патча. Меньше лучше.
- `orig` — точность на исходных правилах по всем контекстам после патча. Больше лучше.
- `dens` — плотность реально обновляемых градиентов/масок. Меньше = более sparse.

## Итог v4

`error_wave_v4` почти догнал полный backprop по обучению нового патча, но забывает намного меньше.

```text
full_bp:
  patch  = 0.952
  retain = 0.612
  forget = 0.316

random_sparse:
  patch  = 0.598
  retain = 0.827
  forget = 0.103

error_wave_v4:
  patch  = 0.944
  retain = 0.802
  forget = 0.128
```

Главный вывод:

```text
error_wave_v4 обновляет мало весов, но попадает в нужные пути лучше random_sparse.
```

Это пока не SOTA и не доказательство для больших моделей. Но это хороший исследовательский сигнал для направления:

```text
localized patch learning / continual learning without strong forgetting
```

## Что внутри v4

- context-aware blame: `blame_c0 - retain_penalty * blame_retain`;
- EMA маски;
- selectivity без бонуса мёртвым нейронам;
- weight-level mask для первого слоя;
- decay плотности в patch-фазе.

## Запуск

```bash
bash experiments/error_wave_sparse_learning/scripts/run_v4_quick.sh
```

Или напрямую:

```bash
python -u experiments/error_wave_sparse_learning/error_wave_sparse_learning_v4_weightlevel.py \
  --device cuda \
  --amp fp16 \
  --seeds 0,1,2 \
  --pretrain-steps 400 \
  --patch-steps 200 \
  --batch 128 \
  --mask-batch 128 \
  --eval-batch 512 \
  --eval-every 100 \
  --contexts 4 \
  --block-dim 12 \
  --hidden 96 \
  --layers 4 \
  --lr 2e-3 \
  --wave-frac 0.25 \
  --patch-wave-frac 0.08 \
  --patch-wave-frac-end 0.04 \
  --random-patch-density 0.014 \
  --input-frac 0.35 \
  --context-blame \
  --retain-penalty 0.5 \
  --wave-ema 0.90 \
  --selectivity-power 1.0 \
  --activity-threshold 0.05 \
  --weight-level-first \
  --frac-decay \
  --fisher-guard 0.0 \
  --results-csv experiments/error_wave_sparse_learning/results/error_wave_sparse_learning_v4_weightlevel.csv \
  | tee experiments/error_wave_sparse_learning/logs/error_wave_sparse_learning_v4_weightlevel.log
```

## Что делать дальше

Следующий честный шаг:

1. сравнять compute/плотность ещё строже;
2. добавить conflict-score / gradient-surgery;
3. сравнить с EWC/SI/LoRA-style локальным патчингом;
4. проверить на более сложной multi-context задаче.
