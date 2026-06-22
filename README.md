# SRCF Pregeometry Lab

Эксперименты с **Self-Referential Closure Field (SRCF)** — координатно-свободной моделью для матриц отношений.

## Коротко

SRCF не получает координаты и не обучается на метках аномалий. Он берёт матрицу отношений `R[i,j,c]`, переводит её в скрытое состояние и несколько раз применяет один и тот же оператор к самому состоянию:

```text
R0 -> H0 -> F(H0) -> F(F(H0)) -> ... -> H*
```

Цель обучения без labels:

- малое повреждение того же состояния должно сходиться в тот же basin;
- другое состояние должно оставаться отличимым;
- финальное состояние должно быть устойчивым;
- после шума оно должно восстанавливаться;
- модель не должна схлопываться в константу.

## Главные файлы

- `self_referential_closure_field_v6_basin_dna.py` — основное обучение SRCF на synthetic/DNA relation matrices.
- `srcf_benchmark_v6_real.py` — synthetic + real DNA benchmark.
- `srcf_realdata_benchmark_v1.py` — новый реальный benchmark на встроенных sklearn-датасетах без синтетики.
- `pregeometric_self_query_field_v2.py` — supervised toy-прототип координатно-свободных self-query operators.
- `field_archs_v2_diagnostic.py` — диагностические демо/baseline.

## Что уже видно

1. На DNA k-mer relation states SRCF показывает реальную basin dynamics без labels: `contract < 1`, `h_contract < 1`, `move > 0`, кривая шагов убывает, операторы не схлопываются.
2. На synthetic hard anomalies closure-сигнал сильный, но benchmark искусственный.
3. На real DNA пока нет уверенной победы над простыми baseline по всем типам аномалий.
4. Поэтому добавлен новый real-data benchmark: breast cancer, digits, wine. Он нужен, чтобы быстро понять, где closure-score даёт практический выигрыш против обычных методов.

## Реальный benchmark без синтетики

Запуск:

```bash
bash scripts/run_real_bench.sh
```

Или явно:

```bash
python -u srcf_realdata_benchmark_v1.py \
  --device cuda --amp fp16 \
  --dataset all \
  --pretrain-steps 200 \
  --batch 16 --eval-batch 32 \
  --dim 48 --ops 8 --iters 6 \
  --results-csv results/realdata_summary.csv \
  | tee srcf_realdata_v1.log
```

Датасеты:

- `breast_cancer`: benign = normal, malignant = anomaly.
- `digits`: выбранная цифра = normal, остальные = anomaly.
- `wine`: выбранный класс = normal, остальные = anomaly.

Сравнение:

- `closure_typicality` — двусторонняя типичность closure features.
- `closure_instability` — простая нестабильность/восстановление.
- `embedding_dist` — расстояние SRCF descriptor до нормального центра.
- `raw_dist` — расстояние исходных признаков до центра нормального класса.
- `pca_recon` — ошибка восстановления PCA, обученной на normal.
- `isolation_forest` — sklearn IsolationForest, обученный на normal.

## Как читать метрики

Для training:

- `contract < 1` — near-состояния стягиваются.
- `h_contract < 1` — hidden-state тоже стягивается.
- `far_keep >= 0.7` — другие состояния не схлопываются слишком сильно.
- `move > 0.1` — модель не identity.
- `state_var` не должен падать к нулю.
- `curve start -> end` должна убывать.

Для benchmark:

- `high` — обычное направление score: выше = аномальнее.
- `low` — обратное направление.
- `best` — есть ли разделимость вообще.

Практически нужен высокий `high`. Если высокий только `best`, значит сигнал есть, но его надо калибровать.

## Что считать успехом

SRCF становится реально полезным, если на реальном benchmark:

```text
closure_typicality AUROC >= raw_dist / PCA / IsolationForest / embedding_dist
```

или если он явно выигрывает хотя бы на одном типе данных, где обычные расстояния проваливаются.
