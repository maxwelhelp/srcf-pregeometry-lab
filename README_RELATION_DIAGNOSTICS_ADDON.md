# SRCF Relation Diagnostics v1

Один файл проверяет три вещи:

1. **Other / foreign relation detection** — наше текущее сильное место.
2. **Graph-level memory** — настоящая проверка памяти на выборе basin, а не repair одного графа.
3. **Non-graph relation matrices** — проверка тезиса “это не про графы, а про матрицы отношений”.

Главный файл:

```bash
closure_relation_diagnostics_v1.py
```

## Быстрый графовый тест

```bash
python -u closure_relation_diagnostics_v1.py \
  --mode other --device cuda --amp fp16 \
  --K-list 2,4 --seeds 0 --steps 120 \
  --batch 24 --eval-batch 48 --eval-batches 2 \
  --n 32 --dim 48 --hidden 96 --iters 5 \
  --rel-mode raw --eval-every 30 \
  --results-csv results/closure_relation_other_v1.csv \
  | tee closure_relation_other_v1.log
```

## Non-graph тест на co-occurrence матрицах из локальных файлов репозитория

```bash
python -u closure_relation_diagnostics_v1.py \
  --mode nongraph --device cuda --amp fp16 \
  --K-list 2,4 --seeds 0 --steps 120 \
  --batch 24 --eval-batch 48 --eval-batches 2 \
  --n 32 --dim 48 --hidden 96 --iters 5 \
  --rel-mode raw --text-root . --eval-every 30 \
  --results-csv results/closure_relation_nongraph_v1.csv \
  | tee closure_relation_nongraph_v1.log
```

## Что смотреть

- `cl_no_foreign` — closure без памяти на чужих связях.
- `cl_mem_foreign` — graph-level memory.
- `raw_foreign` — nearest-clean baseline.
- `deep_foreign` — сильный deep baseline.
- `deg_best`, `cn_best` — дешёвые локальные baseline. Если они дают 0.90+, значит задача тривиальна.
- `cl_mem_acc > cl_no_acc` — память реально помогает retrieval.
- `cl_no_foreign > deg_best/cn_best` — closure делает больше, чем локальный degree/CN трюк.

## Честные выводы

- Если `cl_no_foreign` высокий, а `deg_best/cn_best` низкий — у closure есть реальный self/other signal.
- Если `cl_mem_acc` не выше `cl_no_acc/raw/deep` — новая память пока не доказана.
- Если non-graph режим работает — можно говорить “матрица отношений”, а не только граф.
