# Closure Basin Memory Benchmark v1

Это первый честный тест идеи **closure-memory**.

Старый graph repair тест не доказывал память: там в одном run был один clean-граф, поэтому AttractorBank мог быть просто bias-регуляризатором.

Новый тест:

```text
есть K clean graph basins
query = damaged(A) + random false edges + foreign edges from B
модель должна:
1. выбрать правильный basin A
2. восстановить clean A
3. откинуть foreign edges из B
```

## Baseline, который обязательно надо побить

`raw_nearest`:

```text
сравнить query adjacency с каждым clean adjacency
выбрать ближайший по cosine
```

Если closure-memory не лучше `raw_nearest`, новый вид памяти не доказан.

## Модели

- `raw_nearest`: прямое сравнение с clean graph memories.
- `deep_big`: большой MLP-ретривер и восстановитель.
- `closure_no_mem`: closure/triangle без памяти.
- `closure_mem`: closure/triangle + AttractorBank.

## Главные метрики

- `retrieval_acc`: правильно ли выбран basin.
- `edge_auc`: восстановление clean graph.
- `foreign_auc`: отбрасывание чужих рёбер из другого basin.
- `slot_cos`, `slot_pmax`, `slot_entropy`: диагностика, схлопнулась ли память.
- `curve_ratio`: насколько closure-состояние сжимается по итерациям.

## Что считать успехом

Минимально:

```text
closure_mem retrieval_acc > raw_nearest retrieval_acc
closure_mem foreign_auc > raw_nearest foreign_auc
closure_mem > closure_no_mem по foreign_auc или retrieval_acc
```

Сильный результат:

```text
при K=2,4,8 closure_mem деградирует медленнее raw_nearest/deep_big
foreign_auc стабильно выше у closure_mem
slot_entropy не схлопывается в 0 слишком рано
```
