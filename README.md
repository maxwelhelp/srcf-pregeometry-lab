# SRCF Pregeometry Lab

Исследования **Self-Referential Closure Field**: координатно-свободные relation-сети, где модель не получает координаты/объекты как основу, а учится самоприменением формировать устойчивые basin-состояния.

## Главная идея

Обычная сеть:

```text
данные -> признаки -> класс/ответ
```

SRCF:

```text
relation tensor R[i,j,c]
-> encoder
-> один и тот же learned closure-оператор применяется к своему состоянию несколько раз
-> near-возмущения должны стягиваться в один basin
-> far-состояния должны оставаться различимыми
```

Никаких supervised labels на этапе pretrain нет.

## Файлы

- `self_referential_closure_field_v6_basin_dna.py` — основной SRCF trainer: synthetic + real DNA 3-mer relation states.
- `srcf_benchmark_v6_real.py` — hard synthetic anomaly + real DNA benchmark.
- `pregeometric_self_query_field_v2.py` — supervised coordinate-free prototype.
- `field_archs_v2_diagnostic.py` — диагностические демо/baseline.

## Что уже получилось

DNA self-supervised run на E. coli 200k bases:

- `contract` держится ниже 1: near-состояния стягиваются.
- `h_contract` ниже 1: стягивается не только descriptor, но и hidden-state.
- `move` высокий: это не ленивый identity.
- `curve` убывает: самоприменение сходится.
- `eff_ops` около 7.6-7.9: операторы не схлопнулись в один.
- `perm` около 1e-6: нет скрытой зависимости от порядка узлов.

См. `results/summary_seed.md`.

## Что было исправлено в v6 benchmark

Старый anomaly score ошибочно считал:

```text
аномалия = высокая instability
```

Но hard anomaly показал обратный режим:

```text
часть аномалий = слишком стабильные / over-closed / слишком простые
```

Поэтому v6 добавляет:

- `closure_distance` — обычное отклонение от нормы.
- `closure_typicality` — ловит и слишком высокое, и слишком низкое отклонение.
- `closure_energy_typicality` — проверяет typical set, а не только расстояние от центра.
- real DNA benchmark против baseline:
  - `embedding_dist`
  - `raw_summary_dist`
  - `raw_flat_dist`
  - `kmer_freq_dist`

## Команды запуска

### 1. DNA training

```bash
python -u self_referential_closure_field_v6_basin_dna.py \
  --device cuda --amp fp16 \
  --steps 300 --batch-size 4 --eval-batch-size 4 \
  --data dna --dna-bases 200000 --dna-window 2048 --kmer 3 \
  --dim 48 --ops 8 --iters 6 --eval-every 25 \
  --metrics-csv results/srcf_v6_dna_metrics.csv \
  --save-path ./srcf_v6_dna.pt \
  | tee srcf_v6_dna.log
```

### 2. Synthetic hard anomaly benchmark

```bash
python -u srcf_benchmark_v6_real.py \
  --device cuda --amp fp16 \
  --task anomaly --pretrain-steps 150 \
  --batch 8 --eval-batch 8 \
  --n 32 --dim 48 --ops 8 --iters 6 \
  --results-csv results/summary.csv \
  | tee srcf_benchmark_v6_anomaly.log
```

### 3. Real DNA benchmark

```bash
python -u srcf_benchmark_v6_real.py \
  --device cuda --amp fp16 \
  --task dna --pretrain-steps 150 \
  --batch 4 --eval-batch 4 \
  --n 64 --dim 48 --ops 8 --iters 6 \
  --dna-bases 200000 --dna-window 2048 --kmer 3 \
  --results-csv results/summary.csv \
  | tee srcf_benchmark_v6_dna.log
```

## Как читать AUROC

Benchmark печатает:

```text
high = чем выше score, тем аномальнее
low  = обратное направление
best = есть ли разделимость вообще
```

Цель:

```text
closure_typicality high > embedding_dist high
closure_typicality high > raw_summary_dist high
```

Если `best` высокий, а `high` низкий — сигнал есть, но направление score надо калибровать.
