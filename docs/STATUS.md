# STATUS / текущие выводы

## Последний DNA training run

Файл: `results/srcf_v5_dna_metrics.csv`

Последняя точка:

| step | contract | h_contract | far_keep | move | state_var | desc_var | eff_ops | perm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 0.548 | 0.746 | 0.791 | 0.409 | 0.357 | 1.23e-04 | 7.88 | 1.4e-06 |

Итоги по run:

- min contract: `0.423`
- min h_contract: `0.678`
- min far_keep: `0.561`
- max move: `0.409`
- min state_var: `0.240`
- max eff_ops: `7.91`

Интерпретация:

- `contract < 1` и `h_contract < 1` — near DNA relation states реально стягиваются в basin.
- `move ~0.4` — это не identity.
- `eff_ops ~7.8` — операторы не схлопнулись.
- `perm ~1e-6` — нет зависимости от скрытого порядка узлов.
- `far_keep` иногда падает, но не до collapse; это надо мониторить.

## Старый hard anomaly benchmark v5

Главные результаты:

| metric | high | low | best |
|---|---:|---:|---:|
| calibrated_closure | 0.024 | 0.976 | 0.976 |
| instability | 0.019 | 0.981 | 0.981 |
| recovery | 0.019 | 0.981 | 0.981 |
| embedding_dist | 0.375 | 0.625 | 0.625 |
| raw_summary_dist | 0.825 | 0.175 | 0.825 |

Вывод:

- closure signal сильный (`best ~0.98`), но направление было инвертировано.
- Это означает, что часть hard anomalies являются не chaotic/unstable, а **over-closed / suspiciously stable**.
- v6 benchmark добавляет typicality-score, который должен ловить оба направления.

## Следующее

1. Запустить `srcf_benchmark_v6_real.py --task anomaly`.
2. Запустить `srcf_benchmark_v6_real.py --task dna`.
3. Сравнить `closure_typicality` против `embedding_dist`, `raw_summary_dist`, `raw_flat_dist`, `kmer_freq_dist`.
