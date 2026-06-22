# Seed results notes

Результаты пишем кратко. `.log` и `.pt` не коммитим.

| date | file | data | steps | contract | h_contract | far_keep | move | state_var | eff_ops | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026-06-22 | self_referential_closure_field_v5_basin_dna.py | DNA E.coli 3-mer | 300 | 0.548 | 0.746 | 0.791 | 0.409 | 0.357 | 7.88 | basin contraction работает; нужен real DNA benchmark |
| 2026-06-22 | srcf_benchmark_v5_hard.py | synthetic hard anomaly | 150 pretrain | - | - | - | - | - | - | closure best ~0.976, но high/low инверсия; v6 чинит scoring |
