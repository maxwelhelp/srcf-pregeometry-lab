# ClosureGraph v4 MEMORY

Цель v4: проверить, помогает ли **AttractorBank / Hopfield-style memory** внутри closure-layer.

Идея:

```text
H_t -> triangle update -> closure step
H_t -> AttractorBank(H_t) -> притяжение к выученным relation-прототипам
```

Проверяем не философию, а метрики:

- `closure` — основной слой: triangle + memory, если включён `--use-memory`.
- `closure_no_mem` — тот же triangle closure, но без памяти, включается через `--ablate-memory`.
- `closure_no_tri` — ablation без triangle update, включается через `--ablate-tri`.
- `memΔ=edge/path2/comm` — AUC(closure) - AUC(closure_no_mem). Если положительный, память помогает.
- `triΔ=edge/path2/comm` — AUC(closure) - AUC(closure_no_tri). Если положительный, triangle помогает.

Где память должна быть полезна:

1. OOD corruption: когда граф повреждён сильнее, чем на train.
2. community/basin: когда нужно восстановить тип структуры, а не одно ребро.
3. raw-only режим: когда нет ручных common-neighbors/jaccard/path2 features.

Если `memΔ` отрицательный — память мешает или слишком сильно тянет всё к прототипам. Тогда уменьшать `--memory-strength` или `--memory-beta`.
