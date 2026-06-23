# Circuit-Target Signal Birth для Qwen

Эта папка содержит экспериментальный пакет для проверки идеи **circuit-target signal birth**: рождение новых внутренних структурных сигналов/операторов из residual точных circuit targets трансформера.

Идея: вместо того чтобы смотреть на отдельные `Wq/Wk/Wv/Wo` или только на runtime activations, мы строим точные circuit targets:

```python
M_qk_aug[h, delta] = Wq_aug[h].T @ R_delta @ Wk_aug[kv] / sqrt(head_dim)
C_vo_aug[h]        = Wo[h] @ Wv_aug[kv]
```

Дальше:

1. базовый операторный словарь пытается объяснить `M_qk_aug` и `C_vo_aug`;
2. остаток `residual` майнится через PCA/SVD;
3. найденные BirthOps добавляются как новые structural signals;
4. качество проверяется на heldout heads/prompts и functional метриках `A_rel/KL/top1/Y_all/H_after`;
5. найденные BirthOps используются как protected subspace во время LoRA fine-tune.

## Что лежит в папке

- `signal_birth_test_v1_3_hard.py` — контролируемый hard compose test: birth собирает новый сигнал из примитивов, а не выбирает готовый hidden-кандидат.
- `real_qwen_joint_heads_circuit_compare_v2.py` — реальный joint all-head тест: сравнение слоя целиком, `Y_all = sum_h Y_h`, а не одной головы.
- `real_qwen_circuit_signal_birth_v1_2.py` — основной real-Qwen circuit signal birth: QK shared birth + VO per-head prompt validation.
- `circuit_birth_finetune_protect_v1_3.py` — первый fine-tune A/B: baseline LoRA vs LoRA + circuit-birth protected-subspace regularizer.
- `RUN_COMMANDS_RU.md` — команды запуска.
- `RESULTS_RU.md` — сводка результатов и честные выводы.

## Главный результат

На реальном `Qwen/Qwen2.5-0.5B-Instruct` без planted hidden:

- QK BirthOps стабильно уменьшают heldout residual `M_qk` примерно на 70%+.
- Routing proxy улучшается: `A_rel`, `KL`, `top1` резко лучше.
- VO требует per-head/prompt validation, а не shared split по головам.
- В fine-tune stress test protected regularizer сохранил edit accuracy 100%, но снизил drift защищённых QK/VO BirthOps примерно на 98.5%.

## Статус

Это исследовательский proof-of-concept. Уже доказано, что circuit-target signal birth работает как **structural signal channel** и может использоваться как circuit-aware regularizer. Но ещё не доказано на большом downstream benchmark, что он стабильно улучшает общую точность обучения или снижает forgetting на реальных задачах.
