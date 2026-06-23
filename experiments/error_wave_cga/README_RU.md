# Error-Wave CGA v8

Файлы пакета:

- `error_wave_sparse_learning_v8_cga.py` — тест CGA vs Error-Wave v4.
- `docs/CGA_DRCF_ROADMAP_RU.md` — дорожная карта Error-Wave / CGA / DRCF.

Метрики в тесте:

- `patch_gain` — насколько хорошо выучен новый context=0.
- `retain_drop` — насколько сломались старые contexts 1..C-1; меньше лучше.
- `density` — доля реально обновлённых весов.
