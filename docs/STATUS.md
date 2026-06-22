# Статус проекта

## Текущий вывод

SRCF уже показал, что может обучать самозамыкающуюся динамику без labels. Самый сильный текущий результат — DNA training:

```text
step 300:
contract=0.55
h_contract=0.75
far_keep=0.79
move=0.409
state_var=0.357
eff_ops=7.88
curve убывает
```

Это значит, что near DNA relation states входят в общий basin, модель не identity, операторы не умерли.

## Главная нерешённая проблема

Пока не доказано, что SRCF стабильно лучше простых baseline на реальных задачах. На real DNA baseline вроде k-mer frequency/raw-flat часто сильные.

## Что добавлено в v7

Добавлен `srcf_realdata_benchmark_v1.py` — полностью реальный benchmark без синтетических anomaly generators.

Он проверяет:

- breast cancer anomaly detection;
- digits one-class anomaly detection;
- wine one-class anomaly detection;
- сравнение SRCF closure scores с PCA, IsolationForest, raw distance и embedding distance.

## Следующий критерий

Если closure_typicality выигрывает у raw/PCA/IsolationForest хотя бы на одном реальном датасете — это практический сигнал.
Если нет — нужно искать другую область применения или менять objective/scoring.
