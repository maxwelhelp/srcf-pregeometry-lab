# SRCF Dynamic Systems, Relation Memory, and Diagnostics

Дата: 2026-06-22  
Репозиторий: `maxwelhelp/srcf-pregeometry-lab`

## 1. Изучение режимов динамических систем

Если есть time-series, не обязательно подавать её как sequence.

Можно построить relation matrix:

```text
sensor_i -> sensor_j
lag correlation
transition influence
co-activation
phase relation
causal proxy
```

SRCF может учить режимы системы:

```text
режим A
режим B
переходный режим
устойчивый basin
неустойчивый basin
```

Это полезно не только для anomaly detection.

Главные применения:

- режимная кластеризация;
- прогнозирование состояния системы;
- раннее предупреждение о смене режима;
- контроль / управление;
- понимание, какие связи держат режим;
- анализ переходов между режимами;
- поиск связей, которые первыми меняются перед сменой режима.

Примеры данных:

```text
промышленные сенсоры
биосигналы
климатические системы
робототехника
сетевой трафик
финансовые режимы
сервисная инфраструктура
мозговые / нейронные сигналы
```

SRCF здесь даёт не просто “странно / не странно”, а карту:

```text
какой relation basin активен,
как система входит в него,
какие связи держат устойчивость,
и когда basin начинает разрушаться.
```

## 2. Relation memory / associative memory

SRCF похож на аттракторную память, но не для векторов, а для отношений.

Обычная память:

```text
query -> nearest embedding
```

Relation memory:

```text
неполная система отношений -> восстановить целую
шумная система отношений -> вернуть в basin
частичный граф -> достроить контекст
```

Где полезно:

- knowledge graph completion;
- memory retrieval;
- case-based reasoning;
- структурная аналогия;
- восстановление missing context;
- восстановление неполного плана;
- восстановление связей между файлами в коде;
- восстановление контекста агента по фрагменту задачи;
- восстановление цепочки фактов в RAG;
- восстановление graph context после partial observation.

Главная разница:

```text
обычная память ищет похожий embedding;
SRCF-memory ищет, какое самосогласованное отношение восстановится из фрагмента.
```

Это делает SRCF не просто encoder’ом, а relation-associative memory.

## 3. Что SRCF даёт, чего не даёт обычный метод

SRCF не даёт просто “класс” и не даёт просто “расстояние”.

Он даёт несколько новых диагностик динамики.

### 3.1. `contract`

Показывает, насколько близкие варианты состояния сходятся в один basin.

```text
near(R) -> same basin?
```

Это важно для self-supervised learning: модель должна понимать, что малое повреждение того же состояния не должно менять identity системы.

### 3.2. `h_contract`

Показывает, сходятся ли скрытые состояния, а не только выходные дескрипторы.

Это защищает от ложной метрики, когда descriptor выглядит нормально, но сама скрытая система не сходится.

### 3.3. `recovery`

Показывает, восстанавливается ли состояние после шума.

```text
H* + noise -> F(...) -> H*
```

Это главный мост к denoising / repair / associative memory.

### 3.4. `fixed`

Показывает, стало ли состояние устойчивым.

```text
F(H*) ≈ H*
```

Если fixed плохой, значит closure не завершился или динамика нестабильна.

### 3.5. `move`

Показывает, изменилась ли система реально или модель просто identity.

```text
move = ||H* - H0||
```

Высокий `move` при нормальном `fixed/recovery` означает, что модель реально преобразует систему, а не копирует вход.

### 3.6. `far_keep`

Показывает, не схлопывает ли модель разные системы вместе.

```text
far states должны оставаться различимыми
```

Это анти-collapse диагностика.

### 3.7. `overclosed / underclosed`

Показывает, является ли система слишком стабильной или слишком нестабильной.

```text
underclosed = не замыкается, плавает, плохо восстанавливается
overclosed  = слишком быстро и слишком просто схлопывается
```

Это отдельный сигнал, которого нет у обычного distance baseline.

### 3.8. `closure trajectory`

Показывает, как именно система приходит к устойчивости.

```text
curve_start
curve_end
curve_decay
```

Траектория может показать:

- быстрое нормальное замыкание;
- медленное замыкание;
- неустойчивую динамику;
- слишком раннее схлопывание;
- отсутствие реального движения;
- переходный режим между basin.

## 4. Почему это не замена baseline, а новый слой

SRCF не обязан заменить все стандартные методы.

Правильная роль:

```text
baseline даёт distance / classifier / embedding;
SRCF даёт closure dynamics profile.
```

То есть SRCF добавляет новый слой анализа:

```text
как relation system замыкается,
восстанавливается,
сохраняет различимость,
переходит между basin,
и где она overclosed / underclosed.
```

## 5. MVP: Dynamic Regime Learning

Цель:

```text
из time-series строить relation matrices и учить режимы системы.
```

Файлы:

```text
srcf_timeseries_relation_builder.py
srcf_dynamic_regime_benchmark.py
srcf_regime_transition_report.py
```

Relation channels:

```text
correlation
lag_correlation
phase_relation
co_activation
transition_influence
causal_proxy
physical_adjacency
```

Метрики:

```text
regime clustering quality
transition early warning
basin stability
relation edges that change before transition
forecast improvement from closure_profile
recovery after synthetic perturbation
```

## 6. MVP: Relation Memory

Цель:

```text
проверить, может ли SRCF восстанавливать полный relation context из неполного фрагмента.
```

Файлы:

```text
srcf_relation_memory.py
srcf_partial_graph_completion.py
srcf_memory_retrieval_benchmark.py
srcf_structural_analogy_report.py
```

Задачи:

```text
partial graph -> full graph
partial facts -> consistent fact basin
partial plan -> full plan skeleton
partial code context -> missing dependencies
partial RAG context -> missing claim/source
```

Метрики:

```text
missing context recovery
knowledge graph completion
case retrieval accuracy
structural analogy retrieval
memory repair quality
relation reconstruction error
```

## 7. Короткий вывод

Эти два направления усиливают главный тезис SRCF:

```text
SRCF — это не anomaly detector.
SRCF — это relation closure engine.
```

В dynamic systems он учит:

```text
режимы и переходы между basins.
```

В relation memory он учит:

```text
восстановление самосогласованной системы отношений из фрагмента.
```

А диагностический профиль SRCF показывает:

```text
contract,
h_contract,
recovery,
fixed,
move,
far_keep,
overclosed / underclosed,
closure trajectory.
```

Это не заменяет baseline, а добавляет слой, которого обычные методы не дают.
