# SRCF Relation Closure Roadmap

Дата: 2026-06-22  
Репозиторий: `maxwelhelp/srcf-pregeometry-lab`

## 1. Главное переименование направления

SRCF не надо фиксировать как `anomaly detector`.

Правильное название направления:

```text
SRCF = Self-Supervised Relation Closure Engine
```

Или короче:

```text
SRCF = Relation Closure Field
```

Проверки на аномалии были полезны только как дешевый способ измерить сигнал. Но смысл системы шире:

```text
данные -> relation tensor R[i,j,c]
R -> hidden relation state H
H -> F(H) -> F(F(H)) -> ... -> H*
```

SRCF изучает не класс объекта, а поведение системы отношений под самоприменением:

- сходится ли состояние в basin;
- восстанавливается ли оно после шума;
- сохраняет ли различимость от других состояний;
- не схлопывается ли в константу;
- не является ли оно слишком стабильным / overclosed;
- не является ли оно слишком нестабильным / underclosed;
- какие relation-связи усиливаются или гасятся после closure.

Поэтому основная ценность SRCF:

```text
не “найти аномалию”,
а получить closure dynamics descriptor для системы отношений.
```

## 2. Что реально уже сделано

Текущая система уже реализует рабочий прототип self-supervised relation closure:

- входом является матрица отношений `R[i,j,c]`;
- узлы анонимны, координаты не подаются напрямую;
- модель кодирует `R` в скрытое состояние;
- один и тот же learned operator применяется к состоянию несколько раз;
- loss заставляет near-состояния сходиться, far-состояния оставаться различимыми, fixed point быть устойчивым, а state variance не схлопываться.

На DNA basin training уже виден сильный механизм:

```text
contract   ≈ 0.548
h_contract ≈ 0.746
far_keep   ≈ 0.791
move       ≈ 0.409
state_var  ≈ 0.357
eff_ops    ≈ 7.88
perm       ≈ 1e-6
```

Это значит:

- near DNA relation states реально стягиваются;
- hidden states тоже стягиваются;
- модель не identity, потому что `move` высокий;
- модель не схлопнулась, потому что `state_var` нормальный;
- операторы живые, потому что effective ops почти 8;
- скрытого порядка узлов нет, потому что permutation error около нуля.

Это хороший результат как демонстрация механизма.

## 3. Что пока не доказано

Пока не доказано, что SRCF лучше стандартных методов на выбранном DNA anomaly benchmark.

По текущей картине:

```text
closure_typicality_best = 0.678
embedding_dist_best     = 0.709
raw_flat_dist_best      = 0.749
kmer_freq_dist_best     = 0.742
```

Вывод:

```text
SRCF basin dynamics работает,
но выбранный DNA anomaly benchmark пока не показывает практического преимущества над простыми baseline.
```

Это не провал. Это нормальный исследовательский результат.

SRCF не обязан сразу побеждать:

- PCA;
- IsolationForest;
- raw distance;
- k-mer frequency;
- embedding distance;
- ручные статистики.

Стандартные методы спрашивают:

```text
далеко ли объект от нормального облака?
```

SRCF спрашивает другое:

```text
как система отношений ведет себя при самозамыкании?
```

Это другой тип сигнала.

## 4. Главное открытие: overclosed / underclosed

Самый важный вывод из synthetic hard benchmark:

```text
аномалия не всегда нестабильна.
```

Иногда аномалия выглядит как overclosed state:

- слишком быстро сходится;
- слишком стабильна;
- слишком проста;
- слишком центральна;
- подозрительно хорошо восстанавливается;
- имеет слишком низкую instability.

Два режима:

```text
underclosed = система не замыкается, шумит, не восстанавливается, плавает;
overclosed  = система слишком быстро схлопывается, слишком проста, слишком стабильна.
```

Это уже не обычный distance-based anomaly detection.

Это можно использовать как общий диагностический слой для систем отношений.

## 5. Новый правильный объект выхода

Не надо сводить SRCF к одному score.

Правильный выход:

```text
closure_profile = {
  basin_id,
  closure_speed,
  fixed_point_distance,
  recovery_score,
  underclosed_score,
  overclosed_score,
  multi_basin_score,
  collapse_risk,
  state_variance,
  relation_channel_importance,
  changed_edges_after_closure,
  operator_usage,
  curve_start,
  curve_end,
  curve_decay
}
```

То есть SRCF должен стать не просто detector, а генератором `closure dynamics descriptor`.

Этот descriptor можно отдавать дальше:

- agent planner;
- RAG router;
- code inspector;
- memory selector;
- small classifier;
- scientific graph analyzer;
- relation-system dashboard.

## 6. Где SRCF даст максимальную пользу

### 6.1. Агент: проверка смысла задачи / вопроса

Самое сильное направление.

Для агента relation matrix можно строить из:

```text
goal_i -> constraint_j
goal_i -> available_tool_j
goal_i -> required_file_j
fact_i -> expected_output_j
missing_info_i -> risk_j
user_request_i -> previous_context_j
plan_step_i -> tool_j
```

SRCF может отвечать:

- задача замкнута или нет;
- можно ли действовать без уточнений;
- не противоречат ли ограничения друг другу;
- не развалится ли план при малом изменении контекста;
- есть ли несколько разных basins трактовки;
- не слишком ли задача overclosed, то есть ответ шаблонный и уже очевидный;
- не underclosed ли задача, то есть не хватает файлов, фактов, web, execution или user input.

Пример применения:

```text
Пользователь: “почини recovery loop”

R = связи между:
  scanner,
  controller,
  executor,
  recovery metric,
  logs,
  expected candidate,
  expected edge recovery,
  files,
  tests.

SRCF-profile:
  same_basin?        да/нет
  missing_edges?     какие связи не замыкаются
  underclosed?       не хватает runtime trace
  overclosed?        агент слишком уверен по статике
  multi_basin?       scanner или controller или metric
```

Польза:

```text
SRCF может стать предварительным мозгом агента:
“понимаю ли я задачу достаточно, чтобы действовать?”
```

### 6.2. Кодовые базы и архитектура проекта

Это второе самое сильное направление.

Для твоих проектов relation matrix естественная:

```text
file_i -> file_j:
  imports
  function calls
  shared config keys
  shared metrics
  shared tensor shapes
  writes/reads artifacts
  test coverage relation
  runtime trace relation
  error propagation
  command dependency
  model checkpoint dependency
```

SRCF может давать:

- какие файлы лежат в одном closure basin;
- какие модули должны меняться вместе;
- где loop не замкнут;
- где есть orphan-файл;
- где связь искусственная;
- где recovery не доходит до metric;
- где scanner/controller/executor/memory расходятся;
- где изменение одного файла не восстанавливается остальной системой;
- где архитектура overclosed и всё слишком жестко завязано на один путь;
- где архитектура underclosed и нет настоящего runtime пути.

Это не замена static analyzer.

Это слой:

```text
architectural closure diagnostics
```

Минимальный эксперимент:

```text
1. Спарсить repo graph.
2. Построить R[file_i,file_j,c].
3. Обучить SRCF на нормальных snapshots / текущей архитектуре.
4. Внести controlled breaks:
   - удалить import;
   - сломать metric route;
   - заменить config key;
   - сломать artifact path;
   - сломать recovery edge.
5. Проверить, какие breaks дают underclosed / overclosed / multi-basin profile.
```

### 6.3. LLM embeddings и semantic basins

LLM embeddings сами по себе дают cosine similarity.

SRCF может смотреть не на близость отдельных текстов, а на замыкание всей системы смыслов.

Relation matrix:

```text
chunk_i -> chunk_j:
  cosine similarity
  entailment score
  contradiction score
  shared entities
  temporal dependency
  causal dependency
  topic transition
  reference/coreference
  tool-needed relation
  source reliability relation
```

Применения:

- проверка, достаточно ли retrieved context для ответа;
- поиск противоречий в RAG;
- определение, какие chunks реально входят в один reasoning basin;
- context compression не по similarity, а по closure-важности;
- определение missing chunk;
- routing между memory/web/files/code;
- проверка, является ли вопрос хорошо поставленным;
- multi-hop reasoning: замыкается ли цепочка фактов.

Ключевая задача:

```text
RAG context -> relation closure -> stable answer basin?
```

Если closure плохой, агент не должен уверенно отвечать.

### 6.4. Память агента

Обычная память агента:

```text
query -> vector search -> top-k memories
```

SRCF-подход:

```text
current_task + memories -> relation graph -> closure profile
```

Relation matrix:

```text
memory_i -> current_task_j:
  semantic relevance
  project relevance
  temporal relevance
  user preference relation
  unresolved issue relation
  contradiction relation
  dependency relation
  stale/updated relation
```

Что можно получить:

- какие memories реально нужны;
- какие memories мешают;
- какие устарели;
- где память overclosed и тащит старый шаблон;
- где задача underclosed без конкретного прошлого контекста;
- какие memory nodes надо объединить;
- какие memory nodes надо забыть или понизить.

Это сильнее, чем просто top-k retrieval.

### 6.5. Научные графы: paper -> idea -> method -> result

SRCF можно использовать для карты исследований.

Relation matrix:

```text
paper_i -> paper_j:
  citation direction
  method similarity
  benchmark overlap
  assumption overlap
  result dependency
  contradiction
  improvement relation
  shared dataset
  shared failure mode
```

Вопросы:

- идея входит в существующий basin или реально новая;
- где теория не замыкается;
- какие статьи образуют устойчивую область;
- где bridge paper между двумя basins;
- какая гипотеза underclosed;
- какая гипотеза overclosed и просто переобъясняет известное;
- какие missing links нужны для proof chain.

Для research-проектов по matrix/operator/field это очень полезно.

### 6.6. Биология шире DNA

DNA была удобной, потому что relation matrix естественная:

- k-mer transition;
- PMI;
- reverse-complement;
- GC relation;
- Hamming relation;
- frequency outer product;
- symmetric / antisymmetric transition.

Но биология шире:

```text
protein residues -> contact / distance / coevolution matrix
genes -> regulatory graph
cell types -> expression relation
pathways -> activation/inhibition relation
microbiome species -> co-occurrence relation
motifs -> transition relation
```

Здесь SRCF полезен не как “найти аномалию”, а как:

```text
биологическая система отношений самозамыкается или нет?
```

### 6.7. Сенсоры, промышленные системы, биосигналы

Relation matrix:

```text
sensor_i -> sensor_j:
  correlation
  lag correlation
  causality proxy
  phase relation
  failure co-occurrence
  physical adjacency
  recovery relation
```

Польза:

- система в нормальном basin;
- один сенсор выпал из closure;
- слишком синхронное движение = overclosed;
- шумное рассогласование = underclosed;
- recovery после perturbation;
- где relation graph сменил режим.

Это лучше подходит SRCF, чем таблички без структуры отношений.

### 6.8. Сетевой трафик и сервисные зависимости

Relation matrix:

```text
host_i -> host_j:
  request count
  latency relation
  error propagation
  port/service relation
  auth relation
  retry relation
  temporal co-activation
```

Польза:

- normal service basin;
- degraded basin;
- cascading failure basin;
- suspicious overclosed bot-like pattern;
- underclosed broken dependency;
- recovery quality after incident.

### 6.9. Финансовые/рыночные relation systems

Осторожно: не как “предсказатель цены”.

Relation matrix:

```text
asset_i -> asset_j:
  correlation
  lag relation
  volatility spillover
  sector relation
  news co-movement
  liquidity relation
```

Польза:

- market relation regime;
- correlation breakdown;
- overclosed market-wide synchronization;
- underclosed sector divergence;
- recovery after shock;
- regime descriptor for downstream model.

## 7. Где SRCF плохо подходит

SRCF не надо пихать везде.

Плохо подходит:

- обычные таблички, где признаки независимы;
- задачи, где raw distance уже почти идеально решает;
- простая классификация без relation structure;
- данные, где relation matrix искусственная и не несет смысла;
- задачи, где нужны labels и финальный supervised classifier уже дешевле и точнее;
- задачи, где нет нормального понятия basin/recovery/closure.

Критерий:

```text
Если объект важен как набор независимых признаков — SRCF слабее.
Если объект важен как система отношений — SRCF имеет смысл.
```

## 8. Главная рекомендация по развитию

Не добавлять еще 10 случайных датасетов.

Следующий этап:

```text
интерпретация closure dynamics
```

Надо понять, что именно SRCF меняет в relation matrix.

Сделать инструменты:

### 8.1. Edge delta inspector

Считать:

```text
Delta = H* - H0
```

И выводить:

- top changed edges `(i,j)`;
- top changed channels `c`;
- какие связи усилились;
- какие связи погасились;
- какие relation patterns входят в fixed point.

Для DNA:

```text
какие k-mer пары усилились/погасли после closure?
```

Для кода:

```text
какие file->file связи SRCF считает недозамкнутыми?
```

Для агента:

```text
какие task/tool/file/fact связи делают вопрос решаемым или нерешаемым?
```

### 8.2. Channel attribution

Для каждого relation channel:

- удалить канал;
- зашумить канал;
- заменить канал median нормального состояния;
- посмотреть, как меняется closure profile.

Вывод:

```text
какие каналы реально важны для basin dynamics.
```

### 8.3. Basin clustering

Собрать descriptors `H*` и кластеризовать:

- normal closure basin;
- underclosed basin;
- overclosed basin;
- multi-basin;
- collapsed basin;
- noisy/recoverable basin.

### 8.4. Closure curve dashboard

Для каждого объекта показывать:

```text
curve_start
curve_end
curve_decay
move
recovery
fixed
contract
h_contract
state_var
```

Это должно стать основной диагностикой, а не один AUROC.

## 9. Лучший следующий MVP

Самый полезный следующий MVP:

```text
SRCF Agent Task Closure Benchmark
```

Цель:

```text
проверить, может ли SRCF определить, замкнута ли задача для агента.
```

### 9.1. Типы задач

Собрать 200-500 задач:

```text
closed             = можно делать сразу
underdefined       = не хватает данных
needs_file         = нужен файл/репо
needs_web          = нужна актуальная информация
needs_execution    = надо запустить код
contradictory      = ограничения конфликтуют
too_broad          = задача слишком широкая
multi_basin        = несколько возможных трактовок
trivial_overclosed = ответ почти шаблонный / уже есть в контексте
```

### 9.2. Relation nodes

Узлы:

```text
user_goal
constraints
known_facts
missing_facts
tools
files
code_symbols
expected_output
risks
past_context
plan_steps
```

### 9.3. Relation channels

Каналы:

```text
semantic_similarity
requires
blocks
contradicts
supports
same_project
same_file
tool_can_resolve
needs_current_info
needs_execution
has_source
missing_dependency
```

### 9.4. Модель

Не надо сразу делать сложный агент.

Достаточно:

```text
R[task_nodes, task_nodes, channels]
SRCF pretrain без labels
closure_profile extraction
маленький classifier/router сверху
```

### 9.5. Что измерять

Не только accuracy.

Метрики:

```text
closed_vs_underclosed AUROC
needs_tool routing accuracy
contradiction detection
multi_basin detection
calibration: когда агент уверен, closure должен быть stable
ablation: без SRCF profile vs с SRCF profile
```

Главный вопрос:

```text
дает ли closure_profile пользу агенту сверх embeddings и rules?
```

## 10. Второй MVP: Code Closure Inspector

Цель:

```text
проверить SRCF на реальном repo graph.
```

### 10.1. Построение R

Узлы:

```text
files
classes
functions
configs
metrics
artifacts
commands
tests
```

Каналы:

```text
imports
calls
reads_config
writes_artifact
reads_artifact
shares_metric
shares_tensor_shape
same_directory
covered_by_test
mentioned_in_logs
runtime_dependency
```

### 10.2. Controlled breaks

Внести искусственные поломки:

```text
remove import edge
remove metric edge
remove artifact write edge
change config key
remove test coverage edge
break recovery metric route
break scanner->controller edge
break executor->artifact edge
```

### 10.3. Ожидаемый результат

SRCF должен показывать:

```text
underclosed score растет
recovery падает
fixed point меняется
edge delta указывает на сломанный участок
multi-basin появляется при неоднозначной архитектуре
```

Это ближе к твоим реальным задачам, чем DNA anomaly.

## 11. Третий MVP: RAG Context Closure

Цель:

```text
проверить, достаточно ли retrieved context для ответа.
```

### 11.1. Узлы

```text
question
retrieved_chunks
known_facts
entities
claims
sources
answer_requirements
```

### 11.2. Каналы

```text
similarity
entailment
contradiction
same_entity
temporal_order
source_support
claim_dependency
missing_claim
```

### 11.3. Labels только для оценки

Типы:

```text
answerable
missing_context
contradictory_context
irrelevant_context
too_many_unrelated_chunks
```

### 11.4. Проверка

Сравнить:

```text
embedding top-k only
rules only
SRCF closure profile
embedding + SRCF profile
```

Если SRCF улучшает detection missing/contradictory context, это сильный результат.

## 12. Изменения в README/позиционировании

Текущий README лучше поправить в сторону:

```text
SRCF is not primarily an anomaly detector.
It is a self-supervised relation closure model.
Anomaly benchmarks are diagnostic tests, not the final goal.
```

Добавить раздел:

```text
Primary use cases:
1. Agent task closure
2. Code architecture closure
3. RAG context closure
4. Memory relation closure
5. Scientific/biological relation systems
```

Добавить честный статус:

```text
What is proven:
- basin dynamics can be trained on DNA relation matrices;
- near states contract;
- far states remain separated;
- fixed/recovery losses are meaningful;
- overclosed/underclosed distinction exists.

What is not proven:
- superiority over standard DNA anomaly baselines;
- general practical advantage on arbitrary tabular datasets;
- optimal relation construction for each domain.
```

## 13. Что не делать сейчас

Не делать:

- не гнаться за еще десятками случайных датасетов;
- не продавать SRCF как universal anomaly detector;
- не пытаться побить все baselines на любых данных;
- не делать сразу огромную архитектуру;
- не смешивать все domains в один непонятный benchmark;
- не оценивать только одним AUROC;
- не забывать про интерпретацию edge/channel/curve.

## 14. Что делать прямо сейчас

Приоритетный план:

### Шаг 1. Зафиксировать статус

Создать `STATUS.md` или обновить README:

```text
SRCF = relation closure engine.
DNA basin training works.
Real DNA anomaly advantage not proven.
Overclosed/underclosed signal is the main new observation.
Next focus: interpretation and agent/code/RAG closure tasks.
```

### Шаг 2. Добавить interpreters

Файлы:

```text
srcf_edge_delta_inspector.py
srcf_channel_ablation.py
srcf_basin_cluster_report.py
srcf_closure_profile_export.py
```

### Шаг 3. Сделать Agent Task Closure MVP

Файлы:

```text
srcf_agent_task_closure_dataset.py
srcf_agent_task_relation_builder.py
srcf_agent_task_closure_benchmark.py
```

### Шаг 4. Сделать Code Closure MVP

Файлы:

```text
srcf_code_graph_builder.py
srcf_code_closure_benchmark.py
srcf_code_break_injector.py
```

### Шаг 5. Только потом возвращаться к DNA/protein/sensors

После того как появится нормальная интерпретация closure dynamics.

## 15. Короткий финальный вывод

SRCF сейчас надо развивать как:

```text
self-supervised relation closure model
```

А не как:

```text
universal anomaly detector
```

Максимальная польза ожидается там, где важна не отдельная точка, а структура отношений:

```text
агентные задачи,
кодовые архитектуры,
RAG-контекст,
память агента,
научные графы,
биологические relation systems,
сенсоры,
сервисные зависимости.
```

Самый сильный следующий эксперимент:

```text
SRCF Agent Task Closure Benchmark
```

Потому что там SRCF может отвечать на реально важный вопрос:

```text
эта задача для агента замкнута, решаема и устойчива,
или она недоопределена, противоречива, overclosed/underclosed?
```

Это намного ценнее, чем продолжать гонку “еще один anomaly dataset”.
