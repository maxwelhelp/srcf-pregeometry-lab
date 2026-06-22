# Closure graph benchmark v3 triangle

Что изменено:

- triangle update внутри closure-layer: `sum_k h[i,k] * h[k,j]`;
- raw-only ablation: `--rel-mode raw`;
- directed stress-test: `--directed`;
- optional closure_no_tri ablation: `--ablate-tri`;
- fixed-point curve ratio логируется; `--fixed-w` можно включить, но по умолчанию выключено ради скорости;
- deep_big и GCN baseline сохранены.

Главные проверки:

1. full features + triangle против deep_big/GCN;
2. raw-only: если closure остаётся сильным, значит дело не только в ручных фичах;
3. ablate-tri: если `closure` > `closure_no_tri`, значит помогает настоящий треугольный/path update;
4. directed: проверка на направленные relation-графы.
