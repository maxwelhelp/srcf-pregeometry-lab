# Seed results notes

Формат для коротких результатов:

| date | experiment | dataset | steps | key metric | result | conclusion |
|---|---|---|---:|---|---:|---|
| 2026-06-22 | SRCF v5 DNA train | E.coli k=3 | 300 | contract/h_contract/far_keep/move | 0.55 / 0.75 / 0.79 / 0.409 | basin dynamics работает |
| 2026-06-22 | SRCF v6 synthetic anomaly | synthetic hard | 150 | closure best | 1.000 | сильный, но synthetic слишком лёгкий |
| 2026-06-22 | SRCF v6 real DNA | E.coli anomalies | 150 | closure best overall | 0.6785 | пока не бьёт raw/kmer baseline |
