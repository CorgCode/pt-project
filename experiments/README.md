# Эксперименты

Все методы на одних данных и эталонах одной командой.

```bash
python -m experiments.end_to_end --fixture --workdir runs/demo                  # самопроверка на тестовых данных
python -m experiments.end_to_end --netflow data/raw/netflow_day-02.bz2 \
    --config cfg.json --workdir runs/lanl --data-kind lanl --source-note "..." --methods all
```

Настоящие данные — по шагам в [`lanl/README.md`](lanl/README.md).

## Что получается в `<workdir>/run`

| файл | что |
|---|---|
| `scoreboard_test.md` | таблица результатов |
| `report.md`, `figures/` | отчёт с графиками |
| `results/` | все метрики по методам |
| `audit.json` | проверки валидности |

## Другие команды

| команда | что делает |
|---|---|
| `experiments.tune_methods` | подбор параметров на validation |
| `experiments.feature_ablation` | что дают порты (#43) |
| `experiments.robustness` | устойчивость к удалению связей (#40) |
| `experiments.dc_sage.run` | DC-SAGE: обучение, пять сидов, абляции; [запуск](dc_sage/README.md), [результаты](../results/dc_sage_day02/README.md) |
| `experiments.hybrid_dc.run` | лидер + линейный агрегат соседей, подбор веса на validation; [запуск](hybrid_dc/README.md) |
| `experiments.make_demo_bundle` | пересобрать демо для дашборда и сайта |

Перед выводами по числам — [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md).
