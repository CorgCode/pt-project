"""Подбор гиперпараметров графового метода ТОЛЬКО на validation.

    python -m experiments.tune_methods --processed-dir data/processed --reference-dir data/reference \
        --method node2vec --grid '{"p": [0.25, 1, 4], "q": [0.25, 1, 4]}' --metric pair_roc_auc --out runs/tune_node2vec.json

Каждая точка сетки обучается на train и оценивается на reference/validation; test не читается
(--target test запрещён). Метрику выбора фиксируйте ДО просмотра результатов; отчёт пишет, сколько точек
перепробовано (больше точек — оптимистичнее лучший результат). Выбранную конфигурацию замораживаем
и один раз прогоняем на test.
"""
from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from pathlib import Path

from benchmark import methods as graph_methods
from evaluation.evaluate import run
from pipeline.data import ProcessedData

MAX_CONFIGS = 64


def tune(processed_dir: Path, reference_dir: Path, method: str, grid: dict[str, list], *, dim: int = 16,
         seed: int = 0, metric: str = "pair_roc_auc", fast: bool = False, n_boot: int = 0) -> dict:
    reference_dir = Path(reference_dir)
    ref = reference_dir / "validation"
    if not ref.exists():
        raise ValueError(f"{ref} не найден: для подбора нужен эталон validation (pipeline.reference с окном validation)")
    if method not in graph_methods.available():
        raise ValueError(f"неизвестный метод {method!r}; есть: {sorted(graph_methods.available())}")
    keys = sorted(grid)
    configs = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))] or [{}]
    if len(configs) > MAX_CONFIGS:
        raise ValueError(f"{len(configs)} конфигураций больше предела {MAX_CONFIGS} "
                         "(каждая лишняя делает лучший результат на validation оптимистичнее)")
    data = ProcessedData(processed_dir)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, cfg in enumerate(configs):
            m = graph_methods.create([method], dim, seed, {method: cfg}, fast)[0].fit(data)
            path = Path(tmp) / f"{i}.csv"
            m.embed(data, "train").to_csv(path)
            res = run(path, ref, f"{method}#{i}", n_boot=n_boot, seed=seed)
            if metric not in res["metrics"]:
                raise ValueError(f"метрики {metric!r} нет на эталоне validation; "
                                 f"выберите из {sorted(k for k in res['metrics'] if not k.endswith(('_std', '_median')))}")
            rows.append({"config": cfg, "metrics": res["metrics"], "fit_seconds": m.info()["fit_seconds"]["train"]})
    best = max(rows, key=lambda r: r["metrics"][metric])
    return {
        "method": method, "selected_by": metric, "n_configurations": len(rows), "target_reference": "validation",
        "test_reference_read": False, "best_config": best["config"], "best_validation_score": best["metrics"][metric],
        "dim": dim, "seed": seed, "fast": fast, "results": rows,
        "warning": f"best of {len(rows)} configurations on the validation window: its score is optimistically biased; "
                   "report the frozen configuration's TEST score, not this number",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--grid", required=True, help='JSON {param: [values]} or a path to a JSON file')
    ap.add_argument("--metric", default="pair_roc_auc", help="selection metric, fixed in advance")
    ap.add_argument("--target", default="validation", help="must be 'validation' (the test reference is never used)")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.target != "validation":
        ap.error("tuning on the test reference is refused; use --target validation")
    g = Path(a.grid)
    grid = json.loads(g.read_text(encoding="utf-8") if g.exists() else a.grid)
    out = tune(a.processed_dir, a.reference_dir, a.method, grid, dim=a.dim, seed=a.seed, metric=a.metric, fast=a.fast)
    text = json.dumps(out, indent=2, sort_keys=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("method", "selected_by", "n_configurations", "best_config",
                                          "best_validation_score", "warning")}, indent=2))


if __name__ == "__main__":
    main()
