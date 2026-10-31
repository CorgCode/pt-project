"""Вероятности узнавания хоста по трафику как дополнительный обученный блок."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cloudpickle
import numpy as np

from experiments.multiday_gain import MultiDayEncoder, RECIPES, check_chronology, evaluate, restore_training_features
from experiments.overnight.model import T
from experiments.overnight.search import safe_to_continue
from pipeline.data import ProcessedData


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--extra", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/multiday_posterior"))
    ap.add_argument("--validation-window", default="test_day04")
    ap.add_argument("--reference-dir", type=Path, default=Path("scoreboard/data/reference/test_day04"))
    ap.add_argument("--features-from", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = ProcessedData(args.base)
    training = [data, *[ProcessedData(p) for p in args.extra]]
    check_chronology(training, T._span(data, args.validation_window)[0])
    model = (restore_training_features(args.features_from, data, training) if args.features_from
             else MultiDayEncoder().fit_days(training))
    ref = args.reference_dir
    best = -np.inf
    for epochs in [60, 160]:
        if not safe_to_continue():
            raise RuntimeError("Недостаточно ресурсов")
        model.train_head(data, "classifier", epochs)
        for mix in [0.5, 1.0, 2.0]:
            model.posterior_mix, model._parts_cache = mix, {}
            for i, recipe in enumerate(RECIPES[:2]):
                model.recipe, model.pca_dim = recipe, 384
                model.fit_pca(data)
                name = f"posterior_{epochs}_{mix}_{i}"
                score, result = evaluate(model, data, args.validation_window, ref, args.out / name, name)
                row = {"name": name, "score": score, "identity": result["metrics"]["identity_top1"],
                       "epochs": epochs, "mix": mix, "recipe": recipe,
                       "features_from": str(args.features_from) if args.features_from else None}
                with (args.out / "trials.jsonl").open("a") as out:
                    out.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                if score > best:
                    best = score
                    with (args.out / "best.pkl").open("wb") as out:
                        cloudpickle.dump(model, out)
                    (args.out / "best.json").write_text(json.dumps(row, indent=2) + "\n")


if __name__ == "__main__":
    main()
