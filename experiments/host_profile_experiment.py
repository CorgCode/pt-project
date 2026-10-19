"""Агрегировать LANL чанками; подобрать профиль на validation; один test выбранного варианта.

python -m experiments.host_profile_experiment prepare
python -m experiments.host_profile_experiment run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from evaluation import scoreboard
from evaluation.evaluate import run as evaluate
from evaluation.identity import identity_retrieval
from evaluation.io import load_embeddings
from evaluation.validate import validate_submission
from methods.aggregated_host_profile import AggregatedHostProfile, aggregate
from pipeline.windows import window_mask

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "scoreboard"
CONFIGS = {
    "raw_undirected": dict(weighting="raw", direction=False, center=False),
    "raw_directed": dict(weighting="raw", direction=True, center=False),
    "log_directed": dict(weighting="log", direction=True, center=False),
    "tfidf_undirected": dict(weighting="tfidf", direction=False, center=False),
    "tfidf_directed": dict(weighting="tfidf", direction=True, center=False),
    "tfidf_centered": dict(weighting="tfidf", direction=True, center=True),
    "tfidf_struct": dict(weighting="tfidf", direction=True, center=True, struct_weight=.5),
}
SELECTION_METRIC = "partner_new_mrr"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def prepare(source, out, chunksize=500_000):
    """Подокна восстанавливаются только из целиком покрывающего их исходного окна."""
    target = json.loads((BOARD / "data/processed/manifest.json").read_text())
    source_manifest_path = source.parent.parent / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    target_windows = target["config"]["windows"]
    containing = source_manifest["config"]["windows"][source.stem]
    if not all(containing[0] <= start and end <= containing[1] for start, end in target_windows.values()):
        raise ValueError("Исходное окно не покрывает целевые окна целиком")
    if source_manifest["inputs"] != target["inputs"]:
        raise ValueError("Разные исходные архивы: сверка inputs/sha256 не пройдена")
    destination = out / "cache"
    if (destination / "manifest.json").exists():
        raise ValueError(f"{destination} уже подготовлен; используйте run")
    destination.mkdir(parents=True, exist_ok=True)
    profiles = {w: [] for w in target_windows}
    edges = {w: [] for w in target_windows}
    events = {w: pd.Series(dtype=float) for w in target_windows}
    n_flows = {w: 0 for w in target_windows}
    n_self = {w: 0 for w in target_windows}
    scanned = 0
    cols = ["src", "dst", "src_port", "dst_port", "proto", "ts", "duration"]
    dtypes = {c: str for c in ["src", "dst", "src_port", "dst_port", "proto"]}
    # Весь файл читается: порядок строк не предполагается и не влияет на полноту.
    t0 = time.perf_counter()
    for i, chunk in enumerate(pd.read_csv(source, usecols=cols, dtype=dtypes, chunksize=chunksize)):
        scanned += len(chunk)
        for w, bounds in target_windows.items():
            part = chunk.loc[window_mask(chunk.ts.to_numpy(), chunk.duration.to_numpy(), bounds, True)]
            if part.empty:
                continue
            n_flows[w] += len(part)
            real = part[part.src != part.dst]
            n_self[w] += len(part) - len(real)
            events[w] = events[w].add(pd.concat([real.src, real.dst]).value_counts(), fill_value=0)
            profiles[w].append(aggregate(part))
            edges[w].append(real[["src", "dst"]].drop_duplicates())
        if i % 10 == 0:
            log(f"прочитано {scanned:,}; потоки целевых окон: {n_flows}")
            for w in target_windows:
                if profiles[w]:
                    profiles[w] = [pd.concat(profiles[w]).groupby(["host", "key", "w"], as_index=False)["n"].sum()]
                    edges[w] = [pd.concat(edges[w]).drop_duplicates()]
    cohort = pd.read_csv(BOARD / "data/processed/cohort.csv", dtype={"host_id": str}).set_index("host_id")
    threshold = target["config"]["min_events_per_window"]
    selected = sorted(events["train"][events["train"] >= threshold].index)
    if selected != cohort.index.tolist():
        raise ValueError("Восстановленная train-когорта не совпала с scoreboard")
    for w in target_windows:
        if n_flows[w] != target["windows"][w]["n_flows"]:
            raise ValueError(f"Не совпало число потоков {w}: {n_flows[w]}")
        if n_self[w] != target["counts"][f"self_flows_{w}"]:
            raise ValueError(f"Не совпало число self-потоков {w}")
        actual = events[w].reindex(cohort.index, fill_value=0).astype(int)
        np.testing.assert_array_equal(actual.to_numpy(), cohort[f"events_{w}"].to_numpy())
        if w != "train":
            expected = pd.read_csv(BOARD / f"data/reference/{w}/eval_hosts.csv", dtype=str).host_id.tolist()
            if actual[actual >= threshold].index.tolist() != expected:
                raise ValueError(f"Не совпали eval_hosts {w}")
        pd.concat(profiles[w]).groupby(["host", "key", "w"], as_index=False)["n"].sum().to_csv(destination / f"{w}_counts.csv", index=False)
        pd.concat(edges[w]).drop_duplicates().to_csv(destination / f"{w}_edges.csv", index=False)
        log(f"{w}: число потоков, self-потоков и все счётчики когорты совпали")
    pd.DataFrame({"host_id": selected}).to_csv(destination / "hosts.csv", index=False)
    dump(destination / "manifest.json", {
        "source": str(source), "source_sha256": sha(source),
        "source_manifest_sha256": sha(source_manifest_path), "original_inputs": target["inputs"],
        "target_manifest_sha256": sha(BOARD / "data/processed/manifest.json"),
        "cohort_sha256": target["cohort"]["sha256"], "windows": target_windows,
        "n_flows": n_flows, "n_self_flows": n_self, "n_hosts": len(selected),
        "exact_flow_counts_and_cohort_event_counts": True, "rows_scanned": scanned,
        "window_sec": 1800, "seconds": time.perf_counter() - t0,
        "cache_files_sha256": {p.name: sha(p) for p in sorted(destination.glob("*.csv"))},
    })


def read_profile(cache, window):
    return (pd.read_csv(cache / f"{window}_counts.csv", dtype={"host": str}),
            pd.read_csv(cache / f"{window}_edges.csv", dtype=str))


def attach_identity(result, a, b, reference, bootstrap):
    hosts = pd.read_csv(reference / "eval_hosts.csv", dtype=str).host_id.tolist()
    modes = {mode: identity_retrieval(a.loc[hosts], b.loc[hosts], alignment=mode, n_boot=bootstrap, seed=0)
             for mode in ("none", "procrustes")}
    ident = modes["procrustes"]  # та же процедура, что в общей таблице
    result["metrics"].update(identity_top1=ident["top1"], identity_mrr=ident["mrr"], identity_pair_roc_auc=ident["pair_roc_auc"])
    if ident.get("top1_ci"):
        result["metrics_ci"]["identity_top1"] = list(ident["top1_ci"])
    result["metric_details"]["identity"] = {**modes, "shared_coordinate_system_declared": True}
    result["metric_status"]["identity"] = "ok"
    result["summary"].append({"group": "Узнаваемость", "metric": "identity_top1", "title": "Узнаваемость, Top-1",
                              "value": ident["top1"], "ci": result["metrics_ci"].get("identity_top1"), "chance": ident["chance_top1"]})


def experiment(out, bootstrap=200):
    cache = out / "cache"
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["target_manifest_sha256"] != sha(BOARD / "data/processed/manifest.json"):
        raise ValueError("Манифест общей таблицы изменился")
    for filename, expected in manifest.get("cache_files_sha256", {}).items():
        if sha(cache / filename) != expected:
            raise ValueError(f"Изменился файл кэша {filename}")
    if (out / "selection.json").exists():
        raise ValueError("Эксперимент уже выбирал модель; используйте новую out-dir для нового протокола")
    hosts = pd.read_csv(cache / "hosts.csv", dtype=str).host_id.tolist()
    train, train_edges = read_profile(cache, "train")
    val, val_edges = read_profile(cache, "validation")
    ref = BOARD / "data/reference"
    dump(out / "protocol.json", {"configs": CONFIGS, "dim": 32, "seed": 0,
                                 "selection_metric": SELECTION_METRIC, "selection_reference": "validation",
                                 "trained_on_windows": ["train"], "bootstrap": bootstrap,
                                 "python": platform.python_version(), "cache": manifest,
                                 "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                                              "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
                                 "host_profile_sha256": sha(ROOT / "host_profile.py"),
                                 "adapter_sha256": sha(ROOT / "methods/aggregated_host_profile.py"),
                                 "runner_sha256": sha(Path(__file__))})
    rows, models = [], {}
    for name, kw in CONFIGS.items():
        t0 = time.perf_counter()
        method = f"host_profile_{name}"
        log(f"обучение {method}")
        model = AggregatedHostProfile(dim=32, seed=0, **kw).fit_profile(train, hosts, train_edges)
        a, b = model.transform_profile(train, train_edges), model.transform_profile(val, val_edges)
        folder = out / "submissions" / method
        folder.mkdir(parents=True, exist_ok=True)
        a.to_csv(folder / "embeddings.csv")
        b.to_csv(folder / "validation.csv")
        (folder / "SHARED_COORDINATES").touch()
        dump(folder / "provenance.json", {"trained_on_windows": ["train"], "config": kw,
                                         "cohort_sha256": manifest["cohort_sha256"]})
        validate_submission(str(folder / "embeddings.csv"))
        r = evaluate(folder / "embeddings.csv", ref / "validation", method, n_boot=bootstrap, seed=0)
        attach_identity(r, a, b, ref / "validation", bootstrap)
        r["method_info"] = {"config": kw, "fit_windows": ["train"], "shared_coordinate_system": True}
        dump(out / "results/validation" / f"{method}.json", r)
        rows.append(r)
        models[method] = model
        log(f"{method}: {SELECTION_METRIC}={r['metrics'][SELECTION_METRIC]:.6f}, {time.perf_counter()-t0:.1f} с")
    best = max(rows, key=lambda r: r["metrics"][SELECTION_METRIC])
    selected = best["method"]
    # Выбор фиксируется на диске до первого чтения тестового профиля и эталона.
    dump(out / "selection.json", {"method": selected, "config": best["method_info"]["config"],
                                   "metric": SELECTION_METRIC, "validation_score": best["metrics"][SELECTION_METRIC],
                                   "n_configurations": len(rows), "test_read_at_selection": False})
    (out / "validation.md").write_text(scoreboard.headline_table(rows))
    log(f"зафиксирован {selected}; единственная оценка выбранного профиля на test")
    test, test_edges = read_profile(cache, "test")
    model = models[selected]
    folder = out / "submissions" / selected
    a = load_embeddings(folder / "embeddings.csv")
    b = model.transform_profile(test, test_edges)
    b.to_csv(folder / "test.csv")
    r = evaluate(folder / "embeddings.csv", ref / "test", selected, n_boot=bootstrap, seed=0,
                 fit_reference_dir=ref / "validation")
    attach_identity(r, a, b, ref / "test", bootstrap)
    r["method_info"] = best["method_info"]
    dump(out / "results/test" / f"{selected}.json", r)
    baselines = [json.loads(p.read_text()) for p in sorted((BOARD / "results").glob("*.json"))]
    scoreboard.check_compatible([*baselines, r])
    (out / "comparison_test.md").write_text(scoreboard.headline_table([*baselines, r]))
    scoreboard.full_table([*baselines, r]).to_csv(out / "comparison_test.csv", index=False)
    log(f"готово: {out / 'comparison_test.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["prepare", "run"])
    ap.add_argument("--source", type=Path, default=ROOT / "data/processed/flows/train.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs/host_profile")
    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.out_dir)
    else:
        experiment(args.out_dir, args.bootstrap)


if __name__ == "__main__":
    main()
