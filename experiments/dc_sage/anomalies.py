"""Диагностика дрейфа: замена поведения, сканирование и удвоение активности."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from evaluation.uncertainty import auc_with_ci
from experiments.dc_sage.data import load_stats, merge, summarize, universe
from experiments.dc_sage.run import BOARD, ROOT, dump, load, log, frozen_model
from methods.graph._dc_sage import infer, prepare_graph
from pipeline.windows import window_mask


def _save(path, value):
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=5)


def combine(original, changes):
    """changes — (агрегаты, знак); вычитание — точное удаление исходных потоков."""
    pieces = [original]
    for stats, sign in changes:
        copy = {key: table.copy() for key, table in stats.items()}
        if sign != 1:
            for table in copy.values():
                # Времена/bin — ключи, их знак не меняем.
                columns = [c for c in table.select_dtypes(include="number").columns if c not in ("ts", "bin")]
                table[columns] *= sign
        pieces.append(copy)
    result = merge(pieces)
    for key in result:
        table = result[key]
        if "n" in table:
            if (table.n < 0).any():
                raise ValueError(f"Отрицательные счётчики после вмешательства: {key}")
            result[key] = table[table.n > 0].reset_index(drop=True)
        elif key == "volumes":
            result[key] = table[(table.out_count + table.in_count) > 0].reset_index(drop=True)
    return result


def prepare(work):
    out = work / "anomalies"
    out.mkdir(exist_ok=True)
    if (out / "manifest.json").exists():
        return
    data = load(work / "features.pkl")
    prep = load(work / "preprocessor.pkl")
    model, _ = frozen_model(work / "models/dc_sage/seed0")
    weights = prepare_graph(data["train"]["counts"], data["idf"], model.config)
    train_z, _, _ = infer(model, data["train"]["x"], weights)
    cohort = data["cohort_names"]
    groups = KMeans(8, random_state=2026, n_init=10).fit_predict(train_z[data["cohort"]])
    cluster = dict(zip(cohort, groups))
    eval_hosts = pd.read_csv(BOARD / "reference/test/eval_hosts.csv", dtype=str).host_id.tolist()
    rng = np.random.default_rng(2026)
    selected = sorted(rng.choice(eval_hosts, round(.05*len(eval_hosts)), replace=False))
    donors = []
    for host in selected:
        candidates = [h for h in eval_hosts if cluster[h] != cluster[host] and h not in selected]
        donors.append(str(rng.choice(candidates)))
    mapping = pd.DataFrame({"donor": donors, "replacement": selected})
    cache_meta = json.loads((work / "cache/manifest.json").read_text())
    bounds = cache_meta["windows"]["test"]
    source = Path(cache_meta["source"])
    removed, replaced = [], []
    # Дублируются все потоки, в которых выбранный хост — любая сторона, один раз.
    for i, chunk in enumerate(pd.read_csv(source, dtype={c: str for c in ("src", "dst", "src_port", "dst_port", "proto")}, chunksize=400_000)):
        f = chunk.loc[window_mask(chunk.ts.to_numpy(), chunk.duration.to_numpy(), bounds, True)]
        if f.empty:
            continue
        affected = f[f.src.isin(selected) | f.dst.isin(selected)]
        if len(affected):
            removed.append(summarize(affected, bounds))
        a = f.merge(mapping, left_on="src", right_on="donor")
        same = a.dst == a.donor
        a.loc[same, "dst"] = a.loc[same, "replacement"]
        a["src"] = a.replacement
        b = f.merge(mapping, left_on="dst", right_on="donor")
        b = b[b.src != b.donor]  # self-поток уже скопирован в a
        b["dst"] = b.replacement
        clones = pd.concat([a[f.columns], b[f.columns]], ignore_index=True)
        if len(clones):
            replaced.append(summarize(clones, bounds))
        if len(removed) > 8:
            removed = [merge(removed)]
        if len(replaced) > 8:
            replaced = [merge(replaced)]
    original = load_stats(work / "cache", "test")
    subtraction, addition = merge(removed), merge(replaced)
    role = combine(original, [(subtraction, -1), (addition, 1)])
    benign = combine(original, [(subtraction, 1)])
    # Каждому выбранному хосту добавляем 200 коротких попыток к случайным узлам.
    peer_pool = universe(original)
    src = np.repeat(selected, 200)
    dst = rng.choice(peer_pool, len(src))
    same = src == dst
    while same.any():
        dst[same] = rng.choice(peer_pool, int(same.sum()))
        same = src == dst
    scan_flows = pd.DataFrame({"src": src, "dst": dst, "src_port": "PortSCANNER", "dst_port": "445", "proto": "6",
                               "ts": rng.integers(int(bounds[0]), int(bounds[1])-1, len(src)), "duration": 0,
                               "bytes": 64, "packets": 1, "resp_bytes": 0, "resp_packets": 0})
    scanner = combine(original, [(summarize(scan_flows, bounds), 1)])
    for name, value in (("role_change", role), ("scanner", scanner), ("benign_x2", benign)):
        _save(out / f"{name}.pkl", value)
    dump(out / "manifest.json", {"seed": 2026, "selected_hosts": selected, "fraction": .05,
                                  "donors": dict(zip(selected, donors)), "donor_clusters": "KMeans8 of frozen seed0 TRAIN vectors",
                                  "scanner_attempts_per_host": 200, "scanner_port": "445",
                                  "role_change": "remove all incident selected-host flows, clone donor flows with donor replaced",
                                  "benign": "duplicate every incident selected-host flow once",
                                  "parameters_tuned_on_injections": False})
    log(f"инъекции подготовлены: {len(selected)} хоста, три сценария")


def drift(a, b):
    return 1 - np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-12)


def evaluate(work, seeds):
    out = work / "anomalies"
    data, prep = load(work / "features.pkl"), load(work / "preprocessor.pkl")
    meta = json.loads((out / "manifest.json").read_text())
    hosts = pd.read_csv(BOARD / "reference/test/eval_hosts.csv", dtype=str).host_id.tolist()
    label = np.isin(hosts, meta["selected_hosts"]).astype(int)
    train_indices = data["hosts"].get_indexer(hosts)
    results = []
    for scenario in ("role_change", "scanner", "benign_x2"):
        if (out / f"{scenario}_results.json").exists():
            results.extend(json.loads((out / f"{scenario}_results.json").read_text()))
            continue
        with (out / f"{scenario}.pkl").open("rb") as stream:
            stats = pickle.load(stream)
        window_hosts, x, p, counts = prep.transform(stats)
        indices = window_hosts.get_indexer(hosts)
        if (indices < 0).any():
            # Замена может убрать единственное общение контрагента: сохраняем нулевой профиль с флагом.
            window_hosts, x, p, counts = prep.transform(stats, universe(stats, hosts))
            indices = window_hosts.get_indexer(hosts)
        baseline = drift(data["train"]["x"][train_indices], x[indices])
        base_auc = auc_with_ci(label, baseline, n_boot=200, seed=0)
        rows = []
        for seed in seeds:
            model, _ = frozen_model(work / "models/dc_sage" / f"seed{seed}")
            train_z, _, _ = infer(model, data["train"]["x"], prepare_graph(data["train"]["counts"], data["idf"], model.config))
            z, _, _ = infer(model, x, prepare_graph(counts, prep.idf_neighbors(window_hosts), model.config))
            distance = drift(train_z[train_indices], z[indices])
            auc = auc_with_ci(label, distance, n_boot=200, seed=0)
            row = {"scenario": scenario, "seed": seed, "n_hosts": len(hosts), "n_injected": int(label.sum()),
                   "embedding_drift": auc, "profile_drift": base_auc,
                   "interpretation": "benign_x2 should be near 0.5; malicious scenarios should be high"}
            rows.append(row)
            log(f"инъекции {scenario}/seed{seed}: AUC={auc['roc_auc']:.4f}, профиль={base_auc['roc_auc']:.4f}")
        dump(out / f"{scenario}_results.json", rows)
        results.extend(rows)
    dump(out / "results.json", results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "evaluate"])
    ap.add_argument("--workdir", type=Path, default=ROOT / "runs/dc_sage")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    args = ap.parse_args()
    torch.set_num_threads(4)
    prepare(args.workdir) if args.stage == "prepare" else evaluate(args.workdir, args.seeds)


if __name__ == "__main__":
    main()
