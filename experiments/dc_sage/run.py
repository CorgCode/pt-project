"""python -m experiments.dc_sage.run {features,pilot,train,evaluate} --workdir runs/dc_sage"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from evaluation import scoreboard, submit
from evaluation.evaluate import run as evaluate
from evaluation.io import load_embeddings
from evaluation.future_links import load_candidate_edges
from evaluation.uncertainty import auc_with_ci, host_index
from experiments.dc_sage.data import Preprocessor, load_stats, universe, sha, prepare
from methods.graph._dc_sage import Config, DCSAGE, Decoder, LinkSampler, infer, prepare_graph, train_epoch

ROOT = Path(__file__).resolve().parents[2]
BOARD = ROOT / "scoreboard/data"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load(path):
    with path.open("rb") as file:
        return pickle.load(file)


def frame(values, hosts):
    return pd.DataFrame(values, index=pd.Index(hosts, name="host_id"), columns=[f"emb_{i}" for i in range(values.shape[1])])


def write_embeddings(path, x, hosts):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame(x, hosts).to_csv(path)


def features(work):
    meta = json.loads((work / "cache/manifest.json").read_text())
    if meta["target_manifest_sha256"] != sha(BOARD / "processed/manifest.json"):
        raise ValueError("Общая подготовка данных изменилась")
    cohort = pd.read_csv(BOARD / "processed/cohort.csv", dtype=str).host_id.tolist()
    training = load_stats(work / "cache", "train")
    validation = load_stats(work / "cache", "validation")
    hosts = universe(training, universe(validation, cohort))
    prep = Preprocessor().fit(training, cohort)
    data = {"hosts": hosts, "cohort": hosts.get_indexer(cohort), "cohort_names": cohort}
    for w, stats in (("train", training), ("validation", validation)):
        _, x, p, counts = prep.transform(stats, hosts)
        data[w] = {"x": x, "p": p, "counts": counts}
    del training, validation
    for w in ("s1", "s2"):
        stats = load_stats(work / "cache", w)
        _, x, p, counts = prep.transform(stats, hosts)
        data[w] = {"x": x, "p": p, "counts": counts}
        del stats
    one, two = data["s1"]["counts"], data["s2"]["counts"]
    event1 = np.asarray(one.sum(0) + one.sum(1).T).ravel()
    event2 = np.asarray(two.sum(0) + two.sum(1).T).ravel()
    data["positives"] = data["cohort"][(event1[data["cohort"]] >= 30) & (event2[data["cohort"]] >= 30)]
    data["idf"] = prep.idf_neighbors(hosts)
    with (work / "features.pkl").open("wb") as file:
        pickle.dump(data, file, protocol=5)
    with (work / "preprocessor.pkl").open("wb") as file:
        pickle.dump(prep, file, protocol=5)
    dump(work / "feature_manifest.json", {"nodes": len(hosts), "cohort": len(cohort), "contrastive_hosts": len(data["positives"]),
                                         "tokens": len(prep.vocab), "input_dim": data["train"]["x"].shape[1],
                                         "behavior_columns": prep.behavior_names, "svd_dimensions": prep.svd.n_components,
                                         "fit_windows": ["train"], "slices": meta["windows"],
                                         "schema": "role-service TFIDF/SVD64 + 31 behavioural features; fixed train z-score, clip5"})
    log(f"признаки готовы: {len(hosts)} узлов, {len(data['positives'])} контрастных хостов, {data['train']['x'].shape[1]} признаков")


def score_folder(folder, window, bootstrap=0):
    reference = BOARD / "reference" / window
    result = evaluate(folder / "embeddings.csv", reference, folder.name, n_boot=bootstrap, seed=0,
                      fit_reference_dir=BOARD / "reference/validation" if window == "test" else None)
    submit.add_identity(result, folder / "embeddings.csv", folder / f"{window}.csv", reference, shared=True, n_boot=bootstrap)
    cols = {row["metric"]: row["title"] for row in result["summary"]}
    score = float(scoreboard.overall_score([result], cols).iloc[0])
    return score, result


def fit_one(work, name, config, seed, checkpoints=(25, 50), data=None):
    folder = work / "models" / name / f"seed{seed}"
    if (folder / "selection.json").exists():
        log(f"уже обучено: {name}/seed{seed}")
        return json.loads((folder / "selection.json").read_text())
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = load(work / "features.pkl") if data is None else data
    model = DCSAGE(data["train"]["x"].shape[1], config)
    decoder = Decoder(config.dim)
    optimizer = torch.optim.AdamW([*model.parameters(), *decoder.parameters()], lr=config.lr, weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    weights = {w: prepare_graph(data[w]["counts"], data["idf"], config) for w in ("train", "s1", "s2", "validation")}
    sampler = LinkSampler(data["s1"]["counts"], data["s2"]["counts"], data["cohort"], config)
    tensors = {w: torch.tensor(data[w]["x"], dtype=torch.float32) for w in ("s1", "s2")}
    profile = torch.tensor(data["train"]["p"], dtype=torch.float32)
    trials, history = [], []
    started = time.perf_counter()
    best = -np.inf
    for epoch in range(1, config.epochs + 1):
        stats = train_epoch(model, decoder, optimizer, tensors["s1"], tensors["s2"], weights["s1"], weights["s2"],
                            data["positives"], profile, sampler, rng)
        schedule.step()
        history.append({"epoch": epoch, **stats})
        if epoch == 1 or epoch % 10 == 0:
            log(f"{name} seed={seed} epoch={epoch}: loss={stats['loss']:.4f}, cos={stats['mean_cosine']:.3f}, graph/own={stats['graph_own_ratio']:.3f}, elapsed={time.perf_counter()-started:.0f}s")
        if epoch not in checkpoints:
            continue
        for w in ("train", "validation"):
            z, _, _ = infer(model, data[w]["x"], weights[w])
            indices = data["cohort"]
            write_embeddings(folder / ("embeddings.csv" if w == "train" else "validation.csv"), z[indices], data["cohort_names"])
        score, result = score_folder(folder, "validation")
        trials.append({"epoch": epoch, "validation_score": score, "metrics": result["metrics"]})
        dump(folder / f"validation_epoch{epoch}.json", result)
        log(f"{name} seed={seed} epoch={epoch}: validation score={score:.3f}")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "decoder": decoder.state_dict(), "config": config.dictionary(),
                        "seed": seed, "epoch": epoch, "inputs": data["train"]["x"].shape[1]}, folder / "best.pt")
            dump(folder / "best_validation.json", result)
    dump(folder / "history.json", history)
    selection = {"variant": name, "seed": seed, "config": config.dictionary(), "trials": trials,
                 "best_validation_score": best, "seconds": time.perf_counter()-started, "test_read": False}
    dump(folder / "selection.json", selection)
    return selection


def pilot(work):
    if (work / "frozen_config.json").exists():
        log("конфигурация уже зафиксирована")
        return
    data = load(work / "features.pkl")
    records = []
    # Бюджет заранее фиксирован: три веса связей, остальные настройки инструкции.
    dump(work / "protocol.json", {"method": "DC-SAGE", "selection": "official validation score; no test access",
                                  "pilot_link_weights": [.1, .3, 1.], "checkpoints": [25, 50], "seeds": list(range(5)),
                                  "primary_dimension": 64, "ablation_dimensions": [32, 64],
                                  "available_test_windows": ["test"], "missing_test_windows": ["test_day03", "test_day04"],
                                  "training_step": "one full-graph update per epoch; all contrastive batches contribute",
                                  "torch": torch.__version__, "python": platform.python_version()})
    for weight in (.1, .3, 1.):
        name = f"pilot_link{weight:g}"
        records.append(fit_one(work, name, replace(Config(), link_weight=weight), 0, data=data))
    best = max(records, key=lambda r: r["best_validation_score"])
    dump(work / "frozen_config.json", {"config": best["config"], "selected_pilot": best["variant"],
                                       "validation_score": best["best_validation_score"], "test_read": False})
    log(f"зафиксирован {best['variant']}, validation={best['best_validation_score']:.3f}")


def variants(config):
    return {"dc_sage": config, "skip_only": replace(config, graph=False),
            "undirected": replace(config, directed=False), "no_idf": replace(config, hub_idf=False),
            "no_contrastive": replace(config, contrastive=False), "no_links": replace(config, link_weight=0),
            "no_mask": replace(config, false_negative_mask=False), "one_layer": replace(config, layers=1),
            "dc_sage32": replace(config, dim=32)}


def train(work, names=None, seeds=range(5)):
    selected = json.loads((work / "frozen_config.json").read_text())
    config = Config(**selected["config"])
    data = load(work / "features.pkl")
    for name, c in variants(config).items():
        if names and name not in names:
            continue
        for seed in seeds:
            if name == "dc_sage" and seed == 0:
                destination = work / "models/dc_sage/seed0"
                if not destination.exists():
                    shutil.copytree(work / "models" / selected["selected_pilot"] / "seed0", destination)
            fit_one(work, name, c, seed, data=data)


def frozen_model(folder):
    state = torch.load(folder / "best.pt", weights_only=True)
    model = DCSAGE(state["inputs"], Config(**state["config"]))
    model.load_state_dict(state["model"])
    decoder = Decoder(state["config"]["dim"])
    decoder.load_state_dict(state["decoder"])
    return model, decoder


def baseline_embeddings(data, test, out, window="test"):
    """Все простые бейзлайны — 64 координаты; преобразования подбираются на train."""
    cohort = data["cohort"]
    def reduced(train, later):
        d = min(64, train.shape[1] - 1, len(cohort)-1)
        svd = TruncatedSVD(d, random_state=0).fit(train[cohort])
        return svd.transform(train), svd.transform(later)
    p1, p2 = data["train"]["p"], test["p"]
    b1, b2 = data["train"]["x"][:, p1.shape[1]:], test["x"][:, p2.shape[1]:]
    def aggregate(x, c):
        c = c + c.T
        c.data = np.log1p(c.data)
        c = sp.diags(1 / np.maximum(np.asarray(c.sum(1)).ravel(), 1)) @ c
        return np.hstack([x, .5 * (c @ x)])
    x1, x2 = data["train"]["x"], test["x"]
    linear = reduced(aggregate(x1, data["train"]["counts"]), aggregate(x2, test["counts"]))
    pairs = {"profile": (p1, p2), "behavioral": (b1, b2), "linear": linear,
             "random": (np.random.default_rng(0).normal(size=(len(x1), 64)), np.random.default_rng(1).normal(size=(len(x2), 64)))}
    for name, (a, b) in pairs.items():
        folder = out / name
        for filename, x, indices in (("embeddings.csv", a, cohort), (f"{window}.csv", b, test["cohort"])):
            x = x[indices]
            if x.shape[1] < 64:
                x = np.pad(x, ((0, 0), (0, 64-x.shape[1])))
            zero = np.linalg.norm(x, axis=1) == 0
            x[zero, 0] = 1  # одинаковый fallback для пустого профиля, без идентификаторов
            write_embeddings(folder / filename, x, data["cohort_names"])


def baseline_validation(work):
    data = load(work / "features.pkl")
    val = {**data["validation"], "cohort": data["cohort"]}
    baseline_embeddings(data, val, work / "baselines_validation", "validation")
    for name in ("profile", "behavioral", "linear", "random"):
        score, result = score_folder(work / "baselines_validation" / name, "validation")
        dump(work / "results/validation_baselines" / f"{name}.json", result)
        log(f"validation {name}: score={score:.3f}")


@torch.no_grad()
def score_decoder(decoder, folder, data, bootstrap):
    edges = load_candidate_edges(BOARD / "reference/test/edges.csv")
    edges = edges[edges.split == "test"]
    embedding = load_embeddings(folder / "embeddings.csv")
    z = torch.tensor(embedding.to_numpy(), dtype=torch.float32)
    a = embedding.index.get_indexer(edges.host_a)
    b = embedding.index.get_indexer(edges.host_b)
    logdeg = torch.tensor(np.log1p(data["train"]["counts"].getnnz(axis=0)[data["cohort"]]), dtype=torch.float32)
    # Эталон связи не направлен: вероятность общения — максимум двух направлений.
    score = torch.maximum(decoder(z[a], z[b], logdeg[b]), decoder(z[b], z[a], logdeg[a])).numpy()
    result = auc_with_ci(edges.label.to_numpy(int), score, n_boot=bootstrap, seed=0,
                         hosts=host_index(edges.host_a, edges.host_b))
    return {**result, "direction_reduction": "max(u->v, v->u)", "degree_window": "train"}


def freeze(work):
    if not (work / "frozen_config.json").exists():
        raise ValueError("Сначала pilot и train")
    if not (work / "evaluation_freeze.json").exists():
        expected = [(name, seed) for name in variants(Config()) for seed in range(5)]
        if any(not (work / "models" / name / f"seed{seed}/selection.json").exists() for name, seed in expected):
            raise ValueError("До чтения test требуется завершить все 9 вариантов × 5 сидов")
        primary = [json.loads((work / "models/dc_sage" / f"seed{s}/selection.json").read_text()) for s in range(5)]
        best_seed = max(primary, key=lambda r: r["best_validation_score"])["seed"]
        dump(work / "evaluation_freeze.json", {
            "selected_export_seed_by_validation": best_seed,
            "models_sha256": {f"{name}/seed{seed}": sha(work / "models" / name / f"seed{seed}/best.pt") for name, seed in expected},
            "test_used_for_selection": False, "available_test_windows": ["test"],
            "missing_test_windows": ["test_day03", "test_day04"],
        })


def evaluate_all(work, names=None, seeds=range(5), bootstrap=200):
    # Начинается только после заморозки всех обучений/остановок, без подбора по результатам.
    freeze(work)
    data = load(work / "features.pkl")
    prep = load(work / "preprocessor.pkl")
    stats = load_stats(work / "cache", "test")
    hosts = pd.Index([*data["hosts"], *sorted(set(universe(stats)) - set(data["hosts"]))])
    _, x, p, counts = prep.transform(stats, hosts)
    test = {"x": x, "p": p, "counts": counts, "cohort": hosts.get_indexer(data["cohort_names"])}
    all_results = []
    for name in variants(Config()):
        if names and name not in names:
            continue
        for seed in seeds:
            source = work / "models" / name / f"seed{seed}"
            if not (source / "best.pt").exists():
                raise ValueError(f"Не завершено обучение {source}")
            folder = work / "submissions" / f"{name}_seed{seed}"
            result_path = work / "results/test" / f"{name}_seed{seed}.json"
            if result_path.exists():
                all_results.append(json.loads(result_path.read_text()))
                continue
            model, decoder = frozen_model(source)
            for window, inputs, count, window_hosts, indices in (
                ("train", data["train"]["x"], data["train"]["counts"], data["hosts"], data["cohort"]),
                ("test", x, counts, hosts, test["cohort"]),
            ):
                weights = prepare_graph(count, prep.idf_neighbors(window_hosts), model.config)
                z, own, surrounding = infer(model, inputs, weights)
                write_embeddings(folder / ("embeddings.csv" if window == "train" else "test.csv"), z[indices], data["cohort_names"])
                np.savez_compressed(folder / f"{window}_parts.npz", own=own[indices], graph=surrounding[indices])
            score, r = score_folder(folder, "test", bootstrap)
            r["metric_details"]["dc_sage_decoder"] = score_decoder(decoder, folder, data, bootstrap)
            r["method_info"] = {"config": model.config.dictionary(), "seed": seed, "fit_windows": ["train"],
                                 "available_test_windows": ["test"], "incomplete_multiday": True}
            dump(result_path, r)
            all_results.append(r)
            log(f"test {name}/seed{seed}: score={score:.3f}")
    if not names:
        baseline_embeddings(data, test, work / "submissions")
        for name in ("profile", "behavioral", "linear", "random"):
            result_path = work / "results/test" / f"{name}.json"
            if result_path.exists():
                r = json.loads(result_path.read_text())
            else:
                score, r = score_folder(work / "submissions" / name, "test", bootstrap)
                dump(result_path, r)
                log(f"test {name}: score={score:.3f}")
            all_results.append(r)
    scoreboard.check_compatible(all_results)
    (work / "comparison_test.md").write_text(scoreboard.headline_table(all_results))
    scoreboard.full_table(all_results).to_csv(work / "comparison_test.csv", index=False)


def test_baselines(work, bootstrap=200):
    freeze(work)
    data, prep = load(work / "features.pkl"), load(work / "preprocessor.pkl")
    stats = load_stats(work / "cache", "test")
    hosts = pd.Index([*data["hosts"], *sorted(set(universe(stats))-set(data["hosts"]))])
    _, x, p, counts = prep.transform(stats, hosts)
    later = {"x": x, "p": p, "counts": counts, "cohort": hosts.get_indexer(data["cohort_names"])}
    baseline_embeddings(data, later, work / "submissions")
    for name in ("profile", "behavioral", "linear", "random"):
        path = work / "results/test" / f"{name}.json"
        if not path.exists():
            score, result = score_folder(work / "submissions" / name, "test", bootstrap)
            dump(path, result)
            log(f"test {name}: score={score:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "features", "baselines", "pilot", "train", "freeze", "evaluate", "test-baselines"])
    ap.add_argument("--workdir", type=Path, default=ROOT / "runs/dc_sage")
    ap.add_argument("--variants", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--source", type=Path, default=ROOT / "data/processed/flows/train.csv")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.stage == "prepare":
        prepare(args.source, BOARD, args.workdir / "cache")
    elif args.stage == "features":
        features(args.workdir)
    elif args.stage == "baselines":
        baseline_validation(args.workdir)
    elif args.stage == "pilot":
        pilot(args.workdir)
    elif args.stage == "train":
        train(args.workdir, args.variants, args.seeds)
    elif args.stage == "freeze":
        freeze(args.workdir)
    elif args.stage == "test-baselines":
        test_baselines(args.workdir, args.bootstrap)
    else:
        evaluate_all(args.workdir, args.variants, args.seeds, args.bootstrap)


if __name__ == "__main__":
    main()
