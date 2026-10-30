"""Проверка новых хостов: полное исключение 30% когорты до обучения признаков и сети."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.frozen_neighbors import evaluate_frozen_neighbors
from evaluation.partners import load_partners, partner_ranking
from evaluation.uncertainty import bootstrap_ci
from experiments.dc_sage.data import Preprocessor, load_stats, merge, summarize, universe
from experiments.dc_sage.run import BOARD, ROOT, dump, frame, load, log, frozen_model, write_embeddings
from methods.graph._dc_sage import Config, DCSAGE, Decoder, LinkSampler, infer, prepare_graph, train_epoch
from pipeline.windows import window_mask


def prepare(work):
    out = work / "inductive"
    out.mkdir(exist_ok=True)
    if (out / "features.pkl").exists():
        return
    cohort = pd.read_csv(BOARD / "processed/cohort.csv", dtype=str).host_id.tolist()
    selected = np.random.default_rng(2026).choice(cohort, round(.3*len(cohort)), replace=False)
    banned = set(selected)
    known = [h for h in cohort if h not in banned]
    dump(out / "holdout.json", {"seed": 2026, "hosts": sorted(banned), "fraction": .3,
                                "excluded_from": ["flows", "graph", "SVD", "IDF", "normalization", "neural_loss"],
                                "hyperparameters": "frozen main-experiment configuration; no inductive test tuning"})
    windows = json.loads((work / "cache/manifest.json").read_text())["windows"]
    parts = {w: [] for w in ("train", "s1", "s2")}
    for i, chunk in enumerate(pd.read_csv(work / "cache/train_flows.csv", dtype={c: str for c in ("src", "dst", "src_port", "dst_port", "proto")}, chunksize=400_000)):
        chunk = chunk[~chunk.src.isin(banned) & ~chunk.dst.isin(banned)]
        for w in parts:
            p = chunk.loc[window_mask(chunk.ts.to_numpy(), chunk.duration.to_numpy(), windows[w], True)]
            if len(p):
                parts[w].append(summarize(p, windows[w]))
        if i % 5 == 0:
            for w in parts:
                if parts[w]:
                    parts[w] = [merge(parts[w])]
    stats = {w: merge(value) for w, value in parts.items()}
    for value in stats.values():
        assert not set(value["volumes"].h) & banned
        assert not (set(value["edges"].src) | set(value["edges"].dst)) & banned
    prep = Preprocessor().fit(stats["train"], known)
    hosts = universe(stats["train"], known)
    data = {"hosts": hosts, "cohort_names": known, "cohort": hosts.get_indexer(known), "idf": prep.idf_neighbors(hosts)}
    for w in stats:
        _, x, p, c = prep.transform(stats[w], hosts)
        data[w] = {"x": x, "p": p, "counts": c}
    e1 = np.asarray(data["s1"]["counts"].sum(0) + data["s1"]["counts"].sum(1).T).ravel()
    e2 = np.asarray(data["s2"]["counts"].sum(0) + data["s2"]["counts"].sum(1).T).ravel()
    data["positives"] = data["cohort"][(e1[data["cohort"]] >= 30) & (e2[data["cohort"]] >= 30)]
    for name, value in (("features", data), ("preprocessor", prep)):
        with (out / f"{name}.pkl").open("wb") as file:
            pickle.dump(value, file, protocol=5)
    log(f"новые хосты: исключены {len(banned)}, осталось {len(known)} обучающих хостов")


def train(work, seeds):
    out = work / "inductive"
    data = load(out / "features.pkl")
    config = Config(**json.loads((work / "frozen_config.json").read_text())["config"])
    main_pilot = json.loads((work / "frozen_config.json").read_text())["selected_pilot"]
    fixed_epoch = torch.load(work / "models" / main_pilot / "seed0/best.pt", weights_only=True)["epoch"]
    weights = {w: prepare_graph(data[w]["counts"], data["idf"], config) for w in ("s1", "s2")}
    tensors = {w: torch.tensor(data[w]["x"], dtype=torch.float32) for w in ("s1", "s2")}
    profile = torch.tensor(data["train"]["p"], dtype=torch.float32)
    sampler = LinkSampler(data["s1"]["counts"], data["s2"]["counts"], data["cohort"], config)
    for seed in seeds:
        folder = out / f"seed{seed}"
        if (folder / "best.pt").exists():
            continue
        folder.mkdir(exist_ok=True)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model, decoder = DCSAGE(tensors["s1"].shape[1], config), Decoder(config.dim)
        opt = torch.optim.AdamW([*model.parameters(), *decoder.parameters()], lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epochs)
        history = []
        for epoch in range(1, fixed_epoch+1):
            history.append({"epoch": epoch, **train_epoch(model, decoder, opt, tensors["s1"], tensors["s2"], weights["s1"], weights["s2"], data["positives"], profile, sampler, rng)})
            scheduler.step()
        torch.save({"model": model.state_dict(), "decoder": decoder.state_dict(), "config": config.dictionary(),
                    "seed": seed, "epoch": fixed_epoch, "inputs": tensors["s1"].shape[1]}, folder / "best.pt")
        dump(folder / "history.json", history)
        log(f"новые хосты: обучен seed={seed}, {fixed_epoch} эпох, test не использовался")


def evaluate(work, seeds):
    out = work / "inductive"
    prep = load(out / "preprocessor.pkl")
    holdout = set(json.loads((out / "holdout.json").read_text())["hosts"])
    hosts = pd.read_csv(BOARD / "reference/test/eval_hosts.csv", dtype=str).host_id.tolist()
    query = [h for h in hosts if h in holdout]
    train_stats = load_stats(work / "cache", "train")
    universe_hosts, x, p, count = prep.transform(train_stats)
    # Впервые добавляем отложенные хосты в граф при применении уже обученной модели.
    partners = load_partners(BOARD / "reference/test/partners.csv")
    partners = partners[(partners.window == "source") | partners.host_id.isin(query)]
    baseline = p[universe_hosts.get_indexer(hosts)].copy()
    baseline[np.linalg.norm(baseline, axis=1) == 0, 0] = 1
    baseline_emb = frame(baseline, hosts)
    _, baseline_neighbors = evaluate_frozen_neighbors(baseline_emb, BOARD / "reference/test/neighbors.csv", ks=(10,), return_per_host=True)
    baseline_partner = partner_ranking(baseline_emb, partners, n_boot=200, seed=0)
    dump(out / "profile_baseline.json", {"neighbors@10": float(baseline_neighbors[10].loc[query, "consistency"].mean()),
                                          "new_partners": baseline_partner["new"], "heldout_test_hosts": len(query),
                                          "fit_excludes_all_heldout_flows": True})
    outputs = []
    for seed in seeds:
        folder = out / f"seed{seed}"
        if (folder / "result.json").exists():
            outputs.append(json.loads((folder / "result.json").read_text()))
            continue
        model, _ = frozen_model(folder)
        weights = prepare_graph(count, prep.idf_neighbors(universe_hosts), model.config)
        z, _, _ = infer(model, x, weights)
        embedding = frame(z[universe_hosts.get_indexer(hosts)], hosts)
        _, per = evaluate_frozen_neighbors(embedding, BOARD / "reference/test/neighbors.csv", ks=(10,), return_per_host=True)
        values = per[10].loc[query, "consistency"].to_numpy()
        partner = partner_ranking(embedding, partners, n_boot=200, seed=0)
        r = {"seed": seed, "heldout_train_hosts": len(holdout), "heldout_test_hosts": len(query),
             "neighbors@10": float(values.mean()),
             "neighbors@10_ci": bootstrap_ci(lambda i: float(values[i].mean()), len(values), n_boot=200, seed=0),
             "new_partners": partner["new"], "fit_excludes_all_heldout_flows": True,
             "neighbor_gallery": "all evaluable test hosts; only query hosts are held out",
             "source_profile_window": "full original train; inference after weights frozen"}
        dump(folder / "result.json", r)
        outputs.append(r)
        log(f"новые хосты seed={seed}: neighbors={r['neighbors@10']:.4f}, partners={r['new_partners']['embedding']['mrr']:.4f}")
    dump(out / "results.json", outputs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "train", "evaluate"])
    ap.add_argument("--workdir", type=Path, default=ROOT / "runs/dc_sage")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.stage == "prepare":
        prepare(args.workdir)
    elif args.stage == "train":
        train(args.workdir, args.seeds)
    else:
        evaluate(args.workdir, args.seeds)


if __name__ == "__main__":
    main()
