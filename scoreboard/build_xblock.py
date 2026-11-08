"""Дополнительный блок для склейки: «отпечаток» (случайная проекция полной TF-IDF) и «диффузия» (сервисы соседей).

    python scoreboard/build_xblock.py --name xblock_fp_diff                # оба вида, все окна
    python scoreboard/build_xblock.py --kind fp --name xblock_fp           # только отпечаток
    python scoreboard/build_xblock.py --kind diff --name xblock_diff       # только диффузия

Результат: runs/submissions/<name>/ в формате отправки (embeddings.csv и <окно>.csv). Его принимает склейка как готовый блок:
    python scoreboard/gnn_combine_v3.py --gnn runs/submissions/<name> ...   (вместо gnn_big; вес подбирает Optuna)

Что в блоке (каждая часть приведена к единичной длине строки, части равноправны):
  fp   peers_fp, services_fp — полная разреженная TF-IDF «хост × собеседник» и «хост × сервис» (тот же словарь, IDF, нормировка,
       что в build_blocks.py, всё по train), спроецированная случайной матрицей на --m-peers / --m-services чисел. Случайная
       проекция сохраняет косинусное сходство строк, но, в отличие от SVD, не отбрасывает редкие токены: нужна для Identity.
  diff diff_out, diff_in — какими сервисами пользуются те, к кому хост ходит (out) и кто ходит к нему (in): строки TF-IDF сервисов
       соседей усредняются по рёбрам (вес log1p числа потоков, нормировка по строке), затем проекция на --m-diff чисел.
       Соседи считаются по всем хостам графа окна (и вне когорты). Это аналог «диффузии» у temporal_encoder.
Вход: runs/features/<окно>/{nodes,edges,services}.csv, runs/processed/cohort.csv, код TF-IDF из build_blocks.py. Обучения нет.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_blocks import TfidfSvd, peer_triplets, svc_triplets  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def l1_rows(a: sp.csr_matrix) -> sp.csr_matrix:
    s = np.asarray(a.sum(1)).ravel()
    s[s == 0] = 1.0
    return (sp.diags(1.0 / s) @ a).tocsr()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="xblock")
    ap.add_argument("--kind", choices=["fp", "diff", "both"], default="both")
    ap.add_argument("--m-peers", dest="m_peers", type=int, default=512)
    ap.add_argument("--m-services", dest="m_services", type=int, default=256)
    ap.add_argument("--m-diff", dest="m_diff", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", default=str(ROOT / "runs" / "features"))
    ap.add_argument("--processed", default=str(ROOT / "runs" / "processed"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "submissions"))
    a = ap.parse_args()
    t0 = time.time()
    feat, out = Path(a.features), Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(Path(a.processed) / "cohort.csv", dtype={"host_id": str}).set_index("host_id")
    hosts = list(cohort.index)
    hidx = {h: i for i, h in enumerate(hosts)}
    windows = ["train"] + sorted(d.name for d in feat.iterdir() if d.is_dir() and d.name != "train" and (d / "nodes.csv").exists())
    rng = np.random.default_rng(a.seed)
    use_fp, use_diff = a.kind in ("fp", "both"), a.kind in ("diff", "both")
    models, proj, dims = {}, {}, {}

    for w in windows:
        log(f"== окно {w} ({time.time() - t0:.0f} с)")
        edges = pd.read_csv(feat / w / "edges.csv", dtype={"a": str, "b": str})
        svc = pd.read_csv(feat / w / "services.csv", dtype={"host": str})
        pt, st = peer_triplets(edges), svc_triplets(svc)
        if w == "train":                                   # словарь и IDF — только по train (SVD нам не нужен: k=1)
            models["peers"], models["services"] = TfidfSvd(1), TfidfSvd(1)
            models["peers"].fit(pt, hidx)
            models["services"].fit(st, hidx)
            vp, vs = len(models["peers"].vocab), len(models["services"].vocab)
            proj["peers"] = (rng.standard_normal((vp, a.m_peers)) / np.sqrt(a.m_peers)).astype(np.float32)
            proj["services"] = (rng.standard_normal((vs, a.m_services)) / np.sqrt(a.m_services)).astype(np.float32)
            proj["diff"] = (rng.standard_normal((vs, a.m_diff)) / np.sqrt(a.m_diff)).astype(np.float32)
            log(f"  словарь: peers {vp:,}, services {vs:,}")
        parts = {}
        if use_fp:
            xp = models["peers"]._weight(models["peers"]._matrix(pt, hidx))
            xs = models["services"]._weight(models["services"]._matrix(st, hidx))
            parts["peers_fp"] = unit(np.asarray(xp @ proj["peers"]))
            parts["services_fp"] = unit(np.asarray(xs @ proj["services"]))
        if use_diff:
            nodes = pd.read_csv(feat / w / "nodes.csv", dtype={"host_id": str}).set_index("host_id")
            hall = {h: i for i, h in enumerate(nodes.index)}
            n = len(hall)
            s_all = models["services"]._weight(models["services"]._matrix(st, hall))        # сервисы всех хостов графа окна
            e = edges[edges["a"].isin(hall) & edges["b"].isin(hall)]
            ia, ib = e["a"].map(hall).to_numpy(), e["b"].map(hall).to_numpy()
            fl, ab = e["flows"].to_numpy(float), e["flows_ab"].to_numpy(float)
            a_out = sp.csr_matrix((np.log1p(np.concatenate([ab, fl - ab])), (np.concatenate([ia, ib]), np.concatenate([ib, ia]))),
                                  shape=(n, n))
            a_out.eliminate_zeros()
            rows = np.array([hall[h] for h in hosts])
            for key, mat in (("diff_out", a_out), ("diff_in", a_out.T.tocsr())):
                d = (l1_rows(mat) @ s_all)[rows]
                parts[key] = unit(np.asarray(d @ proj["diff"]))
        z = np.hstack(list(parts.values()))
        dims = {k: v.shape[1] for k, v in parts.items()}
        ev = hosts if w == "train" else [h for h in hosts if bool(cohort.loc[h, f"evaluable_{w}"])]
        idx = np.array([hidx[h] for h in ev])
        df = pd.DataFrame(z[idx], index=pd.Index(ev, name="host_id")).add_prefix("emb_")
        df.to_csv(out / ("embeddings.csv" if w == "train" else f"{w}.csv"), float_format="%.5g")
        log(f"  записано: {len(ev):,} хостов, {z.shape[1]} чисел")

    bar = "=" * 14
    print(f"\n{bar} КРАТКИЙ ИТОГ {bar}")
    print(f"Блок {a.name}: части {dims}; всего {sum(dims.values())} чисел; окна {', '.join(windows)}")
    print(f"Папка: {out}; всего {time.time() - t0:.0f} с")
    print("=" * 83)


if __name__ == "__main__":
    main()
