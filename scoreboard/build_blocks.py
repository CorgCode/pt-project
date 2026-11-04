"""Блоки признаков хоста для склейки: поведение, TF-IDF "с кем общается", TF-IDF "какими сервисами пользуется".

    python scoreboard/build_blocks.py

Вход:  runs/features/<окно>/{nodes,edges,services}.csv (их делает build_features.py), runs/processed/cohort.csv
Выход: runs/blocks/<окно>.npz  (hosts, behav, peers, services — по строке на хост когорты, порядок как в cohort.csv)
       runs/blocks/fit.json     (что обучилось на train: размеры словарей, число компонент, сингулярные числа)

Что в блоках:
  behav    66 признаков профиля хоста, стандартизованы по train, обрезаны до ±5.
  peers    таблица "хост × собеседник" с направлением (o:X — хост ходит к X, i:X — X ходит к хосту).
  services таблица "хост × сервис" (u:proto/port — пользуется, s:proto/port — сам обслуживает; port=anon — порт анонимный).
  Для peers и services: вес = log(1+число потоков) -> умножение на IDF (то, что есть почти у всех хостов, весит меньше
  редкого) -> нормировка строки -> SVD. Словарь, IDF и SVD обучаются на train и применяются к любому окну
  (одни и те же оси координат). Партнёры/сервисы, которых не было в train, в других окнах отбрасываются.
  Компоненты SVD упорядочены по убыванию сингулярного числа, поэтому первые k столбцов = SVD длины k
  (длину потом подбирает gnn_combine_v3.py).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(msg, flush=True)


def peer_triplets(e: pd.DataFrame) -> pd.DataFrame:
    """edges.csv (a<b; flows_ab — потоки a->b) -> строки (host, col, n)."""
    fl, ab = e["flows"].to_numpy(float), e["flows_ab"].to_numpy(float)
    ba = fl - ab
    parts = [(e["a"], "o:", e["b"], ab), (e["a"], "i:", e["b"], ba), (e["b"], "o:", e["a"], ba), (e["b"], "i:", e["a"], ab)]
    out = []
    for host, pre, peer, n in parts:
        d = pd.DataFrame({"host": host.to_numpy(), "col": (pre + peer.astype(str)).to_numpy(), "n": n})
        out.append(d[d["n"] > 0])
    return pd.concat(out, ignore_index=True)


def svc_triplets(s: pd.DataFrame) -> pd.DataFrame:
    port = s["port"].fillna(-1).astype(int)
    plabel = port.astype(str).where(port != -1, "anon")
    col = s["role"].map({0: "u:", 1: "s:"}) + s["proto"].fillna(-1).astype(int).astype(str) + "/" + plabel
    d = pd.DataFrame({"host": s["host"].to_numpy(), "col": col.to_numpy(), "n": s["flows"].to_numpy(float)})
    return d.groupby(["host", "col"], as_index=False)["n"].sum()


def row_norm(x: sp.csr_matrix) -> sp.csr_matrix:
    n = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    n[n == 0] = 1.0
    return (sp.diags(1.0 / n) @ x).tocsr()


class TfidfSvd:
    """Словарь + IDF + SVD, обученные на train; transform() — для любого окна."""

    def __init__(self, kmax: int, min_df: int = 2):
        self.kmax, self.min_df = kmax, min_df

    def _matrix(self, trip: pd.DataFrame, hidx: dict) -> sp.csr_matrix:
        t = trip[trip["host"].isin(hidx)]
        r = t["host"].map(hidx).to_numpy()
        c = t["col"].map(self.vocab)
        ok = c.notna().to_numpy()
        return sp.csr_matrix((np.log1p(t["n"].to_numpy(float)[ok]), (r[ok], c.to_numpy()[ok].astype(int))),
                             shape=(len(hidx), len(self.vocab)))

    def _weight(self, x: sp.csr_matrix) -> sp.csr_matrix:
        return row_norm((x @ sp.diags(self.idf)).tocsr())

    def fit(self, trip: pd.DataFrame, hidx: dict) -> np.ndarray:
        t = trip[trip["host"].isin(hidx)].drop_duplicates(["host", "col"])
        dfc = t.groupby("col").size()
        cols = sorted(dfc[dfc >= self.min_df].index)         # токен хотя бы у min_df хостов
        self.vocab = {c: i for i, c in enumerate(cols)}
        x = self._matrix(trip, hidx)
        df = np.asarray((x > 0).sum(0)).ravel()
        self.idf = np.log((1 + x.shape[0]) / (1 + df)) + 1.0
        xw = self._weight(x).astype(np.float64)
        k = min(self.kmax, min(xw.shape) - 1)
        _, s, vt = svds(xw, k=k)
        o = np.argsort(-s)
        self.s, self.vt = s[o], vt[o]
        return self.transform(trip, hidx)

    def transform(self, trip: pd.DataFrame, hidx: dict) -> np.ndarray:
        return np.asarray(self._weight(self._matrix(trip, hidx)) @ self.vt.T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "runs" / "features"))
    ap.add_argument("--processed", default=str(ROOT / "runs" / "processed"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "blocks"))
    ap.add_argument("--k-peers", type=int, default=256, help="максимальная длина SVD для peers")
    ap.add_argument("--k-services", type=int, default=128, help="максимальная длина SVD для services")
    a = ap.parse_args()
    t0 = time.time()
    feat, out = Path(a.features), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(Path(a.processed) / "cohort.csv", dtype={"host_id": str}).set_index("host_id")
    hosts = list(cohort.index)
    hidx = {h: i for i, h in enumerate(hosts)}
    windows = sorted(d.name for d in feat.iterdir() if d.is_dir() and (d / "nodes.csv").exists())
    if "train" not in windows:
        raise SystemExit(f"нет {feat}/train — сначала запустите build_features.py")
    windows = ["train"] + [w for w in windows if w != "train"]
    log(f"когорта: {len(hosts):,} хостов; окна: {windows}")

    peers_m, svc_m = TfidfSvd(a.k_peers), TfidfSvd(a.k_services)
    stat: dict = {}
    for i, w in enumerate(windows, 1):
        log(f"[{i}/{len(windows)}] окно {w}")
        nodes = pd.read_csv(feat / w / "nodes.csv", dtype={"host_id": str}).set_index("host_id")
        edges = pd.read_csv(feat / w / "edges.csv", dtype={"a": str, "b": str})
        svc = pd.read_csv(feat / w / "services.csv", dtype={"host": str})
        cols = [c for c in nodes.columns if c != "in_cohort"]
        x = nodes[cols].reindex(hosts).fillna(0.0).to_numpy(float)
        pt, st = peer_triplets(edges), svc_triplets(svc)
        if w == "train":
            stat["behav_cols"] = cols
            stat["mean"], stat["std"] = x.mean(0), x.std(0) + 1e-6
            log("  обучаю TF-IDF + SVD на train (peers) ...")
            pz = peers_m.fit(pt, hidx)
            log("  обучаю TF-IDF + SVD на train (services) ...")
            sz = svc_m.fit(st, hidx)
            log(f"  peers: словарь {len(peers_m.vocab):,}, компонент {pz.shape[1]}; services: словарь {len(svc_m.vocab):,}, компонент {sz.shape[1]}")
        else:
            if cols != stat["behav_cols"]:
                raise SystemExit(f"колонки профиля в {w} не совпадают с train")
            pz, sz = peers_m.transform(pt, hidx), svc_m.transform(st, hidx)
        bz = np.clip(np.nan_to_num((x - stat["mean"]) / stat["std"]), -5, 5)
        np.savez_compressed(out / f"{w}.npz", hosts=np.array(hosts, dtype=str), behav=bz.astype(np.float32),
                            peers=pz.astype(np.float32), services=sz.astype(np.float32))
        log(f"  записано {out / (w + '.npz')} ({time.time() - t0:.0f} с)")
    (out / "fit.json").write_text(json.dumps({
        "hosts": len(hosts), "behav_dim": len(stat["behav_cols"]),
        "peers": {"vocab": len(peers_m.vocab), "k": int(len(peers_m.s)), "top_sv": [round(float(v), 3) for v in peers_m.s[:8]]},
        "services": {"vocab": len(svc_m.vocab), "k": int(len(svc_m.s)), "top_sv": [round(float(v), 3) for v in svc_m.s[:8]]},
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    bar = "=" * 14
    print(f"\n{bar} КРАТКИЙ ИТОГ {bar}")
    print(f"Хостов в когорте: {len(hosts):,}; окна: {', '.join(windows)}")
    print(f"peers: словарь {len(peers_m.vocab):,}, SVD {len(peers_m.s)}; services: словарь {len(svc_m.vocab):,}, SVD {len(svc_m.s)}")
    print(f"Файлы: {out}/<окно>.npz, fit.json; всего {time.time() - t0:.0f} с")
    print("=" * 83)


if __name__ == "__main__":
    main()
