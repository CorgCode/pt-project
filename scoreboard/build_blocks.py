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

    def fit(self, trip: pd.DataFrame, hidx: dict, extra: tuple = ()) -> np.ndarray:
        """Словарь, IDF и SVD по train и (необязательно) дополнительным окнам extra: строки "хост × окно" ставятся друг под другом.
        Вектор хоста всё равно считается по одному окну (transform); дополнительные окна нужны только для осей."""
        trips = [trip, *extra]
        dfc = pd.concat([t[t["host"].isin(hidx)].drop_duplicates(["host", "col"]) for t in trips]).groupby("col").size()
        cols = sorted(dfc[dfc >= self.min_df].index)         # токен хотя бы в min_df строках "хост × окно"
        self.vocab = {c: i for i, c in enumerate(cols)}
        x = sp.vstack([self._matrix(t, hidx) for t in trips]).tocsr()
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

    def project(self, trip: pd.DataFrame, hidx: dict, r: np.ndarray) -> np.ndarray:
        """«Отпечаток»: полная (без SVD) взвешенная TF-IDF-строка хоста, умноженная на случайную гауссову матрицу r (словарь × k)."""
        return np.asarray(self._weight(self._matrix(trip, hidx)) @ r)


def rp_matrix(n: int, k: int, seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).normal(size=(n, k)) / np.sqrt(k)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "runs" / "features"))
    ap.add_argument("--processed", default=str(ROOT / "runs" / "processed"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "blocks"))
    ap.add_argument("--k-peers", type=int, default=256, help="максимальная длина SVD для peers")
    ap.add_argument("--k-services", type=int, default=128, help="максимальная длина SVD для services")
    ap.add_argument("--extra-fit", nargs="*", default=[], metavar="DIR",
                    help="папки окон (runs/features_multi/test_day03 ...) только для обучения словаря, IDF и SVD вместе с train; "
                         "блоки для них не пишутся")
    ap.add_argument("--rp-peers", type=int, default=0, help="длина случайной проекции полной TF-IDF собеседников (0 — не считать); поле peers_rp в блоках")
    ap.add_argument("--rp-services", type=int, default=0, help="то же для сервисов; поле services_rp")
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
    extra_p, extra_s = [], []
    for d in map(Path, a.extra_fit):
        if not (d / "edges.csv").exists():
            raise SystemExit(f"нет {d}/edges.csv — это окно не посчитано build_features.py")
        extra_p.append(peer_triplets(pd.read_csv(d / "edges.csv", dtype={"a": str, "b": str})))
        extra_s.append(svc_triplets(pd.read_csv(d / "services.csv", dtype={"host": str})))
        log(f"доп. окно для обучения осей: {d}")
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
            pz = peers_m.fit(pt, hidx, tuple(extra_p))
            log("  обучаю TF-IDF + SVD на train (services) ...")
            sz = svc_m.fit(st, hidx, tuple(extra_s))
            rp_p = rp_matrix(len(peers_m.vocab), a.rp_peers, 0) if a.rp_peers > 0 else None
            rp_s = rp_matrix(len(svc_m.vocab), a.rp_services, 1) if a.rp_services > 0 else None
            log(f"  peers: словарь {len(peers_m.vocab):,}, компонент {pz.shape[1]}; services: словарь {len(svc_m.vocab):,}, компонент {sz.shape[1]}")
        else:
            if cols != stat["behav_cols"]:
                raise SystemExit(f"колонки профиля в {w} не совпадают с train")
            pz, sz = peers_m.transform(pt, hidx), svc_m.transform(st, hidx)
        bz = np.clip(np.nan_to_num((x - stat["mean"]) / stat["std"]), -5, 5)
        extra_arrays = {}
        if rp_p is not None:
            extra_arrays["peers_rp"] = peers_m.project(pt, hidx, rp_p).astype(np.float32)
        if rp_s is not None:
            extra_arrays["services_rp"] = svc_m.project(st, hidx, rp_s).astype(np.float32)
        np.savez_compressed(out / f"{w}.npz", hosts=np.array(hosts, dtype=str), behav=bz.astype(np.float32),
                            peers=pz.astype(np.float32), services=sz.astype(np.float32), **extra_arrays)
        log(f"  записано {out / (w + '.npz')} ({time.time() - t0:.0f} с)")
    (out / "fit.json").write_text(json.dumps({
        "hosts": len(hosts), "extra_fit": list(a.extra_fit), "behav_dim": len(stat["behav_cols"]),
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
