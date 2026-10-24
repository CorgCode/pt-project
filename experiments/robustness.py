"""Устойчивость эмбеддинга к удалению связей (идея 2 из #40, автор — Тихон).

    python -m experiments.robustness --processed-dir data/processed --out-dir runs/robustness \
        [--methods deepwalk node2vec] [--fractions 0.1 0.2 0.3] [--fast-methods]

Если эмбеддинг отражает поведение хоста, то после удаления части связей хост должен
остаться примерно там же. Для каждого метода:
  1. обучаем на полном train-окне — эталон;
  2. удаляем долю f случайных пар хостов (все потоки между ними) и обучаем заново;
  3. сравниваем с эталоном:
       overlap@k  — доля общих k ближайших соседей хоста (выравнивание не нужно);
       procrustes — средний косинус после поворота (нужна одна размерность).
Фоновый шум — эталон, обученный с другим seed: ниже него «устойчивость» не опустится
даже без удаления. Пишет robustness.json и robustness.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

from methods import graph as graph_methods
from methods.baselines.temporal import default_baselines
from pipeline.data import ProcessedData

ROOT = Path(__file__).resolve().parents[1]


def knn_overlap(a: pd.DataFrame, b: pd.DataFrame, k: int = 10) -> float:
    """Средняя доля общих k ближайших соседей (по косинусу) в двух эмбеддингах одних хостов."""
    hosts = sorted(set(a.index) & set(b.index))
    k = min(k, len(hosts) - 1)

    def top(x):
        x = x.loc[hosts].to_numpy(float)
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        s = x @ x.T
        np.fill_diagonal(s, -np.inf)
        return np.argpartition(-s, k - 1, axis=1)[:, :k]

    ta, tb = top(a), top(b)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ta, tb)]))


def procrustes_cosine(a: pd.DataFrame, b: pd.DataFrame) -> float | None:
    """Средний косинус строк после поворота b к a. None, если размерности разные."""
    hosts = sorted(set(a.index) & set(b.index))
    A, B = a.loc[hosts].to_numpy(float), b.loc[hosts].to_numpy(float)
    if A.shape[1] != B.shape[1]:
        return None
    A, B = A - A.mean(0), B - B.mean(0)
    R, _ = orthogonal_procrustes(B, A)
    B = B @ R
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-12)))


class ReducedData(ProcessedData):
    """Тот же датасет, но в train-окне удалена доля пар хостов."""

    def __init__(self, base: ProcessedData, fraction: float, seed: int):
        self.__dict__.update(base.__dict__)
        flows = base.training_flows()
        a, b = flows.src.astype(str), flows.dst.astype(str)
        pair = np.where(a < b, a + "\t" + b, b + "\t" + a)
        uniq = np.unique(pair)
        drop = np.random.default_rng(seed).choice(len(uniq), size=int(round(fraction * len(uniq))), replace=False)
        self._train = flows[~np.isin(pair, uniq[drop])].reset_index(drop=True)
        self.n_pairs, self.n_dropped = len(uniq), len(drop)

    def training_flows(self) -> pd.DataFrame:
        return self._train.copy()

    def flows(self, window: str) -> pd.DataFrame:
        return self.training_flows() if window == "train" else super().flows(window)


def _make(name: str, dim: int, seed: int, fast: bool):
    if name == "hybrid_tfidf":                             # лидер таблицы; размерность у него своя (313)
        import importlib.util
        spec = importlib.util.spec_from_file_location("hybrid_tfidf", ROOT / "scoreboard" / "hybrid_tfidf.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.HybridTfidf(seed)
    base = {b.name: b for b in default_baselines(dim, seed)}
    return base[name] if name in base else graph_methods.create([name], dim, seed, fast=fast)[0]


def robustness(processed_dir: Path, out_dir: Path, methods: list[str], *, fractions=(0.1, 0.2, 0.3),
               dim: int = 16, seed: int = 0, k: int = 10, fast: bool = False) -> dict:
    data = ProcessedData(processed_dir)
    reduced = {f: ReducedData(data, f, seed) for f in fractions}
    out = {"fractions": list(fractions), "k": k, "dim": dim, "seed": seed, "data_kind": data.kind, "methods": {}}
    for name in methods:
        ref = _make(name, dim, seed, fast).fit(data).embed(data, "train")
        noise = _make(name, dim, seed + 1, fast).fit(data).embed(data, "train")
        row = {"noise_floor": {"overlap": knn_overlap(ref, noise, k), "procrustes": procrustes_cosine(ref, noise)},
               "by_fraction": {}}
        for f, d in reduced.items():
            e = _make(name, dim, seed, fast).fit(d).embed(d, "train")
            row["by_fraction"][str(f)] = {"overlap": knn_overlap(ref, e, k), "procrustes": procrustes_cosine(ref, e),
                                          "pairs_dropped": d.n_dropped, "pairs_total": d.n_pairs}
        out["methods"][name] = row
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "robustness.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "robustness.md").write_text(report(out), encoding="utf-8")
    return out


def report(r: dict) -> str:
    fr = r["fractions"]
    L = [f"# Устойчивость к удалению связей (данные `{r['data_kind']}`)\n",
         f"Доля общих {r['k']} ближайших соседей с эмбеддингом на полном графе. "
         "«Шум» — тот же метод с другим seed без удаления: это потолок, выше которого ждать нечего. "
         "Метод, не смотрящий на данные (random), «устойчив» тривиально — это не достоинство.\n",
         "| метод | шум | " + " | ".join(f"−{f:.0%} пар" for f in fr) + " |",
         "|---|---|" + "---|" * len(fr)]
    for m, v in r["methods"].items():
        cells = [f"{v['by_fraction'][str(f)]['overlap']:.3f}" for f in fr]
        L.append(f"| `{m}` | {v['noise_floor']['overlap']:.3f} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", default=["behavioral", "graph_spectral"],
                    help="бейзлайны, hybrid_tfidf и/или графовые методы: " + ", ".join(sorted(graph_methods.available())))
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--fast-methods", action="store_true", help="крошечные гиперпараметры (только для проверки)")
    a = ap.parse_args()
    robustness(a.processed_dir, a.out_dir, a.methods, fractions=tuple(a.fractions), dim=a.dim, seed=a.seed,
               k=a.k, fast=a.fast_methods)
    print((a.out_dir / "robustness.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
