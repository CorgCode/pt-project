"""Временной графовый энкодер поверх hybrid_tfidf (v5) и сжатие PCA (v6).

    pip install torch
    PT_DATA=/путь/к/lanl/processed python scoreboard/temporal_encoder.py

Части эмбеддинга (каждая нормируется: среднее 0, средняя длина 1, затем умножается на вес):
  behavioral ×2 — профиль трафика (methods.baselines.temporal);
  peers ×1, services ×1 — TF-IDF/SVD 128, как в hybrid_tfidf;
  diffusion ×0.5 — сервисы соседей по графу: исходящие, входящие и через два шага;
  neural ×1 — нейросеть 512→256→128, обученная узнавать один хост в двух половинах train.
Итог 825 чисел сжимается PCA до 384 (v6). Всё обучается только на train.
Рецепт и число эпох выбраны на validation дня 02 (results/fast_temporal_day02/).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import submit  # noqa: E402
from methods.baselines.temporal import BehavioralBaseline  # noqa: E402
from pipeline.data import ProcessedData  # noqa: E402
from pipeline.speed import progress  # noqa: E402

NAME = "temporal_encoder"
RECIPE = {"behavioral": 2.0, "peers": 1.0, "services": 1.0, "diffusion": 0.5, "neural": 1.0}
SVD_DIM, CONTEXT_VOCAB, PCA_DIM = 128, 32768, 384
EPOCHS, BATCH, TEMPERATURE, MIN_FLOWS = 20, 256, 0.15, 20
SEED = 17
STATS = ("bytes", "packets", "resp_bytes", "resp_packets", "duration")
KINDS = ("peers", "services", "context")


def _block(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0)
    return x / max(1e-12, np.sqrt((x ** 2).sum(1).mean()))


def _span(data: ProcessedData, window: str) -> tuple[float, float]:
    w = data.manifest["windows"][window]
    return float(w["start"]), float(w["end"])


def view(flows: pd.DataFrame, hosts: list[str], seconds: float) -> dict:
    """Счётчики токенов (хост, токен, n) трёх видов и 12 агрегатов активности за секунду окна."""
    tokens = {k: [] for k in KINDS}
    stats = np.zeros((len(hosts), 2 * (1 + len(STATS))))
    index = pd.Index(hosts)
    for j, (side, end, peer) in enumerate((("o", "src", "dst"), ("i", "dst", "src"))):
        f = flows.loc[flows[end].isin(index)]
        # один groupby по сочетаниям, строки токенов — только для уникальных сочетаний
        g = f.groupby([end, peer, "proto", "dst_port"], observed=True, sort=False).size().reset_index(name="n")
        h, p = g[end].astype(str), side + ":" + g[peer].astype(str)
        svc = g["proto"].astype(str) + "/" + g["dst_port"].astype(str)
        for kind, t in (("peers", p), ("services", ("u:" if side == "o" else "s:") + svc), ("context", p + "|" + svc)):
            tokens[kind].append(pd.DataFrame({"h": h, "t": t, "n": g["n"]}))
        rows = index.get_indexer(f[end].astype(str))
        off = j * (1 + len(STATS))
        stats[:, off] = np.bincount(rows, minlength=len(hosts))
        for k, col in enumerate(STATS, 1):
            stats[:, off + k] = np.bincount(rows, weights=f[col].to_numpy(float), minlength=len(hosts))
    counts = {k: pd.concat(v).groupby(["h", "t"], sort=False)["n"].sum() for k, v in tokens.items()}
    return {"counts": counts, "activity": np.log1p(stats / seconds), "flows": stats[:, 0] + stats[:, 1 + len(STATS)]}


def vocabulary(counts: pd.Series, top: int | None = None) -> pd.Index:
    """Токены хотя бы у двух хостов (самые частые, если задан top), по алфавиту."""
    df = counts.groupby(level="t").size()
    df = df[df >= 2]
    if top is not None and len(df) > top:
        df = df.sort_index(kind="stable").sort_values(ascending=False, kind="stable").iloc[:top]
    return pd.Index(sorted(df.index))


def matrix(counts: pd.Series, hosts: list[str], vocab: pd.Index, log: bool = True) -> sp.csr_matrix:
    r = pd.Index(hosts).get_indexer(counts.index.get_level_values("h"))
    c = vocab.get_indexer(counts.index.get_level_values("t"))
    keep = (r >= 0) & (c >= 0)
    v = counts.to_numpy(float)[keep]
    return sp.csr_matrix((np.log1p(v) if log else v, (r[keep], c[keep])), shape=(len(hosts), len(vocab)))


def adjacencies(counts: pd.Series, hosts: list[str], vocab: pd.Index) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Граф «кто к кому ходит» внутри когорты (по словарю peers); веса больших хабов ослаблены."""
    x = matrix(counts, hosts, vocab, log=False)
    target = pd.Index(hosts).get_indexer([t[2:] if t.startswith("o:") else "" for t in vocab])
    keep = target >= 0
    x = x[:, keep].tocoo()
    a = sp.csr_matrix((np.log1p(x.data), (x.row, target[keep][x.col])), shape=(len(hosts), len(hosts)))
    a.setdiag(0)
    a.eliminate_zeros()

    def norm(m):
        degree = np.asarray(m.sum(0)).ravel()
        return normalize(m @ sp.diags(1 / np.sqrt(np.maximum(degree, 1))), norm="l1", axis=1)
    return norm(a), norm(a.T.tocsr())


def _encoder(inputs: int):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(inputs, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 128))
            self.shortcut = nn.Linear(inputs, 128, bias=False)

        def forward(self, x):
            return F.normalize(self.net(x) + 0.2 * self.shortcut(x), dim=1)
    torch.manual_seed(SEED)
    return Encoder()


class TemporalEncoder:
    """fit(data) учится только на train; embed(data, window, hosts) — как у бейзлайнов."""
    name = NAME
    shared_coordinate_system = True

    def __init__(self, recipe: dict | None = None, epochs: int = EPOCHS, pca_dim: int | None = PCA_DIM):
        self.recipe, self.epochs, self.pca_dim = dict(recipe or RECIPE), epochs, pca_dim

    def _features(self, v: dict) -> dict:
        """Признаки окна в координатах train; input — вход нейросети после стандартизации train."""
        z = {}
        for kind in KINDS:
            tfidf, svd = self.models[kind]
            z[kind] = svd.transform(normalize(tfidf.transform(matrix(v["counts"][kind], self.hosts, self.vocab[kind]))))
        out, inn = adjacencies(v["counts"]["peers"], self.hosts, self.vocab["peers"])
        o, i = out @ z["services"], inn @ z["services"]
        z["diffusion"] = np.hstack([o, i, 0.5 * (out @ i + inn @ o)])
        z["raw"] = np.hstack([z[k] for k in (*KINDS, "diffusion")] + [v["activity"]])
        if hasattr(self, "scaler"):
            z["input"] = np.clip(self.scaler.transform(z["raw"]), -8, 8).astype(np.float32)
        return z

    def _neural(self, x: np.ndarray) -> np.ndarray:
        import torch
        self.net.eval()
        with torch.no_grad():
            return np.concatenate([self.net(torch.from_numpy(x[i:i + 512])).numpy() for i in range(0, len(x), 512)])

    def fit(self, data: ProcessedData) -> "TemporalEncoder":
        import torch
        from torch.nn import functional as F
        torch.set_num_threads(2)                         # как в исходном эксперименте: числа воспроизводятся
        self.hosts = list(data.cohort)
        self.beh = BehavioralBaseline().fit(data)
        flows = data.training_flows()
        start, end = _span(data, "train")
        mid = (start + end) / 2                          # один хост в двух половинах train — положительная пара
        full = view(flows, self.hosts, end - start)
        halves = [view(flows.loc[m], self.hosts, (end - start) / 2) for m in (flows.ts < mid, flows.ts >= mid)]
        self.vocab = {k: vocabulary(full["counts"][k], CONTEXT_VOCAB if k == "context" else None) for k in KINDS}
        self.models = {}
        for kind in KINDS:
            x = matrix(full["counts"][kind], self.hosts, self.vocab[kind])
            tfidf = TfidfTransformer().fit(x)
            dim = min(SVD_DIM, x.shape[0] - 1, x.shape[1] - 1)
            seed = SEED if kind == "context" else 0      # peers и services — те же, что в hybrid_tfidf
            self.models[kind] = tfidf, TruncatedSVD(dim, random_state=seed).fit(normalize(tfidf.transform(x)))
        self._train = self._features(full)
        self.scaler = StandardScaler().fit(self._train["raw"])
        self._train["input"] = np.clip(self.scaler.transform(self._train["raw"]), -8, 8).astype(np.float32)
        a, b = (torch.from_numpy(self._features(h)["input"]) for h in halves)
        active = np.flatnonzero((halves[0]["flows"] >= MIN_FLOWS) & (halves[1]["flows"] >= MIN_FLOWS))

        self.net = _encoder(a.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        behavior = normalize(self.beh.embed(data, "train", self.hosts).to_numpy()).astype(np.float32)
        services = normalize(self._train["services"]).astype(np.float32)
        teacher = torch.from_numpy(np.hstack([behavior, services]) / np.float32(np.sqrt(2)))
        rng = np.random.default_rng(SEED)
        for _ in progress(range(self.epochs), desc=f"{NAME}: эпохи", leave=False):
            self.net.train()
            order = rng.permutation(active)
            for off in range(0, len(order), BATCH):
                ix = torch.from_numpy(order[off:off + BATCH])
                if len(ix) < 2:
                    continue
                za, zb = self.net(a[ix]), self.net(b[ix])
                logits, labels = za @ zb.T / TEMPERATURE, torch.arange(len(ix))
                contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
                target = teacher[ix] @ teacher[ix].T     # сохранить сходство по поведению и сервисам
                relational = 0.5 * (F.mse_loss(za @ za.T, target) + F.mse_loss(zb @ zb.T, target))
                loss = contrastive + 0.5 * relational
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()

        self.pca = None
        if self.pca_dim:
            x = self._concat(data, "train", self.hosts)
            n = min(self.pca_dim, *x.shape)
            self.pca = PCA(n_components=n, svd_solver="randomized", random_state=29).fit(x)
        return self

    def _concat(self, data: ProcessedData, window: str, hosts: list[str]) -> np.ndarray:
        if window == "train":
            z = self._train
        else:
            start, end = _span(data, window)
            z = self._features(view(data.flows(window), self.hosts, end - start))
        rows = pd.Index(self.hosts).get_indexer(hosts)
        parts = {"behavioral": self.beh.embed(data, window, hosts).to_numpy(),
                 "neural": self._neural(z["input"])[rows]}
        parts.update({k: z[k][rows] for k in ("peers", "services", "context", "diffusion")})
        return np.hstack([_block(parts[k]) * w for k, w in self.recipe.items()])

    def embed(self, data: ProcessedData, window: str, hosts: list[str] | None = None) -> pd.DataFrame:
        hosts = list(hosts) if hosts is not None else list(data.cohort)
        x = self._concat(data, window, hosts)
        if self.pca is not None:
            x = self.pca.transform(x)
        return pd.DataFrame(x, index=pd.Index(hosts, name="host_id")).add_prefix("emb_")


def _write(model: TemporalEncoder, data: ProcessedData, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    jobs = [("train", data.cohort, "embeddings.csv")] + [(w, data.evaluable(w), submit.window_file(w))
                                                          for w in data.test_windows]
    if "validation" in data.windows:                  # для подбора гибридов; в git не кладётся
        jobs.append(("validation", data.evaluable("validation"), "validation.csv"))
    for window, hosts, fname in jobs:
        model.embed(data, window, hosts).to_csv(out / fname, float_format="%.5g")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pca", action="store_true", help="без сжатия (v5, 825 чисел)")
    a = ap.parse_args()
    data = ProcessedData(os.environ.get("PT_DATA", ROOT / "scoreboard" / "data" / "processed"))
    model = TemporalEncoder(pca_dim=None if a.no_pca else PCA_DIM).fit(data)
    out = ROOT / "scoreboard" / "submissions" / NAME
    _write(model, data, out)
    (out / "SHARED_COORDINATES").touch()
    (out / "AUTHOR").write_text("physcorgi\n", encoding="utf-8")
    (out / "params.json").write_text(json.dumps({"recipe": model.recipe, "epochs": model.epochs, "pca_dim": model.pca_dim,
                                                 "tuned_on": "validation, день 02"}, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
    r = submit.score(out)
    submit.render()
    print(f"\n{NAME}: " + ", ".join(f"{s['title']} = {s['value']:.3f}" for s in r["summary"] if s["value"] is not None))


if __name__ == "__main__":
    main()
