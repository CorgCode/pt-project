"""Временной графовый энкодер поверх hybrid_tfidf (v5) и сжатие PCA (v6).

    pip install torch
    PT_DATA=/путь/к/lanl/processed python scoreboard/temporal_encoder.py          # рецепт дня 02
    PT_DATA=/путь/к/lanl/processed python scoreboard/temporal_encoder.py --tune   # подобрать на validation → temporal_encoder_tuned

Части эмбеддинга (каждая нормируется: среднее 0, средняя длина 1, затем умножается на вес):
  behavioral ×2 — профиль трафика (methods.baselines.temporal);
  peers ×1, services ×1 — TF-IDF/SVD 128, как в hybrid_tfidf;
  diffusion ×0.5 — сервисы соседей по графу: исходящие, входящие и через два шага;
  neural ×1 — нейросеть 512→256→128, обученная узнавать один хост в двух половинах train.
Итог 825 чисел сжимается PCA до 384 (v6). Всё обучается только на train.
Рецепт и число эпох выбраны на validation дня 02 (results/fast_temporal_day02/).
--encoder cca: вместо нейросети — линейная проекция (CCA): оси, по которым хост одинаков в разных
кусках train. Без torch, обучается за секунды → temporal_cca.
--tune: train делится на 2 или 4 куска (больше пар «тот же хост в другое время»), веса частей
перебираются по сетке GRID; выбор — по баллу на validation со всеми четырьмя задачами, test не смотрим.
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
CCA_DIM, CCA_REG = 128, 0.1
GRID = {"splits": (2, 4), "behavioral": (2.0, 3.0, 4.0), "neural": (0.5, 1.0, 2.0), "diffusion": (0.0, 0.5)}
STATS = ("bytes", "packets", "resp_bytes", "resp_packets", "duration")
KINDS = ("peers", "services", "context")


def _block(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0)
    return x / max(1e-12, np.sqrt((x ** 2).sum(1).mean()))


def _span(data: ProcessedData, window: str) -> tuple[float, float]:
    w = data.manifest["windows"][window]
    return float(w["start"]), float(w["end"])


def _positions(col: pd.Series, index: pd.Index) -> np.ndarray:
    """Номер хоста в index для каждой строки; категории (PT_COMPACT=1) — без перевода всех строк в str."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        return index.get_indexer(col.cat.categories.astype(str))[col.cat.codes.to_numpy()]
    return index.get_indexer(col.astype(str))


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
        rows = _positions(f[end], index)
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

    def __init__(self, recipe: dict | None = None, epochs: int = EPOCHS, pca_dim: int | None = PCA_DIM,
                 splits: int = 2, encoder: str = "mlp"):
        if encoder not in ("mlp", "cca"):
            raise ValueError(f"encoder: mlp или cca, а не {encoder!r}")
        self.recipe, self.epochs, self.pca_dim, self.splits = dict(recipe or RECIPE), epochs, pca_dim, splits
        self.encoder = encoder
        self._parts_cache: dict = {}

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
        if self.encoder == "cca":
            return normalize((x @ self.proj) * self.corr)
        import torch
        self.net.eval()
        with torch.no_grad():
            return np.concatenate([self.net(torch.from_numpy(x[i:i + 512])).numpy() for i in range(0, len(x), 512)])

    def fit(self, data: ProcessedData) -> "TemporalEncoder":
        self.hosts = list(data.cohort)
        self.beh = BehavioralBaseline().fit(data)
        flows = data.training_flows()
        start, end = _span(data, "train")
        # train режется на куски по времени; один хост в двух кусках — положительная пара
        edges = np.linspace(start, end, self.splits + 1)
        edges[-1] = np.inf
        full = view(flows, self.hosts, end - start)
        slices = [view(flows.loc[(flows.ts >= a) & (flows.ts < b)], self.hosts, (end - start) / self.splits)
                  for a, b in zip(edges[:-1], edges[1:])]
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
        xs = [self._features(v)["input"] for v in slices]
        ok = np.stack([v["flows"] >= MIN_FLOWS * 2 / self.splits for v in slices])   # хост активен в куске
        empty = [i for i, o in enumerate(ok) if not o.any()]
        if empty:                                     # например, окно в манифесте шире, чем данные в нём
            raise ValueError(f"{NAME}: в кусках train {empty} нет активных хостов — проверьте окно train в манифесте")
        active = np.flatnonzero(ok.all(0) if self.splits == 2 else ok.sum(0) >= 2)
        if self.encoder == "cca":
            self._fit_cca(xs, ok)
        else:
            self._fit_mlp(data, xs, ok, active)
        self._parts_cache = {}
        self.fit_pca(data)
        return self

    def _fit_cca(self, xs: list[np.ndarray], ok: np.ndarray) -> None:
        """Линейная замена нейросети: направления, в которых хост одинаков в разных кусках train (CCA).
        Пары кусков берутся в обе стороны, поэтому проекция одна; канонические корреляции — веса осей."""
        a, b = [], []
        for i in range(self.splits):
            for j in range(i + 1, self.splits):
                both = ok[i] & ok[j]
                a += [xs[i][both], xs[j][both]]
                b += [xs[j][both], xs[i][both]]
        A, B = np.vstack(a).astype(float), np.vstack(b).astype(float)
        mu = A.mean(0)
        A, B = A - mu, B - mu
        n = len(A)
        caa = A.T @ A / n + CCA_REG * np.eye(A.shape[1])
        evals, evecs = np.linalg.eigh(caa)
        w = evecs @ np.diag(evals ** -0.5) @ evecs.T     # обеление: caa^(-1/2)
        m = w @ (A.T @ B / n) @ w
        corr, v = np.linalg.eigh((m + m.T) / 2)
        top = np.argsort(corr)[::-1][:CCA_DIM]
        self.proj, self.corr = w @ v[:, top], np.clip(corr[top], 0, None)

    def _fit_mlp(self, data: ProcessedData, xs: list[np.ndarray], ok: np.ndarray, active: np.ndarray) -> None:
        import torch
        from torch.nn import functional as F
        torch.set_num_threads(2)                         # как в исходном эксперименте: числа воспроизводятся
        xs = [torch.from_numpy(x) for x in xs]
        pair_rng = np.random.default_rng(SEED + 1)    # отдельный поток: при splits=2 числа как в исследовании

        self.net = _encoder(xs[0].shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        behavior = normalize(self.beh.embed(data, "train", self.hosts).to_numpy()).astype(np.float32)
        services = normalize(self._train["services"]).astype(np.float32)
        teacher = torch.from_numpy(np.hstack([behavior, services]) / np.float32(np.sqrt(2)))
        rng = np.random.default_rng(SEED)
        for _ in progress(range(self.epochs), desc=f"{NAME}: эпохи", leave=False):
            self.net.train()
            order = rng.permutation(active)
            for off in range(0, len(order), BATCH):
                i, j = (0, 1) if self.splits == 2 else pair_rng.choice(self.splits, 2, replace=False)
                batch = order[off:off + BATCH]
                ix = torch.from_numpy(batch[ok[i, batch] & ok[j, batch]])
                if len(ix) < 2:
                    continue
                za, zb = self.net(xs[i][ix]), self.net(xs[j][ix])
                logits, labels = za @ zb.T / TEMPERATURE, torch.arange(len(ix))
                contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
                target = teacher[ix] @ teacher[ix].T     # сохранить сходство по поведению и сервисам
                relational = 0.5 * (F.mse_loss(za @ za.T, target) + F.mse_loss(zb @ zb.T, target))
                loss = contrastive + 0.5 * relational
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()

    def fit_pca(self, data: ProcessedData) -> None:
        """PCA по train для текущего рецепта (после смены весов вызвать заново)."""
        self.pca = None
        if self.pca_dim:
            x = self._concat(data, "train", self.hosts)
            n = min(self.pca_dim, *x.shape)
            self.pca = PCA(n_components=n, svd_solver="randomized", random_state=29).fit(x)

    def _parts(self, data: ProcessedData, window: str, hosts: list[str]) -> dict:
        """Нормированные части окна; запоминаются, чтобы перебор весов не пересчитывал признаки."""
        key = (window, tuple(hosts))
        if key not in self._parts_cache:
            if window == "train":
                z = self._train
            else:
                start, end = _span(data, window)
                z = self._features(view(data.flows(window), self.hosts, end - start))
            rows = pd.Index(self.hosts).get_indexer(hosts)
            parts = {"behavioral": self.beh.embed(data, window, hosts).to_numpy(),
                     "neural": self._neural(z["input"])[rows]}
            parts.update({k: z[k][rows] for k in ("peers", "services", "context", "diffusion")})
            self._parts_cache = {k: v for k, v in self._parts_cache.items() if k[0] in ("train", "validation")}
            self._parts_cache[key] = {k: _block(v) for k, v in parts.items()}
        return self._parts_cache[key]

    def _concat(self, data: ProcessedData, window: str, hosts: list[str]) -> np.ndarray:
        parts = self._parts(data, window, hosts)
        return np.hstack([parts[k] * w for k, w in self.recipe.items() if w])

    def embed(self, data: ProcessedData, window: str, hosts: list[str] | None = None) -> pd.DataFrame:
        hosts = list(hosts) if hosts is not None else list(data.cohort)
        x = self._concat(data, window, hosts)
        if self.pca is not None:
            x = self.pca.transform(x)
        return pd.DataFrame(x, index=pd.Index(hosts, name="host_id")).add_prefix("emb_")


def _write(model: TemporalEncoder, data: ProcessedData, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    submit.keep_vectors_local(out)
    jobs = [("train", data.cohort, "embeddings.csv")] + [(w, data.evaluable(w), submit.window_file(w))
                                                          for w in data.test_windows]
    if "validation" in data.windows:                  # для подбора гибридов; в git не кладётся
        jobs.append(("validation", data.evaluable("validation"), "validation.csv"))
    for window, hosts, fname in jobs:
        model.embed(data, window, hosts).to_csv(out / fname, float_format="%.5g")


def _recipes(grid: dict):
    for wb in grid["behavioral"]:
        for wn in grid["neural"]:
            for wd in grid["diffusion"]:
                yield {**RECIPE, "behavioral": wb, "neural": wn, "diffusion": wd}


def tune(data: ProcessedData, grid: dict | None = None, pca_dim: int | None = PCA_DIM,
         encoder: str = "mlp") -> tuple[TemporalEncoder, list]:
    """Перебор GRID на validation (test не используется); возвращает обученную лучшую модель и таблицу."""
    import tempfile

    grid = grid or GRID
    rows, best = [], None
    with tempfile.TemporaryDirectory() as tmp:
        for splits in grid["splits"]:
            model = TemporalEncoder(pca_dim=pca_dim, splits=splits, encoder=encoder).fit(data)   # не зависит от весов
            for recipe in _recipes(grid):
                model.recipe = recipe
                model.fit_pca(data)
                d = Path(tmp) / "c"
                for window, hosts, fname in (("train", data.cohort, "embeddings.csv"),
                                             ("validation", data.evaluable("validation"), "validation.csv")):
                    d.mkdir(exist_ok=True)
                    model.embed(data, window, hosts).to_csv(d / fname, float_format="%.5g")
                score, tasks = submit.validation_score(d / "embeddings.csv", d / "validation.csv")
                row = {"splits": splits, **{k: recipe[k] for k in ("behavioral", "neural", "diffusion")},
                       "score": round(score, 2), **{k: round(v, 1) for k, v in tasks.items()}}
                rows.append(row)
                print("  " + ", ".join(f"{k} {v:g}" for k, v in row.items()), flush=True)
                if best is None or score > best[0]:
                    best = (score, splits, recipe)
    score, splits, recipe = best
    print(f"лучшее на validation: куски {splits}, {recipe} ({score:.1f})", flush=True)
    model = TemporalEncoder(recipe=recipe, pca_dim=pca_dim, splits=splits, encoder=encoder).fit(data)
    return model, rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pca", action="store_true", help="без сжатия (v5, 825 чисел)")
    ap.add_argument("--tune", action="store_true", help="подобрать куски train и веса на validation")
    ap.add_argument("--encoder", choices=("mlp", "cca"), default="mlp",
                    help="mlp — нейросеть (нужен torch); cca — линейная проекция, без torch, секунды")
    a = ap.parse_args()
    data = ProcessedData(os.environ.get("PT_DATA", ROOT / "scoreboard" / "data" / "processed"))
    pca_dim = None if a.no_pca else PCA_DIM
    name = NAME if a.encoder == "mlp" else "temporal_cca"
    if a.tune:
        name += "_tuned"
        model, rows = tune(data, pca_dim=pca_dim, encoder=a.encoder)
    else:
        rows = None
        model = TemporalEncoder(pca_dim=pca_dim, encoder=a.encoder).fit(data)
    out = ROOT / "scoreboard" / "submissions" / name
    _write(model, data, out)
    (out / "SHARED_COORDINATES").touch()
    (out / "AUTHOR").write_text("physcorgi\n", encoding="utf-8")
    params = {"recipe": model.recipe, "encoder": model.encoder, "splits": model.splits, "epochs": model.epochs,
              "pca_dim": model.pca_dim,
              "tuned_on": "validation этого прогона" if a.tune else "validation, день 02"}
    (out / "params.json").write_text(json.dumps(params, ensure_ascii=False) + "\n", encoding="utf-8")
    if rows:
        (out / "tuning.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    r = submit.score(out)
    submit.render()
    print(f"\n{name}: " + ", ".join(f"{s['title']} = {s['value']:.3f}" for s in r["summary"] if s["value"] is not None))


if __name__ == "__main__":
    main()
