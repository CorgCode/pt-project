"""host_profile.py: эмбеддинги хостов по NetFlow.

Метод: профиль «хост x (роль-в-обмене, протокол/сервис)» с учётом направления,
TF-IDF (ослабляет «всеобщие» сервисы вроде DNS), усреднение по окнам, SVD,
центрирование (против хабов в kNN), опционально блок структуры соседей
и skip-gram по контекстам «кто обращался к сервису в одну минуту».

Использование:
    python host_profile.py selftest
    python host_profile.py embed --flows netflow_day-02.bz2 \
        --cols time=0,src=2,dst=3,proto=4,sport=5,dport=6 \
        --train 0:3600 --test 3600:5400 --out-dir runs/profile

ВАЖНО: номера колонок для LANL проверьте по описанию набора. Порты в LANL
могут быть деидентифицированы (строки вида "Port123"): такие значения
считаются «высокими» (эфемерными), well-known порты остаются числами.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

ROLE_P, ROLE_C, ROLE_S = 0, 1, 2  # нейтральный, клиент сервиса, сервер сервиса


# ------------------------------------------------------------------ токены
def _num(series, fill):
    return pd.to_numeric(series, errors="coerce").fillna(fill).astype(np.int64).to_numpy()


def _tokenize(df: pd.DataFrame, direction: bool, well_known: int = 1024) -> pd.DataFrame:
    """Длинная таблица (host, key, t, role): по строке на каждую сторону потока.

    Сервисный порт = меньший из двух, если он < well_known. Сторона с сервисным
    портом считается сервером, противоположная клиентом. Если оба порта
    «высокие», обе стороны нейтральные.
    """
    sport, dport = _num(df["sport"], 65535), _num(df["dport"], 65535)
    proto = np.clip(_num(df["proto"], -1), -1, 254)
    lo = np.minimum(sport, dport)
    is_svc = lo < well_known
    svc = np.where(is_svc, lo, -1)
    src_srv = is_svc & (sport < dport)
    dst_srv = is_svc & (dport < sport)
    role_src = np.where(src_srv, ROLE_S, np.where(dst_srv, ROLE_C, ROLE_P))
    role_dst = np.where(dst_srv, ROLE_S, np.where(src_srv, ROLE_C, ROLE_P))
    base = (proto + 1) * 2048 + (svc + 1)

    def key(role):
        return (role if direction else 0) * 1_000_000 + base

    t = pd.to_numeric(df["time"], errors="coerce").to_numpy(float)
    return pd.concat(
        [
            pd.DataFrame({"host": df["src"].astype(str).to_numpy(), "key": key(role_src), "t": t, "role": role_src}),
            pd.DataFrame({"host": df["dst"].astype(str).to_numpy(), "key": key(role_dst), "t": t, "role": role_dst}),
        ],
        ignore_index=True,
    )


# ------------------------------------------------------- структура соседей
def _struct(df: pd.DataFrame, hosts: pd.Index) -> np.ndarray:
    """Свой лог-степень + статистики лог-степеней соседей (mean/std/max/q90).

    Распределение степеней соседей в литературе (Jin et al.) само по себе
    сильный baseline для структурных ролей, поэтому держим как отдельный блок.
    """
    src, dst = df["src"].astype(str), df["dst"].astype(str)
    codes, uniq = pd.factorize(pd.concat([src, dst], ignore_index=True))
    n = len(df)
    a, b = codes[:n], codes[n:]
    e = pd.DataFrame({"a": np.r_[a, b], "b": np.r_[b, a]}).drop_duplicates()
    e = e[e.a != e.b]
    deg = e.groupby("a").size()
    e["lb"] = np.log1p(deg.reindex(e.b).to_numpy())
    gb = e.groupby("a").lb
    S = pd.DataFrame(
        {"own": np.log1p(deg), "mean": gb.mean(), "std": gb.std().fillna(0.0), "max": gb.max(), "q90": gb.quantile(0.9)}
    )
    S.index = uniq.take(S.index.to_numpy())
    return S.reindex(hosts).fillna(0.0).to_numpy()


# ------------------------------------------------------------------ модель
class HostProfileEmbedder:
    def __init__(
        self,
        dim: int = 32,
        window_sec: float = 1800.0,
        weighting: str = "tfidf",  # "raw" | "log" | "tfidf"
        direction: bool = True,
        center: bool = True,
        struct_weight: float = 0.0,
        skipgram_weight: float = 0.0,
        skipgram_dim: int = 16,
        skipgram_bucket_sec: float = 60.0,
        min_flows: int = 5,
        seed: int = 0,
    ):
        assert weighting in ("raw", "log", "tfidf")
        self.__dict__.update(locals())
        del self.__dict__["self"]

    # --- профиль хост x токен, усреднённый по активным окнам
    def _matrix(self, long: pd.DataFrame) -> sp.csr_matrix:
        h = self.hosts_.get_indexer(long["host"])
        k = self.vocab_.get_indexer(long["key"])
        w = (long["t"] // self.window_sec).fillna(-1).astype(np.int64).to_numpy()
        m = (h >= 0) & (k >= 0)
        g = pd.DataFrame({"h": h[m], "k": k[m], "w": w[m]}).groupby(["h", "k", "w"]).size().rename("n").reset_index()
        tf = g["n"].to_numpy(float)
        g["tf"] = tf if self.weighting == "raw" else np.log1p(tf)
        nwin = g[["h", "w"]].drop_duplicates().groupby("h").size()
        agg = g.groupby(["h", "k"], as_index=False)["tf"].sum()
        agg["tf"] = agg["tf"].to_numpy() / nwin.reindex(agg["h"]).to_numpy()
        return sp.csr_matrix((agg["tf"], (agg["h"], agg["k"])), shape=(len(self.hosts_), len(self.vocab_)))

    def _weight(self, X: sp.csr_matrix) -> sp.csr_matrix:
        if self.weighting == "tfidf":
            X = (X @ sp.diags(self.idf_)).tocsr()
        if self.weighting in ("log", "tfidf"):
            X = normalize(X)
        return X

    def fit(self, flows: pd.DataFrame) -> "HostProfileEmbedder":
        long = _tokenize(flows, self.direction)
        cnt = long.groupby("host").size()
        self.hosts_ = pd.Index(sorted(cnt[cnt >= self.min_flows].index))
        long = long[long["host"].isin(self.hosts_)]
        self.vocab_ = pd.Index(np.sort(long["key"].unique()))
        X = self._matrix(long)
        n = X.shape[0]
        df_ = np.asarray(X.getnnz(axis=0), float)
        self.idf_ = np.log((n + 1) / (df_ + 1)) + 1.0
        X = self._weight(X)
        d = max(2, min(self.dim, X.shape[1] - 1, n - 1))
        self.svd_ = TruncatedSVD(d, random_state=self.seed).fit(X)
        Z = self.svd_.transform(X)
        act = np.asarray(X.getnnz(axis=1)) > 0
        self.mean_ = Z[act].mean(axis=0) if self.center else np.zeros(d)
        if self.struct_weight > 0:
            S = _struct(flows, self.hosts_)
            self.s_mu_, self.s_sd_ = S[act].mean(0), S[act].std(0) + 1e-9
        return self

    def _skipgram(self, long: pd.DataFrame) -> np.ndarray:
        """Skip-gram по «предложениям»: клиенты одного сервиса за одну минуту.

        Обучается заново на каждом окне, поэтому для сравнения между окнами
        нужен Procrustes-выравнивающий шаг по общим хостам.
        """
        from gensim.models import Word2Vec

        c = long[(long["role"] == ROLE_C) & long["host"].isin(self.hosts_)].copy()
        c["b"] = (c["t"] // self.skipgram_bucket_sec).fillna(-1).astype(np.int64)
        sents = c.sort_values(["key", "b", "t"]).groupby(["key", "b"])["host"].agg(list)
        sents = [s[:200] for s in sents if len(s) >= 2]
        out = np.zeros((len(self.hosts_), self.skipgram_dim))
        if not sents:
            return out
        w2v = Word2Vec(sents, vector_size=self.skipgram_dim, window=5, min_count=1, sg=1,
                       negative=5, epochs=5, workers=1, seed=self.seed)
        for i, hname in enumerate(self.hosts_):
            if hname in w2v.wv:
                out[i] = w2v.wv[hname]
        return out

    def transform(self, flows: pd.DataFrame):
        """Возвращает (E, active): матрица эмбеддингов (по self.hosts_) и маска активных хостов."""
        long = _tokenize(flows, self.direction)
        X = self._weight(self._matrix(long[long["host"].isin(self.hosts_)]))
        act = np.asarray(X.getnnz(axis=1)) > 0
        Z = self.svd_.transform(X)
        Z[act] -= self.mean_
        blocks = [normalize(Z)]
        if self.struct_weight > 0:
            S = (_struct(flows, self.hosts_) - self.s_mu_) / self.s_sd_
            S[~act] = 0.0
            blocks.append(self.struct_weight * normalize(S))
        if self.skipgram_weight > 0:
            G = self._skipgram(long)
            blocks.append(self.skipgram_weight * normalize(G))
        E = normalize(np.hstack(blocks))
        E[~act] = 0.0
        return E, act


# ------------------------------------------------------------------- метрики
def _topk(E: np.ndarray, k: int) -> np.ndarray:
    sim = E @ E.T
    np.fill_diagonal(sim, -np.inf)
    return np.argpartition(-sim, k, axis=1)[:, :k]


def knn_purity(E, labels, k=10) -> float:
    idx = _topk(E, k)
    return float((labels[idx] == labels[:, None]).mean())


def hubness_skew(E, k=10) -> float:
    from scipy.stats import skew

    idx = _topk(E, k)
    return float(skew(np.bincount(idx.ravel(), minlength=len(E))))


def recognition_top1(E_train, E_test) -> float:
    return float((np.argmax(E_test @ E_train.T, axis=1) == np.arange(len(E_test))).mean())


# ------------------------------------------------------------------ синтетика
def make_synthetic(seed: int = 0, t_end: float = 5400.0, rate_scale: float = 1.0) -> tuple[pd.DataFrame, pd.Series]:
    """Сеть с заранее заложенными ролями. Истина (роли) задана независимо от признаков.

    rate_scale < 1 делает трафик разреженным: меньше потоков на хост, задача сложнее.
    """
    rng = np.random.default_rng(seed)
    sizes = dict(dev=200, off=200, file=30, web=30, db=20, dc=20, ext=100)
    g = {r: np.array([f"{r}{i:03d}" for i in range(n)]) for r, n in sizes.items()}
    rows = []

    def emit(clients, servers, dport, rate, proto=6, prefs=3):
        for c in clients:
            pref = rng.choice(servers, size=min(prefs, len(servers)), replace=False)
            n = rng.poisson(rate * rate_scale * rng.lognormal(0, 0.5) * t_end / 60)
            if n:
                rows.append(pd.DataFrame({
                    "time": rng.uniform(0, t_end, n), "src": c, "dst": rng.choice(pref, n),
                    "proto": proto, "sport": rng.integers(49152, 65535, n), "dport": dport}))

    ws = np.r_[g["dev"], g["off"]]
    emit(ws, g["dc"], 88, 0.5); emit(ws, g["dc"], 389, 0.5); emit(ws, g["dc"], 53, 2.0, proto=17)
    emit(ws, g["ext"], 443, 1.5, prefs=8)
    emit(g["dev"], g["web"], 22, 0.4); emit(g["dev"], g["web"], 8080, 0.8)
    emit(g["off"], g["file"], 445, 1.0); emit(g["off"], g["web"], 443, 0.5); emit(g["off"], g["ext"], 80, 0.8, prefs=8)
    emit(g["web"], g["db"], 5432, 2.0); emit(g["web"], g["dc"], 88, 0.3); emit(g["web"], g["dc"], 53, 1.0, proto=17)
    emit(g["file"], g["dc"], 88, 0.3); emit(g["file"], g["dc"], 389, 0.3)
    emit(g["db"], g["dc"], 88, 0.3)
    emit(g["dc"], g["dc"], 389, 1.0, prefs=5)
    for p in (135, 137, 139, 161, 5353):  # шум: редкие случайные сервисы
        emit(ws, np.r_[g["file"], g["web"], g["dc"]], p, 0.03, prefs=2)
    flows = pd.concat(rows, ignore_index=True)
    labels = pd.Series({h: r for r in g if r != "ext" for h in g[r]})
    return flows, labels


def selftest(seed: int = 0, rate_scale: float = 1.0) -> pd.DataFrame:
    flows, labels = make_synthetic(seed, rate_scale=rate_scale)
    tr, te = flows[flows.time < 3600], flows[flows.time >= 3600]
    configs = {
        "raw, без направления": dict(weighting="raw", direction=False, center=False),
        "raw, направление": dict(weighting="raw", direction=True, center=False),
        "log+L2, направление": dict(weighting="log", direction=True, center=False),
        "tfidf, без направления": dict(weighting="tfidf", direction=False, center=False),
        "tfidf, направление": dict(weighting="tfidf", direction=True, center=False),
        "tfidf, напр., центр.": dict(weighting="tfidf", direction=True, center=True),
        "+ структура соседей": dict(weighting="tfidf", direction=True, center=True, struct_weight=0.5),
    }
    try:
        import gensim  # noqa: F401
        configs["+ skip-gram"] = dict(weighting="tfidf", direction=True, center=True, skipgram_weight=0.5)
    except ImportError:
        print("gensim не установлен: skip-gram пропущен")
    res = []
    for name, kw in configs.items():
        m = HostProfileEmbedder(seed=seed, **kw).fit(tr)
        Etr, atr = m.transform(tr)
        Ete, ate = m.transform(te)
        keep = atr & ate & m.hosts_.isin(labels.index)
        y = labels.reindex(m.hosts_).to_numpy()[keep]
        row = {
            "вариант": name,
            "purity@10": knn_purity(Ete[keep], y),
            "hubness_skew": hubness_skew(Ete[keep]),
            "recognition@1": np.nan if kw.get("skipgram_weight") else recognition_top1(Etr[keep], Ete[keep]),
        }
        res.append(row)
    rnd = np.random.default_rng(seed).normal(size=(int(keep.sum()), 32))
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    res.append({"вариант": "random", "purity@10": knn_purity(rnd, y), "hubness_skew": hubness_skew(rnd),
                "recognition@1": recognition_top1(rnd, rnd[np.random.default_rng(1).permutation(len(rnd))])})
    return pd.DataFrame(res)


# ----------------------------------------------------------------------- CLI
def load_flows(path: str, cols: dict[str, int]) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, usecols=list(cols.values()), dtype=str)
    df.columns = [n for n, _ in sorted(cols.items(), key=lambda kv: kv[1])]
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df.dropna(subset=["time"])


def _write(path, hosts, E, act):
    d = pd.DataFrame(E[act], columns=[f"emb_{i}" for i in range(E.shape[1])])
    d.insert(0, "host_id", np.asarray(hosts)[act])
    d.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--rate-scale", type=float, default=1.0, help="<1: разреженный трафик, задача сложнее")
    e = sub.add_parser("embed")
    e.add_argument("--flows", required=True)
    e.add_argument("--cols", default="time=0,src=2,dst=3,proto=4,sport=5,dport=6")
    e.add_argument("--train", required=True, help="t0:t1 секунд")
    e.add_argument("--test", required=True, help="t0:t1 секунд")
    e.add_argument("--out-dir", default="runs/profile")
    e.add_argument("--dim", type=int, default=32)
    e.add_argument("--struct-weight", type=float, default=0.0)
    e.add_argument("--skipgram-weight", type=float, default=0.0)
    a = ap.parse_args()
    if a.cmd == "selftest":
        pd.set_option("display.width", 120)
        print(selftest(rate_scale=a.rate_scale).round(3).to_string(index=False))
        return
    cols = {k: int(v) for k, v in (p.split("=") for p in a.cols.split(","))}
    fl = load_flows(a.flows, cols)
    (t0, t1), (u0, u1) = (map(float, a.train.split(":")), map(float, a.test.split(":")))
    tr, te = fl[(fl.time >= t0) & (fl.time < t1)], fl[(fl.time >= u0) & (fl.time < u1)]
    m = HostProfileEmbedder(dim=a.dim, struct_weight=a.struct_weight, skipgram_weight=a.skipgram_weight).fit(tr)
    os.makedirs(a.out_dir, exist_ok=True)
    for name, part in (("train", tr), ("embeddings", te)):
        E, act = m.transform(part)
        _write(os.path.join(a.out_dir, f"{name}.csv"), m.hosts_, E, act)
    print(f"хостов в когорте: {len(m.hosts_)}; файлы в {a.out_dir}/ (embeddings.csv = test-окно)")


if __name__ == "__main__":
    main()
