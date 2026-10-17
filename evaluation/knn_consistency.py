"""Совпадение соседей (k-NN consistency): сохраняет ли эмбеддинг поведенческое соседство хостов.

Для каждого хоста берём k ближайших по эталонной похожести (из логов) и k ближайших по эмбеддингу
и сравниваем два множества. Больше пересечение — лучше эмбеддинг сохраняет поведенческую похожесть.

Утечка: эталон не может быть самим оцениваемым эмбеддингом. Если метод учился на похожих признаках,
лучше брать эталон из другого окна времени или другие группы признаков.

    build_reference_features(flows, hosts=None) -> DataFrame   поведенческий профиль
    evaluate_embedding(embedding, reference, ks=(5, 10, 20))    сводка по k
    neighborhood_metrics(ref_nn, emb_nn)                         метрики по каждому хосту
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# 1. Эталонная похожесть (без участия эмбеддингов)
# --------------------------------------------------------------------------

def _row_normalize(x: pd.DataFrame) -> pd.DataFrame:
    s = x.sum(axis=1).replace(0, 1.0)
    return x.div(s, axis=0)


BLOCKS = ("volume", "peers", "ratio", "ports", "proto", "hourly")


def build_reference_features(
    flows: pd.DataFrame,
    hosts: list | None = None,
    top_ports: int = 20,
    weights: dict | None = None,
    fitted: dict | None = None,
    return_fitted: bool = False,
    exclude_blocks: tuple[str, ...] = (),
):
    """Поведенческий профиль хоста: строка на хост.

    fitted / return_fitted: нормировку, список портов и протоколов подбираем на одном окне
    (return_fitted=True вернёт (признаки, fitted)) и применяем к другому (fitted=...) —
    тогда оба окна в одной системе координат.

    exclude_blocks: блоки признаков, которые не считаются вовсе, например ("ports",) — профиль без портов.
    Каждый блок нормируется отдельно, поэтому оставшиеся колонки совпадают с полным профилем (есть тест).

    Колонки flows: src, dst, dst_port, proto, bytes, packets, ts. Блоки (одинаковая норма, затем веса):
      volume - log1p числа потоков, байт, пакетов (исходящих и входящих)
      peers  - log1p числа разных собеседников (только количество, не кто именно)
      ratio  - доля исходящего для потоков и байт
      ports  - распределение по популярным портам назначения
      proto  - распределение по протоколам
      hourly - активность по 24 часам
    Кто именно собеседники — специально не используется: графовые методы, обученные на том же графе,
    воспроизвели бы это по построению.
    """
    exclude = tuple(sorted(set(exclude_blocks)))
    unknown = set(exclude) - set(BLOCKS)
    if unknown:
        raise ValueError(f"неизвестные блоки {sorted(unknown)}; есть {BLOCKS}")
    if len(exclude) == len(BLOCKS):
        raise ValueError("нельзя убрать все блоки")
    if fitted is not None and tuple(fitted.get("excluded", ())) != exclude:
        raise ValueError(f"fitted построен без блоков {tuple(fitted.get('excluded', ()))}, "
                         f"а не без {exclude}")
    f = flows.copy()
    if not np.issubdtype(f["ts"].dtype, np.datetime64):
        f["ts"] = pd.to_datetime(f["ts"], unit="s")
    f["hour"] = f["ts"].dt.hour
    if hosts is None:
        hosts = sorted(set(f["src"]) | set(f["dst"]))
    idx = pd.Index(hosts)

    out = f.groupby("src").agg(out_flows=("dst", "size"), out_bytes=("bytes", "sum"),
                               out_pkts=("packets", "sum"), out_peers=("dst", "nunique"))
    inn = f.groupby("dst").agg(in_flows=("src", "size"), in_bytes=("bytes", "sum"),
                               in_pkts=("packets", "sum"), in_peers=("src", "nunique"))
    base = out.join(inn, how="outer").reindex(idx).fillna(0.0)

    volume = np.log1p(base[["out_flows", "out_bytes", "out_pkts",
                            "in_flows", "in_bytes", "in_pkts"]])
    peers = np.log1p(base[["out_peers", "in_peers"]])
    ratio = pd.DataFrame({
        "flow_out_share": base.out_flows / (base.out_flows + base.in_flows).replace(0, 1),
        "byte_out_share": base.out_bytes / (base.out_bytes + base.in_bytes).replace(0, 1),
    }, index=idx)

    def dist(col, keep=None):
        both = pd.concat([f[["src", col]].rename(columns={"src": "h"}),
                          f[["dst", col]].rename(columns={"dst": "h"})])
        if keep is not None:
            both = both[both[col].isin(keep)]
        return _row_normalize(both.groupby(["h", col]).size().unstack(fill_value=0)
                              .reindex(idx).fillna(0.0))

    ports = proto = hourly = None
    port_list = []
    if "ports" not in exclude:
        port_list = (fitted["ports"] if fitted is not None
                     else list(f["dst_port"].value_counts().head(top_ports).index))
        ports = dist("dst_port", port_list)
        if fitted is not None:
            ports = ports.reindex(columns=port_list, fill_value=0.0)
    if "proto" not in exclude:
        proto = dist("proto")
        if fitted is not None:
            proto = proto.reindex(columns=fitted["protos"], fill_value=0.0)
    if "hourly" not in exclude:
        hourly = _row_normalize(pd.concat([
            f[["src", "hour"]].rename(columns={"src": "h"}),
            f[["dst", "hour"]].rename(columns={"dst": "h"})]).groupby(["h", "hour"]).size()
            .unstack(fill_value=0).reindex(index=idx, columns=range(24)).fillna(0.0))

    w = {"volume": 1.0, "peers": 1.0, "ratio": 1.0, "ports": 1.0, "proto": 0.5, "hourly": 1.0}
    w.update(weights or {})
    blocks = {"volume": volume, "peers": peers, "ratio": ratio,
              "ports": ports, "proto": proto, "hourly": hourly}
    parts, scalers = [], {}
    for name, b in blocks.items():
        if name in exclude:
            continue
        if fitted is not None:
            sc = fitted["scalers"][name]
            z = sc.transform(b.values)
        else:
            sc = StandardScaler().fit(b.values)
            z = sc.transform(b.values)
        scalers[name] = sc
        z = np.nan_to_num(z)
        z = z / np.sqrt(max(z.shape[1], 1))          # у каждого блока одинаковая суммарная дисперсия
        parts.append(pd.DataFrame(z * w[name], index=idx).add_prefix(f"{name}_"))
    features = pd.concat(parts, axis=1)
    if return_fitted:
        state = fitted if fitted is not None else {
            "ports": port_list, "protos": list(proto.columns) if proto is not None else [],
            "scalers": scalers, "excluded": exclude}
        return features, state
    return features


# --------------------------------------------------------------------------
# 2. Поиск соседей (сам хост исключён)
# --------------------------------------------------------------------------

def top_k_neighbors(X: np.ndarray, k: int, metric: str = "cosine") -> np.ndarray:
    """Индексы соседей (n, k) по убыванию похожести, без самого хоста."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    if not 1 <= k < n:
        raise ValueError(f"need 1 <= k < n_hosts, got k={k}, n={n}")
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute").fit(X)
    idx = nn.kneighbors(X, return_distance=False)
    out = np.empty((n, k), dtype=int)
    for i, row in enumerate(idx):
        row = row[row != i]          # из-за дубликатов сам хост может оказаться не на первом месте
        out[i] = row[:k]
    return out


# --------------------------------------------------------------------------
# 3. Метрики
# --------------------------------------------------------------------------

def neighborhood_metrics(ref_nn: np.ndarray, emb_nn: np.ndarray) -> pd.DataFrame:
    """Consistency@k, Jaccard@k и nDCG@k по каждому хосту.

    consistency = |R ∩ E| / k
    jaccard     = |R ∩ E| / |R ∪ E|
    ndcg        = DCG списка по эмбеддингу с весом (k - ранг в эталоне) для эталонных соседей,
                  делённый на идеальный DCG: награда за то, что ближайшие эталонные соседи стоят первыми.
    """
    n, k = ref_nn.shape
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = (np.arange(k, 0, -1) * discounts).sum()
    cons, jac, ndcg = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        rel = {h: k - r for r, h in enumerate(ref_nn[i])}
        inter = len(rel.keys() & set(emb_nn[i]))
        cons[i] = inter / k
        jac[i] = inter / (2 * k - inter)
        gains = np.array([rel.get(h, 0) for h in emb_nn[i]], dtype=float)
        ndcg[i] = (gains * discounts).sum() / ideal
    return pd.DataFrame({"consistency": cons, "jaccard": jac, "ndcg": ndcg})


def summarize(per_host: pd.DataFrame) -> pd.DataFrame:
    return per_host.agg(["mean", "median", "std"]).T


def evaluate_embedding(
    embedding: pd.DataFrame | np.ndarray,
    reference: pd.DataFrame | np.ndarray,
    ks=(5, 10, 20),
    emb_metric: str = "cosine",
    ref_metric: str = "cosine",
    return_per_host: bool = False,
):
    """Сравнить эмбеддинг с эталоном для нескольких k.

    Таблицы выравниваются по индексу. Возвращает сводку с индексом (k, метрика) и колонками
    mean/median/std и, по желанию, {k: таблица по хостам}.
    """
    if isinstance(embedding, pd.DataFrame) and isinstance(reference, pd.DataFrame):
        for name, df in (("embedding", embedding), ("reference", reference)):
            if df.index.has_duplicates:
                dups = df.index[df.index.duplicated()].unique().tolist()[:5]
                raise ValueError(f"в {name} повторяющиеся host_id, например {dups}")
        if set(embedding.index) != set(reference.index):
            only_e = list(set(embedding.index) - set(reference.index))[:5]
            only_r = list(set(reference.index) - set(embedding.index))[:5]
            raise ValueError("в эмбеддинге и эталоне разные host_id "
                             f"(только в эмбеддинге: {only_e}, только в эталоне: {only_r})")
        reference = reference.loc[embedding.index]
        hosts = embedding.index
    else:
        hosts = None
    E, R = np.asarray(embedding, float), np.asarray(reference, float)
    if len(E) != len(R):
        raise ValueError("эмбеддинг и эталон должны покрывать одних и тех же хостов")

    rows, per_host = [], {}
    for k in ks:
        ph = neighborhood_metrics(top_k_neighbors(R, k, ref_metric),
                                  top_k_neighbors(E, k, emb_metric))
        if hosts is not None:
            ph.index = hosts
        per_host[k] = ph
        s = summarize(ph)
        s.index = pd.MultiIndex.from_product([[k], s.index], names=["k", "metric"])
        rows.append(s)
    summary = pd.concat(rows)
    return (summary, per_host) if return_per_host else summary
