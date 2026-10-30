"""Потоковая подготовка окон; признаки и все преобразования обучаются только на train."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from pipeline.windows import window_mask

KEYS = {
    "tokens": ["h", "token"], "edges": ["src", "dst"],
    "volumes": ["h"], "times": ["h", "ts"], "bins": ["h", "bin"],
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tokens(flows):
    """Точное правило спецификации: равные порты нейтральны, оба PortNN — сервис dst."""
    srcp = pd.to_numeric(flows.src_port, errors="coerce").to_numpy(float)
    dstp = pd.to_numeric(flows.dst_port, errors="coerce").to_numpy(float)
    src = (srcp < 1024) & (np.isnan(dstp) | (srcp < dstp))
    dst = (dstp < 1024) & (np.isnan(srcp) | (dstp < srcp))
    anon = np.isnan(srcp) & np.isnan(dstp)
    service = np.full(len(flows), "other", dtype=object)
    service[src] = srcp[src].astype(int).astype(str)
    service[dst] = dstp[dst].astype(int).astype(str)
    service[anon] = "a:" + flows.dst_port.to_numpy(str)[anon]
    role_src = np.where(src, "server", np.where(dst | anon, "client", "neutral"))
    role_dst = np.where(dst | anon, "server", np.where(src, "client", "neutral"))
    base = flows.proto.to_numpy(str) + "/" + service + "/"
    return pd.DataFrame({"h": np.r_[flows.src.to_numpy(str), flows.dst.to_numpy(str)],
                         "token": np.r_[base + role_src, base + role_dst]})


def summarize(flows, bounds):
    f = flows.reset_index(drop=True)
    n = len(f)
    ends = pd.DataFrame({"h": np.r_[f.src.to_numpy(str), f.dst.to_numpy(str)],
                         "ts": np.tile(f.ts.to_numpy(), 2)})
    vol = ends[["h"]].copy()
    vol["out_count"], vol["in_count"] = np.r_[np.ones(n), np.zeros(n)], np.r_[np.zeros(n), np.ones(n)]
    # LANL — двунаправленные потоки: учитываем ответные байты и пакеты.
    for kind, col, response in (("bytes", "bytes", "resp_bytes"), ("packets", "packets", "resp_packets")):
        a = f[col].to_numpy(float)
        b = f[response].to_numpy(float) if response in f else np.zeros(n)
        vol[f"out_{kind}"], vol[f"in_{kind}"] = np.r_[a, b], np.r_[b, a]
    vol["out_duration"], vol["in_duration"] = np.r_[f.duration, np.zeros(n)], np.r_[np.zeros(n), f.duration]
    bins = ends[["h"]].copy()
    bins["bin"] = np.minimum(5, (6 * (ends.ts - bounds[0]) / (bounds[1] - bounds[0])).astype(int))
    real = f[f.src != f.dst]
    return {
        "tokens": tokens(f).groupby(["h", "token"]).size().rename("n").reset_index(),
        "edges": real.groupby(["src", "dst"]).size().rename("n").reset_index(),
        "volumes": vol.groupby("h").sum().reset_index(),
        "times": ends.groupby(["h", "ts"]).size().rename("n").reset_index(),
        "bins": bins.groupby(["h", "bin"]).size().rename("n").reset_index(),
    }


def merge(parts):
    return {key: pd.concat([p[key] for p in parts]).groupby(cols, as_index=False).sum()
            for key, cols in KEYS.items()}


def prepare(source: Path, board: Path, out: Path, excluded=()):
    """Восстановить текущие day02-окна из более широкого локального train, проверить каждый хост."""
    target = json.loads((board / "processed/manifest.json").read_text())
    original = json.loads((source.parent.parent / "manifest.json").read_text())
    windows = {w: tuple(target["config"]["windows"][w]) for w in ("train", "validation", "test")}
    midpoint = sum(windows["train"]) / 2
    windows.update(s1=(windows["train"][0], midpoint), s2=(midpoint, windows["train"][1]))
    container = original["config"]["windows"][source.stem]
    if not all(container[0] <= a < b <= container[1] for a, b in windows.values()):
        raise ValueError("Исходный файл не покрывает нужные окна целиком")
    if original["inputs"][0] != target["inputs"][0]:
        raise ValueError("Манифесты указывают разные исходные архивы дня 02")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("Кэш уже существует; используйте другой каталог")
    parts = {w: [] for w in windows}
    rows = {w: 0 for w in windows}
    banned = set(excluded)
    # Полные потоки train сохраняются для отдельного протокола нового хоста.
    save_train = not banned
    for i, chunk in enumerate(pd.read_csv(source, dtype={c: str for c in ("src", "dst", "src_port", "dst_port", "proto")}, chunksize=400_000)):
        for w, bounds in windows.items():
            mask = window_mask(chunk.ts.to_numpy(), chunk.duration.to_numpy(), bounds, True)
            f = chunk.loc[mask]
            if banned and w in ("train", "s1", "s2"):
                f = f[~f.src.isin(banned) & ~f.dst.isin(banned)]
            if f.empty:
                continue
            if save_train and w == "train":
                f.to_csv(out / "train_flows.csv", mode="a" if rows[w] else "w", header=not rows[w], index=False)
            rows[w] += len(f)
            parts[w].append(summarize(f, bounds))
        if i % 10 == 0:
            print(f"подготовка: {(i+1)*400_000:,} прочитано; {rows}", flush=True)
            for w in windows:
                if parts[w]:
                    parts[w] = [merge(parts[w])]
    cohort = pd.read_csv(board / "processed/cohort.csv", dtype={"host_id": str}).set_index("host_id")
    audit = {}
    for w in windows:
        value = merge(parts[w])
        if w in ("train", "validation", "test") and not banned:
            assert rows[w] == target["windows"][w]["n_flows"], (w, rows[w])
            edges = value["edges"]
            activity = edges.groupby("src").n.sum().add(edges.groupby("dst").n.sum(), fill_value=0)
            np.testing.assert_array_equal(activity.reindex(cohort.index, fill_value=0).to_numpy(), cohort[f"events_{w}"].to_numpy())
            audit[w] = {"flow_count": rows[w], "all_cohort_events_match": True}
        with (out / f"{w}.pkl").open("wb") as file:
            pickle.dump(value, file, protocol=5)
        parts[w] = []
    meta = {"windows": windows, "audit": audit, "source": str(source), "source_sha256": sha(source),
            "target_manifest_sha256": sha(board / "processed/manifest.json"),
            "cohort_sha256": target["cohort"]["sha256"], "excluded_hosts": sorted(banned),
            "available_test_windows": ["test"], "missing_test_windows": ["test_day03", "test_day04"]}
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_stats(cache, window):
    with (Path(cache) / f"{window}.pkl").open("rb") as file:
        return pickle.load(file)


def universe(stats, extra=()):
    return pd.Index(sorted(set(stats["volumes"].h) | set(extra)))


def graph(stats, hosts):
    e = stats["edges"]
    return sp.csr_matrix((e.n.to_numpy(np.float32), (hosts.get_indexer(e.src), hosts.get_indexer(e.dst))), shape=(len(hosts),) * 2)


def behavior(stats, hosts):
    v = stats["volumes"].set_index("h").reindex(hosts).fillna(0)
    x = {}
    for col in ("out_count", "in_count", "out_bytes", "in_bytes", "out_packets", "in_packets"):
        x[col] = np.log1p(v[col].to_numpy())
    for side in ("out", "in"):
        count = np.maximum(v[f"{side}_count"].to_numpy(), 1)
        x[f"{side}_mean_bytes"] = np.log1p(v[f"{side}_bytes"].to_numpy() / count)
        x[f"{side}_mean_duration"] = np.log1p(v[f"{side}_duration"].to_numpy() / count)
    edges = stats["edges"]
    out_peers = edges.groupby("src").size().reindex(hosts, fill_value=0).to_numpy()
    in_peers = edges.groupby("dst").size().reindex(hosts, fill_value=0).to_numpy()
    x["out_peers"], x["in_peers"] = np.log1p(out_peers), np.log1p(in_peers)
    total = (v.out_count + v.in_count).to_numpy()
    x["out_share"] = v.out_count.to_numpy() / np.maximum(total, 1)
    x["out_peer_share"] = out_peers / np.maximum(out_peers + in_peers, 1)
    tk = stats["tokens"].copy()
    for role in ("client", "server"):
        x[f"services_{role}"] = np.log1p(tk[tk.token.str.endswith("/" + role)].groupby("h").size().reindex(hosts, fill_value=0).to_numpy())
    sums = tk.groupby("h").n.transform("sum")
    p = tk.n / sums
    entropy = (-(p * np.log(np.maximum(p, 1e-12)))).groupby(tk.h).sum()
    x["service_entropy"] = entropy.reindex(hosts, fill_value=0).to_numpy()
    proto = tk.token.str.split("/").str[0]
    for name, mask in (("tcp", proto == "6"), ("udp", proto == "17"), ("icmp", proto.isin(["1", "58"])), ("other", ~proto.isin(["6", "17", "1", "58"]))):
        counts = tk[mask].groupby("h").n.sum().reindex(hosts, fill_value=0).to_numpy()
        x["proto_" + name] = counts / np.maximum(total, 1)
    bins = stats["bins"].pivot(index="h", columns="bin", values="n").reindex(index=hosts, columns=range(6)).fillna(0).to_numpy()
    for b in range(6):
        x[f"time_{b}"] = bins[:, b] / np.maximum(total, 1)
    x["active_bins"] = (bins > 0).mean(1)
    times = stats["times"].sort_values(["h", "ts"]).copy()
    gaps = times.groupby("h").ts.diff().fillna(0)
    times["gap2"] = gaps ** 2
    t = times.groupby("h").agg(first=("ts", "min"), last=("ts", "max"), n=("n", "sum"), sq=("gap2", "sum")).reindex(hosts).fillna(0)
    mean = (t["last"] - t["first"]) / np.maximum(t.n - 1, 1)
    sd = np.sqrt(np.maximum(t.sq / np.maximum(t.n - 1, 1) - mean ** 2, 0))
    x["burstiness"] = ((sd - mean) / np.maximum(sd + mean, 1e-12)).to_numpy()
    x["events"] = np.log1p(total)
    x["empty"] = (total == 0).astype(float)
    return pd.DataFrame(x, index=hosts).astype(np.float32)


class Preprocessor:
    def __init__(self, dim=64, min_df=2):
        self.dim, self.min_df = dim, min_df

    def token_matrix(self, stats, hosts):
        t = stats["tokens"]
        a, b = hosts.get_indexer(t.h), self.vocab.get_indexer(t.token)
        keep = (a >= 0) & (b >= 0)
        return sp.csr_matrix((t.n.to_numpy(float)[keep], (a[keep], b[keep])), shape=(len(hosts), len(self.vocab)))

    def weighted_tokens(self, stats, hosts):
        x = self.token_matrix(stats, hosts)
        x.data = np.log1p(x.data)
        return normalize(x.multiply(self.idf).tocsr())

    def fit(self, stats, cohort):
        hosts = universe(stats, cohort)
        t = stats["tokens"]
        df = t[t.h.isin(cohort)].groupby("token").size()
        self.vocab = pd.Index(sorted(df[df >= self.min_df].index))
        self.idf = np.log((len(cohort) + 1) / (df.reindex(self.vocab).to_numpy() + 1)) + 1
        x = self.weighted_tokens(stats, pd.Index(cohort))
        d = min(self.dim, x.shape[0] - 1, x.shape[1] - 1)
        self.svd = TruncatedSVD(d, random_state=0).fit(x)
        raw = np.hstack([self.svd.transform(x), behavior(stats, pd.Index(cohort)).to_numpy()])
        self.mean, self.sd = raw.mean(0), np.maximum(raw.std(0), 1e-6)
        self.behavior_names = behavior(stats, pd.Index(cohort)).columns.tolist()
        e = stats["edges"]
        peers = pd.concat([e[e.src.isin(cohort)][["src", "dst"]].rename(columns={"src": "cohort", "dst": "peer"}),
                           e[e.dst.isin(cohort)][["dst", "src"]].rename(columns={"dst": "cohort", "src": "peer"})]).drop_duplicates()
        df_peer = peers.groupby("peer").size()
        self.peer_idf = np.log((len(cohort) + 1) / (df_peer + 1)) + 1
        self.max_idf = np.log(len(cohort) + 1) + 1
        return self

    def transform(self, stats, hosts=None):
        hosts = universe(stats) if hosts is None else pd.Index(hosts)
        profile = self.svd.transform(self.weighted_tokens(stats, hosts))
        b = behavior(stats, hosts)
        raw = np.hstack([profile, b.to_numpy()])
        x = np.clip((raw - self.mean) / self.sd, -5, 5).astype(np.float32)
        empty = b["empty"].to_numpy() > 0
        x[empty] = 0
        # Флаг пустоты сохраняет буквальные 0/1, а не z-score.
        x[:, -1] = empty
        return hosts, x, profile.astype(np.float32), graph(stats, hosts)

    def idf_neighbors(self, hosts):
        return self.peer_idf.reindex(hosts, fill_value=self.max_idf).to_numpy(np.float32)
