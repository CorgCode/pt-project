"""Признаки хостов и рёбер из потоков: вход для GNN.

    python scoreboard/build_features.py                      # все окна из manifest.json (train первым)
    python scoreboard/build_features.py --windows train      # только одно окно (быстрая проверка)

Читает runs/processed/flows/<окно>.csv (кусками, память не растёт), пишет runs/features/<окно>/:
  nodes.csv    — одна строка на хост: профиль (66 признаков, уже в логарифмах и долях, см. ниже)
  edges.csv    — одна строка на пару хостов (a < b): сырые счётчики общения (flows, bytes, packets, ...)
  services.csv — хост × роль × (протокол, порт): для варианта V2 (граф «хост — сервис»)
и runs/features/ports.json — 20 самых частых портов train (одинаковый набор для всех окон).

В граф попадают пары, где хотя бы один хост — из когорты (как в эталонах проекта). В nodes.csv — хосты когорты
и их собеседники. Профиль считается по ВСЕМ потокам хоста (число собеседников — только по рёбрам графа, поэтому
у хостов вне когорты оно неполное).

Профиль хоста (nodes.csv):
  log_flows, log_flows_use, log_flows_serve, serve_share  — потоки: всего, как клиент (src), как сервер (dst), доля сервера
  log_bytes_sent/recv, log_pkts_sent/recv, sent_share, pktsize_sent/recv — объёмы и средний размер пакета
  use_p<порт>, use_p_other, use_p_anon    — доли потоков хоста-клиента по 20 топ-портам (anon — порт вида PortN)
  serve_p<порт>, serve_p_other, serve_p_anon — то же, когда хост — сторона dst
  proto_6, proto_17, proto_1, proto_other — доли протоколов
  tbin0..tbin5                            — доли активности по шести равным отрезкам окна
  log_partners                            — log(1 + число собеседников в графе)
Порт — это dst_port; src_port у LANL анонимизирован (PortN) и не используется.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N_TOP_PORTS = 20
N_BINS = 6
PROTOS = [6, 17, 1]
USE = ["src", "dst", "dst_port", "proto", "bytes", "packets", "ts", "duration", "resp_bytes", "resp_packets"]
DT = {"src": str, "dst": str, "dst_port": str, "proto": str, "bytes": "float64", "packets": "float64",
      "ts": "float64", "duration": "float64", "resp_bytes": "float64", "resp_packets": "float64"}


def log(msg: str) -> None:
    print(msg, flush=True)


class Acc:
    """Копит суммы по ключам кусками и время от времени сворачивает их, чтобы не раздувать память."""

    def __init__(self, limit: int = 3_000_000):
        self.parts, self.n, self.limit = [], 0, limit

    def add(self, df: pd.DataFrame) -> None:
        self.parts.append(df)
        self.n += len(df)
        if self.n > self.limit:
            self.compact()

    def compact(self) -> None:
        if len(self.parts) > 1:
            df = pd.concat(self.parts)
            df = df.groupby(level=list(range(df.index.nlevels)), sort=False).sum()
            self.parts, self.n = [df], len(df)
            self.limit = max(self.limit, 2 * len(df))

    def result(self) -> pd.DataFrame:
        self.compact()
        return self.parts[0]


def num(s: pd.Series) -> np.ndarray:
    """Числовой код порта/протокола; всё, что не число (например, Port35968), получает -1."""
    return pd.to_numeric(s, errors="coerce").fillna(-1).astype("int64").to_numpy()


def svc_frame(host, role, port, proto, sent, recv, ps, pr) -> pd.DataFrame:
    return pd.DataFrame({"host": host, "role": np.full(len(host), role, dtype=np.int8), "proto": proto, "port": port,
                         "flows": 1.0, "bytes_sent": sent, "bytes_recv": recv, "pkts_sent": ps, "pkts_recv": pr})


def read_window(path: Path, w: str, cohort: set, start: float, end: float, chunksize: int):
    S, T, E = Acc(), Acc(), Acc()
    st = dict(rows=0, self=0, resp=0, dur=0, anon=0)
    t0 = time.time()
    for i, ch in enumerate(pd.read_csv(path, chunksize=chunksize, usecols=USE, dtype=DT), 1):
        st["rows"] += len(ch)
        ch = ch[ch.src != ch.dst]
        src, dst = ch.src.to_numpy(), ch.dst.to_numpy()
        port, proto = num(ch.dst_port), num(ch.proto)
        by, rb = ch.bytes.fillna(0).to_numpy(), ch.resp_bytes.fillna(0).to_numpy()
        pk, rp = ch.packets.fillna(0).to_numpy(), ch.resp_packets.fillna(0).to_numpy()
        st["resp"] += int(((rb > 0) | (rp > 0)).sum())
        st["dur"] += int((ch.duration.fillna(0).to_numpy() > 0).sum())
        st["anon"] += int((port == -1).sum())
        st["kept"] = st.get("kept", 0) + len(ch)

        both = pd.concat([svc_frame(src, 0, port, proto, by, rb, pk, rp),
                          svc_frame(dst, 1, port, proto, rb, by, rp, pk)])
        S.add(both.groupby(["host", "role", "proto", "port"], sort=False).sum())

        b = np.clip(((ch.ts.to_numpy() - start) / (end - start) * N_BINS).astype(int), 0, N_BINS - 1)
        T.add(pd.DataFrame({"host": np.concatenate([src, dst]), "bin": np.concatenate([b, b]), "flows": 1.0})
              .groupby(["host", "bin"], sort=False).sum())

        m = (ch.src.isin(cohort) | ch.dst.isin(cohort)).to_numpy()
        if m.any():
            s, d = src[m], dst[m]
            a = np.where(s < d, s, d)
            bb = np.where(s < d, d, s)
            E.add(pd.DataFrame({"a": a, "b": bb, "proto": proto[m], "port": port[m], "flows": 1.0,
                                "flows_ab": (s == a).astype(float), "bytes": by[m] + rb[m], "packets": pk[m] + rp[m]})
                  .groupby(["a", "b", "proto", "port"], sort=False).sum())
        log(f"  [{w}] кусок {i}: прочитано {st['rows']:,} строк, {time.time() - t0:.0f} с")
    st["self"] = st["rows"] - st["kept"]
    return S.result(), T.result(), E.result(), st


def edge_table(E: pd.DataFrame) -> pd.DataFrame:
    e = E.reset_index()
    e["tcp"] = np.where(e.proto == 6, e.flows, 0.0)
    e["udp"] = np.where(e.proto == 17, e.flows, 0.0)
    g = e.groupby(["a", "b"], sort=False).agg(flows=("flows", "sum"), flows_ab=("flows_ab", "sum"),
                                              bytes=("bytes", "sum"), packets=("packets", "sum"),
                                              flows_tcp=("tcp", "sum"), flows_udp=("udp", "sum"),
                                              n_services=("flows", "size"))
    top = e.sort_values("flows", ascending=False).drop_duplicates(["a", "b"]).set_index(["a", "b"])
    g["main_port"] = top["port"]
    g["main_port_share"] = top["flows"] / g["flows"]
    return g.reset_index()


def node_table(S: pd.DataFrame, T: pd.DataFrame, deg: pd.Series, top_ports: list[int]) -> pd.DataFrame:
    S = S.reset_index()
    cols = ["flows", "bytes_sent", "bytes_recv", "pkts_sent", "pkts_recv"]
    tot = S.groupby("host")[cols].sum()
    rf = S.pivot_table(index="host", columns="role", values="flows", aggfunc="sum", fill_value=0)
    rf = rf.reindex(columns=[0, 1], fill_value=0)
    f = pd.DataFrame(index=tot.index)
    f["log_flows"] = np.log1p(tot.flows)
    f["log_flows_use"] = np.log1p(rf[0])
    f["log_flows_serve"] = np.log1p(rf[1])
    f["serve_share"] = rf[1] / tot.flows
    for c in cols[1:]:
        f["log_" + c] = np.log1p(tot[c])
    f["sent_share"] = (tot.bytes_sent / (tot.bytes_sent + tot.bytes_recv)).fillna(0.5)
    f["pktsize_sent"] = np.log1p(tot.bytes_sent / tot.pkts_sent.clip(lower=1))
    f["pktsize_recv"] = np.log1p(tot.bytes_recv / tot.pkts_recv.clip(lower=1))

    labels = [str(p) for p in top_ports] + ["other", "anon"]
    S["plabel"] = np.where(S.port.isin(top_ports), S.port.astype(str), np.where(S.port == -1, "anon", "other"))
    for role, name in ((0, "use"), (1, "serve")):
        pv = (S[S.role == role].pivot_table(index="host", columns="plabel", values="flows", aggfunc="sum", fill_value=0)
              .reindex(columns=labels, fill_value=0))
        pv = pv.div(pv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        pv.columns = [f"{name}_p{c}" for c in labels]
        f = f.join(pv, how="left")

    S["pl"] = np.where(S.proto.isin(PROTOS), S.proto.astype(str), "other")
    pr = (S.pivot_table(index="host", columns="pl", values="flows", aggfunc="sum", fill_value=0)
          .reindex(columns=[str(p) for p in PROTOS] + ["other"], fill_value=0))
    pr = pr.div(pr.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    pr.columns = [f"proto_{c}" for c in pr.columns]
    f = f.join(pr, how="left")

    tb = (T.reset_index().pivot_table(index="host", columns="bin", values="flows", aggfunc="sum", fill_value=0)
          .reindex(columns=range(N_BINS), fill_value=0))
    tb = tb.div(tb.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    tb.columns = [f"tbin{c}" for c in tb.columns]
    f = f.join(tb, how="left")
    f["log_partners"] = np.log1p(deg.reindex(f.index).fillna(0))
    return f.fillna(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=ROOT / "runs" / "processed")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "features")
    ap.add_argument("--windows", nargs="+", default=None, help="по умолчанию — все окна из manifest.json")
    ap.add_argument("--chunksize", type=int, default=500_000)
    a = ap.parse_args()

    manifest = json.loads((a.processed / "manifest.json").read_text(encoding="utf-8"))
    windows = a.windows or sorted(manifest["windows"], key=lambda w: (w != "train", manifest["windows"][w]["start"]))
    cohort = set(pd.read_csv(a.processed / "cohort.csv", dtype={"host_id": str})["host_id"])
    log(f"окна: {windows}; когорта: {len(cohort):,} хостов")
    a.out.mkdir(parents=True, exist_ok=True)
    ports_file = a.out / "ports.json"
    summary = []
    t_all = time.time()

    for w in windows:
        t0 = time.time()
        log(f"== окно {w}")
        info = manifest["windows"][w]
        S, T, E, st = read_window(a.processed / "flows" / f"{w}.csv", w, cohort, info["start"], info["end"], a.chunksize)
        log(f"  [{w}] строю таблицы ...")
        if w == "train" or not ports_file.exists():
            Sr = S.reset_index()
            top = Sr[Sr.port != -1].groupby("port").flows.sum().nlargest(N_TOP_PORTS).index.tolist()
            ports_file.write_text(json.dumps({"ports": [int(p) for p in top], "from_window": w}), encoding="utf-8")
        top_ports = json.loads(ports_file.read_text(encoding="utf-8"))["ports"]
        edges = edge_table(E)
        deg = pd.concat([edges["a"], edges["b"]]).value_counts()
        keep = sorted(cohort | set(edges["a"]) | set(edges["b"]))
        nodes = node_table(S, T, deg, top_ports).reindex(keep).fillna(0)
        nodes.insert(0, "in_cohort", nodes.index.isin(cohort))
        nodes.index.name = "host_id"
        services = S.reset_index()
        services = services[services.host.isin(set(keep))]

        out = a.out / w
        out.mkdir(parents=True, exist_ok=True)
        nodes.to_csv(out / "nodes.csv")
        edges.to_csv(out / "edges.csv", index=False)
        services.to_csv(out / "services.csv", index=False)
        dt = time.time() - t0
        n = max(st["kept"], 1)
        summary.append((w, st["rows"], len(nodes), len(edges), len(services), st["resp"] / n, st["dur"] / n, st["anon"] / n, dt))
        log(f"  [{w}] готово за {dt:.0f} с: {len(nodes):,} хостов, {len(edges):,} пар, {len(services):,} строк services")

    bar = "=" * 14
    print(f"\n{bar} КРАТКИЙ ИТОГ (скопируйте всё отсюда и пришлите мне) {bar}")
    print(f"Топ-{N_TOP_PORTS} портов train: {top_ports}")
    print(f"  {'окно':11s} {'потоков':>11s} {'хостов':>8s} {'пар':>9s} {'services':>9s} {'resp≠0':>7s} {'dur≠0':>6s} {'порт=PortN':>10s} {'с':>5s}")
    for w, rows, nn, ne, ns, r, d, an, dt in summary:
        print(f"  {w:11s} {rows:>11,} {nn:>8,} {ne:>9,} {ns:>9,} {r:>7.1%} {d:>6.1%} {an:>10.1%} {dt:>5.0f}")
    print("Колонки resp≠0 и dur≠0 — доля потоков с ненулевым ответным трафиком и длительностью;")
    print("порт=PortN — доля потоков, где dst_port не число (анонимный).")
    print(f"Всего: {time.time() - t_all:.0f} с. Результат: {a.out}")
    print("=" * 83)


if __name__ == "__main__":
    main()
