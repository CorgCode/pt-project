"""Многодневные срезы окон: поведение, контакты, сервисы и рёбра.

Из подготовленных потоков (``pipeline.prepare``) строит по одному файлу ``<окно>.npz``
на окно, общий ``vocab.json`` и ``manifest.json``. Словари (партнёры, порты, сервисы,
протоколы) строятся ТОЛЬКО по train-окну и дальше не меняются — иначе поздние окна
протекают в признаки.

    python -m methods.mdtgr.shards --processed <processed> --out <shards>

Формат среза (все массивы — в порядке ``cohort.csv``):

    behavior      float32 (N, F)  — профиль, стандартизованный по самому окну (якорь Behavior)
    beh_shared    float32 (N, F)  — тот же профиль в train-системе координат (вход модели)
    evaluable     bool (N,)       — хост active в окне (как в cohort.csv)
    peer_out_*    CSR (N, P)      — контакты src → dst: счётчики и байты
    peer_in_*     CSR (N, P)      — контакты dst → src: счётчики и байты
    service_*     CSR (N, S)      — (протокол, порт) исходящих потоков
    edge_*        массивы рёбер   — неориентированные рёбра внутри когорты

Партнёры в CSR — словарь train-окна (внешние имена и серверы включены), как в
``evaluation/partners.py``. Петли (src == dst) в контакты и рёбра не входят; в
поведенческом профиле они остаются, как в публичном эталоне.

``train`` обязателен: по нему фиксируются словари и нормировки. ``peer_out + peer_in``
по train-окну — это в точности матрица ``M`` из метрики собеседников.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from methods.mdtgr.features import Schema, behavior_pair, fit_schema

SCHEMA_VERSION = 2
USECOLS = ["src", "dst", "dst_port", "proto", "bytes", "packets", "ts", "resp_bytes"]
READ_DTYPE = {"src": "category", "dst": "category", "dst_port": "category", "proto": "category",
              "bytes": np.int64, "packets": np.int64, "ts": np.int64, "resp_bytes": np.int64}


def sha256_file(path: Path) -> str:
    """sha256 файла: провенанс среза без хранения самих потоков."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_flows(path: Path) -> pd.DataFrame:
    """Потоки окна с категориальными строками: память и коды словарей без лишних копий."""
    return pd.read_csv(path, usecols=USECOLS, dtype=READ_DTYPE)


def _partner_vocab(train: pd.DataFrame) -> list[str]:
    """Все имена, встречавшиеся в train-окне как источник или получатель."""
    names = set(train["src"].cat.categories) | set(train["dst"].cat.categories)
    return sorted(str(n) for n in names)


def _service_vocab(train: pd.DataFrame, limit: int) -> list[tuple[str, str]]:
    """Топ-``limit`` пар (протокол, порт) по частоте; порядок детерминирован."""
    proto_codes = train["proto"].cat.codes.to_numpy(np.int64)
    port_codes = train["dst_port"].cat.codes.to_numpy(np.int64)
    key = pd.MultiIndex.from_arrays([proto_codes, port_codes])
    counts = pd.Series(np.ones(len(key), np.int64), index=key).groupby(level=[0, 1]).size()
    proto_names = [str(p) for p in train["proto"].cat.categories]
    port_names = [str(p) for p in train["dst_port"].cat.categories]
    items = [((proto_names[int(a)], port_names[int(b)]), int(c)) for (a, b), c in counts.items()]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return [pair for pair, _ in items[:limit]]


def _codes(series: pd.Series, categories: list[str]) -> np.ndarray:
    """Коды значений в фиксированном словаре; неизвестное — -1."""
    return series.cat.set_categories(pd.Index(categories)).cat.codes.to_numpy(np.int64)


def _pair_codes(proto_codes: np.ndarray, port_codes: np.ndarray, proto_names: list[str],
                port_names: list[str], vocab: list[tuple[str, str]]) -> np.ndarray:
    """Коды пар (протокол, порт) в словаре сервисов; неизвестная пара — -1."""
    table = np.full((len(proto_names), len(port_names)), -1, dtype=np.int64)
    proto_index = {name: i for i, name in enumerate(proto_names)}
    port_index = {name: i for i, name in enumerate(port_names)}
    for j, (proto, port) in enumerate(vocab):
        i, k = proto_index.get(proto), port_index.get(port)
        if i is not None and k is not None:
            table[i, k] = j
    out = np.full(len(proto_codes), -1, dtype=np.int64)
    ok = (proto_codes >= 0) & (port_codes >= 0)
    out[ok] = table[proto_codes[ok], port_codes[ok]]
    return out


def _csr(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, shape: tuple[int, int],
         n_values: int = 1) -> tuple[sparse.csr_matrix, ...]:
    """CSR из троек (row, col, value...). Дубли суммируются, индексы сортируются."""
    mats = []
    for k in range(n_values):
        mat = sparse.coo_matrix((values[:, k], (rows, cols)), shape=shape).tocsr()
        mat.sum_duplicates()
        mat.sort_indices()
        mats.append(mat)
    return tuple(mats)


def build_window(flows: pd.DataFrame, hosts: list[str], partner_vocab: list[str],
                 service_vocab: list[tuple[str, str]], schema: Schema) -> dict:
    """Один срез: поведение, контакты в обе стороны, сервисы и рёбра когорты."""
    n = len(hosts)
    n_partners, n_services = len(partner_vocab), len(service_vocab)

    src = _codes(flows["src"], hosts)
    dst = _codes(flows["dst"], hosts)
    dst_partner = _codes(flows["dst"], partner_vocab)
    src_partner = _codes(flows["src"], partner_vocab)
    byt = flows["bytes"].to_numpy(np.float64) + flows["resp_bytes"].to_numpy(np.float64)
    pkt = flows["packets"].to_numpy(np.float64)
    real = src != dst

    behavior, beh_shared = behavior_pair(flows, hosts, schema)

    mask = real & (src >= 0) & (dst_partner >= 0)
    n_mask = int(mask.sum())
    out_counts, out_bytes = _csr(src[mask], dst_partner[mask],
                                 np.column_stack([np.ones(n_mask), byt[mask]]),
                                 (n, n_partners), n_values=2)
    mask = real & (dst >= 0) & (src_partner >= 0)
    n_mask = int(mask.sum())
    in_counts, in_bytes = _csr(dst[mask], src_partner[mask],
                               np.column_stack([np.ones(n_mask), byt[mask]]),
                               (n, n_partners), n_values=2)

    svc_codes = _pair_codes(flows["proto"].cat.codes.to_numpy(np.int64),
                            flows["dst_port"].cat.codes.to_numpy(np.int64),
                            [str(p) for p in flows["proto"].cat.categories],
                            [str(p) for p in flows["dst_port"].cat.categories],
                            service_vocab)
    mask = (src >= 0) & (svc_codes >= 0)
    n_mask = int(mask.sum())
    service, = _csr(src[mask], svc_codes[mask], np.ones((n_mask, 1)), (n, n_services))

    mask = real & (src >= 0) & (dst >= 0)
    n_mask = int(mask.sum())
    lo = np.minimum(src[mask], dst[mask])
    hi = np.maximum(src[mask], dst[mask])
    edges_count, edges_bytes, edges_packets = _csr(
        lo, hi, np.column_stack([np.ones(n_mask), byt[mask], pkt[mask]]), (n, n), n_values=3)
    coo = edges_count.tocoo()
    return {
        "behavior": behavior,
        "beh_shared": beh_shared,
        "peer_out_indptr": out_counts.indptr.astype(np.int64),
        "peer_out_indices": out_counts.indices.astype(np.int32),
        "peer_out_counts": out_counts.data.astype(np.float32),
        "peer_out_bytes": out_bytes.data.astype(np.float32),
        "peer_in_indptr": in_counts.indptr.astype(np.int64),
        "peer_in_indices": in_counts.indices.astype(np.int32),
        "peer_in_counts": in_counts.data.astype(np.float32),
        "peer_in_bytes": in_bytes.data.astype(np.float32),
        "service_indptr": service.indptr.astype(np.int64),
        "service_indices": service.indices.astype(np.int32),
        "service_counts": service.data.astype(np.float32),
        "edge_src": coo.row.astype(np.int32), "edge_dst": coo.col.astype(np.int32),
        "edge_flows": coo.data.astype(np.float32),
        "edge_bytes": edges_bytes.tocoo().data.astype(np.float32),
        "edge_packets": edges_packets.tocoo().data.astype(np.float32),
    }


def build_shards(processed: Path, out: Path, windows: list[str] | None = None, *,
                 service_limit: int = 1024) -> dict:
    """Собрать срезы окон, словари и манифест; вернуть манифест.

    ``windows`` позволяет достроить отдельные окна: словари и нормировки всё равно
    фиксируются по train-окну, а уже собранные срезы из ``manifest.json`` сохраняются.
    ``train`` обязан быть среди потоков, но не обязан быть в списке сборки.
    """
    processed, out = Path(processed), Path(out)
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    table = pd.read_csv(processed / "cohort.csv", dtype={"host_id": "string"})
    hosts = table["host_id"].astype(str).tolist()
    n = len(hosts)

    names = list(windows) if windows else list(manifest["windows"])
    missing = [w for w in names if not (processed / "flows" / f"{w}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"нет потоков для окон {missing} в {processed / 'flows'}")
    if not (processed / "flows" / "train.csv").exists():
        raise FileNotFoundError(f"нет потоков train-окна в {processed / 'flows'}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "hosts.txt").write_text("\n".join(hosts) + "\n", encoding="utf-8")
    previous: dict = {}
    if (out / "manifest.json").exists():
        previous = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    train = load_flows(processed / "flows" / "train.csv")
    schema = fit_schema(train, hosts)
    partner_vocab = _partner_vocab(train)
    service_vocab = _service_vocab(train, service_limit)
    shards: dict[str, dict] = {}

    for name in names:
        flows = train if name == "train" else load_flows(processed / "flows" / f"{name}.csv")
        arrays = build_window(flows, hosts, partner_vocab, service_vocab, schema)
        column = f"evaluable_{name}"
        if name == "train" or column not in table.columns:
            evaluable = np.ones(n, dtype=bool)
        else:
            evaluable = table[column].astype(str).str.lower().eq("true").to_numpy()
        arrays["evaluable"] = evaluable
        np.savez(out / f"{name}.npz", **arrays)
        shards[name] = {
            "n_flows": int(len(flows)), "n_evaluable": int(evaluable.sum()),
            "n_edges": int(len(arrays["edge_src"])),
            "n_peer_edges": int(len(arrays["peer_out_counts"]) + len(arrays["peer_in_counts"])),
            "n_service_edges": int(len(arrays["service_counts"])),
            "behavior_width": int(arrays["behavior"].shape[1]),
            "sha256": sha256_file(processed / "flows" / f"{name}.csv"),
        }
        del flows, arrays

    vocab = {"partners": partner_vocab,
             "services": [f"{a}:{b}" for a, b in service_vocab],
             "proto_pairs": [[a, b] for a, b in service_vocab],
             "ports": schema.ports, "protos": schema.protos}
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False) + "\n", encoding="utf-8")

    all_shards = {**previous.get("shards", {}), **shards}
    order = [w for w in previous.get("windows", []) if w in all_shards]
    order += [w for w in all_shards if w not in order]
    times = dict(previous.get("window_times", {}))
    for name in names:
        info = manifest["windows"].get(name)
        if info:
            times[name] = [float(info["start"]), float(info["end"])]
    order.sort(key=lambda w: times.get(w, [0.0, 0.0])[0])       # окна — по времени
    result = {
        "schema_version": SCHEMA_VERSION,
        "processed": str(processed).replace("\\", "/"),
        "cohort_sha256": hashlib.sha256("\n".join(hosts).encode()).hexdigest(),
        "n_hosts": n, "windows": order,
        "window_times": times,
        "behavior_width": int(schema.width),
        "schema": schema.to_json(),
        "vocab": {"n_partners": len(partner_vocab), "n_services": len(service_vocab),
                  "n_ports": len(schema.ports), "n_protos": len(schema.protos),
                  "partners_sha256": hashlib.sha256("\n".join(partner_vocab).encode()).hexdigest()},
        "shards": all_shards,
        "notes": ("behavior — профиль, стандартизованный по окну (якорь Behavior); "
                  "beh_shared — тот же профиль в train-системе координат (вход модели). "
                  "Петли исключены из контактов и рёбер; словари только по train."),
    }
    (out / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Срезы окон для MD-TGR")
    ap.add_argument("--processed", type=Path, required=True, help="каталог pipeline.prepare")
    ap.add_argument("--out", type=Path, required=True, help="куда писать .npz и manifest.json")
    ap.add_argument("--windows", help="через запятую; по умолчанию все окна манифеста")
    ap.add_argument("--service-vocab", type=int, default=1024)
    args = ap.parse_args(argv)
    windows = [w.strip() for w in args.windows.split(",")] if args.windows else None
    result = build_shards(args.processed, args.out, windows, service_limit=args.service_vocab)
    print(json.dumps({w: result["shards"][w] for w in result["windows"]}, indent=2))


if __name__ == "__main__":
    main()
