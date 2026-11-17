"""Источник запросов: прогон, отдельный CSV эмбеддингов или CSV сетевых записей.

Прогон читается только на чтение и только через уже существующие контракты
(`app.results.Bundle`, `evaluation.io.load_embeddings`). Новых форматов не вводим.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.results import Bundle, BundleError
from evaluation.io import load_embeddings
from hostquery import features
from hostquery.errors import HostQueryError
from hostquery.schema import (MAX_CSV_BYTES, MAX_CSV_DIMENSIONS, MAX_CSV_HOSTS, MAX_CSV_ROWS)

_EMB_COLUMN = re.compile(r"emb_(\d+)")
_SRC_ALIASES = ("src", "source", "src_host")
_DST_ALIASES = ("dst", "target", "dst_host")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Dataset:
    """Хосты и их векторы в той же форме, что отдаёт `PTLab.parseCSV`.

    `vectors` — то, что сравнивается косинусом (для трафика уже log1p и общая шкала).
    `log_values` — промежуточные значения трафика без общей шкалы: нужны, чтобы
    сравнивать два окна на одной шкале. Для готовых эмбеддингов их нет.
    """

    ids: list[str]
    vectors: np.ndarray
    norms: np.ndarray
    kind: str
    stats: dict
    feature_names: list[str | None]
    feature_kinds: list[str]
    activity: dict[str, dict] | None = None
    connections: list[dict] | None = None
    log_values: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vectors = np.asarray(self.vectors, dtype=float)
        self.norms = np.asarray(self.norms, dtype=float)
        if self.vectors.ndim != 2:
            raise HostQueryError("векторы должны быть матрицей host × признак")
        if len(self.ids) != self.vectors.shape[0] or len(self.norms) != len(self.ids):
            raise HostQueryError("число хостов не совпадает с числом векторов")
        if len(set(self.ids)) != len(self.ids):
            raise HostQueryError("ID хостов повторяются")
        if not np.isfinite(self.vectors).all() or not np.isfinite(self.norms).all():
            raise HostQueryError("в векторах есть NaN или бесконечность")
        if (self.norms <= 0).any():
            bad = self.ids[int(np.argmax(self.norms <= 0))]
            raise HostQueryError(f"нулевой вектор у хоста {bad!r}: косинус не определён")

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def position(self, host: str) -> int:
        try:
            return self.ids.index(host)
        except ValueError:
            raise HostQueryError(f"неизвестный хост {host!r}",
                                 hint=f"в наборе {self.n} хостов; список — команда hosts") from None

    def unit(self) -> np.ndarray:
        """L2-нормированные строки: косинус — это скалярное произведение."""
        return self.vectors / self.norms[:, None]

    def to_json(self) -> dict:
        items = []
        for i, host in enumerate(self.ids):
            item = {"id": host, "v": [float(x) for x in self.vectors[i]], "norm": float(self.norms[i])}
            if self.activity is not None:
                item["activity"] = {k: _json_number(v) for k, v in self.activity[host].items()}
            items.append(item)
        out = {"dataset": items, "stats": {k: _json_number(v) for k, v in self.stats.items()}, "kind": self.kind}
        if self.connections is not None:
            out["connections"] = [dict(edge) for edge in self.connections]
        return out


def _json_number(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


# ------------------------------------------------------------------------------------------ CSV


def _parse_rows(text: str) -> tuple[list[str], list[list[str]]]:
    """Либеральный разбор CSV как в core.js: BOM, `;`, `,`, `\t`, кавычки, пустые строки."""
    if len(text.encode("utf-8")) > MAX_CSV_BYTES:
        raise HostQueryError(f"CSV больше {MAX_CSV_BYTES // (1024 * 1024)} МБ",
                             hint="для больших наборов используйте прогон или --embeddings")
    text = text.lstrip("\ufeff")
    first = text.split("\n", 1)[0].rstrip("\r")
    delimiter = sorted([";", ",", "\t"], key=lambda d: -first.count(d))[0]
    rows: list[list[str]] = []
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        for raw in reader:
            cells = [cell.strip() for cell in raw]
            if any(cells):
                rows.append(cells)
    except csv.Error as error:
        raise HostQueryError(f"CSV не разобран: {error}") from None
    if len(rows) < 2:
        raise HostQueryError("нужны заголовок и хотя бы одна строка данных")
    if len(rows) - 1 > MAX_CSV_ROWS:
        raise HostQueryError(f"в CSV больше {MAX_CSV_ROWS} строк",
                             hint="это ограничение браузерного примера; для больших наборов — прогон")
    columns = [cell.strip().lower() for cell in rows[0]]
    columns[0] = columns[0].lstrip("\ufeff")
    if len(set(columns)) != len(columns):
        raise HostQueryError("имена столбцов повторяются")
    body = rows[1:]
    for index, row in enumerate(body, start=2):
        if len(row) != len(columns):
            raise HostQueryError(f"строка {index}: полей {len(row)}, а в заголовке {len(columns)}")
    return columns, body


def _finite(raw: str, where: str) -> float:
    if raw == "":
        raise HostQueryError(f"{where}: пустое значение в векторе")
    try:
        value = float(raw)
    except ValueError:
        raise HostQueryError(f"{where}: не число {raw!r}") from None
    if not np.isfinite(value):
        raise HostQueryError(f"{where}: не конечное число")
    return value


def _nonnegative(raw: str, where: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise HostQueryError(f"{where}: не число {raw!r}") from None
    if not np.isfinite(value) or value < 0:
        raise HostQueryError(f"{where}: ожидается неотрицательное число")
    return value


def _vector_dataset(columns: list[str], rows: list[list[str]]) -> Dataset:
    host_col = columns.index("host_id")
    dims = sorted((int(m.group(1)), i) for i, name in enumerate(columns)
                  if (m := _EMB_COLUMN.fullmatch(name)))
    if any(number != position for position, (number, _) in enumerate(dims)):
        raise HostQueryError("координаты должны идти подряд от emb_0")
    if not 1 <= len(dims) <= MAX_CSV_DIMENSIONS:
        raise HostQueryError(f"нужно от 1 до {MAX_CSV_DIMENSIONS} столбцов emb_0, emb_1, …")
    if len(rows) > MAX_CSV_HOSTS:
        raise HostQueryError(f"в наборе больше {MAX_CSV_HOSTS} хостов")
    ids: list[str] = []
    seen: set[str] = set()
    vectors = np.empty((len(rows), len(dims)), dtype=float)
    for r, row in enumerate(rows, start=2):
        host = row[host_col]
        if not host:
            raise HostQueryError(f"строка {r}: пустой host_id")
        if host in seen:
            raise HostQueryError(f"строка {r}: host_id {host!r} повторяется")
        seen.add(host)
        ids.append(host)
        for c, (_, column_index) in enumerate(dims):
            vectors[r - 2, c] = _finite(row[column_index], f"строка {r}, {columns[column_index]}")
    norms = np.linalg.norm(vectors, axis=1)
    if (norms <= 0).any():
        raise HostQueryError(f"нулевой вектор у хоста {ids[int(np.argmax(norms <= 0))]!r}")
    if len(ids) < 2:
        raise HostQueryError("для сравнения нужны хотя бы два хоста")
    return Dataset(
        ids=ids, vectors=vectors, norms=norms, kind="vectors",
        stats={"hosts": len(ids), "dimensions": len(dims)},
        feature_names=features.embedding_names(len(dims)),
        feature_kinds=features.embedding_kinds(len(dims)),
    )


def _flow_dataset(columns: list[str], rows: list[list[str]]) -> Dataset:
    src_col = next((columns.index(name) for name in _SRC_ALIASES if name in columns), None)
    dst_col = next((columns.index(name) for name in _DST_ALIASES if name in columns), None)
    if src_col is None or dst_col is None:
        raise HostQueryError("нужны столбцы src и dst либо host_id и emb_0, emb_1…")
    bytes_col = columns.index("bytes") if "bytes" in columns else -1
    packets_col = columns.index("packets") if "packets" in columns else -1
    port_col = columns.index("dst_port") if "dst_port" in columns else -1
    proto_col = columns.index("proto") if "proto" in columns else -1

    order: list[str] = []
    values: dict[str, np.ndarray] = {}
    peers: dict[str, set[str]] = {}
    links: dict[tuple[str, str], dict] = {}

    def host(name: str, where: str) -> str:
        if not name:
            raise HostQueryError(f"{where}: src и dst не должны быть пустыми")
        if name not in values:
            if len(order) >= MAX_CSV_HOSTS:
                raise HostQueryError(f"в наборе больше {MAX_CSV_HOSTS} хостов")
            order.append(name)
            values[name] = np.zeros(features.N_FEATURES, dtype=float)
            peers[name] = set()
        return name

    total_bytes = 0.0
    for r, row in enumerate(rows, start=2):
        a = host(row[src_col], f"строка {r}")
        b = host(row[dst_col], f"строка {r}")
        volume = _nonnegative(row[bytes_col], f"строка {r}, bytes") if bytes_col >= 0 else 0.0
        packets = _nonnegative(row[packets_col], f"строка {r}, packets") if packets_col >= 0 else 0.0
        total_bytes += volume
        values[a][0] += 1
        values[b][1] += 1
        values[a][2] += volume
        values[b][3] += volume
        values[a][4] += packets
        values[b][5] += packets
        peers[a].add(b)
        peers[b].add(a)
        bucket = features.service_bucket(row[proto_col] if proto_col >= 0 else "",
                                         row[port_col] if port_col >= 0 else "")
        values[a][features.N_BASE + bucket] += 1
        values[b][features.N_BASE + features.N_BUCKETS + bucket] += 1
        edge = links.setdefault((a, b), {"src": a, "dst": b, "flows": 0, "bytes": 0.0})
        edge["flows"] += 1
        edge["bytes"] += volume
    if not np.isfinite(total_bytes):
        raise HostQueryError("суммарные объёмы слишком велики для расчёта")
    if len(order) < 2:
        raise HostQueryError("для сравнения нужны хотя бы два хоста")

    activity: dict[str, dict] = {}
    for name in order:
        activity[name] = {
            "outgoing": values[name][0], "incoming": values[name][1],
            "sentBytes": values[name][2], "receivedBytes": values[name][3],
            "contacts": len(peers[name]),
        }
        values[name][6] = len(peers[name])
        values[name][7] = values[name][0] + values[name][1]
    logged = np.log1p(np.vstack([values[name] for name in order]))
    scales = np.maximum(1.0, logged.max(axis=0))
    scaled = logged / scales
    norms = np.linalg.norm(scaled, axis=1)
    return Dataset(
        ids=order, vectors=scaled, norms=norms, kind="flows",
        stats={"hosts": len(order), "flows": len(rows), "bytes": total_bytes,
               "dimensions": features.N_FEATURES},
        feature_names=features.flow_names(),
        feature_kinds=features.flow_kinds(),
        activity=activity,
        connections=[{**edge, "bytes": float(edge["bytes"])} for edge in links.values()],
        log_values=logged,
    )


def dataset_from_csv(text: str) -> Dataset:
    """CSV либо с `host_id, emb_*`, либо с `src, dst, …` — как в лаборатории #156."""
    columns, rows = _parse_rows(text)
    if "host_id" in columns:
        return _vector_dataset(columns, rows)
    return _flow_dataset(columns, rows)


def read_csv_dataset(path: Path) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise HostQueryError(f"нет файла {path}")
    try:
        return dataset_from_csv(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError:
        raise HostQueryError(f"{path}: файл не в UTF-8") from None


# ------------------------------------------------------------------------------------------ прогон


def _header_has_host_id(path: Path) -> bool:
    """Готовые векторы опознаём по `host_id` в заголовке, а не по расширению."""
    try:
        with Path(path).open("r", encoding="utf-8-sig", errors="replace") as handle:
            first = handle.readline()
    except OSError as error:
        raise HostQueryError(f"{path}: не читается ({error.strerror})") from None
    columns = [cell.strip().strip('"').lower() for cell in re.split(r"[;,\t]", first)]
    return "host_id" in columns


class RunStore:
    """Один источник запросов: бандл прогона или отдельный CSV с эмбеддингами."""

    def __init__(self, *, bundle: str | Path | None = None, embeddings: str | Path | None = None,
                 reference_dir: str | Path | None = None, method: str | None = None):
        self.bundle: Bundle | None = None
        self.embeddings_path: Path | None = None
        self.root: Path | None = None
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self._method = method
        self._datasets: dict[str, Dataset] = {}
        self._reference: pd.DataFrame | None | bool = False

        if bundle is not None and embeddings is not None:
            raise HostQueryError("выберите один источник: --bundle или --embeddings")
        if bundle is not None:
            path = Path(bundle)
            try:
                self.bundle = Bundle(path)
            except BundleError as error:
                raise HostQueryError(str(error), hint="--bundle указывает на папку прогона с run/manifest.json") from None
            self.root = self.bundle.root
        elif embeddings is not None:
            self.embeddings_path = Path(embeddings)
            if not self.embeddings_path.exists():
                raise HostQueryError(f"нет файла {self.embeddings_path}")
        else:
            raise HostQueryError("нужен источник данных",
                                 hint="--bundle runs/<имя> или --embeddings <csv>")

    # ------------------------------------------------------------------ сведения
    @property
    def methods(self) -> list[str]:
        if self.bundle is not None:
            found = sorted(p.parent.name for p in (self.bundle.run / "embeddings").glob("*/train.csv"))
            return found
        return [self.embeddings_path.stem] if self.embeddings_path else []

    @property
    def synthetic(self) -> bool:
        return bool(self.bundle and self.bundle.synthetic)

    @property
    def label(self) -> str:
        if self.bundle is not None:
            return self.bundle.label
        return "СВОЙ CSV"

    def resolve_method(self, method: str | None = None) -> str:
        methods = self.methods
        if not methods:
            raise HostQueryError("в источнике нет сохранённых эмбеддингов",
                                 hint="опубликованные results/ хранят только числа; нужен локальный runs/<имя>")
        wanted = method or self._method
        if wanted:
            if wanted not in methods:
                raise HostQueryError(f"метода {wanted!r} нет в источнике",
                                     hint="доступны: " + ", ".join(methods))
            return wanted
        return "behavioral" if "behavioral" in methods else methods[0]

    # ------------------------------------------------------------------ данные
    def dataset(self, method: str | None = None) -> Dataset:
        name = self.resolve_method(method)
        if name in self._datasets:
            return self._datasets[name]
        if self.bundle is None:
            # Свой CSV бывает двух видов: готовые векторы и сырые записи обмена.
            # Вид определяет заголовок, а не имя файла: так одна и та же команда
            # работает и с выгрузкой эмбеддинга, и с выгрузкой netflow.
            path = self.embeddings_path
            if path is not None and _header_has_host_id(path):
                try:
                    frame = load_embeddings(path)
                except (BundleError, ValueError) as error:
                    raise HostQueryError(str(error),
                                         hint="проверьте формат embeddings CSV") from None
                return self._store_vectors(name, path, frame)
            data = read_csv_dataset(path)
            data.provenance = {
                "schema": "hostquery/1", "source": "csv", "bundle": None, "method": name,
                "synthetic": False, "data_kind": data.kind, "embeddings": str(path),
                "embeddings_sha256": sha256_file(path), "reference": None,
                "hosts": data.n, "dimensions": data.dim,
            }
            self._datasets[name] = data
            return data
        try:
            frame = self.bundle.scored_embedding(name)
            path = self.bundle.run / "embeddings" / name / "train.csv"
        except (BundleError, ValueError) as error:
            raise HostQueryError(str(error), hint="проверьте метод и формат embeddings CSV") from None
        return self._store_vectors(name, path, frame)

    def _store_vectors(self, name: str, path: Path, frame: pd.DataFrame) -> Dataset:
        if frame.empty:
            raise HostQueryError(f"эмбеддинг {name!r} пуст")
        ids = [str(host) for host in frame.index]
        # `evaluation.io.load_embeddings` сохраняет порядок столбцов из файла, а
        # `dataset_from_csv` сортирует их по номеру emb_*. Если не привести оба
        # пути к одному порядку, файл с `emb_1,emb_0` сравнится сам с собой как
        # разные векторы. Канонический порядок — числовой, как в core.js.
        dimensions = sorted((int(m.group(1)), str(column)) for column in frame.columns
                            if (m := _EMB_COLUMN.fullmatch(str(column))))
        if not dimensions:
            raise HostQueryError(f"эмбеддинг {name!r}: нет координат emb_0, emb_1, …")
        if any(number != position for position, (number, _) in enumerate(dimensions)):
            raise HostQueryError(f"эмбеддинг {name!r}: координаты должны идти подряд от emb_0")
        vectors = frame[[column for _, column in dimensions]].to_numpy(dtype=float)
        norms = np.linalg.norm(vectors, axis=1)
        reference = self._reference_path()
        provenance = {
            "schema": "hostquery/1",
            "source": "bundle" if self.bundle is not None else "embeddings",
            "bundle": str(self.root) if self.root is not None else None,
            "method": name,
            "synthetic": self.synthetic,
            "data_kind": getattr(self.bundle, "kind", "custom_csv"),
            "embeddings": str(path),
            "embeddings_sha256": sha256_file(path),
            "reference": str(reference) if reference else None,
            "hosts": len(ids),
            "dimensions": int(vectors.shape[1]),
        }
        data = Dataset(
            ids=ids, vectors=vectors, norms=norms, kind="vectors",
            stats={"hosts": len(ids), "dimensions": int(vectors.shape[1])},
            feature_names=features.embedding_names(int(vectors.shape[1])),
            feature_kinds=features.embedding_kinds(int(vectors.shape[1])),
            provenance=provenance,
        )
        self._datasets[name] = data
        return data

    # ------------------------------------------------------------------ эталон
    def _reference_path(self) -> Path | None:
        candidates: list[Path] = []
        if self.reference_dir is not None:
            candidates += [self.reference_dir / "neighbors.csv", self.reference_dir / "test" / "neighbors.csv"]
        if self.bundle is not None:
            candidates.append(self.bundle.ref_dir / "test" / "neighbors.csv")
        return next((p for p in candidates if p.exists()), None)

    def reference_ranks(self, host: str) -> dict[str, int]:
        """Ранг соседа по логам из сохранённого эталона; пусто, если эталона нет."""
        if self._reference is False:
            path = self._reference_path()
            self._reference = pd.read_csv(path, dtype={"host_id": str, "neighbor_id": str}) if path else None
        if self._reference is None:
            return {}
        rows = self._reference[self._reference["host_id"] == host]
        return {str(n): int(r) for n, r in zip(rows["neighbor_id"], rows["rank"])}

    def reference_path(self) -> Path | None:
        return self._reference_path()

    def describe(self) -> dict:
        methods = self.methods
        first = self.dataset(methods[0]) if methods else None
        return {
            "schema": "hostquery/1",
            "source": "bundle" if self.bundle is not None else "embeddings",
            "root": str(self.root) if self.root is not None else str(self.embeddings_path),
            "synthetic": self.synthetic,
            "label": self.label,
            "methods": methods,
            "hosts": first.n if first else 0,
            "dimensions": first.dim if first else 0,
            "reference": str(self._reference_path()) if self._reference_path() else None,
            "features": features.describe_embedding(first.dim) if first else [],
        }


def open_dataset(source: str | Path, *, method: str | None = None,
                 reference_dir: str | Path | None = None) -> Dataset:
    """Папка прогона или CSV — единая точка входа для CLI и сервера."""
    path = Path(source)
    if path.is_dir():
        return RunStore(bundle=path, reference_dir=reference_dir, method=method).dataset()
    if path.suffix.lower() == ".csv":
        data = read_csv_dataset(path)
        data.provenance = {
            "schema": "hostquery/1", "source": "csv", "bundle": None, "method": path.stem,
            "synthetic": False, "data_kind": data.kind, "embeddings": str(path),
            "embeddings_sha256": sha256_file(path), "reference": None,
            "hosts": data.n, "dimensions": data.dim,
        }
        return data
    raise HostQueryError(f"{path}: ожидается папка прогона или CSV",
                         hint="например runs/lanl или embeddings.csv")
