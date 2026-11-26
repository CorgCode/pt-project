"""CLI: те же запросы, что и HTTP, с единым JSON-ответом.

    python -m hostquery similar --bundle runs/lanl --method temporal_encoder --host Comp1 -k 10
    python -m hostquery groups  --bundle runs/lanl --method temporal_encoder -n 6 --json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from hostquery import anomaly, explain, groups, schema, similar, temporal
from hostquery.errors import HostQueryError
from hostquery.store import Dataset, RunStore, open_dataset, read_csv_dataset


def _common() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bundle", type=Path, help="папка прогона с run/manifest.json")
    parser.add_argument("--embeddings", type=Path,
                        help="CSV с host_id, emb_* или с src, dst, bytes, …")
    parser.add_argument("--reference-dir", type=Path, help="папка эталона с neighbors.csv")
    parser.add_argument("--method", help="метод; по умолчанию behavioral или первый")
    parser.add_argument("--json", action="store_true", help="машинный ответ")
    parser.add_argument("--pretty", action="store_true", help="отступы в JSON")
    return parser


def build_parser() -> argparse.ArgumentParser:
    common = _common()
    parser = argparse.ArgumentParser(
        prog="python -m hostquery",
        description="Запросы к хостам по сохранённым артефактам прогона: похожие, группы, "
                    "объяснения, изменения, нетипичность.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("hosts", parents=[common], help="список хостов набора")
    p.add_argument("--limit", type=int, default=0, help="сколько показать (0 — все)")

    p = sub.add_parser("similar", parents=[common], help="похожие хосты")
    p.add_argument("--host")
    p.add_argument("--vector", help="строка чисел или путь к CSV/тексту с одним вектором")
    p.add_argument("-k", "--k", type=int, default=10)
    p.add_argument("--groups", type=int, default=0, help="добавить номер группы при n группах")

    p = sub.add_parser("consensus", parents=[common], help="консенсус по нескольким методам")
    p.add_argument("--host", required=True)
    p.add_argument("-k", "--k", type=int, default=10)
    p.add_argument("--methods", help="через запятую; по умолчанию все методы источника")

    p = sub.add_parser("groups", parents=[common], help="разбить на группы")
    p.add_argument("-n", "--n-groups", type=int, default=3)

    p = sub.add_parser("group", parents=[common], help="профиль одной группы")
    p.add_argument("-n", "--n-groups", type=int, default=3)
    p.add_argument("--group", type=int, required=True, help="номер группы с нуля")

    p = sub.add_parser("pair", parents=[common], help="объяснить похожесть пары")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)

    p = sub.add_parser("changes", help="сравнить два окна")
    p.add_argument("--before", type=Path, required=True, help="папка прогона или CSV")
    p.add_argument("--after", type=Path, required=True, help="папка прогона или CSV")
    p.add_argument("--method")
    p.add_argument("--reference-dir", type=Path)
    p.add_argument("-k", "--k", type=int, default=3, help="групп в первом окне")
    p.add_argument("--alignment", choices=("none", "procrustes"), default="none")
    p.add_argument("--json", action="store_true")
    p.add_argument("--pretty", action="store_true")

    p = sub.add_parser("outliers", parents=[common], help="exploratory-нетипичность")
    p.add_argument("-k", "--k", type=int, default=10)
    p.add_argument("--groups", type=int, default=0, help="добавить номер группы при n группах")

    p = sub.add_parser("serve", parents=[common], help="локальный HTTP API")
    p.add_argument("--port", type=int, default=8847)
    p.add_argument("--compare", type=Path, help="второе окно для /api/lab/changes")
    p.add_argument("--allow-origin", action="append", default=[], metavar="ORIGIN",
                   help="локальный origin лаборатории для CORS, например "
                        "http://127.0.0.1:8845; можно повторять")
    return parser


# ------------------------------------------------------------------------------------------ данные


def _store(args) -> RunStore:
    return RunStore(bundle=args.bundle, embeddings=args.embeddings,
                    reference_dir=args.reference_dir, method=args.method)


def _dataset(args) -> Dataset:
    return _store(args).dataset()


def _read_vector(value: str) -> list[float]:
    path = Path(value)
    if path.exists():
        text = path.read_text(encoding="utf-8-sig")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise HostQueryError(f"{path}: пустой файл с вектором")
        header = [cell.strip().lower() for cell in lines[0].replace(";", ",").split(",")]
        if "host_id" in header or any(cell.startswith("emb_") for cell in header):
            cells = [cell.strip() for cell in lines[1].replace(";", ",").split(",")]
            keep = [index for index, name in enumerate(header) if name != "host_id"]
            keep.sort(key=lambda index: int(header[index][4:])
                      if header[index][4:].isdigit() else index)
            try:
                values = [float(cells[index]) for index in keep if index < len(cells) and cells[index]]
            except ValueError:
                raise HostQueryError(f"{path}: в строке вектора не числа") from None
            if len(values) != len(keep):
                raise HostQueryError(f"{path}: в строке вектора не хватает координат")
            return values
        value = lines[0]
    text = value.strip().strip("[]")
    parts = [part for part in text.replace(";", ",").replace("\t", ",").split(",") if part.strip()]
    try:
        values = [float(part) for part in parts]
    except ValueError:
        raise HostQueryError("вектор должен быть списком чисел через запятую или CSV с emb_*") from None
    if not values:
        raise HostQueryError("пустой вектор")
    if not np.isfinite(values).all():
        raise HostQueryError("в векторе есть NaN или бесконечность")
    return values


# ------------------------------------------------------------------------------------------ команды


def _groups_optional(dataset: Dataset, count: int) -> list[int] | None:
    return groups.fit(dataset, count)["labels"] if count else None


def run(args) -> dict:
    provenance: dict = {}

    if args.command == "hosts":
        store = _store(args)
        dataset = store.dataset()
        hosts = dataset.ids[:args.limit] if args.limit else dataset.ids
        return schema.envelope("hosts", provenance=dataset.provenance,
                               query={"method": store.resolve_method(args.method)},
                               hosts=hosts, count=len(hosts), total=dataset.n)

    if args.command == "similar":
        store = _store(args)
        dataset = store.dataset()
        labels = _groups_optional(dataset, args.groups)
        if bool(args.host) == bool(args.vector):
            raise HostQueryError("нужен ровно один из --host или --vector")
        if args.host:
            result = similar.neighbors(dataset, args.host, args.k, labels=labels,
                                       reference=store.reference_ranks(args.host))
            query = {"host": args.host, "k": args.k}
        else:
            result = similar.similar_to(dataset, _read_vector(args.vector), args.k, labels=labels)
            query = {"vector": True, "k": args.k}
        query["method"] = store.resolve_method(args.method)
        return schema.envelope("similar", provenance=dataset.provenance, query=query, **result)

    if args.command == "consensus":
        store = _store(args)
        names = [name.strip() for name in args.methods.split(",")] if args.methods else store.methods
        items = [(name, store.dataset(name)) for name in names]
        result = similar.consensus(items, args.host, args.k)
        provenance = items[0][1].provenance
        provenance.update({"methods": names, "method": None})
        return schema.envelope("consensus", provenance=provenance,
                               query={"host": args.host, "k": args.k, "methods": names},
                               **result)

    if args.command in {"groups", "group"}:
        dataset = _dataset(args)
        result = groups.fit(dataset, args.n_groups)
        if args.command == "groups":
            profiles = [groups.profile(dataset, result["labels"], group, top=3)
                        for group in range(result["k"])]
            return schema.envelope("groups", provenance=dataset.provenance,
                                   query={"k": args.n_groups, "method": dataset.provenance.get("method")},
                                   **result, profiles=profiles)
        if not 0 <= args.group < result["k"]:
            raise HostQueryError(f"группа должна быть от 0 до {result['k'] - 1}")
        profile = groups.profile(dataset, result["labels"], args.group)
        return schema.envelope("group", provenance=dataset.provenance,
                               query={"group": args.group, "k": args.n_groups}, **profile)

    if args.command == "pair":
        dataset = _dataset(args)
        result = explain.pair(dataset, args.a, args.b)
        return schema.envelope("pair", provenance=dataset.provenance,
                               query={"a": args.a, "b": args.b}, **result)

    if args.command == "changes":
        before = open_dataset(args.before, method=args.method, reference_dir=args.reference_dir)
        after = open_dataset(args.after, method=args.method, reference_dir=args.reference_dir)
        result = temporal.compare(before, after, k=args.k, alignment=args.alignment)
        warnings = result.pop("warnings", [])
        provenance = dict(before.provenance)
        provenance["compare"] = {"before": str(args.before), "after": str(args.after),
                                 "after_sha256": after.provenance.get("embeddings_sha256")}
        return schema.envelope("changes", provenance=provenance, warnings=warnings,
                               query={"k": args.k, "alignment": args.alignment}, **result)

    if args.command == "outliers":
        dataset = _dataset(args)
        labels = _groups_optional(dataset, args.groups)
        result = anomaly.outliers(dataset, args.k, labels=labels)
        warnings = result.pop("warnings", [])
        return schema.envelope("outliers", provenance=dataset.provenance,
                               warnings=warnings, query={"k": args.k}, **result)

    if args.command == "serve":
        from hostquery.server import serve
        return serve(args)

    raise HostQueryError(f"неизвестная команда {args.command!r}")


# ------------------------------------------------------------------------------------------ вывод


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    line = "  ".join(f"{cell:<{width}}" for cell, width in zip(headers, widths)).rstrip()
    body = ["  ".join(f"{cell:<{width}}" for cell, width in zip(row, widths)).rstrip() for row in rows]
    return "\n".join([line, "-" * len(line), *body])


def render(args, result: dict) -> str:
    kind = result["query"]["kind"]
    if kind == "hosts":
        shown = "\n".join(result["hosts"])
        return f"{shown}\n\nвсего хостов: {result['total']}"
    if kind == "similar":
        rows = [[str(item["rank"]), item["id"], f"{item['similarity']:.4f}",
                 str(item.get("group", "")), str(item.get("reference_rank", ""))]
                for item in result["neighbors"]]
        return _table(["#", "хост", "косинус", "группа", "эталон"], rows)
    if kind == "consensus":
        rows = [[item["id"], f"{item['mean_rank']:.2f}", str(item["votes"]),
                 str(item["rank_spread"]), f"{item['mean_similarity']:.4f}"] for item in result["neighbors"]]
        return _table(["хост", "средний ранг", "голосов", "разброс", "косинус"], rows)
    if kind == "groups":
        blocks = []
        for profile in result["profiles"]:
            if profile.get("empty"):
                blocks.append(f"Группа {profile['group']} · 0 хостов (пусто)\n  {profile['note']}")
                continue
            top = ", ".join(f"{item['name'] or item['index']}" for item in profile["top_features"])
            blocks.append(f"Группа {profile['group']} · {profile['size']} хостов\n  сильнее всего: {top}")
        return "\n\n".join(blocks)
    if kind == "group":
        if result.get("empty"):
            return (f"Группа {result['group']} · 0 хостов (пусто)\n"
                    f"  {result['note']}")
        top = ", ".join(f"{item['name'] or item['index']} {item['centroid']:+.3f}"
                        for item in result["top_features"])
        return (f"Группа {result['group']} · {result['size']} хостов\n"
                f"  средний косинус к центроиду: {result['mean_similarity']:.3f}\n"
                f"  сильнее всего: {top}\n  {result['note']}")
    if kind == "pair":
        lines = [f"{result['a']} <-> {result['b']}: косинус {result['cosine']:.4f}", ""]
        for item in result["coordinates"]:
            name = item["name"] or f"координата {item['index']}"
            lines.append(f"  {name:<28} {item['a']:+.3f}  {item['b']:+.3f}")
        if result.get("shared_contacts") is not None:
            shared = ", ".join(result["shared_contacts"]) or "нет"
            lines += ["", f"общие собеседники: {shared}"]
        for direction, traffic in result.get("traffic_comparison", {}).items():
            lines.append("Исходящие записи" if direction == "outgoing" else "Входящие записи")
            for key, label in (("flows", "записи"), ("bytes", "байты"), ("packets", "пакеты")):
                if key in traffic:
                    count = traffic[key]
                    lines.append(f"  {label}: {count['a']:g} / {count['b']:g}")
            for key, label in (("shared_ports", "общие порты назначения"),
                               ("shared_protocols", "общие протоколы")):
                if key in traffic:
                    lines.append(f"  {label}: {', '.join(traffic[key]) or 'нет'}")
        return "\n".join(lines)
    if kind == "changes":
        rows = [[item["id"], f"{item['similarity']:.4f}",
                 f"{item['beforeGroup']} -> {item['afterGroup']}",
                 f"+{len(item.get('added', []))} / -{len(item.get('removed', []))}"]
                for item in result["changes"][:30]]
        header = f"общих: {result['n_common']} · новых: {len(result['newHosts'])} · " \
                 f"пропали: {len(result['missingHosts'])} · выравнивание: {result['alignment']}"
        return header + "\n\n" + _table(["хост", "сходство", "группа", "контакты"], rows)
    if kind == "outliers":
        rows = [[str(index), item["id"], f"{item['score']:.4f}", f"{item['mean_similarity']:.4f}"]
                for index, item in enumerate(result["items"][:30], start=1)]
        return _table(["#", "хост", "нетипичность", "средний косинус"], rows)
    return schema.dumps(result, pretty=True)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # прайм-обёртки в тестах не перенастраиваются
                pass
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
        if args.command == "serve":
            return 0
        print(schema.dumps(result, args.pretty) if args.json else render(args, result))
        return 0
    except HostQueryError as error:
        if getattr(args, "json", False):
            print(schema.dumps(error.as_dict(), True))
        else:
            print(f"Ошибка: {error.message}", file=sys.stderr)
            if error.hint:
                print(f"Подсказка: {error.hint}", file=sys.stderr)
        return 2
