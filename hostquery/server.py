"""Локальный HTTP API для лаборатории #156.

Только 127.0.0.1, только чтение, тот же JSON, что у CLI. Страницы и графику
здесь не рисуем: #157 отдаёт данные, интерфейс остаётся в своей ветке.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from hostquery import anomaly, explain, groups, schema, similar, temporal
from hostquery.errors import HostQueryError
from hostquery.store import RunStore, open_dataset

LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def _normalize_origin(value: str) -> str:
    """Разрешаем только loopback-origin без пути: так API не открывается в сеть."""
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise HostQueryError(f"origin {value!r}: неверный порт") from None
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOCAL_HOSTS:
        raise HostQueryError(f"origin {value!r} должен быть локальным",
                             hint="например http://127.0.0.1:8845 или http://localhost:8845")
    if parsed.username or parsed.password or parsed.path not in {"", "/"} \
            or parsed.query or parsed.fragment:
        raise HostQueryError(f"origin {value!r} должен содержать только схему, хост и порт")
    return f"{parsed.scheme}://{parsed.hostname}" + (f":{port}" if port else "")


def normalize_origins(values) -> tuple[str, ...]:
    """Явный список origin лаборатории; дубликаты схлопываются."""
    seen: list[str] = []
    for value in values or ():
        normalized = _normalize_origin(value)
        if normalized not in seen:
            seen.append(normalized)
    return tuple(seen)

ENDPOINTS = [
    "/api/lab/health",
    "/api/lab/meta",
    "/api/lab/dataset?method=NAME",
    "/api/lab/similar?host=ID&k=10&method=NAME",
    "/api/lab/groups?n=6&method=NAME",
    "/api/lab/group?n=6&group=0&method=NAME",
    "/api/lab/pair?a=ID&b=ID&method=NAME",
    "/api/lab/changes?k=3&alignment=none",
    "/api/lab/outliers?k=10&method=NAME",
]


def _int(params: dict[str, list[str]], name: str, default: int) -> int:
    raw = params.get(name, [None])[0]
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise HostQueryError(f"параметр {name} должен быть целым") from None


def _one(params: dict[str, list[str]], name: str) -> str | None:
    return params.get(name, [None])[0] or None


def make_server(store: RunStore, *, compare=None, port: int = 8847,
                host: str = "127.0.0.1", allowed_origins=()) -> ThreadingHTTPServer:
    origins = normalize_origins(allowed_origins)

    class Handler(BaseHTTPRequestHandler):
        server_version = "hostquery"

        def host_allowed(self) -> bool:
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            return self.headers.get("Host") in allowed

        def cors_origin(self) -> str | None:
            origin = self.headers.get("Origin")
            return origin if origin and origin in self.server.allowed_origins else None

        def trusted(self) -> bool:
            origin = self.headers.get("Origin")
            return self.host_allowed() and (not origin or self.cors_origin() is not None)

        def send(self, status: int, data) -> None:
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
            origin = self.cors_origin()
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self) -> None:  # noqa: N802 - имя задаёт http.server
            """Preflight для fetch из лаборатории: только GET и только разрешённый origin."""
            if not self.host_allowed():
                return self.send(403, {"error": "доступен только локальный интерфейс"})
            origin = self.cors_origin()
            if not origin:
                return self.send(403, {
                    "error": "origin не разрешён",
                    "hint": "запустите serve с --allow-origin http://127.0.0.1:<порт лаборатории>",
                })
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Accept, Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802 - имя задаёт http.server
            if not self.trusted():
                return self.send(403, {"error": "доступен только локальный интерфейс"})
            parsed = urlsplit(self.path)
            params = parse_qs(parsed.query)
            try:
                return self.send(200, self.route(parsed.path, params))
            except HostQueryError as error:
                return self.send(400, error.as_dict())
            except (KeyError, ValueError) as error:
                return self.send(400, {"error": str(error)})
            except Exception:  # noqa: BLE001 - наружу не отдаём трейсбек
                return self.send(500, {"error": "внутренняя ошибка запроса"})

        def do_POST(self) -> None:  # noqa: N802
            self.send(405, {"error": "только чтение: метод не поддерживается"})

        do_PUT = do_POST
        do_DELETE = do_POST
        do_PATCH = do_POST

        def route(self, path: str, params: dict[str, list[str]]) -> dict:
            if path in {"/", "/api", "/api/lab"}:
                return {"schema": schema.SCHEMA, "endpoints": ENDPOINTS,
                        "note": "только чтение; артефакты прогона не изменяются"}
            if path == "/api/lab/health":
                return {"status": "ok", "schema": schema.SCHEMA,
                        "methods": store.methods, "compare": compare is not None}
            if path == "/api/lab/meta":
                return store.describe()

            method = _one(params, "method")
            dataset = store.dataset(method)
            if path == "/api/lab/dataset":
                return schema.envelope("dataset", provenance=dataset.provenance,
                                       query={"method": dataset.provenance.get("method")},
                                       **dataset.to_json())
            if path == "/api/lab/similar":
                host = _one(params, "host")
                if not host:
                    raise HostQueryError("нужен параметр host")
                k = _int(params, "k", 10)
                group_count = _int(params, "groups", 0)
                labels = groups.fit(dataset, group_count)["labels"] if group_count else None
                result = similar.neighbors(dataset, host, k, labels=labels,
                                           reference=store.reference_ranks(host))
                return schema.envelope("similar", provenance=dataset.provenance,
                                       query={"host": host, "k": k,
                                              "method": dataset.provenance.get("method")},
                                       **result)
            if path == "/api/lab/groups":
                count = _int(params, "n", 3)
                result = groups.fit(dataset, count)
                profiles = [groups.profile(dataset, result["labels"], group, top=3)
                            for group in range(result["k"])]
                return schema.envelope("groups", provenance=dataset.provenance,
                                       query={"k": count, "method": dataset.provenance.get("method")},
                                       **result, profiles=profiles)
            if path == "/api/lab/group":
                count = _int(params, "n", 3)
                group = _int(params, "group", -1)
                result = groups.fit(dataset, count)
                if not 0 <= group < result["k"]:
                    raise HostQueryError(f"группа должна быть от 0 до {result['k'] - 1}")
                profile = groups.profile(dataset, result["labels"], group)
                return schema.envelope("group", provenance=dataset.provenance,
                                       query={"group": group, "k": count}, **profile)
            if path == "/api/lab/pair":
                first, second = _one(params, "a"), _one(params, "b")
                if not first or not second:
                    raise HostQueryError("нужны параметры a и b")
                result = explain.pair(dataset, first, second)
                return schema.envelope("pair", provenance=dataset.provenance,
                                       query={"a": first, "b": second}, **result)
            if path == "/api/lab/outliers":
                k = _int(params, "k", 10)
                group_count = _int(params, "groups", 0)
                labels = groups.fit(dataset, group_count)["labels"] if group_count else None
                result = anomaly.outliers(dataset, k, labels=labels)
                warnings = result.pop("warnings", [])
                return schema.envelope("outliers", provenance=dataset.provenance,
                                       warnings=warnings, query={"k": k}, **result)
            if path == "/api/lab/changes":
                if compare is None:
                    raise HostQueryError("второе окно не задано",
                                         hint="запустите сервер с --compare <папка или csv>")
                result = temporal.compare(dataset, compare, k=_int(params, "k", 3),
                                          alignment=_one(params, "alignment") or "none")
                warnings = result.pop("warnings", [])
                provenance = dict(dataset.provenance)
                provenance["compare"] = {"after": compare.provenance.get("embeddings")}
                return schema.envelope("changes", provenance=provenance, warnings=warnings,
                                       query={"k": result["k"]}, **result)
            raise HostQueryError(f"неизвестный маршрут {path}", hint="список путей — GET /")

    server = ThreadingHTTPServer((host, port), Handler)
    server.allowed_origins = origins
    return server


def serve(args) -> None:
    store = RunStore(bundle=args.bundle, embeddings=args.embeddings,
                     reference_dir=args.reference_dir, method=args.method)
    compare = None
    if getattr(args, "compare", None):
        compare = open_dataset(args.compare, method=args.method, reference_dir=args.reference_dir)
    server = make_server(store, compare=compare, port=args.port,
                         allowed_origins=getattr(args, "allow_origin", ()))
    print(f"hostquery: http://127.0.0.1:{server.server_port}/  (только чтение, Ctrl+C — остановить)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
