"""Локальный HTTP API для лаборатории #156 на FastAPI.

Только 127.0.0.1, только чтение, тот же JSON, что у CLI. Модели запросов и ответов
описывают контракт `hostquery/1` в OpenAPI, поэтому UI может проверять ответы по
схеме, а не по примерам в README. Вычисления и загрузка данных остаются в доменных
модулях: здесь только маршрутизация, валидация и единая обработка ошибок.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request

from hostquery import anomaly, explain, groups, schema, similar, temporal
from hostquery.errors import HostQueryError
from hostquery.store import RunStore, open_dataset

LOCAL_HOSTS = {"127.0.0.1", "localhost"}
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}

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
    "/api/lab/openapi.json",
]


# ------------------------------------------------------------------------------------------ origin


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


# ------------------------------------------------------------------------------------------ модели


class ApiModel(BaseModel):
    """Базовая модель: лишние поля сохраняются, иначе ответ разошёлся бы с CLI."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Envelope(ApiModel):
    """Общая часть конверта: версия контракта, запрос и происхождение."""

    schema_version: str = Field(default=schema.SCHEMA, alias="schema")
    query: dict[str, Any]
    provenance: dict[str, Any]


class FeatureInfo(ApiModel):
    index: int
    name: str | None = None
    kind: str


class IndexResponse(ApiModel):
    schema_version: str = Field(default=schema.SCHEMA, alias="schema")
    endpoints: list[str]
    note: str


class HealthResponse(ApiModel):
    status: str
    schema_version: str = Field(default=schema.SCHEMA, alias="schema")
    methods: list[str]
    compare: bool


class MetaResponse(ApiModel):
    schema_version: str = Field(default=schema.SCHEMA, alias="schema")
    source: str
    root: str
    synthetic: bool
    label: str | None = None
    methods: list[str]
    hosts: int
    dimensions: int
    reference: str | None = None
    features: list[FeatureInfo]


class Neighbor(ApiModel):
    id: str
    rank: int
    similarity: float
    group: int | None = None
    reference_rank: int | None = None
    in_reference_top_k: bool | None = None


class SimilarResponse(Envelope):
    host: str
    k: int
    neighbors: list[Neighbor]
    warnings: list[str] = Field(default_factory=list)


class ProjectionPoint(ApiModel):
    x: float
    y: float
    hostId: str


class TopFeature(ApiModel):
    index: int
    name: str | None = None
    kind: str
    centroid: float


class GroupProfile(ApiModel):
    group: int
    size: int
    members: list[str]
    mean_similarity: float | None = None
    centroid: list[float] | None = None
    top_features: list[TopFeature]
    empty: bool
    note: str


class GroupsResponse(Envelope):
    k: int
    labels: list[int]
    sizes: list[int]
    centers: list[list[float]]
    projection: list[ProjectionPoint]
    profiles: list[GroupProfile]
    warnings: list[str] = Field(default_factory=list)


class GroupResponse(Envelope):
    group: int
    size: int
    members: list[str]
    mean_similarity: float | None = None
    centroid: list[float] | None = None
    top_features: list[TopFeature]
    empty: bool
    note: str
    warnings: list[str] = Field(default_factory=list)


class Coordinate(ApiModel):
    index: int
    name: str | None = None
    kind: str
    a: float
    b: float
    delta: float
    product: float


class Contribution(ApiModel):
    index: int
    name: str | None = None
    contribution: float


class PairResponse(Envelope):
    a: str
    b: str
    cosine: float
    coordinates: list[Coordinate]
    top_contributions: list[Contribution]
    shared_contacts: list[str] | None = None
    shared_contacts_note: str | None = None
    warnings: list[str] = Field(default_factory=list)


class DatasetItem(ApiModel):
    id: str
    v: list[float]
    norm: float
    activity: dict[str, Any] | None = None


class DatasetResponse(Envelope):
    dataset: list[DatasetItem]
    stats: dict[str, Any]
    kind: str
    connections: list[dict[str, Any]] | None = None
    warnings: list[str] = Field(default_factory=list)


class HostState(ApiModel):
    id: str
    v: list[float]
    norm: float
    logValues: list[float] | None = None
    activity: dict[str, Any] | None = None


class Change(ApiModel):
    id: str
    similarity: float
    drift: float
    before: HostState
    after: HostState
    beforeGroup: int
    afterGroup: int
    added: list[str] | None = None
    removed: list[str] | None = None
    beforeContacts: list[str] | None = None
    afterContacts: list[str] | None = None


class ChangesResponse(Envelope):
    kind: str
    changes: list[Change]
    newHosts: list[str]
    missingHosts: list[str]
    alignment: str
    k: int
    n_common: int
    warnings: list[str] = Field(default_factory=list)


class OutlierItem(ApiModel):
    id: str
    score: float
    mean_similarity: float
    group: int | None = None


class OutliersResponse(Envelope):
    k: int
    items: list[OutlierItem]
    warnings: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------------------ HTTP


class UTF8JSONResponse(JSONResponse):
    """`charset=utf-8` в Content-Type: тело всегда UTF-8, без догадок клиента."""

    media_type = "application/json; charset=utf-8"


def _json(status: int, payload: dict) -> UTF8JSONResponse:
    return UTF8JSONResponse(payload, status_code=status)


class LocalOnlyMiddleware(BaseHTTPMiddleware):
    """Пускает только loopback: проверяет `Host` и разрешённый `Origin`."""

    def __init__(self, app, allowed_origins: tuple[str, ...] = ()):
        super().__init__(app)
        self.allowed_origins = allowed_origins

    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "")
        if host.split(":")[0].lower() not in LOCAL_HOSTS:
            return _json(403, {"error": "доступен только локальный интерфейс"})
        origin = request.headers.get("origin")
        if origin and origin not in self.allowed_origins:
            return _json(403, {"error": "доступен только локальный интерфейс"})
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Одинаковые защитные заголовки на всех ответах, включая ошибки."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response


def create_app(store: RunStore, *, compare=None, compare_store: RunStore | None = None, allowed_origins=()) -> FastAPI:
    """Собрать приложение. Вычисления берутся из доменных модулей без изменений."""
    origins = normalize_origins(allowed_origins)
    app = FastAPI(
        title="hostquery",
        version=schema.SCHEMA,
        summary="Запросы к сохранённым артефактам прогона: только чтение",
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/lab/openapi.json",
        default_response_class=UTF8JSONResponse,
    )

    def dataset_for(method: str | None):
        return store.dataset(method or None)

    @app.get("/", response_model=IndexResponse, tags=["служебные"])
    @app.get("/api", response_model=IndexResponse, include_in_schema=False)
    @app.get("/api/lab", response_model=IndexResponse, include_in_schema=False)
    def index() -> dict:
        return {"schema": schema.SCHEMA, "endpoints": ENDPOINTS,
                "note": "только чтение; артефакты прогона не изменяются"}

    @app.get("/api/lab/health", response_model=HealthResponse, tags=["служебные"])
    def health() -> dict:
        return {"status": "ok", "schema": schema.SCHEMA,
                "methods": store.methods, "compare": compare is not None or compare_store is not None}

    @app.get("/api/lab/meta", response_model=MetaResponse, tags=["служебные"])
    def meta() -> dict:
        return store.describe()

    @app.get("/api/lab/dataset", response_model=DatasetResponse, tags=["данные"],
             response_model_exclude_unset=True)
    def dataset(method: str | None = Query(None, description="метод; по умолчанию behavioral")) -> dict:
        data = dataset_for(method)
        return schema.envelope("dataset", provenance=data.provenance,
                               query={"method": data.provenance.get("method")},
                               **data.to_json())

    @app.get("/api/lab/similar", response_model=SimilarResponse, tags=["запросы"],
             response_model_exclude_unset=True)
    def similar_endpoint(
        host: str | None = Query(None, description="идентификатор хоста"),
        k: int = Query(10, ge=1, description="сколько соседей вернуть"),
        group_count: int = Query(0, alias="groups", ge=0,
                                 description="добавить номер группы при n группах"),
        method: str | None = Query(None),
    ) -> dict:
        if not host:
            raise HostQueryError("нужен параметр host")
        data = dataset_for(method)
        labels = groups.fit(data, group_count)["labels"] if group_count else None
        result = similar.neighbors(data, host, k, labels=labels,
                                   reference=store.reference_ranks(host))
        return schema.envelope("similar", provenance=data.provenance,
                               query={"host": host, "k": k,
                                      "method": data.provenance.get("method")},
                               **result)

    @app.get("/api/lab/groups", response_model=GroupsResponse, tags=["запросы"],
             response_model_exclude_unset=True)
    def groups_endpoint(
        n: int = Query(3, ge=1, description="число групп"),
        method: str | None = Query(None),
    ) -> dict:
        data = dataset_for(method)
        result = groups.fit(data, n)
        profiles = [groups.profile(data, result["labels"], group, top=3)
                    for group in range(result["k"])]
        return schema.envelope("groups", provenance=data.provenance,
                               query={"k": n, "method": data.provenance.get("method")},
                               **result, profiles=profiles)

    @app.get("/api/lab/group", response_model=GroupResponse, tags=["запросы"],
             response_model_exclude_unset=True)
    def group_endpoint(
        n: int = Query(3, ge=1, description="число групп"),
        group: int = Query(-1, description="номер группы с нуля"),
        method: str | None = Query(None),
    ) -> dict:
        data = dataset_for(method)
        result = groups.fit(data, n)
        if not 0 <= group < result["k"]:
            raise HostQueryError(f"группа должна быть от 0 до {result['k'] - 1}")
        profile = groups.profile(data, result["labels"], group)
        return schema.envelope("group", provenance=data.provenance,
                               query={"group": group, "k": n}, **profile)

    @app.get("/api/lab/pair", response_model=PairResponse, tags=["запросы"],
             response_model_exclude_unset=True)
    def pair_endpoint(
        a: str | None = Query(None, description="первый хост"),
        b: str | None = Query(None, description="второй хост"),
        method: str | None = Query(None),
    ) -> dict:
        if not a or not b:
            raise HostQueryError("нужны параметры a и b")
        data = dataset_for(method)
        result = explain.pair(data, a, b)
        return schema.envelope("pair", provenance=data.provenance,
                               query={"a": a, "b": b}, **result)

    @app.get("/api/lab/changes", response_model=ChangesResponse, tags=["запросы"],
             response_model_exclude_unset=True)
    def changes_endpoint(
        k: int = Query(3, ge=1, description="число групп в первом окне"),
        alignment: str = Query("none", description="none или procrustes"),
        method: str | None = Query(None),
    ) -> dict:
        if compare is None and compare_store is None:
            raise HostQueryError("второе окно не задано",
                                 hint="запустите сервер с --compare <папка или csv>")
        data = dataset_for(method)
        if compare_store is not None:
            selected = data.provenance.get("method") if store.bundle is not None else compare_store.resolve_method()
            after = compare_store.dataset(selected)
        else:
            # A CSV has no model identifier. Treat it as an export for the server's default method.
            if store.bundle is not None and store.resolve_method(method) != store.resolve_method():
                raise HostQueryError("с CSV второго окна сравнивается только метод, выбранный при запуске")
            after = compare
        result = temporal.compare(data, after, k=k, alignment=alignment)
        warnings = result.pop("warnings", [])
        provenance = dict(data.provenance)
        # Контракт совпадает с CLI: путь и sha256 обоих окон. Иначе потребитель
        # не может проверить, что сравнил именно те файлы.
        provenance["compare"] = {
            "before": data.provenance.get("embeddings"),
            "after": after.provenance.get("embeddings"),
            "after_sha256": after.provenance.get("embeddings_sha256"),
        }
        return schema.envelope("changes", provenance=provenance, warnings=warnings,
                               query={"k": result["k"], "alignment": alignment}, **result)

    @app.get("/api/lab/outliers", response_model=OutliersResponse, tags=["запросы"],
             response_model_exclude_unset=True)
    def outliers_endpoint(
        k: int = Query(10, ge=1, description="сколько ближайших соседей усреднять"),
        group_count: int = Query(0, alias="groups", ge=0,
                                 description="добавить номер группы при n группах"),
        method: str | None = Query(None),
    ) -> dict:
        data = dataset_for(method)
        labels = groups.fit(data, group_count)["labels"] if group_count else None
        result = anomaly.outliers(data, k, labels=labels)
        warnings = result.pop("warnings", [])
        return schema.envelope("outliers", provenance=data.provenance,
                               warnings=warnings, query={"k": k}, **result)

    @app.exception_handler(HostQueryError)
    async def hostquery_error(request: Request, error: HostQueryError) -> UTF8JSONResponse:
        return _json(400, error.as_dict())

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request,
                               error: RequestValidationError) -> UTF8JSONResponse:
        first = error.errors()[0] if error.errors() else {"loc": (), "msg": ""}
        location = ".".join(str(part) for part in first["loc"] if part != "query")
        message = f"параметр {location}: неверное значение" if location \
            else "неверные параметры запроса"
        return _json(400, {"error": message,
                           "hint": "проверьте типы и диапазоны: k >= 1, n >= 1, group >= 0"})

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> UTF8JSONResponse:
        # Неизвестный маршрут остаётся 400, как было в контракте CLI и HTTP.
        if error.status_code == 404:
            return _json(400, {"error": f"неизвестный маршрут {request.url.path}",
                               "hint": "список путей — GET /"})
        if error.status_code == 405:
            return _json(405, {"error": "только чтение: метод не поддерживается"})
        return _json(error.status_code, {"error": str(error.detail)})

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> UTF8JSONResponse:
        # Наружу не отдаём трейсбек: только стабильный JSON.
        return _json(500, {"error": "внутренняя ошибка запроса"})

    # Порядок важен: LocalOnly снаружи (иначе preflight с чужим Host прошёл бы),
    # CORS внутри — он добавляет заголовки к ответам маршрутов и ошибкам.
    app.add_middleware(CORSMiddleware, allow_origins=list(origins),
                       allow_methods=["GET", "OPTIONS"],
                       allow_headers=["Accept", "Content-Type"], max_age=600)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(LocalOnlyMiddleware, allowed_origins=origins)
    return app


def serve(args) -> None:
    """Запустить локальный сервер: один процесс, только 127.0.0.1."""
    store = RunStore(bundle=args.bundle, embeddings=args.embeddings,
                     reference_dir=args.reference_dir, method=args.method)
    compare = None
    compare_store = None
    if getattr(args, "compare", None):
        if Path(args.compare).is_dir():
            compare_store = RunStore(bundle=args.compare, method=args.method, reference_dir=args.reference_dir)
        else:
            compare = open_dataset(args.compare, reference_dir=args.reference_dir)
    app = create_app(store, compare=compare, compare_store=compare_store,
                     allowed_origins=getattr(args, "allow_origin", ()))
    print(f"hostquery: http://127.0.0.1:{args.port}/  (только чтение, Ctrl+C — остановить)",
          flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
