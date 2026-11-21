"""Production ASGI entry point. The local preview remains dependency-free."""
from contextlib import asynccontextmanager

import json
import logging
import os
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from threading import Lock
from time import monotonic
from urllib.parse import parse_qs, urlsplit

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from .accounts import Accounts, COOKIE, SESSION_TTL
from .server import ASSETS, MAX_BODY, SITE, STATIC, Store, render_auth_page

STATIC_DIRS = {
    "_nuxt", "audio", "fonts", "hdri", "images", "models", "textures", "vendor",
    "model/assets", "model/reference",
}
PAGES = {"/": SITE / "index.html", "/model/": SITE / "model/index.html"}
FILES = {
    **ASSETS,
    **{"/" + name: STATIC / name for name in ("app.css", "app.js", "workflow.js")},
    **{"/" + file.name: file for file in SITE.iterdir()
       if file.is_file() and file.suffix in {".css", ".js", ".svg", ".jpg", ".png"}},
    **{"/lab/" + file.name: file for file in (SITE / "lab").iterdir()
       if file.is_file() and file.suffix in {".js", ".css"}},
    "/model/model.css": SITE / "model/model.css",
    "/model/model.js": SITE / "model/model.js",
}


def create_app(directory: Path, origin: str, inference_url: str | None = None, *,
               engine=None, allow_registration=True, metrics_token=None) -> FastAPI:
    url = urlsplit(origin)
    if (url.scheme not in {"http", "https"} or not url.hostname
            or url.username or url.password or url.path or url.query or url.fragment
            or (url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1"})):
        raise ValueError("PT_PUBLIC_ORIGIN must be an HTTPS origin, or HTTP on loopback for testing")
    if engine is not None:
        from .database import DatabaseAccounts, DatabaseWorkspaces, check_database
        accounts = DatabaseAccounts(engine)
    else:
        accounts = Accounts(directory)

    @asynccontextmanager
    async def lifespan(_):
        try:
            yield
        finally:
            if engine is not None:
                engine.dispose()
    secure = url.scheme == "https"
    attempts = deque()
    lock = Lock()
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[url.hostname, "127.0.0.1"])

    def error(status, message):
        return JSONResponse({"error": message}, status)

    def user(request):
        return accounts.identify(request.cookies.get(COOKIE))

    def store(user_id):
        if engine is not None:
            return DatabaseWorkspaces(engine, user_id)
        return Store(directory / "users" / user_id / "workspaces.sqlite3")

    @app.middleware("http")
    async def protect(request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get("origin") != origin:
            return error(403, "Запрос должен исходить с этого сайта.")
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        path = request.url.path
        if path.startswith("/api/") or path.startswith("/lab/"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: blob:; frame-ancestors 'none'; form-action 'self'"
            )
        return response

    if engine is not None:
        from .resources import add_routes, embedding_result, finish_job, start_job
        add_routes(app, engine, user, error)

    if metrics_token:
        from .telemetry import instrument
        instrument(app, metrics_token)

    @app.get("/api/ready")
    def ready():
        if engine is not None:
            from sqlalchemy.exc import SQLAlchemyError
            try:
                check_database(engine)
            except SQLAlchemyError:
                logging.getLogger("pt.operator").warning("database readiness check failed")
                return error(503, "Database is unavailable.")
        return {"status": "ready"}

    @app.get("/api/health")
    def health():
        return {"status": "ok", "product": "CorgCode", "workspaceSchema": 2}

    @app.api_route("/lab/", methods=["GET", "HEAD"])
    def workspace(request: Request):
        if not user(request):
            return RedirectResponse("/lab/login/", 303)
        page = (STATIC / "index.html").read_text(encoding="utf-8")
        inference = ' data-ml-enabled="true"' if engine is not None and inference_url else ""
        return HTMLResponse(page.replace("<body>", f'<body data-workspace-mode="account"{inference}>'))

    @app.api_route("/lab/register/", methods=["GET", "HEAD"])
    @app.api_route("/lab/login/", methods=["GET", "HEAD"])
    def sign_in_page(request: Request):
        registering = request.url.path == "/lab/register/"
        if registering and not allow_registration:
            return error(404, "Регистрация закрыта.")
        if user(request):
            return RedirectResponse("/lab/", 303)
        return HTMLResponse(render_auth_page(registering, allow_registration=allow_registration))

    @app.post("/lab/register/")
    @app.post("/lab/login/")
    async def sign_in(request: Request):
        async def form():
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > 4096:
                    raise ValueError
                body.extend(chunk)
            return parse_qs(body.decode(), max_num_fields=4)

        wants_json = request.headers.get("accept") == "application/json"

        registering = request.url.path == "/lab/register/"

        if registering and not allow_registration:
            return error(404, "Регистрация закрыта.")

        def fail(status, message):
            return error(status, message) if wants_json else HTMLResponse(render_auth_page(registering, message, allow_registration=allow_registration), status)

        if request.headers.get("content-type", "").split(";")[0] != "application/x-www-form-urlencoded":
            return fail(400, "Ожидается форма входа.")
        try:
            fields = await form()
        except ValueError:
            return fail(400, "Некорректная форма входа.")
        now = monotonic()
        with lock:
            while attempts and now - attempts[0] >= 60:
                attempts.popleft()
            if len(attempts) >= 30:
                return fail(429, "Слишком много попыток. Попробуйте через минуту.")
            attempts.append(now)
        name, password = fields.get("name", [""])[0], fields.get("password", [""])[0]
        if registering:
            try:
                await run_in_threadpool(accounts.add, name, password)
            except ValueError as exc:
                return fail(400, str(exc))
        token = await run_in_threadpool(accounts.login, name, password)
        if not token:
            return fail(401, "Проверьте имя пользователя и пароль.")
        await run_in_threadpool(accounts.logout, request.cookies.get(COOKIE))
        response = JSONResponse({"signedIn": True}, status_code=201 if registering else 200) if wants_json else RedirectResponse("/lab/", 303)
        response.set_cookie(COOKIE, token, max_age=SESSION_TTL, httponly=True, secure=secure, samesite="strict")
        return response

    @app.post("/lab/logout/")
    def sign_out(request: Request):
        accounts.logout(request.cookies.get(COOKIE))
        response = RedirectResponse("/lab/", 303)
        response.delete_cookie(COOKIE, httponly=True, secure=secure, samesite="strict")
        return response

    @app.get("/api/workspaces")
    @app.get("/api/workspaces/{key}")
    def read_workspace(request: Request, key: str | None = None):
        account = user(request)
        if not account:
            return error(401, "Войдите, чтобы открыть сохранённый набор.")
        storage = store(account)
        if key is None:
            return storage.list()
        saved = storage.get(key)
        if saved is None:
            return error(404, "Набор не найден.")
        if engine is not None and saved.get("inferenceId"):
            saved["inference"] = embedding_result(engine, account, saved["inferenceId"])
        return saved

    @app.post("/api/workspaces", status_code=201)
    @app.put("/api/workspaces/{key}")
    async def write_workspace(request: Request, key: str | None = None):
        account = await run_in_threadpool(user, request)
        if not account:
            return error(401, "Войдите, чтобы сохранить набор.")
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            return error(400, "Ожидается JSON.")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_BODY:
                return error(413, "Набор слишком большой.")
            body.extend(chunk)
        storage = await run_in_threadpool(store, account)
        if key is not None and await run_in_threadpool(storage.get, key) is None:
            return error(404, "Набор не найден.")
        try:
            ident = await run_in_threadpool(storage.save, json.loads(body), key)
        except (ValueError, UnicodeError) as exc:
            return error(400, str(exc))
        return {"id": ident} if ident else error(404, "Набор не найден.")

    if inference_url:
        def run_inference(body, digest=None):
            limit = MAX_BODY
            if digest is not None:
                from deploy.inference.adapter import MAX_EMBEDDING_RESPONSE, validate_response
                limit = MAX_EMBEDDING_RESPONSE
            upstream = urllib.request.Request(
                inference_url, data=body, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(upstream, timeout=60) as response:
                    result = response.read(limit + 1)
                if len(result) > limit:
                    return error(502, "Ответ модели слишком большой.")
                decoded = json.loads(result)
                if digest is not None:
                    try:
                        validate_response(decoded, digest)
                    except (TypeError, ValueError, OverflowError):
                        return error(502, "Модель вернула некорректные эмбеддинги.")
                return JSONResponse(decoded)
            except urllib.error.HTTPError as exc:
                with exc:
                    if exc.code in {400, 413, 422}:
                        return error(exc.code, "Входные данные не подходят для модели.")
                    if exc.code == 502 and digest is not None:
                        return error(502, "Модель вернула некорректные эмбеддинги.")
            except (OSError, ValueError):
                pass
            return error(503, "Модель временно недоступна.")

        @app.post("/api/inference")
        @app.post("/api/inference/embeddings")
        async def inference(request: Request):
            owner = await run_in_threadpool(user, request)
            if not owner:
                return error(401, "Войдите, чтобы запустить модель.")
            if request.headers.get("content-type", "").split(";")[0] != "application/json":
                return error(415, "Ожидается JSON.")
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > MAX_BODY:
                    return error(413, "Входные данные слишком большие.")
                body.extend(chunk)
            digest = None
            if request.url.path.endswith("/embeddings"):
                if engine is None:
                    return error(503, "Хранилище эмбеддингов недоступно.")
                from deploy.inference.adapter import csv_digest, input_csv
                try:
                    digest = csv_digest(input_csv(json.loads(body)))
                except (ValueError, RecursionError):
                    return error(422, "Ожидается CSV до 3 МБ.")
            if engine is None:
                return await run_in_threadpool(run_inference, bytes(body))

            def run_job():
                job = start_job(engine, owner)
                status = "failed"
                result = None
                try:
                    response = run_inference(bytes(body), digest)
                    status = "succeeded" if response.status_code < 400 else "failed"
                    if digest is not None and status == "succeeded":
                        result = json.loads(response.body)
                    response.headers["X-PT-Job-ID"] = job
                    if result is not None:
                        return JSONResponse({**result, "jobId": job}, headers={"X-PT-Job-ID": job})
                    return response
                finally:
                    finish_job(engine, job, status, result)

            return await run_in_threadpool(run_job)

    for prefix in sorted(STATIC_DIRS):
        app.mount("/" + prefix, StaticFiles(directory=SITE / prefix))

    @app.api_route("/{asset:path}", methods=["GET", "HEAD"])
    def serve_asset(asset: str):
        route = "/" + asset
        redirects = {
            "/lab": "/lab/", "/lab/index.html": "/lab/", "/lab/workspace": "/lab/", "/lab/workspace/": "/lab/",
            "/model": "/model/", "/work": "/lab/", "/work/": "/lab/", "/contact": "/model/", "/contact/": "/model/",
            "/product": "/product/", "/demo": "/demo/",
        }
        if route in redirects:
            return RedirectResponse(redirects[route], 308)
        if route == "/demo/":
            return FileResponse(SITE / "lab/index.html")
        if route == "/product/":
            return FileResponse(SITE / "product/index.html")
        file = PAGES.get(route) or FILES.get(route)
        return FileResponse(file) if file else error(404, "Страница не найдена.")

    return app


def from_environment() -> FastAPI:
    database_file = os.getenv("PT_DATABASE_URL_FILE")
    engine = None
    if database_file:
        from .database import connect_database
        engine = connect_database(Path(database_file).read_text().strip())
    token_file = os.getenv("PT_METRICS_TOKEN_FILE")
    return create_app(
        Path(os.environ["PT_DATA_DIR"]), os.environ["PT_PUBLIC_ORIGIN"], os.getenv("PT_INFERENCE_URL"),
        engine=engine, allow_registration=os.getenv("PT_ALLOW_REGISTRATION", "1") == "1",
        metrics_token=Path(token_file).read_text().strip() if token_file else None,
    )
