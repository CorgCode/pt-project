"""Local-only HTTP UI and SQLite workspace storage. Python standard library only."""
import argparse
import json
import mimetypes
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from getpass import getpass
from http.cookies import CookieError, SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .accounts import Accounts, COOKIE, SESSION_TTL

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
SITE = ROOT / "noomo-ui"
WORKSPACE = "/lab/workspace/"
LOGIN = "/lab/login/"
LOGOUT = "/lab/logout/"
MAX_BODY = 8 * 1024 * 1024
ASSETS = {
    "/assets/core.js": ROOT / "noomo-ui/lab/core.js",
    "/assets/onest.ttf": ROOT / "noomo-ui/fonts/onest-400.ttf",
    "/assets/onest-bold.ttf": ROOT / "noomo-ui/fonts/onest-600.ttf",
    "/assets/display.otf": ROOT / "noomo-ui/model/reference/NeueMachina-Regular.otf",
}


def validate(payload):
    if not isinstance(payload, dict):
        raise ValueError("Ожидается описание рабочего набора.")
    name = payload.get("name", "")
    if not isinstance(name, str) or not name.strip() or len(name) > 120:
        raise ValueError("Название должно содержать от 1 до 120 символов.")
    for key in ("csv", "baseline"):
        value = payload.get(key, "")
        if not isinstance(value, str) or len(value.encode("utf-8")) > 3 * 1024 * 1024:
            raise ValueError("Каждый CSV должен быть не больше 3 МБ.")
    if not payload.get("csv", "").strip():
        raise ValueError("Нужен CSV текущего окна.")
    notes = payload.get("notes", {})
    if not isinstance(notes, dict) or len(notes) > 500 or any(
        not isinstance(k, str) or not isinstance(v, str) or len(k) > 500 or len(v) > 10000
        for k, v in notes.items()
    ):
        raise ValueError("Заметки: до 500 устройств и 10 000 символов на устройство.")
    annotations = payload.get("annotations", {})
    if not isinstance(annotations, dict) or len(annotations) > 500:
        raise ValueError("Метки можно задать максимум для 500 устройств.")
    clean_annotations = {}
    for host, item in annotations.items():
        if not isinstance(host, str) or len(host) > 500 or not isinstance(item, dict):
            raise ValueError("Некорректное описание устройства.")
        label, tags = item.get("label", ""), item.get("tags", [])
        status, watch = item.get("status", "new"), item.get("watch", False)
        if not isinstance(label, str) or len(label) > 80:
            raise ValueError("Подпись устройства — до 80 символов.")
        if status not in ("new", "review", "done") or not isinstance(watch, bool):
            raise ValueError("Неизвестный статус или отметка наблюдения.")
        if not isinstance(tags, list) or len(tags) > 8 or any(
            not isinstance(tag, str) or not tag.strip() or len(tag) > 32 for tag in tags
        ):
            raise ValueError("До 8 меток, каждая от 1 до 32 символов.")
        clean_annotations[host] = {"label": label.strip(), "tags": list(dict.fromkeys(tag.strip() for tag in tags)),
                                   "status": status, "watch": watch}
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("Некорректные настройки набора.")
    group_count = settings.get("groupCount", 3)
    if type(group_count) is not int or not 2 <= group_count <= 6:
        raise ValueError("Число групп — от 2 до 6.")
    return {"name": name.strip(), "csv": payload["csv"],
            "baseline": payload.get("baseline", ""), "notes": notes,
            "sample": payload.get("sample") is True, "annotations": clean_annotations,
            "settings": {"groupCount": group_count}}


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS workspaces (id TEXT PRIMARY KEY, name TEXT NOT NULL, payload TEXT NOT NULL, updated TEXT NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def list(self):
        with closing(self.connect()) as db, db:
            return [dict(zip(("id", "name", "updated"), row)) for row in db.execute(
                "SELECT id,name,updated FROM workspaces ORDER BY updated DESC")]

    def get(self, key):
        with closing(self.connect()) as db, db:
            row = db.execute("SELECT payload FROM workspaces WHERE id=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, payload, key=None):
        payload = validate(payload)
        key = key or str(uuid.uuid4())
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO workspaces VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,payload=excluded.payload,updated=excluded.updated",
                       (key, payload["name"], json.dumps(payload, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
        return key


def make_server(port=8846, data_dir=None, *, site=False):
    directory = Path(data_dir or Path.home() / ".corgcode")
    accounts = Accounts(directory) if site else None
    local_store = None if site else Store(directory / "workspaces.sqlite3")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=SITE, **kwargs)

        def list_directory(self, path):
            self.send_error(404)
            return None

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def trusted(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in allowed and (
                not origin or origin in {"http://" + host for host in allowed})

        def send(self, status, data, content_type="application/json; charset=utf-8", headers=()):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def redirect(self, target, cookie=None):
            headers = [("Location", target)]
            if cookie is not None:
                age = SESSION_TTL if cookie else 0
                headers.append(("Set-Cookie", f"{COOKIE}={cookie}; Path=/; Max-Age={age}; HttpOnly; SameSite=Strict"))
            return self.send(303, b"", "text/plain", headers)

        def token(self):
            try:
                item = SimpleCookie(self.headers.get("Cookie", "")).get(COOKIE)
                return item.value if item else None
            except CookieError:
                return None

        def workspace_store(self):
            if accounts is None:
                return local_store
            user = accounts.identify(self.token())
            return Store(directory / "users" / user / "workspaces.sqlite3") if user else None

        def login_page(self, status=200, *, invalid=False):
            page = (STATIC / "login.html").read_text(encoding="utf-8")
            message = '<p class="login-error" role="alert">Проверьте имя пользователя и пароль.</p>' if invalid else ""
            if not accounts.has_users():
                page = page.replace('class="login-setup" hidden', 'class="login-setup"')
            return self.send(status, page.replace("<!-- login-error -->", message).encode(), "text/html; charset=utf-8")

        def authenticate(self, route):
            if not self.trusted() or not self.headers.get("Origin"):
                return self.send(403, {"error": "Откройте форму входа на сайте."})
            if route == LOGOUT:
                accounts.logout(self.token())
                return self.redirect(LOGIN, "")
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096 or self.headers.get_content_type() != "application/x-www-form-urlencoded":
                    return self.login_page(400, invalid=True)
                fields = parse_qs(self.rfile.read(size).decode(), max_num_fields=4)
            except ValueError:
                return self.login_page(400, invalid=True)
            token = accounts.login(fields.get("name", [""])[0], fields.get("password", [""])[0])
            if not token:
                return self.login_page(401, invalid=True)
            accounts.logout(self.token())
            return self.redirect(WORKSPACE, token)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            if not self.trusted():
                return self.send(403, {"error": "Доступен только локальный интерфейс."})
            route = urlsplit(self.path).path
            if route == "/api/workspaces" or route.startswith("/api/workspaces/"):
                store = self.workspace_store()
                if store is None:
                    return self.send(401, {"error": "Войдите в лабораторию снова."})
                if route == "/api/workspaces":
                    return self.send(200, store.list())
                value = store.get(route.rsplit("/", 1)[-1])
                return self.send(200 if value else 404, value or {"error": "Набор не найден."})
            if route == "/api/health":
                return self.send(200, {"status": "ok", "product": "CorgCode", "storage": "local", "workspaceSchema": 2})
            if site and route in (LOGIN.rstrip("/"), WORKSPACE.rstrip("/")):
                return self.redirect(route + "/")
            if site and route == LOGIN:
                return self.redirect(WORKSPACE) if accounts.identify(self.token()) else self.login_page()
            if site and route == WORKSPACE:
                if not accounts.identify(self.token()):
                    return self.redirect(LOGIN)
                page = (STATIC / "index.html").read_text(encoding="utf-8")
                navigation = '<div class="site-navigation"><a class="quiet" href="/lab/">О лаборатории</a><form method="post" action="/lab/logout/"><button class="quiet" type="submit">Выйти</button></form></div>'
                reauth = '<p id="session-expired" class="panel" role="status" hidden>Сессия истекла. <a href="/lab/login/" target="_blank" rel="noopener">Войдите в новой вкладке</a>, затем вернитесь сюда и повторите сохранение.</p>'
                page = page.replace("<!-- site-navigation -->", navigation).replace("<!-- session-expired -->", reauth)
                return self.send(200, page.encode(), "text/html; charset=utf-8")
            files = {"/app.css": STATIC / "app.css", "/app.js": STATIC / "app.js",
                     "/workflow.js": STATIC / "workflow.js", **ASSETS}
            if not site:
                files["/"] = STATIC / "index.html"
            file = files.get(route)
            if file and file.is_file():
                return self.send(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            if site and not route.startswith("/api/"):
                path = Path(self.translate_path(self.path)).resolve()
                if not path.is_relative_to(SITE) or any(
                    part.startswith(".") or part in {"node_modules", "tools", "__pycache__"}
                    for part in path.relative_to(SITE).parts
                ):
                    return self.send_error(404)
                return super().do_HEAD() if self.command == "HEAD" else super().do_GET()
            return self.send(404, {"error": "Страница не найдена."})

        def do_POST(self):
            route = urlsplit(self.path).path
            if site and route in (LOGIN, LOGOUT):
                return self.authenticate(route)
            self.write_workspace(False)

        def do_PUT(self):
            self.write_workspace(True)

        def write_workspace(self, update):
            if not self.trusted():
                return self.send(403, {"error": "Запрос должен исходить из локального интерфейса."})
            store = self.workspace_store()
            if store is None:
                return self.send(401, {"error": "Войдите в лабораторию снова."})
            route = urlsplit(self.path).path
            key = route.rsplit("/", 1)[-1] if update else None
            if (not update and route != "/api/workspaces") or (update and (not route.startswith("/api/workspaces/") or store.get(key) is None)):
                return self.send(404, {"error": "Набор не найден."})
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Ожидается JSON.")
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > MAX_BODY:
                    return self.send(413, {"error": "Набор слишком большой."})
                payload = json.loads(self.rfile.read(size))
                key = store.save(payload, key)
            except (ValueError, UnicodeError) as error:
                return self.send(400, {"error": str(error)})
            return self.send(200 if update else 201, {"id": key})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(*, site=False):
    parser = argparse.ArgumentParser(description="CorgCode — локальное рабочее место SOC")
    parser.add_argument("--port", type=int, default=8845 if site else 8846)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--add-user", metavar="NAME")
    args = parser.parse_args()
    if args.add_user:
        password = getpass("Пароль: ")
        if password != getpass("Повторите пароль: "):
            parser.error("Пароли различаются.")
        try:
            Accounts(args.data_dir or Path.home() / ".corgcode").add(args.add_user, password)
        except ValueError as error:
            parser.error(str(error))
        print("Учётная запись создана.")
        return
    server = make_server(args.port, args.data_dir, site=site)
    print(f"CorgCode: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
