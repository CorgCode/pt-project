"""Local-only HTTP UI and SQLite workspace storage. Python standard library only."""
import argparse
import json
import mimetypes
import sqlite3
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
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
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS workspaces (id TEXT PRIMARY KEY, name TEXT NOT NULL, payload TEXT NOT NULL, updated TEXT NOT NULL)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def list(self):
        with self.connect() as db:
            return [dict(zip(("id", "name", "updated"), row)) for row in db.execute(
                "SELECT id,name,updated FROM workspaces ORDER BY updated DESC")]

    def get(self, key):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM workspaces WHERE id=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, payload, key=None):
        payload = validate(payload)
        key = key or str(uuid.uuid4())
        with self.connect() as db:
            db.execute("INSERT INTO workspaces VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,payload=excluded.payload,updated=excluded.updated",
                       (key, payload["name"], json.dumps(payload, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
        return key


def make_server(port=8846, data_dir=None):
    store = Store(Path(data_dir or Path.home() / ".corgcode") / "workspaces.sqlite3")

    class Handler(BaseHTTPRequestHandler):
        def trusted(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in allowed and (
                not origin or origin in {"http://" + host for host in allowed})

        def send(self, status, data, content_type="application/json; charset=utf-8"):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.trusted():
                return self.send(403, {"error": "Доступен только локальный интерфейс."})
            route = urlsplit(self.path).path
            if route == "/api/workspaces":
                return self.send(200, store.list())
            if route.startswith("/api/workspaces/"):
                value = store.get(route.rsplit("/", 1)[-1])
                return self.send(200 if value else 404, value or {"error": "Набор не найден."})
            if route == "/api/health":
                return self.send(200, {"status": "ok", "product": "CorgCode", "storage": "local", "workspaceSchema": 2})
            files = {"/": STATIC / "index.html", "/app.css": STATIC / "app.css", "/app.js": STATIC / "app.js",
                     "/workflow.js": STATIC / "workflow.js", **ASSETS}
            file = files.get(route)
            if file and file.is_file():
                return self.send(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            return self.send(404, {"error": "Страница не найдена."})

        def do_POST(self):
            self.write_workspace(False)

        def do_PUT(self):
            self.write_workspace(True)

        def write_workspace(self, update):
            if not self.trusted():
                return self.send(403, {"error": "Запрос должен исходить из локального интерфейса."})
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


def main():
    parser = argparse.ArgumentParser(description="CorgCode — локальное рабочее место SOC")
    parser.add_argument("--port", type=int, default=8846)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    server = make_server(args.port, args.data_dir)
    print(f"CorgCode: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
