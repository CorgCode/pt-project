import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.operator.server import Store, make_server, validate


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.server = make_server(0, self.directory.name)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def request(self, route, data=None, method=None, headers=None):
        request = Request(self.base + route, data=json.dumps(data).encode() if data is not None else None,
                          method=method, headers={"Content-Type": "application/json", **(headers or {})})
        with urlopen(request) as response:
            return response.status, response.read(), response.headers

    def test_save_reopen_update_and_persistence(self):
        body = {"name": "Сеть", "csv": "src,dst\na,b", "notes": {"a": "Проверить связь"}, "annotations": {"a": {"status": "review", "label": "Проверить сервер", "watch": True, "tags": ["инфра"]}}, "settings": {"groupCount": 4}}
        code, raw, _ = self.request("/api/workspaces", body)
        self.assertEqual(code, 201)
        key = json.loads(raw)["id"]
        saved = json.loads(self.request("/api/workspaces/" + key)[1])
        self.assertEqual(saved["notes"], body["notes"])
        self.assertEqual(saved["annotations"], body["annotations"])
        self.assertEqual(saved["settings"], body["settings"])
        body["name"] = "Новое имя"
        self.request("/api/workspaces/" + key, body, "PUT")
        self.assertEqual(Store(Path(self.directory.name) / "workspaces.sqlite3").get(key)["name"], "Новое имя")
        self.assertEqual(len(json.loads(self.request("/api/workspaces")[1])), 1)

    def test_legacy_and_invalid_annotations(self):
        base = {"name": "Сеть", "csv": "src,dst\na,b"}
        self.assertEqual(validate(base)["annotations"], {})
        self.assertEqual(validate(base)["settings"], {"groupCount": 3})
        for item in ({"status": "attack"}, {"watch": "true"}, {"tags": ["x"] * 9},
                     {"label": "x" * 81}, {"tags": ["x" * 33]}, {"tags": [""]}):
            with self.assertRaises(ValueError):
                validate({**base, "annotations": {"a": item}})
        for count in (True, 0, 7, "3"):
            with self.assertRaises(ValueError):
                validate({**base, "settings": {"groupCount": count}})
        self.assertEqual(validate({**base, "annotations": {"__proto__": {"watch": True}}})["annotations"]["__proto__"]["watch"], True)

    def test_private_server_paths_and_origin(self):
        for route in ("/../README.md", "/assets/../../.git/config", "/api/workspaces/missing"):
            with self.assertRaises(HTTPError) as caught:
                self.request(route)
            self.assertEqual(caught.exception.code, 404)
            caught.exception.close()
        for headers in ({"Host": "external.test"}, {"Origin": "https://external.test"}):
            with self.assertRaises(HTTPError) as caught:
                self.request("/api/workspaces", headers=headers)
            self.assertEqual(caught.exception.code, 403)
            caught.exception.close()

    def test_invalid_payload_and_assets(self):
        for body in ({"name": "", "csv": "x"}, {"name": "x", "csv": ""}, {"name": "x", "csv": "x", "notes": {"a": []}}, []):
            with self.assertRaises(HTTPError) as caught:
                self.request("/api/workspaces", body)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        for asset in ("/", "/app.js", "/app.css", "/assets/core.js", "/workflow.js", "/assets/onest.ttf", "/assets/display.otf"):
            code, raw, headers = self.request(asset)
            self.assertEqual(code, 200)
            self.assertTrue(raw)
            self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(json.loads(self.request("/api/health")[1])["product"], "CorgCode")


if __name__ == "__main__":
    unittest.main()
