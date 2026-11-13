"""Local preview without stale JavaScript or CSS. Run: python3 serve.py."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class PreviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).resolve().parent), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


if __name__ == "__main__":
    print("PT Network: http://127.0.0.1:8845/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8845), PreviewHandler).serve_forever()
