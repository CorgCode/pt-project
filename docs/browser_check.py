"""Automated visual smoke test of the static GitHub Pages site.

Launches a local HTTP server and headless Chromium; this deliberately never
starts Streamlit or uploads a CSV. The key requirement is that research charts
appear on first page load on both desktop and mobile.
"""
from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent  # emulate branch-based Jekyll deployment from repo root


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass


def main() -> None:
    handler = partial(QuietHandler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    problems = []
    try:
        with sync_playwright() as engine:
            browser = engine.chromium.launch(headless=True, args=["--no-sandbox"])
            try:
                for width, height in ((1440, 900), (390, 844)):
                    page = browser.new_page(viewport={"width": width, "height": height})
                    page.on("pageerror", lambda error: problems.append(str(error)))
                    page.goto(f"http://127.0.0.1:{server.server_port}/index.html",
                              wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_selector("#dashboardContents:not([hidden])",
                                           timeout=30000)
                    assert page.url.rstrip("/").endswith("/docs"), (
                        "Repo root must redirect to /docs/ and show the real dashboard: " + page.url
                    )
                    assert "snapshot loaded" in page.locator("#dashboardStatus").inner_text().lower()
                    assert page.locator("#chartConsistency svg").count() == 1
                    assert page.locator("#chartLinks svg").count() == 1
                    assert page.locator("#chartPairs svg").count() == 1
                    assert page.locator("#chartIdentity svg").count() == 1
                    assert page.locator("#chartHosts svg circle").count() == 300
                    assert page.locator("#snapshotBoard tbody tr").count() == 6
                    assert "synthetic" in page.locator("#researchDashboard").inner_text().lower()
                    assert "node2vec" in page.locator("#missingMethods").inner_text().lower()
                    assert page.locator("#results.hidden").count() == 1, (
                        "upload-playground results must remain hidden before upload"
                    )
                    scores = page.locator("#snapshotBoard tbody").inner_text()
                    assert "0.280" in scores, "reported behavioural score absent"
                    page.locator("#dashboardSplit").select_option("validation")
                    assert page.locator("#chartConsistency svg").count() == 1
                    assert page.locator("#snapshotBoard tbody tr").count() == 6
                    assert "0.284" in page.locator("#snapshotBoard tbody").inner_text()
                    if width == 390:
                        assert page.evaluate(
                            "document.documentElement.scrollWidth <= window.innerWidth + 2"
                        ), "Page introduces horizontal overflow at mobile width"
                    assert not problems, f"Page errors: {problems}"
                    page.close()
                print("OK: repo root redirects to /docs/; five plots + leaderboard render before upload; "
                      "test/validation toggle; desktop/mobile; no page errors.")
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
