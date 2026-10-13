"""Real-browser check of the static site (headless Chromium), served from docs/ exactly as the
Pages workflow publishes it. Never starts Streamlit and never needs an upload for the main page.

Verifies at 1440 / 768 / 390 px: no page errors, no horizontal overflow, all charts drawn on first
load, provenance + pending placeholders visible, split toggle redraws, playground accepts a valid CSV
and rejects invalid ones. Pass --shots DIR to also save full-page screenshots.
"""
from __future__ import annotations

import argparse
import tempfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
SIZES = ((1440, 900), (768, 1024), (390, 844))
CHARTS = ["chartConsistency", "chartLinks", "chartPairs", "chartIdentity", "chartHosts", "leaderboard"]


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *_a) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="")
    args = ap.parse_args()
    shots = Path(args.shots) if args.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(ROOT)))
    Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    problems: list[str] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        for w, h in SIZES:
            page = browser.new_page(viewport={"width": w, "height": h})
            page.on("pageerror", lambda e, w=w: problems.append(f"{w}px pageerror: {e}"))
            page.on("console", lambda m, w=w: m.type == "error" and problems.append(f"{w}px console: {m.text}"))
            page.on("requestfailed", lambda r, w=w: problems.append(f"{w}px request failed: {r.url}"))
            page.goto(base + "/index.html", wait_until="networkidle", timeout=30000)
            page.wait_for_selector("html[data-charts=ready]", timeout=10000)
            for cid in CHARTS:
                n = page.eval_on_selector(f"#{cid}", "e => e.querySelectorAll('svg, table').length")
                if n < 1:
                    problems.append(f"{w}px: #{cid} not rendered")
            sw = page.evaluate("document.documentElement.scrollWidth")
            if sw > w:
                problems.append(f"{w}px: horizontal overflow (scrollWidth {sw})")
            body = page.inner_text("body")
            for needle in ("СИНТЕТИКА — НЕ РЕАЛЬНЫЕ ДАННЫЕ LANL", "Ожидаются результаты эксперимента", "Чего вывести нельзя"):
                if needle not in body:
                    problems.append(f"{w}px: missing text {needle!r}")
            if page.locator("svg text").count() < 20:
                problems.append(f"{w}px: chart labels missing")
            clipped = page.evaluate("""() => [...document.querySelectorAll('.fig svg')].flatMap(s => {
                const r = s.getBoundingClientRect();
                return [...s.querySelectorAll('text')].filter(t => { const b = t.getBoundingClientRect();
                    return b.width && (b.right > r.right + 1 || b.left < r.left - 1); }).map(t => t.textContent); })""")
            if clipped:
                problems.append(f"{w}px: clipped chart labels {clipped[:3]}")
            before = page.inner_text("#leaderboard")
            page.click("button[data-split=validation]")
            if page.inner_text("#leaderboard") == before:
                problems.append(f"{w}px: split toggle did not change leaderboard")
            page.click("button[data-split=test]")
            page.goto(base + "/index.html", wait_until="networkidle")
            page.keyboard.press("Tab")
            if page.evaluate("document.activeElement.className") != "skip":
                problems.append(f"{w}px: skip link not first focus")
            if shots:
                page.screenshot(path=str(shots / f"index_{w}.png"), full_page=True)
            page.close()

        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.on("pageerror", lambda e: problems.append(f"playground pageerror: {e}"))
        page.goto(base + "/playground.html", wait_until="networkidle")
        with page.expect_download() as dl:
            page.click("#sample")
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "embeddings.csv"
            dl.value.save_as(good)
            page.set_input_files("#file", str(good))
            page.click("#run")
            page.wait_for_selector("#results:not([hidden])", timeout=60000)
            if "300 хостов" not in page.inner_text("#status"):
                problems.append("playground: valid CSV not accepted")
            if page.evaluate("document.documentElement.scrollWidth") > 390:
                problems.append("playground: horizontal overflow at 390px")
            if shots:
                page.screenshot(path=str(shots / "playground_390.png"), full_page=True)
            for label, content, expect in (
                ("dup", "host_id,emb_0\n" + "h0,1\n" * 300, "повторяющиеся"),
                ("nan", "host_id,emb_0\n" + "".join(f"h{i},{'x' if i == 3 else 1}\n" for i in range(300)), "нечисловые"),
                ("noid", "id,emb_0\nh0,1\n", "host_id"),
            ):
                bad = Path(tmp) / f"{label}.csv"
                bad.write_text(content, encoding="utf-8")
                page.set_input_files("#file", str(bad))
                page.click("#run")
                page.wait_for_function("document.getElementById('status').textContent.startsWith('Ошибка')", timeout=20000)
                if expect not in page.inner_text("#status") or not page.is_hidden("#results"):
                    problems.append(f"playground: bad CSV '{label}' not rejected properly")
        page.close()
        browser.close()
    server.shutdown()
    if problems:
        raise SystemExit("Browser check FAILED:\n- " + "\n- ".join(problems))
    print("OK: index charts/provenance/pending/toggle at 1440/768/390; playground accepts valid and rejects invalid CSV.")


if __name__ == "__main__":
    main()
