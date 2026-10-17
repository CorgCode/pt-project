"""Проверка сайта в настоящем headless Chromium, раздаётся из docs/ так же, как публикует Pages.

На 1440 / 768 / 390 px: нет ошибок страницы и горизонтальной прокрутки, все графики нарисованы при первой загрузке,
подписи источника и графовые методы видны, вкладки метрик перерисовывают. --shots DIR — ещё и снимки страниц.
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
CHARTS = ["leaderboard", "chartMetric"]


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
            for needle in ("LANL · ДЕНЬ 02 · ИСТОЧНИК ЗАЯВЛЕН", "node2vec", "Какой метод лучше"):
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
            before = page.inner_html("#chartMetric")
            page.click("button[data-metric=links]")
            if page.inner_html("#chartMetric") == before:
                problems.append(f"{w}px: metric tabs did not change chartMetric")
            page.click("button[data-metric=neighbors]")
            page.goto(base + "/index.html", wait_until="networkidle")
            page.keyboard.press("Tab")
            if page.evaluate("document.activeElement.className") != "skip":
                problems.append(f"{w}px: skip link not first focus")
            if shots:
                page.screenshot(path=str(shots / f"index_{w}.png"), full_page=True)
            page.close()

        browser.close()
    server.shutdown()
    if problems:
        raise SystemExit("Browser check FAILED:\n- " + "\n- ".join(problems))
    print("OK: графики, подписи источника и вкладки метрик на 1440/768/390 px.")


if __name__ == "__main__":
    main()
