"""Проверка дашборда в НАСТОЯЩЕМ headless Chromium — то, что увидит пользователь.

    pip install playwright && playwright install chromium
    python -m app.browser_check --out app/screenshots [--bundle demo/synthetic_run]

На ширине 1440 и 390 px открывает каждую вкладку, сохраняет снимок, собирает ошибки консоли и исключения
Streamlit, проверяет горизонтальную прокрутку и сверяет числа на странице с файлами результатов.
Код выхода 1 при любой ошибке. В CI не входит (нужен браузер).
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TABS = ["Обзор", "Таблица", "Прогноз связей", "Соседи", "Во времени", "Аудит",
        "Происхождение", "Свой эмбеддинг"]
VIEWPORTS = {"desktop": (1440, 1900), "phone": (390, 3200)}


LONG = "graphsage_unsupervised_v2_final_FINAL_with_a_very_long_descriptive_method_name"


def stress_bundle(src: Path, dst: Path, scenario: str) -> Path:
    """Скопировать прогон и испортить: очень длинное имя метода или пропавшие файлы."""
    import shutil
    shutil.copytree(src, dst)
    run = dst / "run"
    if scenario == "long_name":
        old = "behavioral_pca"
        for f in sorted(run.rglob(f"*{old}*"), key=lambda p: len(p.parts), reverse=True):   # сначала самые глубокие
            f.rename(f.with_name(f.name.replace(old, LONG)))
        for f in run.rglob("[!.]*.json"):
            f.write_text(f.read_text().replace(old, LONG))
        for f in run.glob("*.csv"):
            f.write_text(f.read_text().replace(old, LONG))
    elif scenario == "missing":
        (run / "audit.json").unlink()
        shutil.rmtree(run / "embeddings")
        shutil.rmtree(run / "identity")
        shutil.rmtree(run / "results" / "validation")
        (run / "scoreboard_validation.csv").unlink()
        for t in ("test", "validation"):
            f = dst / "reference" / t / "neighbors.csv"
            f.unlink(missing_ok=True)
    return dst


EXPECT_MISSING = {"Соседи": "виды по хостам недоступны", "Аудит": "нет audit.json"}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(page, timeout=60_000):
    page.wait_for_selector('[data-testid="stApp"]', timeout=timeout)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=timeout)
    page.wait_for_timeout(1200)


def check(page, name, problems, errors, phone):
    exc = page.locator('[data-testid="stException"]').count()
    if exc:
        problems.append(f"{name}: {exc} Streamlit exception box(es): {page.locator('[data-testid=\"stException\"]').first.inner_text()[:300]}")
    over = page.evaluate("""() => { const e = document.querySelector('[data-testid="stMain"]') || document.scrollingElement;
        return [e.scrollWidth, e.clientWidth]; }""")
    if over[0] > over[1] + 4:
        problems.append(f"{name}: horizontal overflow ({over[0]} > {over[1]} px)")
    txt = page.inner_text("body")
    if re.search(r"\bnan\b|\bNaN\b|undefined|Traceback", txt):
        problems.append(f"{name}: suspicious text (nan/undefined/Traceback) in the page")


def run(bundle: Path, out: Path, shots_for: set[str], *, full_checks: bool = True, expect: dict | None = None,
        tag: str = "") -> dict:
    from playwright.sync_api import sync_playwright
    out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    env = {"PT_RESULTS_DIR": str(bundle), **__import__("os").environ}
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(REPO / "app" / "streamlit_app.py"),
                               "--server.headless", "true", "--server.port", str(port),
                               "--browser.gatherUsageStats", "false"], cwd=REPO, env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report = {"versions": {}, "problems": [], "console_errors": [], "checks": {}, "screenshots": []}
    try:
        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.5)
        results = (json.loads((bundle / "run" / "results" / "test" / "behavioral.json").read_text())["metrics"]
                   if full_checks else {})
        with sync_playwright() as p:
            browser = p.chromium.launch()
            report["versions"] = {"chromium": browser.version, "playwright": __import__("importlib.metadata", fromlist=["version"]).version("playwright")}
            for vp, (w, h) in VIEWPORTS.items():
                ctx = browser.new_context(viewport={"width": w, "height": h})
                page = ctx.new_page()
                page.on("console", lambda m: report["console_errors"].append(f"[{vp}] {m.text}") if m.type == "error" else None)
                page.on("pageerror", lambda e: report["console_errors"].append(f"[{vp}] pageerror: {e}"))
                page.goto(f"http://127.0.0.1:{port}")
                wait_ready(page)
                for tab in TABS:
                    page.get_by_role("tab", name=tab, exact=True).click()
                    wait_ready(page)
                    if tab == "Соседи" and full_checks:
                        page.locator(".js-plotly-plot:visible").first.wait_for(timeout=30_000)
                    if expect and tab in expect and expect[tab] not in page.inner_text("body"):
                        report["problems"].append(f"{vp}/{tab}: expected message {expect[tab]!r} not shown")
                    check(page, f"{vp}/{tab}", report["problems"], report["console_errors"], vp == "phone")
                    if tab in ("Обзор", "Таблица", "Прогноз связей", "Соседи", "Во времени",
                               "Аудит", "Происхождение") and (vp, tab) in shots_for:
                        f = out / f"{tag}{vp}_{tab.lower().replace(' ', '_')}.png"
                        page.screenshot(path=str(f))
                        report["screenshots"].append(f.name)
                    if tab == "Обзор":
                        body = page.inner_text("body")
                        for needle in ("СИНТЕТИКА", "не трафик LANL"):
                            if needle not in body:
                                report["problems"].append(f"{vp}/Overview: banner text {needle!r} missing")
                    if tab == "Таблица" and full_checks:
                        # таблица рисуется на canvas (не читается); подпись под ней — обычный текст DOM
                        honest = {m: json.loads((bundle / "run" / "results" / "test" / f"{m}.json").read_text())["metrics"]
                                  for m in json.loads((bundle / "run" / "manifest.json").read_text())["methods"]
                                  if m != "reference_ceiling"}
                        best_m, best_v = max(((m, x["consistency@10_mean"]) for m, x in honest.items()), key=lambda t: t[1])
                        body = page.inner_text("body")
                        found = re.search(r"Consistency@10: (\S+) ([0-9.]+)", body)
                        ok = bool(found) and found.group(1) == best_m and abs(float(found.group(2)) - best_v) < 0.0005
                        report["checks"][f"{vp}_leaderboard_best_matches_result_files"] = ok
                        if not ok:
                            report["problems"].append(f"{vp}/Leaderboard: best Consistency@10 in the page "
                                                      f"{found and found.groups()} != result files ({best_m} {best_v:.3f})")
                    if tab == "Соседи" and full_checks:
                        cap = page.inner_text("body")
                        m = re.search(r"Среднее совпадение соседей у behavioral при k=10: (\d\.\d+?)\.", cap)
                        ok = bool(m) and abs(float(m.group(1)) - results["consistency@10_mean"]) < 0.0005
                        report["checks"][f"{vp}_per_host_agreement_equals_result_file"] = ok
                        if not ok:
                            report["problems"].append(f"{vp}/Neighbourhoods: per-host mean agreement {m and m.group(1)} "
                                                      f"!= result file {results['consistency@10_mean']:.3f}")
                ctx.close()
            browser.close()
    finally:
        server.terminate()
    report["console_errors"] = sorted(set(report["console_errors"]))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=REPO / "demo" / "synthetic_run")
    ap.add_argument("--out", type=Path, default=REPO / "app" / "screenshots")
    ap.add_argument("--all-shots", action="store_true", help="screenshot every tab at both widths")
    ap.add_argument("--stress", action="store_true", help="also run a very-long-method-name and a missing-artifacts bundle")
    a = ap.parse_args()
    shots = {(v, t) for v in VIEWPORTS for t in TABS} if a.all_shots else {
        ("desktop", "Обзор"), ("desktop", "Таблица"), ("desktop", "Прогноз связей"),
        ("desktop", "Соседи"), ("phone", "Обзор"), ("phone", "Таблица")}
    rep = run(a.bundle, a.out, shots)
    if a.stress:
        import tempfile
        rep["stress"] = {}
        for scenario in ("long_name", "missing"):
            with tempfile.TemporaryDirectory() as tmp:
                b2 = stress_bundle(a.bundle, Path(tmp) / "b", scenario)
                r2 = run(b2, a.out, {("desktop", "Таблица"), ("phone", "Таблица"), ("desktop", "Соседи")},
                         full_checks=False, expect=EXPECT_MISSING if scenario == "missing" else None, tag=f"stress_{scenario}_")
            rep["stress"][scenario] = {"problems": r2["problems"], "console_errors": r2["console_errors"]}
            rep["problems"] += [f"[stress:{scenario}] {x}" for x in r2["problems"]]
    (a.out / "browser_check.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=2))
    return 1 if rep["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
