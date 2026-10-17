"""Статическая проверка сайта GitHub Pages (Python и Node.js только для синтаксиса JS).

Проверяет: страницы и файлы на месте; интерфейс по-русски; нет сторонних скриптов, стилей и шрифтов;
синтетика подписана; все графики и заглушки на месте; внутренние ссылки работают; JS разбирается;
файл данных синтетический и согласован. Визуальную проверку в браузере не заменяет (docs/browser_check.py).
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "synthetic_demo.json"
PAGES = ["index.html", "playground.html"]
GRAPH_METHODS = {"deepwalk", "node2vec", "graphsage", "rolx", "struc2vec"}
CHARTS = ["chartConsistency", "chartLinks", "chartPairs", "chartIdentity", "chartHosts", "leaderboard"]
SECTIONS = ["overview", "readiness", "methods", "baselines", "results", "team", "architecture", "next"]


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.external_assets: list[str] = []
        self.h1 = 0

    def handle_starttag(self, tag, attrs) -> None:
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "h1":
            self.h1 += 1
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])
        for key in ("src", "href"):
            v = a.get(key, "")
            if tag in ("script", "link", "img") and v.startswith(("http:", "https:", "//")):
                self.external_assets.append(v)


def check() -> None:
    info = json.loads(DATA.read_text(encoding="utf-8"))
    meta = info["meta"]
    assert meta["synthetic"] is True and meta["hosts"] == 300
    assert set(meta["missing"]) == GRAPH_METHODS
    assert "LANL" not in set(meta["methods"]) and "reference_ceiling" in meta["methods"]

    pages = {}
    for name in PAGES:
        text = (ROOT / name).read_text(encoding="utf-8")
        p = Page()
        p.feed(text)
        pages[name] = (text, p)
        assert '<html lang="ru">' in text, name
        assert p.h1 == 1, f"{name}: exactly one <h1>"
        assert not p.external_assets, f"{name}: third-party assets {p.external_assets}"
        assert "<title>" in text and 'name="viewport"' in text
        assert "LANL" in text and ("синтет" in text.lower() or "учебн" in text.lower()), name

    idx, ip = pages["index.html"]
    for sid in SECTIONS + CHARTS:
        assert sid in ip.ids, f"index.html: missing #{sid}"
    assert idx.count("СИНТЕТИКА — НЕ РЕАЛЬНЫЕ ДАННЫЕ LANL") >= 6, "every chart needs a provenance label"
    for needle in ("Neighborhood Consistency@k", "результаты эксперимента",
                   "Чего вывести нельзя", "Что измеряется", "Как читать", "pull/62", "pull/64", "pull/65", "issues/60"):
        assert needle in idx, needle
    assert idx.count("Чего вывести нельзя") >= 5
    js = (ROOT / "assets" / "site.js").read_text(encoding="utf-8")
    assert "Ожидаются результаты эксперимента" in js
    assert 'fetch("./data/synthetic_demo.json"' in js
    # графовый метод не должен появляться с числом в файле данных
    for m in GRAPH_METHODS:
        assert m not in info["results"]["test"] and m not in info["identity"], m

    # внутренние ссылки работают
    for name, (_, p) in pages.items():
        for href in p.links:
            if href.startswith(("http:", "https:", "mailto:")):
                continue
            target, _, frag = href.partition("#")
            if target in ("", "./"):
                tp = pages["index.html"][1] if target == "./" else p
                path_ok = True
            else:
                path_ok = (ROOT / target).exists()
                tp = pages[target][1] if target in pages else None
            assert path_ok, f"{name}: broken link {href}"
            if frag and tp is not None:
                assert frag in tp.ids, f"{name}: broken anchor {href}"

    points = info["projection"]["points"]
    assert len(points) == meta["hosts"] and len({p["id"] for p in points}) == len(points)
    assert all(math.isfinite(p["x"]) and math.isfinite(p["y"]) and 0 <= p["agreement"] <= 1 for p in points)
    avg = sum(p["agreement"] for p in points) / len(points)
    assert abs(avg - info["results"]["test"]["behavioral"]["consistency@10_mean"]) < 1e-10
    assert info["results"]["test"]["reference_ceiling"]["consistency@10_mean"] == 1
    assert info["results"]["test"]["random"]["consistency@10_mean"] < 0.1
    for split in ("test", "validation"):
        assert set(info["results"][split]) == set(meta["methods"])
        for method, scores in info["results"][split].items():
            assert all(math.isfinite(v) for v in scores.values()), (split, method)

    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js required to syntax-check GitHub Pages JavaScript")
    for f in sorted((ROOT / "assets").glob("*.js")):
        subprocess.run([node, "--check", str(f)], check=True)
    assert not re.search(r"cdn\.|plot\.ly|jsdelivr|googleapis", idx + (ROOT / "playground.html").read_text(encoding="utf-8") + js)
    print("OK: 2 pages, no third-party assets, links/anchors resolve, charts+provenance present, data consistent.")


if __name__ == "__main__":
    check()
