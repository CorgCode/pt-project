"""Статическая проверка сайта GitHub Pages (Python и Node.js только для синтаксиса JS).

Проверяет: страницы и файлы на месте; интерфейс по-русски; нет сторонних скриптов, стилей и шрифтов;
источник данных подписан на каждом графике; внутренние ссылки работают; JS разбирается;
файл данных согласован. Визуальную проверку в браузере не заменяет (docs/browser_check.py).
"""
from __future__ import annotations

import json
import math
import posixpath
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "results.json"
LABEL = "LANL · ДЕНЬ 02 · ИСТОЧНИК ЗАЯВЛЕН"
PAGES = ["index.html", "report.html", "team.html", "scoreboard/index.html"]
GRAPH_METHODS = {"deepwalk", "node2vec", "graphsage", "rolx", "struc2vec"}
CHARTS = ["leaderboard", "chartMetric", "hostMap"]
SECTIONS = ["overview", "results", "metrics", "map", "methods", "status"]


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
    assert meta["kind"] == "lanl_declared" and meta["synthetic"] is False and meta["hosts"] == len(info["projection"]["points"])
    assert GRAPH_METHODS <= set(meta["methods"]) and not meta["missing"], "все пять графовых методов должны быть посчитаны"
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
        assert "LANL" in text, name

    idx, ip = pages["index.html"]
    for sid in SECTIONS + CHARTS:
        assert sid in ip.ids, f"index.html: missing #{sid}"
    assert idx.count(LABEL) >= len(CHARTS), "у каждого графика должна быть подпись источника данных"
    for needle in ("docs/build_data.py", "Scoreboard", "issues/60", "scoreboard/"):
        assert needle in idx, needle
    js = (ROOT / "assets" / "site.js").read_text(encoding="utf-8")
    for task in info["team"]["tasks"]:
        assert f'"{task}"' in js, f"задача {task} не показана на сайте"
    assert all(set(r["tasks"]) <= set(info["team"]["tasks"]) for r in info["team"]["rows"])
    assert 'fetch("./data/results.json"' in js
    rep = pages["report.html"][0]
    for r in info["team"]["rows"]:
        assert f"<code>{r['method']}</code>" in rep, f"report.html: нет метода {r['method']} — запустите docs/build_report.py"
    # у каждого графового метода есть числа и узнаваемость во времени
    for m in GRAPH_METHODS:
        assert m in info["results"]["test"] and m in info["identity"], m

    # внутренние ссылки работают
    for name, (_, p) in pages.items():
        for href in p.links:
            if href.startswith(("http:", "https:", "mailto:")):
                continue
            target, _, frag = href.partition("#")
            if target:                                        # путь относительно страницы
                target = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
                target = "./" if target == "." else target
            if target in ("", "./"):
                tp = pages["index.html"][1] if target == "./" else p
                path_ok = True
            else:
                path_ok = (ROOT / target).exists()
                tp = pages[target][1] if target in pages else None   # якоря таблицы появляются после загрузки
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
    assert not re.search(r"cdn\.|plot\.ly|jsdelivr|googleapis", idx + js)
    print("OK: страница, без сторонних ресурсов, ссылки работают, графики и подписи источника на месте, данные согласованы.")


if __name__ == "__main__":
    check()
