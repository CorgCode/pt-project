"""Static GitHub Pages regression check. No Python research runtime needed.

Checks that chart JS is syntactically valid in Node, verifies the embedded
snapshot is explicitly synthetic and internally consistent with the stored
per-host PCA data, and ensures charts render on initial page load (no upload).
This is a static check, NOT a substitute for browser visual QA.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "index.html"
DATA = ROOT / "data" / "synthetic_demo.json"
EXPECTED = {"deepwalk", "node2vec", "graphsage", "rolx", "struc2vec"}


class Scripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.inside = False
        self.parts: list[str] = []
        self.scripts: list[str] = []

    def handle_starttag(self, tag, attrs) -> None:
        if tag == "script" and not dict(attrs).get("src"):
            self.inside = True
            self.parts = []

    def handle_data(self, data) -> None:
        if self.inside:
            self.parts.append(data)

    def handle_endtag(self, tag) -> None:
        if tag == "script" and self.inside:
            self.scripts.append("".join(self.parts))
            self.inside = False


def check() -> None:
    html = HTML.read_text(encoding="utf-8")
    info = json.loads(DATA.read_text(encoding="utf-8"))
    meta = info["meta"]

    assert meta["synthetic"] is True
    assert meta["hosts"] == 300
    assert set(meta["missing"]) == EXPECTED
    assert "LANL" not in set(meta["methods"])
    assert "reference_ceiling" in meta["methods"]
    assert '<html lang="ru">' in html
    assert "синтетический пример" in html.lower()
    assert "Исследование представлений хостов" in html
    assert "Сохранение ближайших соседей" in html
    assert "Прогноз будущих связей" in html
    assert "Карта представлений хостов" in html
    assert "Скачать пример CSV" in html
    assert "Оценить embeddings" in html
    root_html = (ROOT.parent / "index.html").read_text(encoding="utf-8")
    assert '<html lang="ru">' in root_html
    assert "Открыть визуализации" in root_html
    assert "id=\"dashboardContents\"" in html
    assert "id=\"dashboardSplit\"" in html
    assert "id=\"chartConsistency\"" in html
    assert "id=\"chartLinks\"" in html
    assert "id=\"chartPairs\"" in html
    assert "id=\"chartIdentity\"" in html
    assert "id=\"chartHosts\"" in html
    assert "id=\"snapshotBoard\"" in html
    assert "fetch(\"./data/synthetic_demo.json\"" in html
    assert "loadExternal" in html  # upload-only third-party libs, not page blocking

    points = info["projection"]["points"]
    assert len(points) == meta["hosts"]
    assert len({p["id"] for p in points}) == len(points)
    assert all(math.isfinite(p["x"]) and math.isfinite(p["y"]) and
               0 <= p["agreement"] <= 1 for p in points)
    avg = sum(p["agreement"] for p in points) / len(points)
    reported = info["results"]["test"]["behavioral"]["consistency@10_mean"]
    assert abs(avg - reported) < 1e-10, (avg, reported)
    assert abs(avg - info["projection"]["per_host_mean_agreement"]) < 1e-10
    assert info["results"]["test"]["reference_ceiling"]["consistency@10_mean"] == 1
    assert info["results"]["test"]["random"]["consistency@10_mean"] < 0.1

    for split in ("test", "validation"):
        assert set(info["results"][split]) == set(meta["methods"])
        for method, scores in info["results"][split].items():
            assert all(math.isfinite(v) for v in scores.values()), (split, method)

    parser = Scripts()
    parser.feed(html)
    assert len(parser.scripts) >= 2, "Expected upload and snapshot JS blocks"
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js required to syntax-check GitHub Pages JavaScript")
    with tempfile.TemporaryDirectory() as root:
        for i, content in enumerate(parser.scripts):
            path = Path(root) / f"script_{i}.js"
            path.write_text(content, encoding="utf-8")
            subprocess.run([node, "--check", str(path)], check=True)
    print("OK: Pages scripts parse; 300 synthetic hosts; demo values match stored research scores.")


if __name__ == "__main__":
    check()
