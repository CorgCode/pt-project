"""Паритет с лабораторией: числа и константы считаются как в `noomo-ui/lab/core.js`.

Лаборатория — витрина, а не отдельная реализация. Если здесь что-то разойдётся,
пользователь увидит на графике не то, что отдаёт API, поэтому проверяются:

1. Константы алгоритма (18 итераций проекции, 20 итераций k-means, 20 признаков).
2. Числа `project`/`group`/`compareWindows` при установленном `node` — побайтово те же
   входные CSV прогоняются через настоящий core.js.
3. Соседи совпадают с `app.results.Bundle.neighborhood` — контракт уже существующего слоя.

Тест с `node` пропускается, если Node.js не установлен: Python-часть остаётся проверяемой.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hostquery import features, groups
from hostquery.similar import neighbors
from hostquery.store import RunStore, dataset_from_csv
from hostquery.temporal import compare

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "origin/physcorgi/ui-soc-story"
CORE = "noomo-ui/lab/core.js"

NODE = shutil.which("node")

FLOW_HEADER = "src,dst,bytes,packets,proto,dst_port"
FLOW_BEFORE = FLOW_HEADER + """
a,b,100,1,6,443
a,c,90,1,6,80
b,a,110,1,6,443
b,c,80,1,17,53
c,a,95,1,6,80
c,b,85,1,17,53
d,a,40,1,6,22
"""

FLOW_AFTER = FLOW_HEADER + """
a,b,130,2,6,443
a,c,90,1,6,80
a,e,700,9,6,8443
b,a,140,2,6,443
b,c,85,1,17,53
c,a,90,1,6,80
c,b,80,1,17,53
d,a,45,1,6,22
e,b,60,1,6,443
"""


def core_source() -> str:
    """core.js из рабочей копии, иначе из ветки лаборатории, иначе пропуск."""
    local = ROOT / CORE
    if local.exists():
        return local.read_text(encoding="utf-8")
    try:
        done = subprocess.run(["git", "show", f"{BRANCH}:{CORE}"], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8", check=True)
    except (OSError, subprocess.CalledProcessError) as error:  # pragma: no cover - среда без git
        pytest.skip(f"{CORE} недоступен: {error}")
    return done.stdout


@pytest.fixture(scope="module")
def source() -> str:
    return core_source()


def test_constants_match_core_js(source):
    """Константы не «примерно такие»: те же строки, что исполняет браузер."""
    assert "Math.sin(j * 1.7 + (previous ? 1.1 : 0.4))" in source
    assert "iter < 18" in source
    assert "iteration<20" in source
    assert "k=Math.min(k,rows.length)" in source
    assert "Math.log1p" in source
    assert "Math.max(1, ...dataset.map((h) => h.v[i]))" in source
    assert "Math.max(1,...all.map(h=>h.logValues[i]))" in source
    assert "a.similarity-b.similarity||a.id.localeCompare(b.id)" in source
    assert "Math.max(-1,Math.min(1" in source
    assert "dimensions: 20" in source
    assert "a.v[8 + (hash % 6)]++" in source
    assert "b.v[14 + (hash % 6)]++" in source

    assert features.N_FEATURES == 20
    assert features.N_BASE == 8
    assert groups.PROJECTION_ITERATIONS == 18
    assert groups.ITERATIONS == 20


def test_core_js_flows_use_the_same_twenty_feature_layout(source):
    """8 простых признаков + 12 хешированных блоков: сдвиги 8 и 14 из core.js."""
    assert len(features.BASE_FEATURES) == 8
    assert len(features.SERVICE_FEATURES) == 12
    assert features.N_BASE == 8
    assert features.N_BASE + features.N_BUCKETS == 14


def _write_parity_driver(tmp_path: Path, source: str) -> Path:
    """Мини-драйвер Node.js: настоящий core.js получает те же CSV, что и Python."""
    core = tmp_path / "core.js"
    core.write_text(source, encoding="utf-8")
    driver = tmp_path / "parity.js"
    driver.write_text(
        """
global.window = {};
require(process.argv[2]);
const fs = require("fs");
const read = (path) => fs.readFileSync(path, "utf8");
const first = window.PTLab.parseCSV(read(process.argv[3]));
const out = { projection: window.PTLab.project(first), group: window.PTLab.group(first, 3) };
if (process.argv[4]) {
  const second = window.PTLab.parseCSV(read(process.argv[4]));
  out.changes = window.PTLab.compareWindows(first, second, 3);
}
process.stdout.write(JSON.stringify(out));
""",
        encoding="utf-8",
    )
    return driver


def _random_vectors_csv(path: Path, count: int = 24, dimensions: int = 4) -> str:
    rng = np.random.default_rng(5)
    lines = ["host_id," + ",".join(f"emb_{i}" for i in range(dimensions))]
    for index in range(count):
        values = rng.normal(size=dimensions).round(4)
        lines.append(",".join([f"h{index:02d}"] + [repr(float(x)) for x in values]))
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


@pytest.mark.skipif(NODE is None, reason="node не установлен")
def test_projection_and_group_are_numerically_identical_to_core_js(tmp_path, source):
    vector_text = _random_vectors_csv(tmp_path / "vectors.csv")
    before = tmp_path / "before.csv"
    after = tmp_path / "after.csv"
    before.write_text(FLOW_BEFORE, encoding="utf-8")
    after.write_text(FLOW_AFTER, encoding="utf-8")
    driver = _write_parity_driver(tmp_path, source)

    done = subprocess.run([NODE, str(driver), str(tmp_path / "core.js"),
                           str(tmp_path / "vectors.csv"), str(before), str(after)],
                          capture_output=True, text=True, encoding="utf-8", check=True)
    reference = json.loads(done.stdout)

    data = dataset_from_csv(vector_text)
    ours = groups.project(data)
    assert [item["hostId"] for item in ours] == [item["hostId"] for item in reference["projection"]]
    np.testing.assert_allclose([item["x"] for item in ours],
                               [item["x"] for item in reference["projection"]], atol=1e-12)
    np.testing.assert_allclose([item["y"] for item in ours],
                               [item["y"] for item in reference["projection"]], atol=1e-12)
    assert groups.fit(data, 3)["labels"] == reference["group"]

    ours_changes = compare(dataset_from_csv(FLOW_BEFORE), dataset_from_csv(FLOW_AFTER), k=3)
    theirs = reference["changes"]
    assert ours_changes["kind"] == theirs["kind"] == "flows"
    assert [item["id"] for item in ours_changes["changes"]] == [item["id"] for item in theirs["changes"]]
    np.testing.assert_allclose([item["similarity"] for item in ours_changes["changes"]],
                               [item["similarity"] for item in theirs["changes"]], atol=1e-12)
    for ours_item, their_item in zip(ours_changes["changes"], theirs["changes"]):
        assert ours_item["beforeGroup"] == their_item["beforeGroup"]
        assert ours_item["afterGroup"] == their_item["afterGroup"]
        assert ours_item["added"] == their_item["added"]
        assert ours_item["removed"] == their_item["removed"]
        assert sorted(ours_item["beforeContacts"]) == sorted(their_item["beforeContacts"])
        assert sorted(ours_item["afterContacts"]) == sorted(their_item["afterContacts"])
    assert sorted(ours_changes["newHosts"]) == sorted(theirs["newHosts"])
    assert sorted(ours_changes["missingHosts"]) == sorted(theirs["missingHosts"])


def test_neighbors_match_bundle_neighborhood(demo, demo_path):
    """Уже существующий слой данных и новый поиск дают одних и тех же соседей."""
    from app.results import Bundle

    bundle = Bundle(demo_path)
    store = RunStore(bundle=demo_path)
    for host in (demo.ids[0], demo.ids[len(demo.ids) // 2]):
        frame = bundle.neighborhood("behavioral", host, 7)
        ours = neighbors(demo, host, 7, reference=store.reference_ranks(host))
        assert [item["id"] for item in ours["neighbors"]] == frame["host_id"].tolist()
        np.testing.assert_allclose([item["similarity"] for item in ours["neighbors"]],
                                   np.clip(frame["cosine"].to_numpy(float), -1.0, 1.0),
                                   atol=1e-12)
        expected = [None if pd.isna(value) else int(value) for value in frame["reference_rank"]]
        assert [item.get("reference_rank") for item in ours["neighbors"]] == expected
