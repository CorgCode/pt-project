"""Источник запросов: прогон и два вида CSV.

Главное, что здесь проверяется, — данные читаются ровно так же, как их увидит
лаборатория: трафик превращается в 20 признаков по правилам core.js, а готовые
векторы не получают придуманных имён координат.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from hostquery import features
from hostquery.errors import HostQueryError
from hostquery.store import (Dataset, RunStore, dataset_from_csv, open_dataset,
                             read_csv_dataset, sha256_file)

DEMO = Path(__file__).resolve().parents[1] / "app" / "testdata" / "synthetic_run"

FLOW_CSV = """src,dst,bytes,packets,proto,dst_port
h1,h2,10,1,6,443
h1,h3,20,2,17,53
h2,h1,30,3,6,443
h3,h1,40,4,6,80
"""

VECTOR_CSV = """host_id,emb_0,emb_1
a,1,0
b,0,2
c,3,4
"""


def flow_dataset() -> Dataset:
    return dataset_from_csv(FLOW_CSV)


def test_bundle_dataset_is_read_only_contract(demo):
    data = RunStore(bundle=DEMO).dataset("behavioral")
    assert data.kind == "vectors" and data.n == 300 and data.dim == 56
    assert data.ids == sorted(data.ids)
    assert data.feature_names == [None] * 56
    assert set(data.feature_kinds) == {"embedding_coordinate"}
    assert data.provenance["synthetic"] is True
    assert data.provenance["source"] == "bundle"
    path = DEMO / "run" / "embeddings" / "behavioral" / "train.csv"
    assert data.provenance["embeddings_sha256"] == sha256_file(path)
    assert data.provenance["reference"].endswith("reference\\test\\neighbors.csv") or \
        data.provenance["reference"].endswith("reference/test/neighbors.csv")


def test_dataset_rejects_bad_shapes():
    with pytest.raises(HostQueryError, match="матрицей"):
        Dataset(ids=["a"], vectors=np.zeros(3), norms=np.ones(1), kind="vectors",
                stats={}, feature_names=[], feature_kinds=[])
    with pytest.raises(HostQueryError, match="повторяются"):
        Dataset(ids=["a", "a"], vectors=np.ones((2, 2)), norms=np.ones(2), kind="vectors",
                stats={}, feature_names=[None] * 2, feature_kinds=["x"] * 2)
    with pytest.raises(HostQueryError, match="нулевой вектор"):
        Dataset(ids=["a", "b"], vectors=np.array([[0.0, 0.0], [1.0, 0.0]]),
                norms=np.array([0.0, 1.0]), kind="vectors", stats={},
                feature_names=[None] * 2, feature_kinds=["x"] * 2)


def test_vectors_csv_bom_semicolon_and_order(tmp_path):
    path = tmp_path / "vectors.csv"
    path.write_text("\ufeffhost_id;emb_0;emb_1\nb;0;2\na;1;0\n", encoding="utf-8")
    data = read_csv_dataset(path)
    assert data.ids == ["b", "a"]                  # порядок строк сохраняется
    assert np.allclose(data.vectors, [[0, 2], [1, 0]])
    assert data.kind == "vectors" and data.stats == {"hosts": 2, "dimensions": 2}
    assert data.feature_names == [None, None]
    opened = open_dataset(path)
    assert opened.provenance["source"] == "csv" and opened.provenance["data_kind"] == "vectors"
    assert opened.provenance["embeddings_sha256"] == sha256_file(path)


def test_flow_csv_builds_same_twenty_features_as_core_js():
    data = flow_dataset()
    assert data.kind == "flows" and data.dim == features.N_FEATURES == 20
    assert data.ids == ["h1", "h2", "h3"]
    assert data.feature_names == features.FEATURE_NAMES
    assert data.feature_kinds == features.FEATURE_KINDS
    assert data.stats["flows"] == 4 and data.stats["bytes"] == 100

    # Сырые показатели: activity хранится ДО log1p, как в core.js.
    assert data.activity["h1"] == {"outgoing": 2.0, "incoming": 2.0, "sentBytes": 30.0,
                                   "receivedBytes": 70.0, "contacts": 2}
    assert data.activity["h2"]["contacts"] == 1 and data.activity["h2"]["outgoing"] == 1.0

    # Признаки 0..7: направления, объёмы, пакеты, собеседники, всего записей.
    raw = np.zeros((3, features.N_FEATURES))
    expected = {
        "h1": [2, 2, 30, 70, 3, 7, 2, 4],
        "h2": [1, 1, 30, 10, 3, 1, 1, 2],
        "h3": [1, 1, 40, 20, 4, 2, 1, 2],
    }
    for row, host in enumerate(data.ids):
        raw[row, :features.N_BASE] = expected[host]

    outgoing = features.service_bucket("6", "443")
    port53 = features.service_bucket("17", "53")
    port80 = features.service_bucket("6", "80")
    raw[0, features.N_BASE + outgoing] += 1          # h1 -> h2, 443
    raw[0, features.N_BASE + port53] += 1            # h1 -> h3, 53
    raw[1, features.N_BASE + outgoing] += 1          # h2 -> h1, 443
    raw[2, features.N_BASE + port80] += 1            # h3 -> h1, 80
    raw[1, features.N_BASE + features.N_BUCKETS + outgoing] += 1
    raw[0, features.N_BASE + features.N_BUCKETS + outgoing] += 1
    raw[2, features.N_BASE + features.N_BUCKETS + port53] += 1
    raw[0, features.N_BASE + features.N_BUCKETS + port80] += 1

    logged = np.log1p(raw)
    scales = np.maximum(1.0, logged.max(axis=0))
    assert np.allclose(data.log_values, logged)
    assert np.allclose(data.vectors, logged / scales)
    assert np.allclose(data.norms, np.linalg.norm(logged / scales, axis=1))

    # Связи агрегированы по направленной паре, как Map в core.js.
    assert {(edge["src"], edge["dst"], edge["flows"], edge["bytes"])
            for edge in data.connections} == {
        ("h1", "h2", 1, 10.0), ("h1", "h3", 1, 20.0),
        ("h2", "h1", 1, 30.0), ("h3", "h1", 1, 40.0),
    }


def test_flow_csv_accepts_aliases_and_optional_columns():
    text = "source,target\nx,y\ny,x\n"
    data = dataset_from_csv(text)
    assert data.ids == ["x", "y"]
    assert data.stats["bytes"] == 0
    assert data.activity["x"]["outgoing"] == 1 and data.activity["x"]["incoming"] == 1


def test_csv_quoted_hosts_are_not_split(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text('host_id,emb_0,emb_1\n"a,1",1,0\n"b;2",0,1\n', encoding="utf-8")
    data = read_csv_dataset(path)
    assert data.ids == ["a,1", "b;2"]


@pytest.mark.parametrize("text, message", [
    ("host_id,emb_0\na,1\na,2\n", "повторяется"),
    ("host_id,emb_0\na,\n", "пустое значение"),
    ("host_id,emb_0\na,nan\n", "не конечное"),
    ("host_id,emb_0\na,abc\n", "не число"),
    ("host_id,emb_0\na,0\n", "нулевой вектор"),
    ("host_id,emb_0\n", "заголовок и хотя бы одна строка"),
    ("host_id,emb_0\n,1\n", "пустой host_id"),
    ("host_id,host_id\n a,b\n", "повторяются"),
    ("host_id,emb_0\na,1,2\n", "полей"),
    ("a,b\nh1,h2\n", "нужны столбцы src и dst"),
    ("src,dst,bytes\nh1,h2,-1\n", "неотрицательное"),
    ("src,dst,bytes\nh1,,1\n", "не должны быть пустыми"),
    ("src,dst,bytes\nh1,h2,abc\n", "не число"),
    ("src,dst\nh1,h1\n", "хотя бы два хоста"),
    ("host_id,emb_0\nonly,1\n", "хотя бы два хоста"),
])
def test_csv_errors_are_human_readable(text, message):
    with pytest.raises(HostQueryError, match=message):
        dataset_from_csv(text)


def test_csv_coordinates_must_be_contiguous():
    with pytest.raises(HostQueryError, match="подряд"):
        dataset_from_csv("host_id,emb_0,emb_2\na,1,2\nb,3,4\n")


def test_run_store_resolves_methods_and_explains_unknowns():
    store = RunStore(bundle=DEMO)
    assert "behavioral" in store.methods and store.resolve_method() == "behavioral"
    assert store.resolve_method("deepwalk") == "deepwalk"
    with pytest.raises(HostQueryError, match="нет в источнике"):
        store.resolve_method("no_such_method")
    with pytest.raises(HostQueryError, match="один источник"):
        RunStore(bundle=DEMO, embeddings=DEMO / "run" / "embeddings" / "behavioral" / "train.csv")
    with pytest.raises(HostQueryError, match="нужен источник"):
        RunStore()
    with pytest.raises(HostQueryError, match="нет run/manifest.json"):
        RunStore(bundle=DEMO.parent / "nope")


def test_run_store_reads_flow_csv_through_same_switch(tmp_path):
    path = tmp_path / "flows.csv"
    path.write_text(FLOW_CSV, encoding="utf-8")
    store = RunStore(embeddings=path)
    data = store.dataset()
    assert data.kind == "flows" and data.n == 3
    assert store.resolve_method() == "flows"
    assert data.provenance["source"] == "csv" and data.provenance["data_kind"] == "flows"
    assert data.provenance["embeddings_sha256"] == sha256_file(path)


def test_reference_ranks_match_saved_neighbors(demo):
    store = RunStore(bundle=DEMO)
    ranks = store.reference_ranks("Comp100014")
    assert ranks and all(rank >= 1 for rank in ranks.values())
    assert "Comp100014" not in ranks
    assert store.reference_path().name == "neighbors.csv"
    assert RunStore(bundle=DEMO, reference_dir=DEMO / "reference").reference_ranks("Comp100014")


def test_missing_sources_and_bad_paths(tmp_path):
    with pytest.raises(HostQueryError, match="нет файла"):
        open_dataset(tmp_path / "nope.csv")
    with pytest.raises(HostQueryError, match="папка прогона или CSV"):
        open_dataset(tmp_path / "file.txt")
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(HostQueryError, match="UTF-8|заголовок"):
        read_csv_dataset(bad)
