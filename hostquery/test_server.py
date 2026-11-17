"""Локальный HTTP API: те же ответы, только чтение и только localhost."""
from __future__ import annotations

import hashlib
import http.client
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from hostquery.errors import HostQueryError
from hostquery.server import ENDPOINTS, make_server, normalize_origins
from hostquery.store import Dataset, RunStore, open_dataset, sha256_file


def perturbed(data: Dataset) -> Dataset:
    rng = np.random.default_rng(11)
    vectors = data.vectors + rng.normal(0, 0.1, data.vectors.shape)
    return Dataset(ids=list(data.ids), vectors=vectors, norms=np.linalg.norm(vectors, axis=1),
                   kind=data.kind, stats=data.stats, feature_names=data.feature_names,
                   feature_kinds=data.feature_kinds,
                   provenance={"source": "test", "embeddings": "in-memory",
                               "embeddings_sha256": "0" * 64})


@pytest.fixture(scope="module")
def running(demo, demo_path):
    server = make_server(RunStore(bundle=demo_path, method="behavioral"),
                         compare=perturbed(demo), port=0,
                         allowed_origins=["http://127.0.0.1:8845"])
    server.RequestHandlerClass.log_message = lambda *args, **kwargs: None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(scope="module")
def plain(demo_path):
    server = make_server(RunStore(bundle=demo_path, method="behavioral"), port=0)
    server.RequestHandlerClass.log_message = lambda *args, **kwargs: None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def get(server, path: str, headers: dict | None = None):
    base = f"http://127.0.0.1:{server.server_port}"
    request = urllib.request.Request(base + path, headers=headers or {})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.load(response), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error), dict(error.headers)


def raw(server, method: str, path: str, headers: dict | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        for key, value in (headers or {}).items():
            connection.putheader(key, value)
        connection.endheaders()
        response = connection.getresponse()
        payload = response.read()
        body = json.loads(payload) if payload else None
        return response.status, body, dict(response.headers)
    finally:
        connection.close()


def test_index_lists_endpoints_and_read_only_note(running):
    status, body, headers = get(running, "/")
    assert status == 200 and body["schema"] == "hostquery/1"
    assert body["endpoints"] == ENDPOINTS and len(body["endpoints"]) == 9
    assert "только чтение" in body["note"]
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    assert headers["Cache-Control"] == "no-store"
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'none'" in headers["Content-Security-Policy"]


def test_health_and_meta(running):
    status, body, _ = get(running, "/api/lab/health")
    assert status == 200 and body["status"] == "ok" and body["compare"] is True
    assert "behavioral" in body["methods"]
    status, body, _ = get(running, "/api/lab/meta")
    assert status == 200 and body["hosts"] == 300 and body["dimensions"] == 56
    assert len(body["features"]) == 56 and body["features"][0]["name"] is None


def test_dataset_endpoint_matches_store_contract(running):
    status, body, _ = get(running, "/api/lab/dataset?method=behavioral")
    assert status == 200 and body["query"]["kind"] == "dataset"
    assert body["kind"] == "vectors" and len(body["dataset"]) == 300
    first = body["dataset"][0]
    assert set(first) == {"id", "v", "norm"} and len(first["v"]) == 56
    assert body["provenance"]["method"] == "behavioral"


@pytest.mark.parametrize("path, kind", [
    ("/api/lab/similar?host=Comp100014&k=3", "similar"),
    ("/api/lab/groups?n=3", "groups"),
    ("/api/lab/group?n=3&group=1", "group"),
    ("/api/lab/pair?a=Comp100014&b=Comp100021", "pair"),
    ("/api/lab/outliers?k=5", "outliers"),
    ("/api/lab/changes?k=3", "changes"),
])
def test_every_query_endpoint_returns_envelope(running, path, kind):
    status, body, _ = get(running, path)
    assert status == 200
    assert body["schema"] == "hostquery/1"
    assert body["query"]["kind"] == kind
    assert len(body["provenance"]["embeddings_sha256"]) == 64
    assert isinstance(body["warnings"], list)


def test_similar_with_groups_uses_zero_based_labels(running):
    _, body, _ = get(running, "/api/lab/similar?host=Comp100014&k=4&groups=3")
    assert len(body["neighbors"]) == 4
    assert all(item["group"] in {0, 1, 2} for item in body["neighbors"])


def test_changes_use_configured_second_window(running):
    _, body, _ = get(running, "/api/lab/changes?k=3")
    assert body["kind"] == "vectors" and body["n_common"] == 300
    assert body["provenance"]["compare"]["after"]
    assert body["changes"][0]["before"]["id"] == body["changes"][0]["after"]["id"]


@pytest.fixture(scope="module")
def two_windows(tmp_path_factory):
    root = tmp_path_factory.mktemp("windows")
    text = "host_id,emb_0,emb_1\na,1,0\nb,0,1\nc,1,1\n"
    before, after = root / "before.csv", root / "after.csv"
    before.write_text(text, encoding="utf-8")
    after.write_text(text, encoding="utf-8")
    server = make_server(RunStore(embeddings=before), compare=open_dataset(after), port=0)
    server.RequestHandlerClass.log_message = lambda *args, **kwargs: None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, before, after
    finally:
        server.shutdown()
        server.server_close()


def test_changes_provenance_has_both_paths_and_hashes(two_windows):
    server, before, after = two_windows
    status, body, _ = get(server, "/api/lab/changes?k=2")
    assert status == 200
    provenance = body["provenance"]
    compare_info = provenance["compare"]
    assert provenance["embeddings_sha256"] == sha256_file(before)
    assert compare_info["before"] == str(before)
    assert compare_info["after"] == str(after)
    assert compare_info["after_sha256"] == sha256_file(after)


def test_errors_are_json_with_400(running):
    for path in (
        "/api/lab/similar?host=NOPE&k=3",
        "/api/lab/similar?host=Comp100014&k=0",
        "/api/lab/similar?k=3",
        "/api/lab/similar?host=Comp100014&k=abc",
        "/api/lab/pair?a=Comp100014",
        "/api/lab/group?n=3&group=9",
        "/api/lab/dataset?method=nope",
        "/api/lab/nope",
    ):
        status, body, _ = get(running, path)
        assert status == 400, path
        assert "error" in body


def test_changes_without_second_window_explains_how_to_start(plain):
    status, body, _ = get(plain, "/api/lab/changes?k=3")
    assert status == 400 and "compare" in (body.get("hint") or "")


def test_non_read_methods_are_rejected(running):
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        status, body, _ = raw(running, method, "/api/lab/health",
                              {"Host": f"127.0.0.1:{running.server_port}"})
        assert status == 405 and "только чтение" in body["error"]


def test_foreign_host_and_origin_are_blocked(running):
    status, body, _ = raw(running, "GET", "/api/lab/health", {"Host": "evil.example"})
    assert status == 403 and "локальный" in body["error"]
    status, body, _ = get(running, "/api/lab/health", {"Origin": "http://evil.example"})
    assert status == 403 and "локальный" in body["error"]
    status, _, _ = get(running, "/api/lab/health",
                       {"Origin": f"http://127.0.0.1:{running.server_port}"})
    assert status == 403


def test_documented_lab_origin_gets_cors_headers(running):
    lab_origin = "http://127.0.0.1:8845"          # тот же origin, что в README
    status, body, headers = get(running, "/api/lab/health", {"Origin": lab_origin})
    assert status == 200 and body["status"] == "ok"
    assert headers["Access-Control-Allow-Origin"] == lab_origin
    assert "Origin" in headers["Vary"]

    # Ошибки тоже читаются fetch-ом: без CORS браузер скрыл бы тело ответа.
    status, body, headers = get(running, "/api/lab/similar?k=3", {"Origin": lab_origin})
    assert status == 400 and "error" in body
    assert headers["Access-Control-Allow-Origin"] == lab_origin


def test_cors_preflight_allows_only_get(running):
    lab_origin = "http://127.0.0.1:8845"
    status, _, headers = raw(running, "OPTIONS", "/api/lab/health", {
        "Host": f"127.0.0.1:{running.server_port}",
        "Origin": lab_origin,
        "Access-Control-Request-Method": "GET",
    })
    assert status == 204
    assert headers["Access-Control-Allow-Origin"] == lab_origin
    assert headers["Access-Control-Allow-Methods"] == "GET, OPTIONS"
    assert headers["Access-Control-Allow-Headers"] == "Accept, Content-Type"

    status, body, _ = raw(running, "OPTIONS", "/api/lab/health", {
        "Host": f"127.0.0.1:{running.server_port}",
        "Origin": "http://evil.example",
        "Access-Control-Request-Method": "GET",
    })
    assert status == 403 and "origin" in body["error"]


def test_origin_allowlist_accepts_only_loopback():
    assert normalize_origins(["http://127.0.0.1:8845", "http://localhost:8845/"]) == (
        "http://127.0.0.1:8845", "http://localhost:8845")
    for value in ("http://example.com", "https://192.168.1.5:8845",
                  "http://127.0.0.1:8845/path", "file://"):
        with pytest.raises(HostQueryError, match="локальным|только схему|порт"):
            normalize_origins([value])


@pytest.fixture(scope="module")
def identical(tmp_path_factory):
    path = tmp_path_factory.mktemp("identical") / "vectors.csv"
    path.write_text("host_id,emb_0,emb_1\na,1,1\nb,1,1\nc,1,1\n", encoding="utf-8")
    server = make_server(RunStore(embeddings=path), port=0)
    server.RequestHandlerClass.log_message = lambda *args, **kwargs: None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_empty_groups_are_described_not_rejected(identical):
    status, body, _ = get(identical, "/api/lab/groups?n=3")
    assert status == 200 and body["sizes"] == [3, 0, 0]
    assert len(body["profiles"]) == 3
    assert body["profiles"][0]["empty"] is False
    empty = body["profiles"][1]
    assert empty["group"] == 1 and empty["size"] == 0 and empty["empty"] is True
    assert empty["members"] == [] and empty["centroid"] is None

    status, body, _ = get(identical, "/api/lab/group?n=3&group=1")
    assert status == 200 and body["group"] == 1 and body["size"] == 0
    assert body["empty"] is True and body["mean_similarity"] is None


def test_artifacts_are_not_modified(demo_path, running):
    path = demo_path / "run" / "embeddings" / "behavioral" / "train.csv"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    for _ in range(3):
        get(running, "/api/lab/similar?host=Comp100014&k=3")
        get(running, "/api/lab/changes?k=3")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
