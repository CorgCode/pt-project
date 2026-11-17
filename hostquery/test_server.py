"""Локальный HTTP API: те же ответы, только чтение и только localhost.

Тесты идут через ASGI (`TestClient`), без открытия сокетов: так их можно запускать
и в среде, где локальные порты запрещены. Транспорт проверяется отдельно, вручную.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hostquery.errors import HostQueryError
from hostquery.server import ENDPOINTS, create_app, normalize_origins
from hostquery.store import Dataset, RunStore, open_dataset, sha256_file

BASE = "http://127.0.0.1:8847"
LAB = "http://127.0.0.1:8845"          # тот же origin, что в README


def perturbed(data: Dataset) -> Dataset:
    rng = np.random.default_rng(11)
    vectors = data.vectors + rng.normal(0, 0.1, data.vectors.shape)
    return Dataset(ids=list(data.ids), vectors=vectors, norms=np.linalg.norm(vectors, axis=1),
                   kind=data.kind, stats=data.stats, feature_names=data.feature_names,
                   feature_kinds=data.feature_kinds,
                   provenance={"source": "test", "embeddings": "in-memory",
                               "embeddings_sha256": "0" * 64})


def client_for(app) -> TestClient:
    return TestClient(app, base_url=BASE, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def running(demo, demo_path):
    client = client_for(create_app(RunStore(bundle=demo_path, method="behavioral"),
                                   compare=perturbed(demo), allowed_origins=[LAB]))
    with client:
        yield client


@pytest.fixture(scope="module")
def plain(demo_path):
    client = client_for(create_app(RunStore(bundle=demo_path, method="behavioral")))
    with client:
        yield client


def body_of(response):
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:                     # preflight отвечает текстом, а не JSON
        return response.text


def headers_of(response):
    # httpx отдаёт имена заголовков в нижнем регистре — сравниваем так же.
    return {name.lower(): value for name, value in response.headers.items()}


def get(client, path: str, headers: dict | None = None):
    response = client.get(path, headers=headers or {})
    return response.status_code, body_of(response), headers_of(response)


def raw(client, method: str, path: str, headers: dict | None = None):
    response = client.request(method, path, headers=headers or {})
    return response.status_code, body_of(response), headers_of(response)


def test_index_lists_endpoints_and_read_only_note(running):
    status, body, headers = get(running, "/")
    assert status == 200 and body["schema"] == "hostquery/1"
    assert body["endpoints"] == ENDPOINTS and len(body["endpoints"]) == 10
    assert "только чтение" in body["note"]
    assert headers["content-type"] == "application/json; charset=utf-8"
    assert headers["cache-control"] == "no-store"
    assert headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in headers["content-security-policy"]


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
    client = client_for(create_app(RunStore(embeddings=before),
                                   compare=open_dataset(after)))
    with client:
        yield client, before, after


def test_changes_provenance_has_both_paths_and_hashes(two_windows):
    client, before, after = two_windows
    status, body, _ = get(client, "/api/lab/changes?k=2")
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
        status, body, _ = raw(running, method, "/api/lab/health", {"Host": "127.0.0.1:8847"})
        assert status == 405 and "только чтение" in body["error"]


def test_foreign_host_and_origin_are_blocked(running):
    status, body, _ = raw(running, "GET", "/api/lab/health", {"Host": "evil.example"})
    assert status == 403 and "локальный" in body["error"]
    status, body, _ = get(running, "/api/lab/health", {"Origin": "http://evil.example"})
    assert status == 403 and "локальный" in body["error"]
    status, _, _ = get(running, "/api/lab/health", {"Origin": BASE})
    assert status == 403


def test_documented_lab_origin_gets_cors_headers(running):
    status, body, headers = get(running, "/api/lab/health", {"Origin": LAB})
    assert status == 200 and body["status"] == "ok"
    assert headers["access-control-allow-origin"] == LAB
    assert "Origin" in headers["vary"]

    # Ошибки тоже читаются fetch-ом: без CORS браузер скрыл бы тело ответа.
    status, body, headers = get(running, "/api/lab/similar?k=3", {"Origin": LAB})
    assert status == 400 and "error" in body
    assert headers["access-control-allow-origin"] == LAB


def test_cors_preflight_allows_only_get(running):
    status, _, headers = raw(running, "OPTIONS", "/api/lab/health", {
        "Host": "127.0.0.1:8847",
        "Origin": LAB,
        "Access-Control-Request-Method": "GET",
    })
    assert status == 200
    assert headers["access-control-allow-origin"] == LAB
    assert set(headers["access-control-allow-methods"].split(", ")) == {"GET", "OPTIONS"}
    assert "Accept" in headers["access-control-allow-headers"]
    assert "Content-Type" in headers["access-control-allow-headers"]

    status, body, _ = raw(running, "OPTIONS", "/api/lab/health", {
        "Host": "127.0.0.1:8847",
        "Origin": "http://evil.example",
        "Access-Control-Request-Method": "GET",
    })
    assert status == 403 and "локальный" in body["error"]


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
    client = client_for(create_app(RunStore(embeddings=path)))
    with client:
        yield client


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


def test_openapi_describes_every_route_and_response_model(running):
    status, body, _ = get(running, "/api/lab/openapi.json")
    assert status == 200 and body["info"]["version"] == "hostquery/1"
    paths = set(body["paths"])
    documented = {path.split("?")[0] for path in ENDPOINTS if not path.endswith("openapi.json")}
    assert documented <= paths
    schemas = body["components"]["schemas"]
    for name in ("SimilarResponse", "GroupsResponse", "GroupResponse", "PairResponse",
                 "ChangesResponse", "OutliersResponse", "DatasetResponse"):
        assert name in schemas
    # Модель ответа — часть контракта, а не только пример в README.
    properties = set(schemas["SimilarResponse"]["properties"])
    assert {"schema", "query", "provenance", "neighbors", "warnings"} <= properties
    assert {"schema", "query", "provenance", "items"} <= set(
        schemas["OutliersResponse"]["properties"])


def test_http_and_cli_return_the_same_json(demo_path, running):
    """Один контракт: транспорт не должен менять ни числа, ни форму ответа."""
    from hostquery.cli import build_parser, run

    cases = [
        (["similar", "--host", "Comp100014", "-k", "3"],
         "/api/lab/similar?host=Comp100014&k=3"),
        (["groups", "-n", "3"], "/api/lab/groups?n=3"),
        (["group", "-n", "3", "--group", "1"], "/api/lab/group?n=3&group=1"),
        (["pair", "--a", "Comp100014", "--b", "Comp100021"],
         "/api/lab/pair?a=Comp100014&b=Comp100021"),
        (["outliers", "-k", "5"], "/api/lab/outliers?k=5"),
    ]
    for argv, path in cases:
        args = build_parser().parse_args(argv + ["--bundle", str(demo_path)])
        expected = run(args)
        status, body, _ = get(running, path)
        assert status == 200, argv
        assert body == expected, argv


def test_artifacts_are_not_modified(demo_path, running):
    path = demo_path / "run" / "embeddings" / "behavioral" / "train.csv"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    for _ in range(3):
        get(running, "/api/lab/similar?host=Comp100014&k=3")
        get(running, "/api/lab/changes?k=3")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
