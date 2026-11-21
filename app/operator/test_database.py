import io
import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from app.operator.accounts import Accounts
from app.operator.database import DatabaseAccounts, connect_database, users
from app.operator.import_sqlite import import_snapshot
from app.operator.server import Store
from app.operator.web import create_app

ORIGIN = "https://pt.example.test"
PASSWORD = "example test passphrase"


@pytest.fixture
def migrated(tmp_path, monkeypatch):
    url = os.getenv("PT_TEST_DATABASE_URL", f"sqlite:///{tmp_path / 'target.sqlite'}")
    secret = tmp_path / "db-url"
    secret.write_text(url)
    monkeypatch.setenv("PT_DATABASE_URL_FILE", str(secret))
    root = Path(__file__).resolve().parents[2]
    config = Config(root / "deploy/alembic.ini")
    engine = connect_database(url)
    command.upgrade(config, "head")
    yield engine
    command.downgrade(config, "base")
    engine.dispose()


def login(client, name):
    return client.post("/lab/login/", data={"name": name, "password": PASSWORD},
                       headers={"origin": ORIGIN, "accept": "application/json"})


def test_import_preserves_credentials_and_owner(tmp_path, migrated, monkeypatch):
    source = tmp_path / "legacy"
    accounts = Accounts(source)
    accounts.add("one", PASSWORD)
    accounts.add("two", PASSWORD)
    old_token = accounts.login("one", PASSWORD)
    owner = accounts.identify(old_token)
    saved_id = Store(source / "users" / owner / "workspaces.sqlite3").save(
        {"name": "Research", "csv": "src,dst\na,b", "notes": {"a": "Комментарий"}},
    )
    assert import_snapshot(source, migrated) == {"users": 2, "workspaces": 1}
    with pytest.raises(ValueError, match="empty"):
        import_snapshot(source, migrated)
    app = create_app(tmp_path / "unused", ORIGIN, "http://inference:8001/v1/predict",
                     engine=migrated, allow_registration=False, metrics_token="a" * 32)
    monkeypatch.setattr("app.operator.web.urllib.request.urlopen",
                        lambda *a, **kw: io.BytesIO(b'{"accepted":true}'))
    with TestClient(app, base_url=ORIGIN) as first, TestClient(app, base_url=ORIGIN) as second:
        assert first.get("/api/ready").status_code == 200
        assert first.get("/demo/").status_code == 200
        assert first.get("/lab/register/").status_code == 404
        assert first.post("/lab/register/", headers={"Origin": ORIGIN}).status_code == 404
        first.cookies.set("corgcode_session", old_token)
        assert first.get("/api/workspaces").status_code == 401
        assert login(first, "one").status_code == 200
        assert login(second, "two").status_code == 200
        route = f"/api/workspaces/{saved_id}"
        assert first.get(route).json()["notes"] == {"a": "Комментарий"}
        assert second.get(route).status_code == 404
        assert second.put(route, headers={"Origin": ORIGIN}, json={"name": "Other", "csv": "x"}).status_code == 404
        assert first.get("/api/datasets").json()[0]["id"] == saved_id
        assert second.get("/api/datasets").json() == []
        assert first.post("/api/inference", json={}).status_code == 403
        result = first.post("/api/inference", headers={"Origin": ORIGIN}, json={})
        assert result.json() == {"accepted": True}
        job = "/api/jobs/" + result.headers["X-PT-Job-ID"]
        assert first.get(job).json()["status"] == "succeeded"
        assert second.get(job).status_code == 404
        assert first.get("/internal/metrics").status_code == 404
        assert "pt_http_requests_total" in first.get(
            "/internal/metrics", headers={"Authorization": "Bearer " + "a" * 32}
        ).text


def test_bad_snapshot_rolls_back(tmp_path, migrated):
    source = tmp_path / "legacy"
    accounts = Accounts(source)
    accounts.add("one", PASSWORD)
    accounts.add("two", PASSWORD)
    owner = accounts.identify(accounts.login("two", PASSWORD))
    store = Store(source / "users" / owner / "workspaces.sqlite3")
    key = store.save({"name": "Research", "csv": "data"})
    with store.connect() as db:
        db.execute("UPDATE workspaces SET payload=? WHERE id=?", ("invalid", key))
    with pytest.raises(ValueError):
        import_snapshot(source, migrated)
    with migrated.begin() as db:
        assert db.execute(users.select()).first() is None


def test_real_embeddings_persist_and_survive_model_switch(tmp_path, migrated, monkeypatch):
    from deploy.inference.service import app as model_app
    from hostquery.store import dataset_from_csv

    csv = "src,dst,bytes,packets,proto,dst_port\na,b,800,10,tcp,443\na,c,200,2,udp,53\nb,c,900,5,tcp,80\n"
    payload = {"csv": csv, "name": "Observed traffic"}
    expected = dataset_from_csv(csv)
    accounts = DatabaseAccounts(migrated)
    for name in ("one", "two"):
        accounts.add(name, PASSWORD)
    monkeypatch.delenv("PT_MODEL_ADAPTER", raising=False)
    monkeypatch.delenv("PT_MODEL_MANIFEST_SHA256", raising=False)
    model = None

    def upstream(request, timeout):
        response = model.post("/v1/predict", content=request.data, headers={"Content-Type": "application/json"})
        assert response.status_code == 200
        return io.BytesIO(response.content)

    monkeypatch.setattr("app.operator.web.urllib.request.urlopen", upstream)
    app = create_app(tmp_path, ORIGIN, "http://model/v1/predict", engine=migrated)
    headers = {"Origin": ORIGIN}
    route = "/api/inference/embeddings"
    with TestClient(app, base_url=ORIGIN) as client, TestClient(app, base_url=ORIGIN) as other:
        assert client.post(route, json=payload, headers=headers).status_code == 401
        assert login(client, "one").status_code == 200
        assert login(other, "two").status_code == 200
        assert client.post(route, json=payload).status_code == 403
        assert client.post(route, json={"csv": "x" * (3 * 1024 * 1024 + 1)}, headers=headers).status_code == 422
        monkeypatch.setenv("PT_MODEL_VERSION", "traffic-profile-v1")
        with TestClient(model_app) as model:
            assert model.post("/v1/predict", json={"csv": "host_id,emb_0\na,1\nb,2"}).status_code == 422
            result = client.post(route, json=payload, headers=headers).json()
            repeated = model.post("/v1/predict", json=payload).json()
            assert result["dataset"] == repeated["dataset"]
        assert result["schema"] == "hostquery/1"
        assert [row["id"] for row in result["dataset"]] == expected.ids
        assert [row["v"] for row in result["dataset"]] == expected.vectors.tolist()
        linked = {**payload, "inferenceId": result["jobId"]}
        assert other.post("/api/workspaces", json=linked, headers=headers).status_code == 400
        assert client.post("/api/workspaces", json={**linked, "csv": csv + "\n"}, headers=headers).status_code == 400
        stored = client.post("/api/workspaces", json=linked, headers=headers)
        assert stored.status_code == 201
        workspace = "/api/workspaces/" + stored.json()["id"]
        assert other.get(workspace).status_code == 404
        assert client.get(workspace).json()["inference"]["dataset"] == result["dataset"]

        monkeypatch.setenv("PT_MODEL_VERSION", "traffic-volume-v1")
        with TestClient(model_app) as model:
            switched = client.post(route, json=payload, headers=headers).json()
        assert switched["provenance"]["model_version"] == "traffic-volume-v1"
        assert switched["stats"]["dimensions"] < result["stats"]["dimensions"]
        assert client.get(workspace).json()["inference"]["provenance"]["model_version"] == "traffic-profile-v1"
