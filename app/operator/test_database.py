import io
import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from app.operator.accounts import Accounts
from app.operator.database import connect_database, users
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
