import io

from fastapi.testclient import TestClient

from app.operator.accounts import Accounts
from app.operator.web import create_app


ORIGIN = "https://pt.example.test"


def sign_in(client, username):
    return client.post(
        "/lab/login/",
        data={"name": username, "password": "example test passphrase"},
        headers={"Origin": ORIGIN, "Accept": "application/json"},
    )


def test_site_workspace_session(tmp_path):
    app = create_app(tmp_path, ORIGIN)
    accounts = Accounts(tmp_path)
    for username in ("one", "two"):
        accounts.add(username, "example test passphrase")
    with TestClient(app, base_url=ORIGIN) as client, TestClient(app, base_url=ORIGIN) as other:
        for route in ("/", "/model/", "/lab/", "/product/", "/app.js", "/assets/core.js", "/pt-favicon.svg"):
            assert client.get(route).status_code == 200, route
        assert client.get("/.git/config").status_code == 404
        assert 'data-workspace-mode="guest"' in client.get("/lab/").text
        assert client.get("/api/workspaces").status_code == 401
        response = sign_in(client, "one")
        assert response.status_code == 200
        assert {"secure", "httponly", "samesite=strict"} <= {
            part.strip().lower() for part in response.headers["set-cookie"].split(";")
        }
        assert 'data-workspace-mode="account"' in client.get("/lab/").text
        payload = {"name": "Shift report", "csv": "src,dst\na,b"}
        assert client.post("/api/workspaces", json=payload).status_code == 403
        created = client.post("/api/workspaces", json=payload, headers={"Origin": ORIGIN})
        assert created.status_code == 201
        route = "/api/workspaces/" + created.json()["id"]
        assert sign_in(other, "two").status_code == 200
        assert other.get(route).status_code == 404
        assert client.get(route).json()["name"] == payload["name"]
        cookie = dict(client.cookies)
        assert client.post("/lab/logout/", headers={"Origin": ORIGIN}, follow_redirects=False).status_code == 303
        assert client.get(route, headers={"Cookie": "; ".join(f"{k}={v}" for k, v in cookie.items())}).status_code == 401


def test_optional_inference_gateway(tmp_path, monkeypatch):
    upstream = "http://inference:8001/v1/predict"
    app = create_app(tmp_path, ORIGIN, upstream)
    Accounts(tmp_path).add("one", "example test passphrase")

    def predict(request, timeout):
        assert request.full_url == upstream
        assert request.data == b'{"sample":"example"}'
        return io.BytesIO(b'{"accepted":true}')

    monkeypatch.setattr("app.operator.web.urllib.request.urlopen", predict)
    with TestClient(app, base_url=ORIGIN) as client:
        headers = {"Origin": ORIGIN}
        payload = {"sample": "example"}
        assert client.post("/api/inference", json=payload, headers=headers).status_code == 401
        assert sign_in(client, "one").status_code == 200
        response = client.post("/api/inference", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.json() == {"accepted": True}

        def unavailable(*args, **kwargs):
            raise TimeoutError

        monkeypatch.setattr("app.operator.web.urllib.request.urlopen", unavailable)
        assert client.post("/api/inference", json=payload, headers=headers).status_code == 503
