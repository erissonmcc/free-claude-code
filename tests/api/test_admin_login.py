from fastapi.testclient import TestClient

from free_claude_code.core.version import package_version
from tests.api.support import create_test_app


def _remote_client():
    return TestClient(
        create_test_app(),
        base_url="https://render.example",
        client=("203.0.113.10", 50000),
    )


def test_admin_login_sets_secure_cookie_and_allows_admin_page(monkeypatch):
    monkeypatch.setenv("FCC_ADMIN_AUTH_TOKEN", "admin-secret")
    client = _remote_client()

    response = client.post("/admin/api/auth/login", json={"token": "admin-secret"})

    assert response.status_code == 200
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "secure" in cookie
    assert "samesite=strict" in cookie
    assert "path=/admin" in cookie
    assert "max-age=43200" in cookie
    assert "Free Claude Code</h1>" in client.get("/admin").text
    asset = client.get(f"/admin/assets/{package_version()}/admin.js")
    assert asset.status_code == 200


def test_invalid_admin_login_does_not_set_cookie(monkeypatch):
    monkeypatch.setenv("FCC_ADMIN_AUTH_TOKEN", "admin-secret")
    response = _remote_client().post(
        "/admin/api/auth/login", json={"token": "wrong-secret"}
    )

    assert response.status_code == 401
    assert "set-cookie" not in response.headers


def test_public_admin_page_shows_login_without_auth(monkeypatch):
    monkeypatch.setenv("FCC_ADMIN_AUTH_TOKEN", "admin-secret")
    client = _remote_client()
    response = client.get("/admin")
    direct_login = client.get("/admin/login")

    assert response.status_code == 200
    assert "Acessar o Admin" in response.text
    assert 'type="password"' in response.text
    assert direct_login.status_code == 200
    assert "Acessar o Admin" in direct_login.text


def test_admin_cookie_and_bearer_can_access_guarded_api(monkeypatch):
    monkeypatch.setenv("FCC_ADMIN_AUTH_TOKEN", "admin-secret")
    client = _remote_client()
    client.post("/admin/api/auth/login", json={"token": "admin-secret"})

    cookie_response = client.get("/admin/api/config")
    bearer_response = _remote_client().get(
        "/admin/api/config", headers={"Authorization": "Bearer admin-secret"}
    )

    assert cookie_response.status_code == 200
    assert bearer_response.status_code == 200


def test_admin_logout_clears_cookie(monkeypatch):
    monkeypatch.setenv("FCC_ADMIN_AUTH_TOKEN", "admin-secret")
    client = _remote_client()
    client.post("/admin/api/auth/login", json={"token": "admin-secret"})

    response = client.post("/admin/api/auth/logout")

    assert response.status_code == 200
    assert "max-age=0" in response.headers["set-cookie"].lower()
    assert client.get("/admin/api/config").status_code == 401
