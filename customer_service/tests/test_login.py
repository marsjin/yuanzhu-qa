"""登录 Cookie：60 分钟有效期、退出后页面即无权限。"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from customer_service.app import service
from customer_service.core import auth
from customer_service.core import config as cfg

HTML = {"accept": "text/html"}


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setattr(cfg, "AUTH_ENABLED", True)
    monkeypatch.setattr(cfg, "AUTH_ADMIN_USERNAME", "admin")
    monkeypatch.setattr(cfg, "AUTH_ADMIN_PASSWORD", "s3cret")
    monkeypatch.setattr(cfg, "AUTH_USER_USERNAME", "user")
    monkeypatch.setattr(cfg, "AUTH_USER_PASSWORD", "change-me-user")
    monkeypatch.setattr(cfg, "AUTH_EXTRA_USERS", "")
    monkeypatch.setattr(cfg, "AUTH_SESSION_MINUTES", 60)


def _login(client: TestClient, username: str, password: str, next_path: str = "/"):
    return client.post(
        "/login",
        data={"username": username, "password": password, "next": next_path},
        follow_redirects=False,
    )


def test_login_sets_cookie_and_opens_pages():
    client = TestClient(service.app)
    assert client.get("/login", headers=HTML).status_code == 200

    resp = _login(client, "admin", "s3cret")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/"
    assert cfg.AUTH_COOKIE_NAME in resp.cookies

    assert client.get("/", headers=HTML).status_code == 200
    assert client.get("/admin", headers=HTML).status_code == 200


def test_login_wrong_password():
    client = TestClient(service.app)
    resp = _login(client, "admin", "wrong")
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/login?error=1")
    assert cfg.AUTH_COOKIE_NAME not in resp.cookies


def test_anonymous_page_redirects_to_login():
    client = TestClient(service.app)
    resp = client.get("/admin", headers=HTML, follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login?next=/admin"


def test_redirect_keeps_source_query():
    client = TestClient(service.app)
    resp = client.get("/?from=github_readme", headers=HTML, follow_redirects=False)
    assert resp.status_code == 303
    from urllib.parse import unquote
    assert unquote(resp.headers["location"]) == "/login?next=/?from=github_readme"


def test_logout_revokes_page_access():
    client = TestClient(service.app)
    _login(client, "admin", "s3cret")
    assert client.get("/admin", headers=HTML).status_code == 200

    out = client.get("/logout", headers=HTML, follow_redirects=False)
    assert out.status_code == 303
    assert out.headers["location"] == "/login"

    after = client.get("/admin", headers=HTML, follow_redirects=False)
    assert after.status_code == 303
    assert after.headers["location"] == "/login?next=/admin"


def test_user_cookie_cannot_open_admin():
    client = TestClient(service.app)
    _login(client, "user", "change-me-user")
    assert client.get("/", headers=HTML).status_code == 200
    assert client.get("/admin", headers=HTML).status_code == 403


def test_login_keeps_next_path():
    client = TestClient(service.app)
    resp = _login(client, "user", "change-me-user", next_path="/admin")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/admin"


def test_external_next_is_ignored():
    client = TestClient(service.app)
    resp = _login(client, "user", "change-me-user", next_path="//evil.example.com")
    assert resp.headers["location"] == "/"


def test_me_returns_current_user():
    client = TestClient(service.app)
    _login(client, "user", "change-me-user")
    resp = client.get("/me")
    assert resp.status_code == 200
    assert resp.json() == {"username": "user", "role": "user"}


def test_me_requires_login():
    client = TestClient(service.app)
    assert client.get("/me").status_code == 401


def test_cookie_identity_wins_over_cached_basic():
    """浏览器可能缓存了 admin 的 Basic 凭证，登录 Cookie 必须优先。"""
    client = TestClient(service.app)
    _login(client, "user", "change-me-user")
    raw = base64.b64encode(b"admin:s3cret").decode("ascii")
    resp = client.get("/me", headers={"Authorization": f"Basic {raw}"})
    assert resp.status_code == 200
    assert resp.json() == {"username": "user", "role": "user"}


def test_token_expires(monkeypatch):
    monkeypatch.setattr(cfg, "AUTH_SESSION_MINUTES", -1)
    assert auth.verify(auth.issue("admin", "admin")) is None


def test_token_tampered():
    token = auth.issue("admin", "admin")
    tampered = token[:-1] + ("0" if token[-1] != "0" else "1")
    assert auth.verify(tampered) is None
    assert auth.verify(None) is None
    assert auth.verify("garbage") is None
