"""页面权限：admin 可访问全部，user 仅可对话。"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from customer_service.app import service
from customer_service.core import config as cfg


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setattr(cfg, "AUTH_ENABLED", True)
    monkeypatch.setattr(cfg, "AUTH_ADMIN_USERNAME", "admin")
    monkeypatch.setattr(cfg, "AUTH_ADMIN_PASSWORD", "s3cret")
    monkeypatch.setattr(cfg, "AUTH_USER_USERNAME", "user")
    monkeypatch.setattr(cfg, "AUTH_USER_PASSWORD", "change-me-user")
    monkeypatch.setattr(cfg, "AUTH_EXTRA_USERS", "jerry:change-me-extra")


def _auth(username: str, password: str) -> dict:
    raw = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {raw}"}


def test_anonymous_rejected():
    client = TestClient(service.app)
    assert client.get("/").status_code == 401
    assert client.get("/admin").status_code == 401


def test_admin_can_access_all():
    client = TestClient(service.app)
    assert client.get("/", headers=_auth("admin", "s3cret")).status_code == 200
    assert client.get("/admin", headers=_auth("admin", "s3cret")).status_code == 200


def test_admin_page_renders():
    client = TestClient(service.app)
    resp = client.get("/admin", headers=_auth("admin", "s3cret"))
    assert "管理后台" in resp.text
    assert "返回对话首页" in resp.text
    assert "/health" in resp.text
    assert "/events/stats" in resp.text


def test_user_can_chat_but_not_admin():
    client = TestClient(service.app)
    user = _auth("user", "change-me-user")
    assert client.get("/", headers=user).status_code == 200
    assert client.get("/health", headers=user).status_code == 200
    assert client.get("/admin", headers=user).status_code == 403
    assert client.get("/events/stats?days=7", headers=user).status_code == 403


def test_user_wrong_password():
    client = TestClient(service.app)
    assert client.get("/", headers=_auth("user", "wrong")).status_code == 401


def test_extra_user_can_chat_but_not_admin():
    client = TestClient(service.app)
    jerry = _auth("jerry", "change-me-extra")
    assert client.get("/", headers=jerry).status_code == 200
    assert client.get("/admin", headers=jerry).status_code == 403
