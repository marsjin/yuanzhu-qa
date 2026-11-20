"""HTTP Basic 鉴权与角色测试。"""

import base64

import pytest

from customer_service.app import service
from customer_service.core import config as cfg


def _header(username: str, password: str) -> str:
    encoded = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {encoded}"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setattr(cfg, "AUTH_ADMIN_USERNAME", "admin")
    monkeypatch.setattr(cfg, "AUTH_ADMIN_PASSWORD", "s3cret")
    monkeypatch.setattr(cfg, "AUTH_USER_USERNAME", "user")
    monkeypatch.setattr(cfg, "AUTH_USER_PASSWORD", "change-me-user")
    monkeypatch.setattr(cfg, "AUTH_EXTRA_USERS", "jerry:change-me-extra")


def test_admin_role():
    assert service._auth_role(_header("admin", "s3cret")) == "admin"


def test_user_role():
    assert service._auth_role(_header("user", "change-me-user")) == "user"


def test_extra_user_role():
    assert service._auth_role(_header("jerry", "change-me-extra")) == "user"


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "Bearer token",
        "Basic not-base64",
        _header("admin", "wrong"),
        _header("user", "s3cret"),
        _header("jerry", "change-me-user"),
        _header("other", "s3cret"),
        _header("admin", ""),
    ],
)
def test_invalid_credentials(header):
    assert service._auth_role(header) is None


def test_unicode_credentials(monkeypatch):
    monkeypatch.setattr(cfg, "AUTH_ADMIN_PASSWORD", "强密码")
    assert service._auth_role(_header("admin", "强密码")) == "admin"
