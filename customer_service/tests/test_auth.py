"""HTTP Basic 鉴权测试。"""

import base64

import pytest

from customer_service.app import service
from customer_service.core import config as cfg


def _header(username: str, password: str) -> str:
    encoded = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {encoded}"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setattr(cfg, "AUTH_USERNAME", "admin")
    monkeypatch.setattr(cfg, "AUTH_PASSWORD", "s3cret")


def test_valid_credentials():
    assert service._valid_basic_auth(_header("admin", "s3cret"))


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "Bearer token",
        "Basic not-base64",
        _header("admin", "wrong"),
        _header("other", "s3cret"),
        _header("admin", ""),
    ],
)
def test_invalid_credentials(header):
    assert not service._valid_basic_auth(header)


def test_unicode_credentials(monkeypatch):
    monkeypatch.setattr(cfg, "AUTH_PASSWORD", "强密码")
    assert service._valid_basic_auth(_header("admin", "强密码"))
