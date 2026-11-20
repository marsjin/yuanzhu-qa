"""登录态令牌：HMAC 签名，默认 60 分钟有效，登录后写入 HttpOnly Cookie。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time

from customer_service.core import config as cfg


def _key() -> bytes:
    """签名密钥由账号口令派生：改密码后旧令牌自动失效。"""
    material = "|".join([
        cfg.AUTH_ADMIN_USERNAME, cfg.AUTH_ADMIN_PASSWORD,
        cfg.AUTH_USER_USERNAME, cfg.AUTH_USER_PASSWORD,
        cfg.AUTH_EXTRA_USERS,
    ])
    return hashlib.sha256(material.encode("utf-8")).digest()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def issue(username: str, role: str) -> str:
    """签发令牌：payload 为 用户名|角色|过期时间戳。"""
    expires = int(time.time()) + cfg.AUTH_SESSION_MINUTES * 60
    payload = _b64(f"{username}|{role}|{expires}".encode("utf-8"))
    signature = hmac.new(_key(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def verify(token: str | None) -> tuple[str, str] | None:
    """校验令牌，返回 (用户名, 角色)；签名不符、已过期或账号被移除时返回 None。"""
    if not token or "." not in token:
        return None
    payload, _, signature = token.partition(".")
    expected = hmac.new(_key(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        username, role, expires = _unb64(payload).decode("utf-8").rsplit("|", 2)
        if int(expires) < int(time.time()):
            return None
    except (ValueError, UnicodeDecodeError):
        return None
    if username not in cfg.auth_users():
        return None
    return (username, role)
