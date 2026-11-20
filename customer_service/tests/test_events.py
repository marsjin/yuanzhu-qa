"""点击统计：事件落库与按类型聚合。"""

from __future__ import annotations

import pytest

from customer_service.app import service
from customer_service.core.session import SessionStore

_LINK = "https://book.douban.com/subject/1322455/"


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "events.db")


def test_log_event_and_stats(store):
    store.log_event("buy_click", "s1", _LINK)
    store.log_event("buy_click", "s2", _LINK)
    store.log_event("other", "s1")
    assert store.event_stats(days=7) == {"buy_click": 2, "other": 1}


def test_event_endpoint_writes_store(monkeypatch, store):
    class _Pipeline:
        pass

    pipeline = _Pipeline()
    pipeline.store = store
    monkeypatch.setitem(service._state, "pipeline", pipeline)

    assert service.log_event(service.EventRequest(type="buy_click", url=_LINK, session_id="s9")) == {"ok": True}
    assert service.event_stats(days=1)["counts"] == {"buy_click": 1}
