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
    store.log_event("click_answer", "s1", _LINK)
    store.log_event("click_answer", "s2", _LINK)
    store.log_event("other", "s1")
    assert store.event_stats(days=7) == {"click_answer": 2, "other": 1}


def test_legacy_buy_click_migrated(tmp_path):
    db = tmp_path / "legacy.db"
    store = SessionStore(db)
    store.log_event("buy_click", "s1", _LINK)
    assert "buy_click" in store.event_stats(days=7)

    reopened = SessionStore(db)
    counts = reopened.event_stats(days=7)
    assert "buy_click" not in counts
    assert counts["click_answer"] == 1


def test_event_endpoint_writes_store(monkeypatch, store):
    class _Pipeline:
        pass

    pipeline = _Pipeline()
    pipeline.store = store
    monkeypatch.setitem(service._state, "pipeline", pipeline)

    assert service.log_event(service.EventRequest(type="impression_default", url=_LINK)) == {"ok": True}
    assert service.event_stats(days=1)["counts"] == {"impression_default": 1}


def test_stats_splits_impression_and_click(monkeypatch, store):
    class _Pipeline:
        pass

    pipeline = _Pipeline()
    pipeline.store = store
    monkeypatch.setitem(service._state, "pipeline", pipeline)

    for kind in ("impression_default", "click_default", "impression_answer", "impression_answer"):
        service.log_event(service.EventRequest(type=kind, url=_LINK))

    stats = service.event_stats(days=1)
    assert stats["impressions"] == {"default": 1, "answer": 2}
    assert stats["clicks"] == {"default": 1, "answer": 0}
    assert stats["ctr"]["default"] == 1.0
    assert stats["ctr"]["answer"] == 0.0
