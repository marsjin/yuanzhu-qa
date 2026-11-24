"""访问来源：来源解析、PV/UV、归因、明细与提问关联。"""

from __future__ import annotations

import time
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

from customer_service.app import service
from customer_service.app.pipeline import AnswerResult
from customer_service.core import config as cfg
from customer_service.core.session import SessionStore

HTML = {"accept": "text/html"}


class _Pipeline:
    """最小流水线桩：既能提供 store，也能响应 /chat。"""

    def __init__(self, store: SessionStore) -> None:
        self.store = store

    def answer(self, session_id: str, message: str) -> AnswerResult:
        return AnswerResult("ok", "来源：测试", "fast", "book_qa", session_id=session_id)


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "visits.db")


@pytest.fixture
def client(monkeypatch, store):
    monkeypatch.setattr(cfg, "AUTH_ENABLED", False)
    monkeypatch.setitem(service._state, "pipeline", _Pipeline(store))
    monkeypatch.setitem(service._last_purge, "ts", 0.0)
    return TestClient(service.app)


def _rows(store: SessionStore) -> list[dict]:
    with store._connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM visits ORDER BY id")]


def test_source_priority_from_over_utm(client, store):
    client.get("/?from=github_readme&utm_source=github&utm_medium=readme", headers=HTML)
    row = _rows(store)[0]
    assert row["source"] == "github_readme"
    assert row["medium"] == "readme"
    assert row["visitor_id"]
    assert client.cookies.get(cfg.VISITOR_COOKIE_NAME)


def test_source_utm_maps_github(client, store):
    client.get("/?utm_source=gh", headers=HTML)
    assert _rows(store)[0]["source"] == "github_unknown"


def test_source_direct_without_signal(client, store):
    client.get("/", headers=HTML)
    row = _rows(store)[0]
    assert row["source"] == "direct"
    assert row["referer_host"] == ""


def test_source_github_from_referer_only(client, store):
    client.get("/", headers={**HTML, "referer": "https://github.com/owner/repo"})
    row = _rows(store)[0]
    assert row["source"] == "github_unknown"
    assert row["referer_host"] == "github.com"


def test_self_referer_is_direct(client, store):
    client.get("/", headers={**HTML, "referer": "http://testserver/"})
    assert _rows(store)[0]["source"] == "direct"


def test_bot_flagged_and_kept(client, store):
    client.get("/", headers={**HTML, "user-agent": "Googlebot/2.1"})
    row = _rows(store)[0]
    assert row["is_bot"] == 1
    assert row["ua_family"] == "Other"


def test_login_redirect_preserves_source_query(monkeypatch, store):
    monkeypatch.setattr(cfg, "AUTH_ENABLED", True)
    monkeypatch.setitem(service._state, "pipeline", _Pipeline(store))
    client = TestClient(service.app)
    resp = client.get("/?from=github_readme", headers=HTML, follow_redirects=False)
    assert resp.status_code == 303
    assert unquote(resp.headers["location"]) == "/login?next=/?from=github_readme"


def test_stats_pv_uv_and_bots(client, store):
    now = time.time()
    for visitor, bot in (("v1", False), ("v1", False), ("v2", False), ("v3", True)):
        store.log_visit(visited_at=now, path="/", source="github_readme",
                        visitor_id=visitor, is_bot=bot)
    data = client.get("/visits/stats?days=1").json()
    assert data["pv"] == 4
    assert data["uv"] == 3
    assert data["bot_pv"] == 1
    assert data["bot_ratio"] == 0.25

    human = {r["source"]: r for r in
             client.get("/visits/stats?days=1&include_bots=0").json()["by_source"]}
    assert human["github_readme"]["pv"] == 3
    assert human["github_readme"]["uv"] == 2


def test_list_visits_pagination(client, store):
    now = time.time()
    for i in range(25):
        store.log_visit(visited_at=now - i, path="/", source="github_readme",
                        visitor_id="v1")
    first = client.get("/visits?days=1&page=1&page_size=10").json()
    assert first["total"] == 25
    assert len(first["items"]) == 10
    third = client.get("/visits?days=1&page=3&page_size=10").json()
    assert len(third["items"]) == 5


def test_visit_questions_join_by_attribution(client, store):
    now = time.time()
    visit_id = store.log_visit(visited_at=now, path="/", source="github_readme",
                               visitor_id="v1")
    store.upsert_attribution("v1", "github_readme", "readme", "book_cs", now, visit_id)
    store.bind_visitor_session("web-1", "v1", "jerry", "user", now)
    store.append("web-1", "user", "丁元英是谁？")
    store.append("web-1", "assistant", "丁元英是……")

    items = client.get("/visits/questions?days=1&source=github").json()["items"]
    assert [it["question"] for it in items] == ["丁元英是谁？"]
    assert items[0]["username"] == "jerry"
    assert client.get("/visits/questions?days=1&source=other").json()["items"] == []


def test_chat_binds_visitor_and_backfills_visit(client, store):
    client.get("/?from=github_readme", headers=HTML)
    resp = client.post("/chat", json={"session_id": "web-1", "message": "丁元英是谁？"})
    assert resp.status_code == 200

    with store._connect() as conn:
        session = dict(conn.execute("SELECT * FROM visitor_sessions").fetchone())
        visit = dict(conn.execute("SELECT * FROM visits").fetchone())
    assert session["visitor_id"] == visit["visitor_id"]
    assert session["message_count"] == 1
    assert visit["session_id"] == "web-1"


def test_direct_attribution_upgraded_by_github(client, store):
    start = time.time()
    first = store.log_visit(visited_at=start, path="/", source="direct", visitor_id="v1")
    store.upsert_attribution("v1", "direct", "", "", start, first)
    second = store.log_visit(visited_at=start + 10, path="/", source="github_readme",
                             visitor_id="v1")
    store.upsert_attribution("v1", "github_readme", "readme", "", start + 10, second)

    with store._connect() as conn:
        row = dict(conn.execute("SELECT * FROM visitor_attribution").fetchone())
    assert row["source"] == "github_readme"
    assert row["first_visit_id"] == second


def test_purge_visits(store):
    store.log_visit(visited_at=time.time() - 100 * 86400, path="/", source="direct",
                    visitor_id="old")
    store.log_visit(visited_at=time.time(), path="/", source="direct", visitor_id="new")
    assert store.purge_visits(time.time() - 90 * 86400) == 1
    assert [r["visitor_id"] for r in _rows(store)] == ["new"]
