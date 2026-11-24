"""会话存储：SQLite 保存历史消息与上一轮实体（支持多轮指代）。"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from customer_service.core import config as cfg


def local_day(ts: float | None = None) -> str:
    """按配置时区把时间戳折算成统计日，避免容器 UTC 造成的跨天偏差。"""
    moment = time.time() if ts is None else ts
    return time.strftime("%Y-%m-%d", time.gmtime(moment + cfg.VISIT_TZ_OFFSET * 3600))


class SessionStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or cfg.SESSION_DB_PATH)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);
                CREATE TABLE IF NOT EXISTS session_state (
                    session_id TEXT PRIMARY KEY,
                    last_question TEXT NOT NULL DEFAULT '',
                    last_entities TEXT NOT NULL DEFAULT '[]',
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    session_id TEXT NOT NULL DEFAULT '',
                    detail TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_kind_time
                    ON events(kind, created_at);
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visited_at REAL NOT NULL,
                    visit_day TEXT NOT NULL,
                    path TEXT NOT NULL DEFAULT '/',
                    source TEXT NOT NULL DEFAULT 'direct',
                    medium TEXT NOT NULL DEFAULT '',
                    campaign TEXT NOT NULL DEFAULT '',
                    referer_host TEXT NOT NULL DEFAULT '',
                    visitor_id TEXT NOT NULL DEFAULT '',
                    visitor_hash TEXT NOT NULL DEFAULT '',
                    session_id TEXT NOT NULL DEFAULT '',
                    username TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL DEFAULT '',
                    ua_family TEXT NOT NULL DEFAULT '',
                    is_bot INTEGER NOT NULL DEFAULT 0,
                    status INTEGER NOT NULL DEFAULT 200
                );
                CREATE INDEX IF NOT EXISTS idx_visits_time
                    ON visits(visited_at);
                CREATE INDEX IF NOT EXISTS idx_visits_day_source
                    ON visits(visit_day, source);
                CREATE INDEX IF NOT EXISTS idx_visits_visitor
                    ON visits(visitor_id, visited_at);
                CREATE INDEX IF NOT EXISTS idx_visits_hash
                    ON visits(visitor_hash, visited_at);
                CREATE TABLE IF NOT EXISTS visitor_attribution (
                    visitor_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL DEFAULT 'direct',
                    medium TEXT NOT NULL DEFAULT '',
                    campaign TEXT NOT NULL DEFAULT '',
                    first_seen REAL NOT NULL,
                    attr_expires_at REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    first_visit_id INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS visitor_sessions (
                    session_id TEXT PRIMARY KEY,
                    visitor_id TEXT NOT NULL DEFAULT '',
                    username TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL DEFAULT '',
                    first_chat_at REAL NOT NULL,
                    last_chat_at REAL NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_vs_visitor
                    ON visitor_sessions(visitor_id, first_chat_at);
                CREATE INDEX IF NOT EXISTS idx_vs_first_chat
                    ON visitor_sessions(first_chat_at);
                """
            )
            conn.execute(
                "UPDATE events SET kind = 'click_answer' WHERE kind = 'buy_click'"
            )

    def ensure(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO session_state(session_id, updated_at) VALUES (?, ?)",
                (session_id, time.time()),
            )

    def append(self, session_id: str, role: str, content: str) -> None:
        self.ensure(session_id)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time()),
            )

    def history(self, session_id: str, limit: int = cfg.HISTORY_WINDOW) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def last_state(self, session_id: str) -> tuple[str, list[str]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_question, last_entities FROM session_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return "", []
        return row["last_question"], json.loads(row["last_entities"])

    def save_state(self, session_id: str, last_question: str, entities: list[str]) -> None:
        self.ensure(session_id)
        with self._connect() as conn:
            conn.execute(
                "UPDATE session_state SET last_question=?, last_entities=?, updated_at=? "
                "WHERE session_id=?",
                (last_question, json.dumps(entities, ensure_ascii=False), time.time(), session_id),
            )

    def reset(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_state WHERE session_id = ?", (session_id,))

    def log_event(self, kind: str, session_id: str = "", detail: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events(kind, session_id, detail, created_at) VALUES (?, ?, ?, ?)",
                (kind, session_id, detail, time.time()),
            )

    def event_stats(self, days: int = 7) -> dict[str, int]:
        since = time.time() - days * 86400
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT kind, COUNT(*) AS n FROM events WHERE created_at >= ? GROUP BY kind",
                (since,),
            ).fetchall()
        return {r["kind"]: r["n"] for r in rows}

    # ---- 访问来源：写入 ----
    def log_visit(self, *, visited_at: float, path: str, source: str, medium: str = "",
                  campaign: str = "", referer_host: str = "", visitor_id: str = "",
                  visitor_hash: str = "", username: str = "", role: str = "",
                  ua_family: str = "", is_bot: bool = False, status: int = 200) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO visits(visited_at, visit_day, path, source, medium, campaign, "
                "referer_host, visitor_id, visitor_hash, username, role, ua_family, is_bot, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (visited_at, local_day(visited_at), path, source, medium, campaign,
                 referer_host, visitor_id, visitor_hash, username, role, ua_family,
                 1 if is_bot else 0, status),
            )
            return int(cursor.lastrowid)

    def upsert_attribution(self, visitor_id: str, source: str, medium: str, campaign: str,
                           visited_at: float, visit_id: int) -> None:
        """首次归因；已有 direct 记录在窗口内被非 direct 来源纠正。"""
        expires = visited_at + cfg.VISIT_ATTRIBUTION_DAYS * 86400
        with self._connect() as conn:
            row = conn.execute(
                "SELECT source, first_seen, attr_expires_at FROM visitor_attribution WHERE visitor_id = ?",
                (visitor_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO visitor_attribution(visitor_id, source, medium, campaign, "
                    "first_seen, attr_expires_at, last_seen, first_visit_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (visitor_id, source, medium, campaign, visited_at, expires,
                     visited_at, visit_id),
                )
                return
            upgrade = (
                row["source"] == "direct" and source != "direct"
                and visited_at <= row["attr_expires_at"]
            )
            if upgrade:
                conn.execute(
                    "UPDATE visitor_attribution SET source = ?, medium = ?, campaign = ?, "
                    "first_seen = ?, attr_expires_at = ?, last_seen = ?, first_visit_id = ? "
                    "WHERE visitor_id = ?",
                    (source, medium, campaign, visited_at, expires, visited_at,
                     visit_id, visitor_id),
                )
            else:
                conn.execute(
                    "UPDATE visitor_attribution SET last_seen = ? WHERE visitor_id = ?",
                    (visited_at, visitor_id),
                )

    def bind_visitor_session(self, session_id: str, visitor_id: str, username: str = "",
                             role: str = "", at: float | None = None) -> None:
        moment = time.time() if at is None else at
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO visitor_sessions(session_id, visitor_id, username, role, "
                "first_chat_at, last_chat_at, message_count) VALUES (?, ?, ?, ?, ?, ?, 1) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "last_chat_at = excluded.last_chat_at, "
                "message_count = visitor_sessions.message_count + 1, "
                "username = CASE WHEN excluded.username <> '' THEN excluded.username "
                "ELSE visitor_sessions.username END",
                (session_id, visitor_id, username, role, moment, moment),
            )
            if visitor_id:
                # 把 30 分钟内的入口访问补上会话 ID，便于明细对照
                conn.execute(
                    "UPDATE visits SET session_id = ? WHERE visitor_id = ? "
                    "AND session_id = '' AND visited_at >= ?",
                    (session_id, visitor_id, moment - 1800),
                )

    # ---- 访问来源：查询 ----
    def visit_totals(self, from_ts: float, to_ts: float) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS pv, COUNT(DISTINCT NULLIF(visitor_id, '')) AS uv, "
                "COALESCE(SUM(is_bot), 0) AS bot_pv FROM visits "
                "WHERE visited_at >= ? AND visited_at < ?",
                (from_ts, to_ts),
            ).fetchone()
        pv = row["pv"] or 0
        return {
            "pv": pv,
            "uv": row["uv"] or 0,
            "bot_pv": row["bot_pv"] or 0,
            "bot_ratio": round((row["bot_pv"] or 0) / pv, 4) if pv else 0.0,
        }

    def visit_by_source(self, from_day: str, to_day: str, include_bots: bool = False) -> list[dict]:
        """来源维度：PV/UV 可切换是否含爬虫，爬虫占比始终以全部访问为分母。"""
        sql = (
            "SELECT source, COUNT(*) AS pv_all, "
            "COALESCE(SUM(CASE WHEN is_bot = 0 THEN 1 ELSE 0 END), 0) AS pv_human, "
            "COUNT(DISTINCT NULLIF(visitor_id, '')) AS uv_all, "
            "COUNT(DISTINCT CASE WHEN is_bot = 0 THEN visitor_id END) AS uv_human, "
            "COALESCE(SUM(is_bot), 0) AS bot_pv FROM visits "
            "WHERE visit_day BETWEEN ? AND ? GROUP BY source ORDER BY pv_all DESC"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, (from_day, to_day)).fetchall()
        result = []
        for row in rows:
            pv = row["pv_all"] if include_bots else row["pv_human"]
            result.append({
                "source": row["source"],
                "pv": pv,
                "uv": row["uv_all"] if include_bots else row["uv_human"],
                "bot_pv": row["bot_pv"],
                "bot_ratio": round(row["bot_pv"] / row["pv_all"], 4) if row["pv_all"] else 0.0,
            })
        return result

    def visit_by_day(self, from_day: str, to_day: str, include_bots: bool = False) -> list[dict]:
        sql = (
            "SELECT visit_day AS day, COUNT(*) AS pv, "
            "COUNT(DISTINCT NULLIF(visitor_id, '')) AS uv FROM visits "
            "WHERE visit_day BETWEEN ? AND ?"
        )
        if not include_bots:
            sql += " AND is_bot = 0"
        sql += " GROUP BY visit_day ORDER BY visit_day"
        with self._connect() as conn:
            rows = conn.execute(sql, (from_day, to_day)).fetchall()
        return [dict(r) for r in rows]

    def visit_chat_uv(self, from_ts: float, to_ts: float) -> dict[str, int]:
        """来源 → 该来源归因访客中真正发过消息的人数（受归因窗口约束）。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT a.source AS source, COUNT(DISTINCT vs.visitor_id) AS n "
                "FROM visitor_sessions vs "
                "JOIN visitor_attribution a ON a.visitor_id = vs.visitor_id "
                "WHERE vs.first_chat_at >= ? AND vs.first_chat_at < ? "
                "AND vs.first_chat_at <= a.attr_expires_at "
                "GROUP BY a.source",
                (from_ts, to_ts),
            ).fetchall()
        return {r["source"]: r["n"] for r in rows}

    def list_visits(self, from_day: str, to_day: str, source: str = "",
                    include_bots: bool = False, page: int = 1,
                    page_size: int = 20) -> tuple[int, list[dict]]:
        clauses = ["visit_day BETWEEN ? AND ?"]
        params: list = [from_day, to_day]
        if source:
            clauses.append("source = ?")
            params.append(source)
        if not include_bots:
            clauses.append("is_bot = 0")
        where = " AND ".join(clauses)
        offset = (page - 1) * page_size
        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) AS n FROM visits WHERE {where}", params
            ).fetchone()["n"]
            rows = conn.execute(
                "SELECT id, visited_at, visit_day, path, source, medium, campaign, "
                "referer_host, visitor_id, session_id, username, role, ua_family, "
                "is_bot, status FROM visits WHERE " + where +
                " ORDER BY visited_at DESC, id DESC LIMIT ? OFFSET ?",
                params + [page_size, offset],
            ).fetchall()
        return int(total or 0), [dict(r) for r in rows]

    def visit_detail(self, visit_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, visited_at, visit_day, path, source, medium, campaign, "
                "referer_host, visitor_id, visitor_hash, session_id, username, role, "
                "ua_family, is_bot, status FROM visits WHERE id = ?",
                (visit_id,),
            ).fetchone()
        return dict(row) if row else None

    def visit_questions(self, from_ts: float, to_ts: float, source: str = "",
                        limit: int = 100) -> list[dict]:
        clauses = [
            "m.role = 'user'",
            "m.created_at >= ?",
            "m.created_at < ?",
            "m.created_at <= a.attr_expires_at",
        ]
        params: list = [from_ts, to_ts]
        if source:
            clauses.append("a.source LIKE ?")
            params.append(source + "%")
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT a.visitor_id, a.source, a.first_seen, vs.session_id, vs.username, "
                "m.content AS question, m.created_at "
                "FROM visitor_attribution a "
                "JOIN visitor_sessions vs ON vs.visitor_id = a.visitor_id "
                "JOIN messages m ON m.session_id = vs.session_id "
                "WHERE " + " AND ".join(clauses) +
                " ORDER BY m.created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def purge_visits(self, before_ts: float) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM visits WHERE visited_at < ?", (before_ts,))
            return int(cursor.rowcount or 0)
