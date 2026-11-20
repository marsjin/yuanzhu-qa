"""会话存储：SQLite 保存历史消息与上一轮实体（支持多轮指代）。"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from customer_service.core import config as cfg


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
                """
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
