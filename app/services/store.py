"""会话与消息持久化：SQLite（默认）/ MySQL（配置 DSN 后自动切换）。"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any

from app.config import settings

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    message_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    intent TEXT DEFAULT '',
    confidence REAL DEFAULT 0,
    citations TEXT DEFAULT '[]',
    trace TEXT DEFAULT '{}',
    latency_ms INTEGER DEFAULT 0,
    feedback INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, created_at);
"""

MYSQL_DDL = [
    """CREATE TABLE IF NOT EXISTS sessions (
        id VARCHAR(48) PRIMARY KEY,
        title VARCHAR(255) NOT NULL,
        category VARCHAR(64) DEFAULT '',
        created_at DOUBLE NOT NULL,
        updated_at DOUBLE NOT NULL,
        message_count INT DEFAULT 0
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS messages (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        session_id VARCHAR(48) NOT NULL,
        role VARCHAR(16) NOT NULL,
        content MEDIUMTEXT NOT NULL,
        intent VARCHAR(32) DEFAULT '',
        confidence DOUBLE DEFAULT 0,
        citations MEDIUMTEXT,
        trace MEDIUMTEXT,
        latency_ms INT DEFAULT 0,
        feedback INT DEFAULT 0,
        created_at DOUBLE NOT NULL,
        INDEX idx_msg_session (session_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]


class Store:
    def __init__(self) -> None:
        self.backend = "sqlite"
        self._lock = threading.RLock()
        self._conn = None
        self._init_sqlite() if settings.db_backend != "mysql" else self._init_mysql()

    # ---------- 初始化 ----------
    def _init_sqlite(self) -> None:
        self._conn = sqlite3.connect(str(settings.sqlite_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SQLITE_DDL)
            self._conn.commit()
        self.backend = "sqlite"

    def _init_mysql(self) -> None:
        try:
            import pymysql

            user, rest = settings.mysql_dsn.split(":", 1)
            pwd, hostpart = rest.split("@", 1)
            hostport, db = hostpart.split("/", 1)
            host, port = (hostport.split(":") + ["3306"])[:2]
            self._conn = pymysql.connect(host=host, port=int(port), user=user, password=pwd, database=db, charset="utf8mb4", autocommit=True)
            with self._lock:
                cur = self._conn.cursor()
                for ddl in MYSQL_DDL:
                    cur.execute(ddl)
                cur.close()
            self.backend = "mysql"
        except Exception:
            self._init_sqlite()

    @property
    def ph(self) -> str:
        return "%s" if self.backend == "mysql" else "?"

    def _exec(self, sql: str, args: tuple = (), fetch: str | None = None):
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute(sql.replace("?", self.ph), args)
                if fetch == "all":
                    return [dict(r) if isinstance(r, dict) else dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]
                if fetch == "one":
                    r = cur.fetchone()
                    if r is None:
                        return None
                    return dict(r) if isinstance(r, dict) else dict(zip([d[0] for d in cur.description], r))
                if self.backend == "sqlite":
                    self._conn.commit()
                return cur.rowcount
            finally:
                cur.close()

    # ---------- 会话 ----------
    def create_session(self, title: str = "新会话", category: str = "") -> dict:
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        self._exec("INSERT INTO sessions (id,title,category,created_at,updated_at,message_count) VALUES (?,?,?,?,?,0)", (sid, title[:120], category, now, now))
        return {"id": sid, "title": title, "category": category, "created_at": now, "updated_at": now, "message_count": 0}

    def list_sessions(self, limit: int = 50) -> list[dict]:
        return self._exec("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,), fetch="all") or []

    def get_session(self, sid: str) -> dict | None:
        return self._exec("SELECT * FROM sessions WHERE id=?", (sid,), fetch="one")

    def touch_session(self, sid: str, title: str | None = None) -> None:
        now = time.time()
        if title:
            self._exec("UPDATE sessions SET updated_at=?, title=?, message_count=message_count+2 WHERE id=?", (now, title[:120], sid))
        else:
            self._exec("UPDATE sessions SET updated_at=?, message_count=message_count+2 WHERE id=?", (now, sid))

    def rename_session(self, sid: str, title: str) -> None:
        self._exec("UPDATE sessions SET title=? WHERE id=?", (title[:120], sid))

    def delete_session(self, sid: str) -> None:
        self._exec("DELETE FROM messages WHERE session_id=?", (sid,))
        self._exec("DELETE FROM sessions WHERE id=?", (sid,))

    def clear_all(self) -> None:
        self._exec("DELETE FROM messages", ())
        self._exec("DELETE FROM sessions", ())

    # ---------- 消息 ----------
    def add_message(self, session_id: str, role: str, content: str, **kw) -> int:
        now = time.time()
        self._exec(
            "INSERT INTO messages (session_id,role,content,intent,confidence,citations,trace,latency_ms,feedback,created_at) VALUES (?,?,?,?,?,?,?,?,0,?)",
            (
                session_id,
                role,
                content,
                kw.get("intent", ""),
                float(kw.get("confidence", 0)),
                json.dumps(kw.get("citations", []), ensure_ascii=False),
                json.dumps(kw.get("trace", {}), ensure_ascii=False),
                int(kw.get("latency_ms", 0)),
                now,
            ),
        )
        row = self._exec("SELECT last_insert_rowid() AS id" if self.backend == "sqlite" else "SELECT LAST_INSERT_ID() AS id", fetch="one")
        return int(row["id"]) if row else 0

    def list_messages(self, session_id: str) -> list[dict]:
        rows = self._exec("SELECT * FROM messages WHERE session_id=? ORDER BY id ASC", (session_id,), fetch="all") or []
        for r in rows:
            try:
                r["citations"] = json.loads(r.get("citations") or "[]")
            except Exception:
                r["citations"] = []
            try:
                r["trace"] = json.loads(r.get("trace") or "{}")
            except Exception:
                r["trace"] = {}
        return rows

    def set_feedback(self, msg_id: int, value: int) -> None:
        self._exec("UPDATE messages SET feedback=? WHERE id=?", (int(value), msg_id))

    # ---------- 统计 ----------
    def stats(self) -> dict:
        s = self._exec("SELECT COUNT(*) AS n FROM sessions", fetch="one") or {"n": 0}
        m = self._exec("SELECT COUNT(*) AS n FROM messages", fetch="one") or {"n": 0}
        f = self._exec("SELECT COUNT(*) AS n FROM messages WHERE feedback=1", fetch="one") or {"n": 0}
        bad = self._exec("SELECT COUNT(*) AS n FROM messages WHERE feedback=-1", fetch="one") or {"n": 0}
        lat = self._exec("SELECT AVG(latency_ms) AS a FROM messages WHERE role='assistant' AND latency_ms>0", fetch="one") or {"a": 0}
        return {"backend": self.backend, "sessions": s["n"], "messages": m["n"], "good": f["n"], "bad": bad["n"], "avg_latency_ms": round(lat["a"] or 0)}


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
