"""SQLite-backed query result cache."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    backend TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (backend, query_hash)
);
"""


class QueryCache:
    """Append-or-overwrite cache of JSON-serializable payloads."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(_SCHEMA)

    def get(self, backend: str, query_hash: str) -> Any | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM cache WHERE backend = ? AND query_hash = ?",
                (backend, query_hash),
            ).fetchone()
        if row is None:
            return None
        payload, expires_at = row
        if expires_at < time.time():
            return None
        return json.loads(payload)

    def set(
        self,
        backend: str,
        query_hash: str,
        payload: Any,
        ttl_seconds: int = 30 * 24 * 3600,
    ) -> None:
        expires_at = time.time() + ttl_seconds
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache "
                "(backend, query_hash, payload, expires_at) VALUES (?, ?, ?, ?)",
                (backend, query_hash, json.dumps(payload), expires_at),
            )
