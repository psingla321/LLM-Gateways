"""Two-layer response cache: in-memory (L1) → SQLite (L2).

The cache key is a SHA-256 hash of the *masked* claim text so PII-stripped
content is cached but the original sensitive text never touches the DB.
"""

import hashlib
import sqlite3
from typing import Optional


class ResponseCache:

    def __init__(self, db_path: str = "usage.db"):
        self._memory: dict = {}          # L1 — dict[key, {response, model}]
        self._db = db_path
        self._init_db()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key   TEXT PRIMARY KEY,
                    response    TEXT    NOT NULL,
                    model       TEXT    NOT NULL,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, masked_text: str) -> Optional[dict]:
        """Return cached {response, model} or None."""
        key = self._make_key(masked_text)
        if key in self._memory:
            return self._memory[key]
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT response, model FROM response_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row:
            hit = {"response": row[0], "model": row[1]}
            self._memory[key] = hit   # promote to L1
            return hit
        return None

    def set(self, masked_text: str, response: str, model: str) -> None:
        """Store response in both L1 and L2."""
        key = self._make_key(masked_text)
        entry = {"response": response, "model": model}
        self._memory[key] = entry
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO response_cache (cache_key, response, model) VALUES (?, ?, ?)",
                (key, response, model),
            )

    def size(self) -> int:
        """Number of cached entries in SQLite."""
        with sqlite3.connect(self._db) as conn:
            return conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:20]
