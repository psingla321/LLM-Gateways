"""SQLite-backed usage logger — every gateway call produces one row."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class RequestLog:
    request_id:       str
    timestamp:        str
    team:             str
    user_id:          str
    task_type:        str      # "simple" | "complex"
    estimated_tokens: int
    pii_masked:       bool
    pii_types_found:  str      # JSON-encoded list, e.g. '["SSN","EMAIL"]'
    model_attempted:  str      # primary model chosen by router
    model_used:       str      # actual model that responded
    fallback_used:    bool
    tokens_in:        int
    tokens_out:       int
    cost_usd:         float
    latency_ms:       int
    cache_hit:        bool
    provider:         str
    summary_length:   int


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS request_logs (
    request_id       TEXT PRIMARY KEY,
    timestamp        TEXT,
    team             TEXT,
    user_id          TEXT,
    task_type        TEXT,
    estimated_tokens INTEGER,
    pii_masked       INTEGER,
    pii_types_found  TEXT,
    model_attempted  TEXT,
    model_used       TEXT,
    fallback_used    INTEGER,
    tokens_in        INTEGER,
    tokens_out       INTEGER,
    cost_usd         REAL,
    latency_ms       INTEGER,
    cache_hit        INTEGER,
    provider         TEXT,
    summary_length   INTEGER
)
"""


class UsageLogger:

    def __init__(self, db_path: str = "usage.db"):
        self._db = db_path
        with sqlite3.connect(self._db) as conn:
            conn.execute(_CREATE_TABLE)

    # ── Write ─────────────────────────────────────────────────────────────────

    def log(self, entry: RequestLog) -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO request_logs VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    entry.request_id, entry.timestamp, entry.team, entry.user_id,
                    entry.task_type, entry.estimated_tokens,
                    int(entry.pii_masked), entry.pii_types_found,
                    entry.model_attempted, entry.model_used,
                    int(entry.fallback_used), entry.tokens_in, entry.tokens_out,
                    entry.cost_usd, entry.latency_ms, int(entry.cache_hit),
                    entry.provider, entry.summary_length,
                ),
            )

    def clear_requests(self) -> int:
        with sqlite3.connect(self._db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]
            conn.execute("DELETE FROM request_logs")
        return count

    # ── Read — Dashboard queries ──────────────────────────────────────────────

    def team_budgets(self) -> List[dict]:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute("""
                SELECT
                    team,
                    COUNT(*)                              AS requests,
                    ROUND(SUM(cost_usd), 6)               AS total_cost,
                    SUM(tokens_in + tokens_out)           AS total_tokens,
                    ROUND(AVG(latency_ms))                AS avg_latency_ms,
                    SUM(cache_hit)                        AS cache_hits,
                    SUM(CASE WHEN task_type='complex' THEN 1 ELSE 0 END) AS complex_count
                FROM request_logs
                GROUP BY team
                ORDER BY total_cost DESC
            """).fetchall()
        cols = ["team", "requests", "total_cost", "total_tokens",
                "avg_latency_ms", "cache_hits", "complex_count"]
        return [dict(zip(cols, r)) for r in rows]

    def model_stats(self) -> List[dict]:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute("""
                SELECT model_used,
                       COUNT(*)              AS requests,
                       ROUND(SUM(cost_usd),6) AS total_cost,
                       ROUND(AVG(latency_ms)) AS avg_latency_ms
                FROM request_logs
                GROUP BY model_used
                ORDER BY requests DESC
            """).fetchall()
        cols = ["model", "requests", "total_cost", "avg_latency_ms"]
        return [dict(zip(cols, r)) for r in rows]

    def summary_stats(self) -> dict:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute("""
                SELECT COUNT(*),
                       ROUND(SUM(cost_usd), 6),
                       ROUND(AVG(latency_ms)),
                       SUM(cache_hit),
                       SUM(pii_masked),
                       SUM(fallback_used)
                FROM request_logs
            """).fetchone()
        total = row[0] or 1
        return {
            "total_requests":  row[0] or 0,
            "total_cost_usd":  row[1] or 0.0,
            "avg_latency_ms":  row[2] or 0,
            "cache_hits":      row[3] or 0,
            "cache_hit_rate":  round((row[3] or 0) / total * 100, 1),
            "pii_masked_count": row[4] or 0,
            "fallback_count":  row[5] or 0,
        }

    def recent_requests(self, limit: int = 25) -> List[dict]:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute("""
                SELECT request_id, timestamp, team, user_id, task_type,
                       model_used, fallback_used, tokens_in, tokens_out,
                       ROUND(cost_usd, 6) AS cost_usd,
                       latency_ms, cache_hit, pii_masked, provider
                FROM request_logs
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()
        cols = ["request_id", "timestamp", "team", "user_id", "task_type",
                "model_used", "fallback_used", "tokens_in", "tokens_out",
                "cost_usd", "latency_ms", "cache_hit", "pii_masked", "provider"]
        return [dict(zip(cols, r)) for r in rows]
