"""Persistent, deployment-configurable query allowance for an anonymous demo."""

import os
import sqlite3
from datetime import datetime, timezone


class DailyLimitReached(Exception):
    pass


def reserve_query():
    limit = int(os.getenv("COMPLIGRAPH_DAILY_QUERY_LIMIT", "0"))
    if limit <= 0:
        return
    path = os.environ["COMPLIGRAPH_USAGE_DB"]
    day = datetime.now(timezone.utc).date().isoformat()
    with sqlite3.connect(path, timeout=5) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS usage (day TEXT PRIMARY KEY, queries INTEGER NOT NULL)"
        )
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT queries FROM usage WHERE day = ?", (day,)).fetchone()
        if row and row[0] >= limit:
            raise DailyLimitReached(
                f"This public demo has reached its shared {limit}-question daily allowance. Please try again after midnight UTC."
            )
        db.execute(
            "INSERT INTO usage(day, queries) VALUES (?, 1) ON CONFLICT(day) DO UPDATE SET queries = queries + 1",
            (day,),
        )
        db.execute("DELETE FROM usage WHERE day < ?", (day,))
