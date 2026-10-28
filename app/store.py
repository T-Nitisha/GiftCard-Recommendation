import asyncio
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

EVENT_WEIGHTS = {"purchase": 3.0, "click": 1.0, "dislike": -2.0}

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT NOT NULL, recipient_id TEXT NOT NULL DEFAULT '',
    interests TEXT NOT NULL DEFAULT '[]', updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, recipient_id)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL, recipient_id TEXT NOT NULL DEFAULT '',
    card_id TEXT NOT NULL, category TEXT NOT NULL DEFAULT '',
    event TEXT NOT NULL, ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_user ON events (user_id, recipient_id);
CREATE INDEX IF NOT EXISTS idx_events_card ON events (card_id);
"""


class ProfileStore:
    """SQLite-backed user/recipient profiles and gift interaction history."""

    def __init__(self, path: str | Path = "data/app.db"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._mem = sqlite3.connect(":memory:", check_same_thread=False) if self.path == ":memory:" else None
        self._run(lambda c: c.executescript(SCHEMA))

    def _run(self, fn):
        conn = self._mem or sqlite3.connect(self.path)
        try:
            with conn:
                return fn(conn)
        finally:
            if not self._mem:
                conn.close()

    async def _arun(self, fn):
        return await asyncio.to_thread(self._run, fn)

    async def set_interests(self, user_id: str, recipient_id: str, interests: list[str]) -> None:
        await self._arun(lambda c: c.execute(
            "INSERT INTO profiles (user_id, recipient_id, interests, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(user_id, recipient_id) DO UPDATE SET interests=excluded.interests, updated_at=excluded.updated_at",
            (user_id, recipient_id, json.dumps(interests), time.time())))

    async def record_event(self, user_id: str, recipient_id: str, card_id: str, category: str, event: str) -> None:
        if event not in EVENT_WEIGHTS:
            raise ValueError(f"event must be one of {sorted(EVENT_WEIGHTS)}")
        await self._arun(lambda c: c.execute(
            "INSERT INTO events (user_id, recipient_id, card_id, category, event, ts) VALUES (?,?,?,?,?,?)",
            (user_id, recipient_id, card_id, category, event, time.time())))

    async def get_profile(self, user_id: str, recipient_id: str = "") -> dict:
        def q(c):
            row = c.execute("SELECT interests FROM profiles WHERE user_id=? AND recipient_id=?",
                            (user_id, recipient_id)).fetchone()
            ev = c.execute("SELECT card_id, category, event FROM events WHERE user_id=? AND recipient_id=?",
                           (user_id, recipient_id)).fetchall()
            return row, ev

        row, events = await self._arun(q)
        gifted = sorted({cid for cid, _, e in events if e == "purchase"})
        disliked = sorted({cid for cid, _, e in events if e == "dislike"})
        cats = Counter()
        for _, cat, e in events:
            if cat:
                cats[cat] += EVENT_WEIGHTS[e]
        return {
            "interests": json.loads(row[0]) if row else [],
            "gifted": gifted,
            "disliked": disliked,
            "category_affinity": {k: v for k, v in cats.items() if v > 0},
        }

    async def popularity(self) -> dict[str, float]:
        """Global weighted interaction score per card."""
        def q(c):
            return c.execute("SELECT card_id, event, COUNT(*) FROM events GROUP BY card_id, event").fetchall()

        scores: Counter = Counter()
        for card_id, event, n in await self._arun(q):
            scores[card_id] += EVENT_WEIGHTS[event] * n
        return dict(scores)
