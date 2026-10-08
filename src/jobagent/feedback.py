"""Local thumbs up/down store. Not yet fed back into ranking (planned: preference-aware prelim)."""
import sqlite3
import time
from pathlib import Path

DB = Path("data/feedback.db")


def _conn():
    DB.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS feedback (job_id TEXT, title TEXT, company TEXT, "
              "vote INTEGER, ts REAL)")
    return c


def record(job_id: str, title: str, company: str, vote: int) -> None:
    with _conn() as c:
        c.execute("INSERT INTO feedback VALUES (?,?,?,?,?)", (job_id, title, company, vote, time.time()))


def counts() -> dict[str, int]:
    with _conn() as c:
        rows = c.execute("SELECT vote, COUNT(*) FROM feedback GROUP BY vote").fetchall()
    return {("up" if v > 0 else "down"): n for v, n in rows}
