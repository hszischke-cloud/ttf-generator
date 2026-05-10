"""
font_registry.py — SQLite-backed registry of user-saved fonts.

Stored inside JOBS_DIR so the registry lives on the same persistent volume
as the job data.  Saved fonts are excluded from the 24-hour auto-cleanup.
"""

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Set

from job_store import JOBS_DIR

_DB_PATH = JOBS_DIR / "font_registry.db"


def _conn() -> sqlite3.Connection:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(_DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def _init_db() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS saved_fonts (
                job_id      TEXT PRIMARY KEY,
                font_name   TEXT NOT NULL,
                mode        TEXT NOT NULL DEFAULT 'print',
                glyph_count INTEGER NOT NULL DEFAULT 0,
                saved_at    REAL NOT NULL
            )
        """)


_init_db()


def save_font(job_id: str, font_name: str, mode: str, glyph_count: int) -> None:
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO saved_fonts "
            "(job_id, font_name, mode, glyph_count, saved_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, font_name, mode, glyph_count, time.time()),
        )


def list_fonts() -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT job_id, font_name, mode, glyph_count, saved_at "
            "FROM saved_fonts ORDER BY saved_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_font(job_id: str) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM saved_fonts WHERE job_id = ?", (job_id,))
    return cur.rowcount > 0


def saved_job_ids() -> Set[str]:
    """Return the set of job_ids in the registry (these are never auto-deleted)."""
    with _conn() as con:
        rows = con.execute("SELECT job_id FROM saved_fonts").fetchall()
    return {row["job_id"] for row in rows}
