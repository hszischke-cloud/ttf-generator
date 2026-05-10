"""
font_registry.py — Supabase-backed registry of user-saved fonts.

Saved fonts are excluded from the 24-hour auto-cleanup so they persist
indefinitely in Supabase Storage.
"""

import time
from typing import Any, Dict, List, Set

from supabase_client import supabase


def save_font(job_id: str, font_name: str, mode: str, glyph_count: int) -> None:
    supabase.table("saved_fonts").upsert({
        "job_id":      job_id,
        "font_name":   font_name,
        "mode":        mode,
        "glyph_count": glyph_count,
        "saved_at":    time.time(),
    }).execute()


def list_fonts() -> List[Dict[str, Any]]:
    res = supabase.table("saved_fonts").select("*").order("saved_at", desc=True).execute()
    return res.data or []


def delete_font(job_id: str) -> bool:
    res = supabase.table("saved_fonts").delete().eq("job_id", job_id).execute()
    return len(res.data) > 0


def saved_job_ids() -> Set[str]:
    """Return the set of job_ids that must not be auto-deleted."""
    res = supabase.table("saved_fonts").select("job_id").execute()
    return {row["job_id"] for row in (res.data or [])}
