"""
job_store.py — Supabase-backed job state + file storage.

State lives in the `jobs` PostgreSQL table (JSONB blob mirrors old state.json).
Files (glyph thumbnails, font binaries) live in Supabase Storage bucket
"ttf-generator" under {job_id}/glyphs/ and {job_id}/output/ prefixes.

Public bucket means font files are downloadable via a stable URL without
going through the FastAPI server.  Thumbnails are in the same bucket; they
contain no sensitive data.
"""

import time
from typing import Any, Dict, Optional, Set

from supabase_client import STORAGE_BUCKET, SUPABASE_URL, supabase

MAX_JOB_AGE_HOURS = 24


def _public_url(path: str) -> str:
    """Return the public Storage URL for a given storage path."""
    return f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{path}"


class JobStore:

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def create_job(self, job_id: str) -> None:
        initial_state: Dict[str, Any] = {
            "status": "pending",
            "progress_pct": 0,
            "error_message": None,
            "created_at": time.time(),
            "glyph_manifest": [],
            "alignment_warnings": [],
            "font_files": {},
        }
        supabase.table("jobs").insert({
            "job_id": job_id,
            "state": initial_state,
        }).execute()

    def job_exists(self, job_id: str) -> bool:
        res = supabase.table("jobs").select("job_id").eq("job_id", job_id).execute()
        return len(res.data) > 0

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_state(self, job_id: str) -> Dict[str, Any]:
        res = supabase.table("jobs").select("state").eq("job_id", job_id).execute()
        if not res.data:
            return {}
        return res.data[0]["state"] or {}

    def update_state(self, job_id: str, **kwargs) -> None:
        """Merge kwargs into the job's JSONB state (top-level keys only)."""
        current = self.get_state(job_id)
        current.update(kwargs)
        supabase.table("jobs").update({"state": current}).eq("job_id", job_id).execute()

    # ------------------------------------------------------------------
    # Storage — glyph thumbnails
    # ------------------------------------------------------------------

    def upload_glyph_png(self, job_id: str, glyph_id: str, png_bytes: bytes) -> None:
        path = f"{job_id}/glyphs/{glyph_id}.png"
        supabase.storage.from_(STORAGE_BUCKET).upload(
            path, png_bytes,
            file_options={"content-type": "image/png", "upsert": "true"},
        )

    def download_glyph_png(self, job_id: str, glyph_id: str) -> Optional[bytes]:
        path = f"{job_id}/glyphs/{glyph_id}.png"
        try:
            return supabase.storage.from_(STORAGE_BUCKET).download(path)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Storage — font output files
    # ------------------------------------------------------------------

    def upload_font_file(self, job_id: str, filename: str, data: bytes, content_type: str) -> str:
        """Upload a font binary. Returns the storage path (not the full URL)."""
        path = f"{job_id}/output/{filename}"
        supabase.storage.from_(STORAGE_BUCKET).upload(
            path, data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        return path

    def get_font_public_url(self, job_id: str, filename: str) -> str:
        """Return the stable public URL for a font file."""
        return _public_url(f"{job_id}/output/{filename}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _delete_storage_files(self, job_id: str) -> None:
        """Remove all Storage objects for a job (best-effort)."""
        for prefix in [f"{job_id}/glyphs", f"{job_id}/output"]:
            try:
                files = supabase.storage.from_(STORAGE_BUCKET).list(prefix)
                if files:
                    paths = [f"{prefix}/{f['name']}" for f in files]
                    supabase.storage.from_(STORAGE_BUCKET).remove(paths)
            except Exception:
                pass

    def cleanup_old_jobs(
        self,
        max_age_hours: int = MAX_JOB_AGE_HOURS,
        skip_ids: Set[str] = None,
    ) -> int:
        """Delete jobs older than max_age_hours from DB and Storage."""
        cutoff = time.time() - max_age_hours * 3600
        skip_ids = skip_ids or set()

        res = supabase.table("jobs").select("job_id, state").execute()
        deleted = 0
        for row in res.data or []:
            job_id = row["job_id"]
            if job_id in skip_ids:
                continue
            state = row.get("state") or {}
            if state.get("created_at", 0) < cutoff:
                self._delete_storage_files(job_id)
                supabase.table("jobs").delete().eq("job_id", job_id).execute()
                deleted += 1
        return deleted


job_store = JobStore()
