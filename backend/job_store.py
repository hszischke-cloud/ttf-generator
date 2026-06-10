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
from typing import Any, Callable, Dict, Optional, Set

from supabase_client import STORAGE_BUCKET, SUPABASE_URL, supabase

MAX_JOB_AGE_HOURS = 24


def _public_url(path: str) -> str:
    """Return the public Storage URL for a given storage path."""
    return f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{path}"


def _retry(fn: Callable, max_attempts: int = 4, initial_delay: float = 1.0):
    """
    Call fn() and retry on any exception with exponential backoff.

    HTTP/2 flow-control errors from httpcore (_read_incoming_data) are
    transient — a fresh connection on the next attempt usually succeeds.
    Delays: 1 s, 2 s, 4 s (three retries after the first attempt).
    """
    delay = initial_delay
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_attempts - 1:
                raise
            print(
                f"[job_store] attempt {attempt + 1}/{max_attempts} failed: "
                f"{type(exc).__name__}: {exc} — retrying in {delay:.0f}s"
            )
            time.sleep(delay)
            delay *= 2


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

    # Whether the server-side `patch_job_state` RPC is available. None = not yet
    # probed; True/False after the first call. Avoids re-attempting the RPC on
    # every update once we've learned it isn't deployed.
    _patch_rpc_ok: Optional[bool] = None

    def update_state(self, job_id: str, **kwargs) -> None:
        """
        Merge kwargs into the job's JSONB state (top-level keys only).

        Prefers the server-side `patch_job_state` RPC, which merges only the
        changed keys in the database — the job state blob carries every glyph's
        svg_paths/pen_paths, so a Python-side read-modify-write round-trips
        several MB just to bump progress_pct. Falls back to read-modify-write if
        the RPC isn't deployed (e.g. schema.sql not re-run yet).
        """
        if JobStore._patch_rpc_ok is not False:
            try:
                supabase.rpc(
                    "patch_job_state", {"p_job_id": job_id, "p_patch": kwargs}
                ).execute()
                JobStore._patch_rpc_ok = True
                return
            except Exception:
                # Function missing (PGRST202) or other RPC failure: fall back
                # and don't try the RPC again this process.
                JobStore._patch_rpc_ok = False

        current = self.get_state(job_id)
        current.update(kwargs)
        supabase.table("jobs").update({"state": current}).eq("job_id", job_id).execute()

    # ------------------------------------------------------------------
    # Storage — glyph thumbnails
    # ------------------------------------------------------------------

    def upload_glyph_png(self, job_id: str, glyph_id: str, png_bytes: bytes) -> None:
        path = f"{job_id}/glyphs/{glyph_id}.png"
        _retry(lambda: supabase.storage.from_(STORAGE_BUCKET).upload(
            path, png_bytes,
            file_options={"content-type": "image/png", "upsert": "true"},
        ))

    def download_glyph_png(self, job_id: str, glyph_id: str) -> Optional[bytes]:
        path = f"{job_id}/glyphs/{glyph_id}.png"
        try:
            return supabase.storage.from_(STORAGE_BUCKET).download(path)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Storage — font output files
    # ------------------------------------------------------------------

    def upload_font_file(
        self,
        job_id: str,
        filename: str,
        data: bytes,
        content_type: str,
        version: Optional[str] = None,
    ) -> str:
        """
        Upload a font binary. Returns the storage path (not the full URL).

        When *version* is given, the file is stored at a content-versioned path
        ({job_id}/output/{version}/{filename}) with a long, immutable
        Cache-Control. Because the path changes whenever the font bytes change
        (the caller derives `version` from a hash of `data`), the browser/CDN
        can cache each build forever and re-previewing the *same* build never
        re-downloads. A genuinely new build lands on a fresh path and is fetched
        once. This replaces the old in-place upsert + `cache-control: 0` scheme,
        which forced a full re-download on every single preview.
        """
        if version:
            path = f"{job_id}/output/{version}/{filename}"
            cache_control = "public, max-age=31536000, immutable"
        else:
            # Legacy in-place path: the CDN keys on path only and ignores query
            # strings, so cache-control "0" is the only way to avoid stale
            # serving when the same path is overwritten on rebuild.
            path = f"{job_id}/output/{filename}"
            cache_control = "0"
        _retry(lambda: supabase.storage.from_(STORAGE_BUCKET).upload(
            path, data,
            file_options={"content-type": content_type, "upsert": "true",
                          "cache-control": cache_control},
        ))
        return path

    def get_font_public_url(self, job_id: str, filename: str) -> str:
        """Return the stable public URL for a legacy (unversioned) font file."""
        return _public_url(f"{job_id}/output/{filename}")

    def public_url(self, path: str) -> str:
        """Return the public Storage URL for an already-resolved storage path."""
        return _public_url(path)

    def download_font_file(self, job_id: str, filename: str) -> Optional[bytes]:
        """Read a built font binary back from storage. Returns None if missing."""
        return self.download_path(f"{job_id}/output/{filename}")

    def download_path(self, path: str) -> Optional[bytes]:
        """Read any storage object by its full path. Returns None if missing."""
        try:
            return supabase.storage.from_(STORAGE_BUCKET).download(path)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _list_files_recursive(self, prefix: str, depth: int = 3) -> list:
        """
        Return full paths of every object under *prefix*, descending into
        subfolders. Font output now lives in content-versioned subfolders
        ({job_id}/output/{version}/Name.otf), so a single non-recursive list of
        {job_id}/output only yields folder placeholders and would leak the
        actual objects on cleanup.
        """
        try:
            entries = supabase.storage.from_(STORAGE_BUCKET).list(prefix)
        except Exception:
            return []
        paths = []
        for e in entries or []:
            name = e.get("name")
            if not name:
                continue
            child = f"{prefix}/{name}"
            # Supabase marks real files with a populated `id`/`metadata`;
            # folders come back with those unset. Recurse into folders.
            if depth > 0 and e.get("id") is None and e.get("metadata") is None:
                paths.extend(self._list_files_recursive(child, depth - 1))
            else:
                paths.append(child)
        return paths

    def _delete_storage_files(self, job_id: str) -> None:
        """Remove all Storage objects for a job (best-effort)."""
        for prefix in [f"{job_id}/glyphs", f"{job_id}/output"]:
            try:
                paths = self._list_files_recursive(prefix)
                if paths:
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
