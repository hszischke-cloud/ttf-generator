"""
job_store.py — Filesystem-based job state management.

Each job lives in JOBS_DIR/{job_id}/ with:
  raw/          — original uploaded file(s)
  pages/        — extracted page images at 300 dpi (page_0.png, page_1.png)
  cells/        — cell crops (grayscale PNGs)
  glyphs/       — tight-cropped binary glyph images + SVG data
  output/       — generated font files (MyFont.otf, MyFont.woff2)
  state.json    — job metadata and status

Designed so that Redis can replace state.json reads/writes later by
swapping out the get_state / update_state methods.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

from filelock import FileLock

JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/tmp/jobs"))
MAX_JOB_AGE_HOURS = 24


class JobStore:
    def __init__(self, jobs_dir: Path = JOBS_DIR):
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def create_job(self, job_id: str) -> Path:
        """Create a new job directory and initial state. Returns job path."""
        job_path = self.jobs_dir / job_id
        for subdir in ["raw", "pages", "cells", "glyphs", "output"]:
            (job_path / subdir).mkdir(parents=True, exist_ok=True)

        self._write_state(job_id, {
            "status": "pending",
            "progress_pct": 0,
            "error_message": None,
            "created_at": time.time(),
            "glyph_manifest": [],
            "alignment_warnings": [],
            "font_files": {},
        })
        return job_path

    def job_path(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def job_exists(self, job_id: str) -> bool:
        return (self.jobs_dir / job_id).is_dir()

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_state(self, job_id: str) -> Dict[str, Any]:
        state_file = self.jobs_dir / job_id / "state.json"
        if not state_file.exists():
            return {}
        with open(state_file, "r") as f:
            return json.load(f)

    def update_state(self, job_id: str, **kwargs) -> None:
        """Atomically update specific fields in state.json."""
        lock_path = self.jobs_dir / job_id / "state.lock"
        with FileLock(str(lock_path)):
            state = self.get_state(job_id)
            state.update(kwargs)
            self._write_state(job_id, state)

    def _write_state(self, job_id: str, state: Dict[str, Any]) -> None:
        state_file = self.jobs_dir / job_id / "state.json"
        tmp = state_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f)
        tmp.replace(state_file)

    # ------------------------------------------------------------------
    # Convenience path helpers
    # ------------------------------------------------------------------

    def raw_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "raw"

    def pages_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "pages"

    def glyphs_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "glyphs"

    def output_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "output"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_old_jobs(self, max_age_hours: int = MAX_JOB_AGE_HOURS) -> int:
        """Delete job directories older than max_age_hours. Returns count deleted."""
        cutoff = time.time() - max_age_hours * 3600
        deleted = 0
        for job_dir in self.jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue
            try:
                state = self.get_state(job_dir.name)
                created_at = state.get("created_at", 0)
                if created_at < cutoff:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    deleted += 1
            except Exception:
                pass
        return deleted


# Singleton instance
job_store = JobStore()
