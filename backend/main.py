"""
main.py — FastAPI application for the handwriting font generator.

Digital-drawing-only flow. Photo/scan upload was removed.

Endpoints:
  GET  /health
  GET  /ui                              — HTML drawing UI
  POST /draw/create                     — make an empty job
  POST /draw/{job_id}/glyph             — save one drawn glyph (upsert)
  GET  /process/{job_id}/status         — poll status (used during finalize)
  GET  /process/{job_id}/glyphs         — list saved glyphs (for review)
  POST /process/{job_id}/finalize       — kick off OTF + line OTF build
  GET  /fonts/{job_id}/{filename}       — serve built font binary
"""

import asyncio
import base64
import io
import json
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from job_store import job_store
from models import (
    DrawGlyphRequest,
    FinalizeRequest, FinalizeResponse,
    GlyphInfo, GlyphsResponse,
    JobStatus, JobStatusResponse,
)
from processing.font_builder import (
    GlyphData, build_otf, otf_to_woff2,
    char_to_glyph_name,
)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sweep stale jobs from previous runs (older than 10 min)
    import shutil, time
    from job_store import JOBS_DIR
    try:
        now = time.time()
        for job_dir in JOBS_DIR.iterdir():
            if job_dir.is_dir() and (now - job_dir.stat().st_mtime) > 600:
                shutil.rmtree(job_dir, ignore_errors=True)
    except Exception:
        pass

    task = asyncio.create_task(_periodic_cleanup())
    yield
    task.cancel()


app = FastAPI(title="Handwriting Font Generator", lifespan=lifespan)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
_origins = ALLOWED_ORIGINS.split(",") if ALLOWED_ORIGINS != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _periodic_cleanup():
    while True:
        await asyncio.sleep(3600)
        try:
            deleted = job_store.cleanup_old_jobs()
            if deleted:
                print(f"[cleanup] Deleted {deleted} old job(s)")
        except Exception as e:
            print(f"[cleanup] Error: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ui", response_class=Response)
async def serve_ui():
    ui_path = Path(__file__).parent.parent / "test_ui.html"
    content = ui_path.read_text(encoding="utf-8")
    # Rewrite API base to the same origin so the static HTML works in any deployment.
    content = content.replace("const API = 'http://localhost:8001';", "const API = '';")
    content = content.replace("const API = 'http://localhost:8000';", "const API = '';")
    return Response(content=content, media_type="text/html")


@app.post("/draw/create")
async def draw_create():
    """Create an empty job for the digital drawing flow."""
    job_id = str(uuid.uuid4())
    job_store.create_job(job_id)
    job_store.update_state(job_id, status="awaiting_review", is_draw_mode=True)
    return {"job_id": job_id}


@app.post("/draw/{job_id}/glyph")
async def draw_submit_glyph(job_id: str, req: DrawGlyphRequest):
    """Save one drawn glyph. Upserts on glyph_id."""
    _require_job(job_id)
    state = job_store.get_state(job_id)
    if state.get("status") != "awaiting_review":
        raise HTTPException(409, "Job is not in awaiting_review state")

    glyphs_dir = job_store.glyphs_dir(job_id)
    png_path = glyphs_dir / f"{req.glyph_id}.png"
    with open(png_path, "wb") as f:
        f.write(base64.b64decode(req.thumbnail_png_b64))

    manifest = state.get("glyph_manifest", [])
    entry = {
        "glyph_id": req.glyph_id,
        "char": req.char,
        "slot": req.slot,
        "has_glyph": True,
        "svg_paths": req.svg_paths,
        "pen_paths": req.pen_paths,
        "svg_width": req.svg_width,
        "svg_height": req.svg_height,
        "baseline_y": req.baseline_y,
        "upscale_factor": req.upscale_factor,
        "form": req.form,
        "entry_x": req.entry_x,
        "exit_x":  req.exit_x,
        "entry_y": req.entry_y,
        "exit_y":  req.exit_y,
    }
    manifest = [e for e in manifest if e["glyph_id"] != req.glyph_id]
    manifest.append(entry)
    job_store.update_state(job_id, glyph_manifest=manifest)
    return {"ok": True}


@app.get("/process/{job_id}/status", response_model=JobStatusResponse)
async def get_status(job_id: str):
    _require_job(job_id)
    state = job_store.get_state(job_id)
    return JobStatusResponse(
        job_id=job_id,
        status=JobStatus(state.get("status", "pending")),
        progress_pct=state.get("progress_pct", 0),
        error_message=state.get("error_message"),
        fea_warning=state.get("fea_warning"),
        line_skipped_glyphs=state.get("line_skipped_glyphs", []),
        has_line_font=state.get("has_line_font", False),
    )


@app.get("/process/{job_id}/glyphs", response_model=GlyphsResponse)
async def get_glyphs(job_id: str):
    _require_job(job_id)
    state = job_store.get_state(job_id)

    if state.get("status") != "awaiting_review":
        raise HTTPException(409, "Glyphs not ready yet")

    manifest = state.get("glyph_manifest", [])
    glyphs_dir = job_store.glyphs_dir(job_id)
    glyph_infos = []

    for entry in manifest:
        img_path = glyphs_dir / f"{entry['glyph_id']}.png"
        if not img_path.exists():
            continue
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        glyph_infos.append(GlyphInfo(
            glyph_id=entry["glyph_id"],
            char=entry["char"],
            slot=entry["slot"],
            image_b64=img_b64,
            accepted=True,
        ))

    return GlyphsResponse(job_id=job_id, glyphs=glyph_infos, alignment_warnings=[])


@app.post("/process/{job_id}/finalize", response_model=FinalizeResponse)
async def finalize(job_id: str, req: FinalizeRequest, background_tasks: BackgroundTasks):
    _require_job(job_id)
    state = job_store.get_state(job_id)

    if state.get("status") not in ("awaiting_review", "complete", "error"):
        raise HTTPException(409, "Job is not in review state")

    font_name = req.font_name.strip() or "MyHandwritingFont"
    job_store.update_state(
        job_id,
        status="finalizing",
        approved_glyph_ids=req.approved_glyph_ids,
        font_name=font_name,
        font_style=req.font_style,
        letter_spacing=req.letter_spacing,
        space_width=req.space_width,
        progress_pct=0,
    )

    background_tasks.add_task(run_in_threadpool, _build_font_job, job_id)

    base = f"/fonts/{job_id}"
    safe_name = font_name.replace(" ", "_")
    return FinalizeResponse(
        job_id=job_id,
        otf_url=f"{base}/{safe_name}.otf",
        woff2_url=f"{base}/{safe_name}.woff2",
        otf_line_url=f"{base}/{safe_name}-Line.otf",
        woff2_line_url=f"{base}/{safe_name}-Line.woff2",
    )


@app.get("/fonts/{job_id}/{filename}")
async def serve_font(job_id: str, filename: str):
    _require_job(job_id)
    state = job_store.get_state(job_id)

    if state.get("status") != "complete":
        raise HTTPException(202, "Font not ready yet")

    font_path = job_store.output_dir(job_id) / filename
    if not font_path.exists():
        raise HTTPException(404, "Font file not found")

    media_type = "font/otf" if filename.endswith(".otf") else "font/woff2"
    return FileResponse(
        str(font_path),
        media_type=media_type,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Content-Disposition": "inline",
            "Cache-Control": "no-store",
        },
    )


# ---------------------------------------------------------------------------
# Font build pipeline
# ---------------------------------------------------------------------------

def _build_font_job(job_id: str):
    """
    Build dimensional OTF + line OTF from the job's drawn glyphs.

    Both share the same family name and OpenType features. The line font
    uses each glyph's pen_paths (raw cursor track from the canvas) which is
    a perfect lossless centerline. If pen_paths is missing for a glyph,
    we emit it only into the dimensional font.
    """
    try:
        state = job_store.get_state(job_id)
        approved_ids = set(state.get("approved_glyph_ids", []))
        font_name = state.get("font_name", "MyHandwritingFont")
        font_style = state.get("font_style", "Regular")
        letter_spacing = state.get("letter_spacing", 0)
        space_width = state.get("space_width", 600)
        manifest = state.get("glyph_manifest", [])

        job_store.update_state(job_id, progress_pct=10)

        lowercase_chars = set("abcdefghijklmnopqrstuvwxyz")
        dimensional_glyphs: List[GlyphData] = []
        line_glyphs: List[GlyphData] = []
        line_skipped: List[str] = []

        # Cursive vs print: any glyph with a non-iso form means cursive flow.
        # In cursive mode, the iso form gets the bare cmap mapping (a → "a")
        # and init/medi/fina forms become named variants ("a.init", etc.)
        # picked up by the calt positional rules.
        is_cursive = any(
            (e.get("form") or "iso") != "iso"
            for e in manifest
            if e.get("has_glyph") and e["glyph_id"] in approved_ids
        )

        from processing.centerline import polyline_paths_to_svg

        # positional[char] -> {"init": "a.init", "medi": "a.medi", "fina": "a.fina"}
        # only filled in cursive mode and only for letters that actually have
        # those forms drawn.
        positional: Dict[str, Dict[str, str]] = {}

        for entry in manifest:
            glyph_id = entry["glyph_id"]
            if glyph_id not in approved_ids:
                continue
            if not entry.get("has_glyph"):
                continue

            char = entry["char"]
            slot = entry["slot"]
            form = entry.get("form") or "iso"
            glyph_name = char_to_glyph_name(char, slot, form)
            svg_w = entry.get("svg_width", 0)
            svg_h = entry.get("svg_height", 0)
            baseline_y = entry.get("baseline_y", 0)
            upscale_factor = entry.get("upscale_factor", 1.0)
            is_lower = char in lowercase_chars

            entry_x = entry.get("entry_x")
            exit_x  = entry.get("exit_x")

            dimensional_glyphs.append(GlyphData(
                char=char, slot=slot, glyph_name=glyph_name,
                svg_paths=entry.get("svg_paths", []),
                svg_width=svg_w, svg_height=svg_h,
                baseline_y_in_svg=baseline_y,
                is_lowercase=is_lower,
                upscale_factor=upscale_factor,
                form=form,
                entry_x=entry_x, exit_x=exit_x,
            ))

            pen_paths = entry.get("pen_paths") or []
            if pen_paths:
                line_svg = polyline_paths_to_svg(pen_paths)
                if line_svg:
                    line_glyphs.append(GlyphData(
                        char=char, slot=slot, glyph_name=glyph_name,
                        svg_paths=line_svg,
                        svg_width=svg_w, svg_height=svg_h,
                        baseline_y_in_svg=baseline_y,
                        is_lowercase=is_lower,
                        upscale_factor=upscale_factor,
                        form=form,
                        entry_x=entry_x, exit_x=exit_x,
                    ))
                else:
                    line_skipped.append(glyph_id)
            else:
                line_skipped.append(glyph_id)

            # Track positional variants per base char (cursive mode only).
            if is_cursive and is_lower and form in ("init", "medi", "fina"):
                positional.setdefault(char, {})[form] = glyph_name

        # If a cursive lowercase letter has positional forms but no iso form
        # was drawn, cmap can't map the Unicode codepoint to anything. Fall
        # back to medi (then fina, then init) so the letter still renders in
        # standalone use.
        if is_cursive:
            existing_iso = {
                g.char for g in dimensional_glyphs
                if g.glyph_name == g.char and g.char in lowercase_chars
            }
            for char, forms in list(positional.items()):
                if char in existing_iso:
                    continue
                fallback_form = next(
                    (f for f in ("medi", "fina", "init") if f in forms),
                    None,
                )
                if not fallback_form:
                    continue
                source_name = forms[fallback_form]
                source_dim = next((g for g in dimensional_glyphs if g.glyph_name == source_name), None)
                source_line = next((g for g in line_glyphs if g.glyph_name == source_name), None)
                if source_dim:
                    dimensional_glyphs.append(GlyphData(
                        char=char, slot=0, glyph_name=char,
                        svg_paths=source_dim.svg_paths,
                        svg_width=source_dim.svg_width, svg_height=source_dim.svg_height,
                        baseline_y_in_svg=source_dim.baseline_y_in_svg,
                        is_lowercase=True,
                        upscale_factor=source_dim.upscale_factor,
                    ))
                if source_line:
                    line_glyphs.append(GlyphData(
                        char=char, slot=0, glyph_name=char,
                        svg_paths=source_line.svg_paths,
                        svg_width=source_line.svg_width, svg_height=source_line.svg_height,
                        baseline_y_in_svg=source_line.baseline_y_in_svg,
                        is_lowercase=True,
                        upscale_factor=source_line.upscale_factor,
                    ))

        job_store.update_state(job_id, progress_pct=25)

        if not dimensional_glyphs:
            raise ValueError("No valid glyphs were approved")

        otf_bytes, fea_warning = build_otf(
            dimensional_glyphs, font_name, font_style, letter_spacing, space_width,
            positional=positional or None,
        )
        job_store.update_state(job_id, progress_pct=50)
        woff2_bytes = otf_to_woff2(otf_bytes)
        job_store.update_state(job_id, progress_pct=65)

        line_otf_bytes = b""
        line_woff2_bytes = b""
        line_fea_warning: Optional[str] = None
        if line_glyphs:
            line_otf_bytes, line_fea_warning = build_otf(
                line_glyphs, font_name, "Line", letter_spacing, space_width,
                positional=positional or None,
            )
            job_store.update_state(job_id, progress_pct=85)
            line_woff2_bytes = otf_to_woff2(line_otf_bytes)

        job_store.update_state(job_id, progress_pct=92)

        safe_name = font_name.replace(" ", "_")
        out_dir = job_store.output_dir(job_id)
        otf_path = out_dir / f"{safe_name}.otf"
        woff2_path = out_dir / f"{safe_name}.woff2"
        with open(otf_path, "wb") as f:
            f.write(otf_bytes)
        with open(woff2_path, "wb") as f:
            f.write(woff2_bytes)

        font_files = {"otf": str(otf_path), "woff2": str(woff2_path)}

        if line_otf_bytes:
            otf_line_path = out_dir / f"{safe_name}-Line.otf"
            woff2_line_path = out_dir / f"{safe_name}-Line.woff2"
            with open(otf_line_path, "wb") as f:
                f.write(line_otf_bytes)
            with open(woff2_line_path, "wb") as f:
                f.write(line_woff2_bytes)
            font_files["otf_line"] = str(otf_line_path)
            font_files["woff2_line"] = str(woff2_line_path)

        extra = {}
        if fea_warning:
            extra["fea_warning"] = fea_warning
        if line_fea_warning and not fea_warning:
            extra["fea_warning"] = line_fea_warning
        if line_skipped:
            extra["line_skipped_glyphs"] = line_skipped

        job_store.update_state(
            job_id,
            status="complete",
            progress_pct=100,
            font_files=font_files,
            has_line_font=bool(line_otf_bytes),
            **extra,
        )

    except Exception:
        job_store.update_state(
            job_id,
            status="error",
            error_message=traceback.format_exc()[:2000],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_job(job_id: str):
    if not job_store.job_exists(job_id):
        raise HTTPException(404, "Job not found")
