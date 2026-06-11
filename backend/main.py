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
import os
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response

from font_registry import (
    save_font as registry_save_font,
    list_fonts as registry_list_fonts,
    delete_font as registry_delete_font,
    rename_font as registry_rename_font,
    saved_job_ids,
)
from job_store import job_store
from models import (
    DrawGlyphBatchRequest, DrawGlyphRequest,
    FinalizeRequest, FinalizeResponse,
    GlyphInfo, GlyphsResponse,
    JobStatus, JobStatusResponse,
    PenStyleRequest, RenameFontRequest, SavedFontInfo,
)
from processing.font_builder import (
    GlyphData, build_otf, compute_glyph_advance,
    char_to_glyph_name, default_bearings_upm,
    CELL_SCALE, CANVAS_PAD,
)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick off hourly cleanup; no local filesystem to sweep (Supabase-backed).
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
            skip = saved_job_ids()
            deleted = job_store.cleanup_old_jobs(skip_ids=skip)
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


def _serve_app() -> Response:
    """Serve the single-file app with the API base rewritten to same-origin."""
    app_path = Path(__file__).parent.parent / "app.html"
    content = app_path.read_text(encoding="utf-8")
    content = content.replace("const API = 'http://localhost:8001';", "const API = '';")
    content = content.replace("const API = 'http://localhost:8000';", "const API = '';")
    # no-store so browsers / edge caches never serve a stale build of the UI.
    return Response(
        content=content,
        media_type="text/html",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.get("/", include_in_schema=False)
async def serve_root():
    return RedirectResponse("/ui", status_code=302)


@app.get("/ui", response_class=Response)
async def serve_ui():
    return _serve_app()


@app.get("/create", response_class=Response)
async def serve_client():
    # Legacy guided-creator route — the consumer flow and the studio are one
    # app now; the dashboard's "share with a client" link points here.
    return _serve_app()


@app.post("/draw/create")
async def draw_create():
    """Create an empty job for the digital drawing flow."""
    job_id = str(uuid.uuid4())
    job_store.create_job(job_id)
    job_store.update_state(job_id, status="awaiting_review", is_draw_mode=True)
    return {"job_id": job_id}


def _manifest_entry(req: DrawGlyphRequest, thumb_path: Optional[str]) -> dict:
    return {
        "glyph_id": req.glyph_id,
        "char": req.char,
        "slot": req.slot,
        "has_glyph": True,
        "thumb_path": thumb_path,
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
        "x_shift": req.x_shift,
        "pen_tool": req.pen_tool,
        "pen_size": req.pen_size,
        "pen_color": req.pen_color,
    }


def _save_glyphs(job_id: str, glyphs: List[DrawGlyphRequest]) -> int:
    """Upload thumbnails (in parallel) and merge entries into the manifest.

    One state read + one state write regardless of how many glyphs arrive —
    the manifest carries every glyph's svg/pen paths, so per-glyph
    read-modify-write made submission O(n²) in transferred bytes.
    """
    state = job_store.get_state(job_id)
    if not state:
        raise HTTPException(404, "Job not found")
    if state.get("status") != "awaiting_review":
        raise HTTPException(409, "Job is not in awaiting_review state")

    def _upload(g: DrawGlyphRequest) -> Tuple[str, Optional[str]]:
        if not g.thumbnail_png_b64:
            return g.glyph_id, None
        path = job_store.upload_glyph_png(
            job_id, g.glyph_id, base64.b64decode(g.thumbnail_png_b64))
        return g.glyph_id, path

    with ThreadPoolExecutor(max_workers=8) as ex:
        thumb_paths = dict(ex.map(_upload, glyphs))

    new_ids = {g.glyph_id for g in glyphs}
    manifest = [e for e in state.get("glyph_manifest", [])
                if e["glyph_id"] not in new_ids]
    manifest.extend(_manifest_entry(g, thumb_paths.get(g.glyph_id)) for g in glyphs)
    job_store.update_state(job_id, glyph_manifest=manifest)
    return len(glyphs)


@app.post("/draw/{job_id}/glyph")
async def draw_submit_glyph(job_id: str, req: DrawGlyphRequest):
    """Save one drawn glyph. Upserts on glyph_id. (Legacy — prefer the batch.)"""
    await run_in_threadpool(_save_glyphs, job_id, [req])
    return {"ok": True}


@app.post("/draw/{job_id}/glyphs/batch")
async def draw_submit_glyphs_batch(job_id: str, req: DrawGlyphBatchRequest):
    """Save many drawn glyphs in one round trip. Upserts on glyph_id."""
    if not req.glyphs:
        return {"ok": True, "count": 0}
    count = await run_in_threadpool(_save_glyphs, job_id, req.glyphs)
    return {"ok": True, "count": count}


@app.get("/process/{job_id}/status", response_model=JobStatusResponse)
async def get_status(job_id: str):
    # Polled every 1–2 s by the UIs: fetch only the status fields instead of
    # the full state blob (which carries every glyph's svg/pen paths).
    fields = await run_in_threadpool(
        job_store.get_state_fields, job_id,
        ["status", "progress_pct", "error_message", "fea_warning",
         "line_skipped_glyphs", "has_line_font"],
    )
    if fields is None:
        raise HTTPException(404, "Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=JobStatus(fields.get("status") or "pending"),
        progress_pct=fields.get("progress_pct") or 0,
        error_message=fields.get("error_message"),
        fea_warning=fields.get("fea_warning"),
        line_skipped_glyphs=fields.get("line_skipped_glyphs") or [],
        has_line_font=bool(fields.get("has_line_font")),
    )


def _thumb_url(job_id: str, entry: dict) -> str:
    """Public CDN URL for a glyph thumbnail (versioned path when available)."""
    path = entry.get("thumb_path") or f"{job_id}/glyphs/{entry['glyph_id']}.png"
    return job_store.public_url(path)


@app.get("/process/{job_id}/glyphs", response_model=GlyphsResponse)
async def get_glyphs(job_id: str):
    state = await run_in_threadpool(_get_state, job_id)

    if state.get("status") != "awaiting_review":
        raise HTTPException(409, "Glyphs not ready yet")

    # Thumbnails are served as public CDN URLs — the browser fetches them in
    # parallel (and caches them), so this endpoint no longer downloads and
    # base64-proxies every PNG through the backend.
    glyph_infos = [
        GlyphInfo(
            glyph_id=entry["glyph_id"],
            char=entry["char"],
            slot=entry["slot"],
            image_url=_thumb_url(job_id, entry),
            accepted=True,
        )
        for entry in state.get("glyph_manifest", [])
        if entry.get("has_glyph")
    ]

    return GlyphsResponse(job_id=job_id, glyphs=glyph_infos, alignment_warnings=[])


@app.post("/process/{job_id}/finalize", response_model=FinalizeResponse)
async def finalize(job_id: str, req: FinalizeRequest, background_tasks: BackgroundTasks):
    fields = await run_in_threadpool(
        job_store.get_state_fields, job_id, ["status"])
    if fields is None:
        raise HTTPException(404, "Job not found")

    if fields.get("status") not in ("awaiting_review", "complete", "error"):
        raise HTTPException(409, "Job is not in review state")

    font_name = req.font_name.strip() or "MyHandwritingFont"
    update_kwargs = dict(
        status="finalizing",
        approved_glyph_ids=req.approved_glyph_ids,
        font_name=font_name,
        font_style=req.font_style,
        letter_spacing=req.letter_spacing,
        space_width=req.space_width,
        progress_pct=0,
    )
    # Persist per-glyph border overrides only when provided, so a plain spacing
    # re-finalize doesn't wipe out adjustments made in the border editor.
    if req.glyph_bearings is not None:
        update_kwargs["glyph_bearings"] = {
            gid: {"lsb": int(b.lsb), "rsb": int(b.rsb)}
            for gid, b in req.glyph_bearings.items()
        }
    # Pen weight only when provided — None means keep the job's current one.
    if req.pen_style is not None:
        style = _normalize_pen_style(req.pen_style)
        if style is None:
            raise HTTPException(422, "pen_style must be 'realistic' or 'realistic-bold'")
        update_kwargs["pen_style"] = style
    job_store.update_state(job_id, **update_kwargs)

    background_tasks.add_task(run_in_threadpool, _build_font_job, job_id)

    base = f"/fonts/{job_id}"
    safe_name = font_name.replace(" ", "_")
    return FinalizeResponse(
        job_id=job_id,
        otf_url=f"{base}/{safe_name}.otf",
        otf_line_url=f"{base}/{safe_name}-Line.otf",
    )


@app.get("/process/{job_id}/proof", response_class=Response)
async def proof_sheet(job_id: str, font: str = "line"):
    """
    Render a built OTF into an SVG contact sheet for visual QA.

    Query param `font`: "line" (default) renders the single-line OTF, anything
    else renders the dimensional OTF. Returns an inline SVG the browser displays
    directly (and can print to PDF).

    The proof inspects the OTF that was last built and uploaded to storage, so
    it gates on that file existing rather than on the live job status. A job
    re-opened for editing flips back to `awaiting_review` (status != complete)
    while its last-built OTF is still on disk — the proof should still render
    it instead of failing with "Font not ready yet".
    """
    state = await run_in_threadpool(
        job_store.get_state_fields, job_id,
        ["font_name", "line_skipped_glyphs", "has_line_font", "status", "font_files"],
    )
    if state is None:
        raise HTTPException(404, "Job not found")

    from processing.proof_sheet import render_proof_svg

    font_name = state.get("font_name") or "Font"
    safe_name = font_name.replace(" ", "_")
    is_line = font == "line"

    if is_line:
        filename = f"{safe_name}-Line.otf"
        title = f"{font_name} — Line (single-line proof)"
        skipped = state.get("line_skipped_glyphs") or []
    else:
        filename = f"{safe_name}.otf"
        title = f"{font_name} — Regular (proof)"
        skipped = []

    otf_bytes = await run_in_threadpool(
        job_store.download_path, _resolve_font_path(state, job_id, filename)
    )
    if not otf_bytes:
        # No built file in storage. Distinguish "this font never had a line
        # version" from "nothing built yet / still building" so the UI can
        # show a sensible message.
        if is_line and not state.get("has_line_font", False):
            raise HTTPException(404, "No line font was built for this job")
        if state.get("status") != "complete":
            raise HTTPException(202, "Font not ready yet — build it first")
        raise HTTPException(404, "Font file not found in storage")

    svg = await run_in_threadpool(
        render_proof_svg, otf_bytes, title, is_line, skipped
    )
    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "no-store"},
    )


def _resolve_font_path(state: dict, job_id: str, filename: str) -> str:
    """
    Map a requested font filename to its actual storage path.

    The build records content-versioned paths in state["font_files"]
    ({"otf": ".../<hash>/Name.otf", "otf_line": ...}). We resolve against that
    so the served path always points at the current build's immutable, cacheable
    object. Falls back to the legacy unversioned path for jobs built before
    versioning existed.
    """
    clean_name = filename.split("?")[0]
    font_files = state.get("font_files") or {}
    kind = "otf_line" if clean_name.endswith("-Line.otf") else "otf"
    path = font_files.get(kind)
    if path:
        return path
    return f"{job_id}/output/{clean_name}"


@app.get("/fonts/{job_id}/{filename}")
async def serve_font(job_id: str, filename: str, request: Request):
    state = await run_in_threadpool(
        job_store.get_state_fields, job_id, ["status", "font_files"])
    if state is None:
        raise HTTPException(404, "Job not found")

    if state.get("status") != "complete":
        raise HTTPException(202, "Font not ready yet")

    url = job_store.public_url(_resolve_font_path(state, job_id, filename))

    # 302 → the browser downloads directly from the Supabase CDN. The redirect
    # itself is no-store (it's a cheap backend hop), but the storage object it
    # points at is content-versioned and immutable, so re-previewing the same
    # build is served from the browser/CDN cache with no re-download. A rebuild
    # changes the resolved path, guaranteeing a fresh fetch — no query-string
    # cache-buster needed (the CDN ignores those anyway).
    return RedirectResponse(url, status_code=302, headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------------------
# Saved-fonts registry endpoints
# ---------------------------------------------------------------------------

@app.post("/fonts/save/{job_id}")
async def save_font(job_id: str):
    """Add a completed font to the persistent saved-fonts registry."""
    state = await run_in_threadpool(_get_state, job_id)
    if state.get("status") != "complete":
        raise HTTPException(409, "Font must be complete before saving")
    manifest = state.get("glyph_manifest", [])
    approved_ids = set(state.get("approved_glyph_ids", []))
    glyph_count = sum(
        1 for e in manifest
        if e["glyph_id"] in approved_ids and e.get("has_glyph")
    )
    mode = (
        "cursive"
        if any((e.get("form") or "iso") != "iso" for e in manifest)
        else "print"
    )
    registry_save_font(job_id, state.get("font_name", "Untitled"), mode, glyph_count)
    return {"ok": True}


@app.get("/fonts/saved")
async def list_saved_fonts():
    """Return all fonts the user has explicitly saved."""
    def _annotate():
        fonts = registry_list_fonts()
        # Annotate with whether the job still exists, and whether a single-line
        # OTF was built (so the UI can offer line proof/download). One small
        # field-select per font — never the full multi-MB state blob.
        annotated = []
        for f in fonts:
            fields = job_store.get_state_fields(
                f["job_id"], ["has_line_font", "pen_style"])
            annotated.append({
                **f,
                "job_exists": fields is not None,
                "has_line_font": bool(fields and fields.get("has_line_font")),
                "pen_style": _normalize_pen_style(
                    fields.get("pen_style") if fields else None) or "realistic",
            })
        return annotated
    return {"fonts": await run_in_threadpool(_annotate)}


@app.delete("/fonts/saved/{job_id}")
async def remove_saved_font(job_id: str):
    """Remove a font from the saved registry (job data is kept on disk)."""
    deleted = await run_in_threadpool(registry_delete_font, job_id)
    if not deleted:
        raise HTTPException(404, "Font not found in saved registry")
    return {"ok": True}


@app.patch("/fonts/saved/{job_id}")
async def rename_saved_font(job_id: str, req: RenameFontRequest):
    """
    Rename a saved font. Updates the registry and the job state immediately;
    the name baked inside the OTF binary updates on the next rebuild
    (finalize reads font_name from state).
    """
    name = req.font_name.strip()[:60]
    if not name:
        raise HTTPException(422, "Font name cannot be empty")

    def _rename():
        ok = registry_rename_font(job_id, name)
        if ok and job_store.get_state_fields(job_id, ["status"]) is not None:
            job_store.update_state(job_id, font_name=name)
        return ok

    if not await run_in_threadpool(_rename):
        raise HTTPException(404, "Font not found in saved registry")
    return {"ok": True, "font_name": name}


@app.post("/process/{job_id}/reopen")
async def reopen_job(job_id: str):
    """Re-open a completed job for editing (sets status back to awaiting_review)."""
    fields = await run_in_threadpool(
        job_store.get_state_fields, job_id, ["status"])
    if fields is None:
        raise HTTPException(404, "Job not found")
    status = fields.get("status")
    if status not in ("complete", "error", "awaiting_review"):
        raise HTTPException(409, f"Cannot reopen job in state '{status}'")
    if status != "awaiting_review":
        await run_in_threadpool(
            lambda: job_store.update_state(job_id, status="awaiting_review"))
    return {"ok": True}


def _normalize_pen_style(value) -> Optional[str]:
    """
    Map any stored/requested pen style onto the two supported weights.

    Every font is inked with the realistic stroker now; "classic" only
    survives as a legacy stored value from before the realistic pen existed
    and builds as the fine weight. Returns None for unrecognised input.
    """
    if value in (None, "", "classic", "fine", "realistic"):
        return "realistic"
    if value in ("bold", "realistic-bold"):
        return "realistic-bold"
    return None


@app.post("/process/{job_id}/pen-style")
async def set_pen_style(job_id: str, req: PenStyleRequest, background_tasks: BackgroundTasks):
    """
    Switch a built font between the fine and bold pen weight and rebuild.

    Both OTFs are regenerated from the job's stored glyph data and settings —
    every glyph's pen_paths are re-stroked with the realistic-ink model at
    the requested weight. Toggling back and forth is lossless because the
    underlying glyph data never changes, and both weights share identical
    advance widths so layout never shifts.
    """
    style = _normalize_pen_style(req.pen_style)
    if style is None:
        raise HTTPException(422, "pen_style must be 'realistic' or 'realistic-bold'")

    fields = await run_in_threadpool(
        job_store.get_state_fields, job_id,
        ["status", "approved_glyph_ids", "font_name", "pen_style"],
    )
    if fields is None:
        raise HTTPException(404, "Job not found")
    if fields.get("status") not in ("complete", "awaiting_review", "error"):
        raise HTTPException(409, "Job is still building — try again shortly")
    if not fields.get("approved_glyph_ids"):
        raise HTTPException(409, "Font has never been built — generate it first")

    # Note: a legacy "classic" job is NOT current even if it normalizes to
    # the requested weight — it was built with the old brush outlines, so a
    # rebuild genuinely changes it. Compare raw stored values for that case.
    stored = fields.get("pen_style")
    if stored == style and fields.get("status") == "complete":
        return {"ok": True, "pen_style": style, "rebuilt": False}

    job_store.update_state(
        job_id, pen_style=style, status="finalizing", progress_pct=0,
    )
    background_tasks.add_task(run_in_threadpool, _build_font_job, job_id)
    return {"ok": True, "pen_style": style, "rebuilt": True}


@app.get("/process/{job_id}/pen-paths")
async def get_pen_paths(job_id: str):
    """
    Return the full glyph data needed to re-open a job for editing.

    Includes svg_paths (original outlines), pen_paths (raw strokes), all
    geometric metadata, and the base64 thumbnail PNG — so the frontend can
    re-submit unmodified glyphs exactly as they were built originally.
    """
    state = await run_in_threadpool(_get_state, job_id)
    manifest = state.get("glyph_manifest", [])
    mode = (
        "cursive"
        if any((e.get("form") or "iso") != "iso" for e in manifest)
        else "print"
    )
    glyphs = []
    for entry in manifest:
        if not entry.get("has_glyph"):
            continue
        glyphs.append({
            "glyph_id":      entry["glyph_id"],
            "char":          entry["char"],
            "slot":          entry["slot"],
            "form":          entry.get("form") or "iso",
            "svg_paths":     entry.get("svg_paths") or [],
            "pen_paths":     entry.get("pen_paths") or [],
            "svg_width":     entry.get("svg_width", 300),
            "svg_height":    entry.get("svg_height", 400),
            "baseline_y":    entry.get("baseline_y", 288),
            "upscale_factor": entry.get("upscale_factor", 1.0),
            "entry_x":       entry.get("entry_x"),
            "exit_x":        entry.get("exit_x"),
            "entry_y":       entry.get("entry_y"),
            "exit_y":        entry.get("exit_y"),
            "x_shift":       entry.get("x_shift", 0.0),
            "pen_tool":      entry.get("pen_tool", "pen"),
            "pen_size":      entry.get("pen_size", 14),
            "pen_color":     entry.get("pen_color", [38, 32, 28]),
            # Thumbnails come from the public CDN now — no per-glyph storage
            # download (this loop used to be serial and dominated load time).
            "image_url":     _thumb_url(job_id, entry),
        })
    return {
        "job_id":    job_id,
        "mode":      mode,
        "font_name": state.get("font_name", ""),
        "glyphs":    glyphs,
    }


@app.get("/process/{job_id}/bearings")
async def get_bearings(job_id: str):
    """
    Return per-glyph side-bearing data for the border editor.

    Only iso (print) glyphs are adjustable — cursive positional forms derive
    their bearings from connection points. Each glyph carries its outline
    paths, geometry, the px↔UPM scale, and the current (override or default)
    lsb/rsb so the editor can render draggable left/right bearing lines.
    """
    state = await run_in_threadpool(_get_state, job_id)
    manifest = state.get("glyph_manifest", [])
    overrides = state.get("glyph_bearings", {}) or {}
    mode = (
        "cursive"
        if any((e.get("form") or "iso") != "iso" for e in manifest)
        else "print"
    )

    glyphs = []
    for entry in manifest:
        if not entry.get("has_glyph"):
            continue
        if (entry.get("form") or "iso") != "iso":
            continue  # only standalone forms are adjustable
        gid = entry["glyph_id"]
        svg_w = entry.get("svg_width", 300)
        upf = entry.get("upscale_factor", 1.0) or 1.0
        def_lsb, def_rsb = default_bearings_upm(svg_w, upf)
        ov = overrides.get(gid)
        glyphs.append({
            "glyph_id":      gid,
            "char":          entry["char"],
            "slot":          entry["slot"],
            "svg_paths":     entry.get("svg_paths") or [],
            "svg_width":     svg_w,
            "svg_height":    entry.get("svg_height", 400),
            "baseline_y":    entry.get("baseline_y", 288),
            "upscale_factor": upf,
            "coord_scale":   (CELL_SCALE / upf if upf > 0 else CELL_SCALE),
            "canvas_pad":    CANVAS_PAD,
            "default_lsb":   def_lsb,
            "default_rsb":   def_rsb,
            "lsb":           int(ov["lsb"]) if ov else def_lsb,
            "rsb":           int(ov["rsb"]) if ov else def_rsb,
            "is_override":   bool(ov),
            "image_url":     _thumb_url(job_id, entry),
        })

    return {
        "job_id":         job_id,
        "mode":           mode,
        "font_name":      state.get("font_name", ""),
        "cell_scale":     CELL_SCALE,
        "canvas_pad":     CANVAS_PAD,
        "letter_spacing": state.get("letter_spacing", 0),
        "space_width":    state.get("space_width", 600),
        "approved_glyph_ids": state.get("approved_glyph_ids", []),
        "has_line_font":  state.get("has_line_font", False),
        "glyphs":         glyphs,
    }


@app.get("/process/{job_id}/bearings/auto")
async def get_auto_bearings(job_id: str, tightness: int = 0):
    """
    Suggest perceived-width-equalizing side bearings for every iso glyph.

    Pure analysis of the stored outlines (HT-Letterspacer-style optical margin
    measurement; see processing/autospace.py) — nothing is persisted. The
    border editor presents these in its editable lsb/rsb fields and the user
    decides whether to apply them via the normal finalize flow. `tightness`
    biases every suggestion looser (+) or tighter (−), in UPM.
    """
    state = await run_in_threadpool(_get_state, job_id)
    manifest = state.get("glyph_manifest", [])

    from processing.autospace import compute_auto_bearings
    bearings = await run_in_threadpool(
        compute_auto_bearings, manifest, tightness)
    return {"job_id": job_id, "bearings": bearings}


@app.get("/process/{job_id}/settings")
async def get_job_settings(job_id: str):
    """Return font-build settings needed to re-finalize an existing job."""
    fields = await run_in_threadpool(
        job_store.get_state_fields, job_id,
        ["font_name", "font_style", "approved_glyph_ids",
         "letter_spacing", "space_width", "has_line_font", "pen_style"],
    )
    if fields is None:
        raise HTTPException(404, "Job not found")
    return {
        "font_name":          fields.get("font_name") or "",
        "font_style":         fields.get("font_style") or "Regular",
        "approved_glyph_ids": fields.get("approved_glyph_ids") or [],
        "letter_spacing":     fields.get("letter_spacing") or 0,
        "space_width":        fields.get("space_width") or 600,
        "has_line_font":      bool(fields.get("has_line_font")),
        "pen_style":          _normalize_pen_style(fields.get("pen_style")) or "realistic",
    }


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
        glyph_bearings = state.get("glyph_bearings", {}) or {}
        pen_style = _normalize_pen_style(state.get("pen_style")) or "realistic"

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

        # Auto-borders on a print job's FIRST build: compute the optical
        # margin-equalizing side bearings (HT-Letterspacer-style, see
        # processing/autospace.py) and persist them as the job's
        # glyph_bearings. Persisting makes every later rebuild — spacing
        # tweaks, pen-weight switches, manual border edits — start from the
        # same values, and the borders editor presents them as the editable
        # baseline. Gated on "never built + no manual bearings" so fonts
        # built before this feature (and any user-adjusted borders) are
        # never silently re-spaced. Cursive is excluded: its spacing comes
        # from connection points and letter_spacing is forced to 0 there.
        if not glyph_bearings and not state.get("font_files") and not is_cursive:
            try:
                from processing.autospace import compute_auto_bearings
                auto = compute_auto_bearings(manifest)
                if auto:
                    glyph_bearings = auto
                    job_store.update_state(job_id, glyph_bearings=glyph_bearings)
            except Exception as exc:
                print(f"[autospace] auto-borders skipped: {exc}")

        # Pick a font-wide base color for COLR layers: most common pen_color
        # across the approved glyphs. Falls back to the default ink color.
        _color_counts: Dict[Tuple[int, int, int], int] = {}
        for _e in manifest:
            if _e["glyph_id"] not in approved_ids or not _e.get("has_glyph"):
                continue
            _c = _e.get("pen_color") or [38, 32, 28]
            try:
                _key = (int(_c[0]), int(_c[1]), int(_c[2]))
            except (TypeError, ValueError, IndexError):
                continue
            _color_counts[_key] = _color_counts.get(_key, 0) + 1
        if _color_counts:
            font_base_color = max(_color_counts.items(), key=lambda kv: kv[1])[0]
        else:
            font_base_color = (38, 32, 28)

        from processing.centerline import polyline_paths_to_svg
        from processing.pen_realistic import (
            BOLD_WIDTH_SCALE, realistic_glyph_outlines,
        )

        pen_width_scale = BOLD_WIDTH_SCALE if pen_style == "realistic-bold" else 1.0

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

            # Manual side-bearing override for this glyph (iso/print only).
            _bearing = glyph_bearings.get(glyph_id) if form == "iso" else None
            lsb_upm = _bearing.get("lsb") if _bearing else None
            rsb_upm = _bearing.get("rsb") if _bearing else None

            # Dimensional outline source: every glyph is re-stroked from its
            # raw pen track with the realistic-ink model at the chosen weight
            # (same coordinate space, so all baseline/advance/bearing math is
            # untouched). The stored svg_paths (legacy canvas brush polygons)
            # are only a fallback for glyphs without pen_paths.
            dim_svg_paths = entry.get("svg_paths", [])
            if entry.get("pen_paths"):
                try:
                    realistic = realistic_glyph_outlines(
                        entry["pen_paths"],
                        pen_size=float(entry.get("pen_size") or 6),
                        seed_key=glyph_id,
                        width_scale=pen_width_scale,
                    )
                    if realistic:
                        dim_svg_paths = realistic
                except Exception as exc:
                    print(f"[pen_realistic] {glyph_id}: fell back to stored outline ({exc})")

            dimensional_glyphs.append(GlyphData(
                char=char, slot=slot, glyph_name=glyph_name,
                svg_paths=dim_svg_paths,
                svg_width=svg_w, svg_height=svg_h,
                baseline_y_in_svg=baseline_y,
                is_lowercase=is_lower,
                upscale_factor=upscale_factor,
                form=form,
                entry_x=entry_x, exit_x=exit_x,
                lsb_upm=lsb_upm, rsb_upm=rsb_upm,
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
                        lsb_upm=lsb_upm, rsb_upm=rsb_upm,
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

        # Compute advance widths from dimensional glyphs (the spacing source of
        # truth).  The line font reuses these exact values so both fonts render
        # with identical inter-letter and space-bar spacing regardless of any
        # floating-point differences in independent computation paths.
        dim_advances: Dict[str, int] = {
            g.glyph_name: compute_glyph_advance(g, letter_spacing)
            for g in dimensional_glyphs
        }
        dim_advances["space"] = space_width

        job_store.update_state(job_id, progress_pct=35)

        # Build both OTFs in parallel — they share no state and each holds the
        # GIL only intermittently (fontTools spends most of its time in C
        # ext modules and I/O), so threading actually overlaps.
        def _build_dim():
            # Realistic ink owns its own edge texture (waviness, fibre
            # notches, pooling are baked into the stroked outlines), so the
            # UPM-space perturbation/divot pass is always skipped — running
            # both would double-texture every edge.
            return build_otf(
                dimensional_glyphs, font_name, font_style, letter_spacing, space_width,
                positional=positional or None,
                perturb=False,
                forced_advances=dim_advances,
                base_color=font_base_color,
            )

        def _build_line():
            if not line_glyphs:
                return b"", None
            return build_otf(
                line_glyphs, font_name, "Line", letter_spacing, space_width,
                positional=positional or None,
                perturb=False,
                forced_advances=dim_advances,
                base_color=font_base_color,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            dim_future = pool.submit(_build_dim)
            line_future = pool.submit(_build_line)
            otf_bytes, fea_warning = dim_future.result()
            job_store.update_state(job_id, progress_pct=70)
            line_otf_bytes, line_fea_warning = line_future.result()

        job_store.update_state(job_id, progress_pct=85)

        safe_name = font_name.replace(" ", "_")

        # Content-versioned upload: each file goes to a path keyed by a hash of
        # its own bytes, with an immutable Cache-Control. Re-previewing the same
        # build serves from cache (no re-download); a rebuild with different
        # spacing/borders produces different bytes → a new path → a single fresh
        # fetch. serve_font resolves the live path from font_files below.
        import hashlib

        def _upload(filename, data):
            version = hashlib.sha1(data).hexdigest()[:12]
            return job_store.upload_font_file(
                job_id, filename, data, "font/otf", version=version
            )

        uploads = [(f"{safe_name}.otf", otf_bytes)]
        if line_otf_bytes:
            uploads.append((f"{safe_name}-Line.otf", line_otf_bytes))

        with ThreadPoolExecutor(max_workers=2) as pool:
            paths = list(pool.map(lambda t: _upload(*t), uploads))

        font_files = {"otf": paths[0]}
        if line_otf_bytes:
            font_files["otf_line"] = paths[1]

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

    except Exception as exc:
        # Full traceback to the server log; a human-readable sentence to the
        # user. Raw tracebacks in the UI read as broken software.
        print(f"[build] job {job_id} failed:\n{traceback.format_exc()}")
        if isinstance(exc, ValueError):
            friendly = str(exc)
        else:
            friendly = (
                "Something went wrong while building your font. "
                "Please try again — if it keeps failing, try redrawing the "
                "last letters you changed."
            )
        job_store.update_state(
            job_id,
            status="error",
            error_message=friendly,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_state(job_id: str) -> dict:
    """Fetch the full job state, 404ing if the job doesn't exist.

    One query instead of the old job_exists + get_state pair. Only for
    endpoints that genuinely need the whole blob (glyph manifest access);
    everything else should use job_store.get_state_fields.
    """
    state = job_store.get_state(job_id)
    if not state:
        raise HTTPException(404, "Job not found")
    return state
