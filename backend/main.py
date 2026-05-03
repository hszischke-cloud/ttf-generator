"""
main.py — FastAPI application for the handwriting font generator.

Endpoints:
  GET  /health
  GET  /template/download
  POST /process/upload
  GET  /process/{job_id}/status
  GET  /process/{job_id}/glyphs
  POST /process/{job_id}/finalize
  GET  /fonts/{job_id}/{filename}
"""

import asyncio
import base64
import io
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

import fitz  # PyMuPDF

from job_store import job_store
from models import (
    DrawGlyphRequest,
    FinalizeRequest, FinalizeResponse,
    GlyphInfo, GlyphsResponse,
    JobStatus, JobStatusResponse,
)
from template import get_template_pdf
from processing.alignment import align_page, align_page_with_manual_points, load_image_from_file
from processing.extraction import extract_all_glyphs, ExtractedGlyph
from processing.vectorize import vectorize_glyph
from processing.centerline import vectorize_centerline
from processing.font_builder import (
    GlyphData, build_otf, otf_to_woff2,
    char_to_glyph_name,
)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Clean up stale job directories from previous server runs (older than 10 min)
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

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Background cleanup
# ---------------------------------------------------------------------------

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
    # Rewrite API calls to use relative paths (same origin, no CORS needed)
    content = content.replace("const API = 'http://localhost:8001';", "const API = '';")
    content = content.replace("const API = 'http://localhost:8000';", "const API = '';")
    content = content.replace("http://localhost:8001/template/download", "/template/download")
    content = content.replace("http://localhost:8000/template/download", "/template/download")
    return Response(content=content, media_type="text/html")


@app.get("/template/download")
async def download_template():
    pdf_bytes = get_template_pdf()
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="handwriting_template.pdf"'},
    )


@app.post("/process/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # Validate file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 50 MB)")

    # Validate file type
    allowed_types = {
        "application/pdf", "image/png", "image/jpeg",
        "image/jpg", "image/tiff", "image/tif",
    }
    ct = (file.content_type or "").lower()
    fname = (file.filename or "").lower()
    is_pdf = ct == "application/pdf" or fname.endswith(".pdf")
    is_image = ct.startswith("image/") or any(
        fname.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]
    )
    if not (is_pdf or is_image):
        raise HTTPException(400, "Unsupported file type. Upload a PDF or image.")

    job_id = str(uuid.uuid4())
    job_store.create_job(job_id)

    # Save raw upload
    raw_path = job_store.raw_dir(job_id) / (file.filename or "upload.pdf")
    with open(raw_path, "wb") as f:
        f.write(content)

    job_store.update_state(job_id, raw_filename=str(raw_path), is_pdf=is_pdf)

    # Start background processing in a subprocess so native crashes don't kill the server
    background_tasks.add_task(_launch_process_job, job_id)

    return {"job_id": job_id}


async def _launch_process_job(job_id: str):
    """Run the processing pipeline in a separate subprocess for crash isolation."""
    import sys
    worker = Path(__file__).parent / "worker.py"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(worker), job_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(Path(__file__).parent),
    )
    _, stderr = await proc.communicate()
    if stderr:
        print(f"[worker {job_id}] {stderr.decode(errors='replace')[-2000:]}", flush=True)
    if proc.returncode != 0:
        err = stderr.decode(errors="replace")[-2000:] if stderr else "Processing crashed unexpectedly"
        job_store.update_state(job_id, status="error", error_message=err)


@app.get("/process/{job_id}/status", response_model=JobStatusResponse)
async def get_status(job_id: str):
    _require_job(job_id)
    state = job_store.get_state(job_id)
    return JobStatusResponse(
        job_id=job_id,
        status=JobStatus(state.get("status", "pending")),
        progress_pct=state.get("progress_pct", 0),
        error_message=state.get("error_message"),
        alignment_failed_image_b64=state.get("alignment_failed_image_b64"),
        fea_warning=state.get("fea_warning"),
        line_skipped_glyphs=state.get("line_skipped_glyphs", []),
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

    return GlyphsResponse(
        job_id=job_id,
        glyphs=glyph_infos,
        alignment_warnings=state.get("alignment_warnings", []),
    )


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

    # Return URLs immediately (files will be ready when status == complete)
    base = f"/fonts/{job_id}"
    safe_name = font_name.replace(" ", "_")
    return FinalizeResponse(
        job_id=job_id,
        otf_url=f"{base}/{safe_name}.otf",
        woff2_url=f"{base}/{safe_name}.woff2",
        otf_line_url=f"{base}/{safe_name}-Line.otf",
        woff2_line_url=f"{base}/{safe_name}-Line.woff2",
    )


@app.post("/draw/create")
async def draw_create():
    """Create an empty job for the digital drawing flow. Returns immediately with job_id."""
    job_id = str(uuid.uuid4())
    job_store.create_job(job_id)
    job_store.update_state(job_id, status="awaiting_review", is_draw_mode=True)
    return {"job_id": job_id}


@app.post("/draw/{job_id}/glyph")
async def draw_submit_glyph(job_id: str, req: DrawGlyphRequest):
    """Save a single drawn glyph (SVG paths + PNG thumbnail) into the job."""
    _require_job(job_id)
    state = job_store.get_state(job_id)
    if state.get("status") != "awaiting_review":
        raise HTTPException(409, "Job is not in awaiting_review state")

    # Write PNG thumbnail for the review page
    glyphs_dir = job_store.glyphs_dir(job_id)
    png_path = glyphs_dir / f"{req.glyph_id}.png"
    with open(png_path, "wb") as f:
        f.write(base64.b64decode(req.thumbnail_png_b64))

    # Upsert manifest entry (replace if same glyph_id already exists)
    manifest = state.get("glyph_manifest", [])
    entry = {
        "glyph_id": req.glyph_id,
        "char": req.char,
        "slot": req.slot,
        "has_glyph": True,
        "svg_paths": req.svg_paths,
        "svg_width": req.svg_width,
        "svg_height": req.svg_height,
        "baseline_y": req.baseline_y,
        "upscale_factor": req.upscale_factor,
    }
    manifest = [e for e in manifest if e["glyph_id"] != req.glyph_id]
    manifest.append(entry)
    job_store.update_state(job_id, glyph_manifest=manifest)

    return {"ok": True}


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
            "Content-Disposition": "inline",  # must be inline so @font-face loads it
            "Cache-Control": "no-store",
        },
    )


# ---------------------------------------------------------------------------
# Background processing pipeline
# ---------------------------------------------------------------------------

def _process_job(job_id: str):
    """Full extraction pipeline: raw file → page images → glyphs → manifest."""
    try:
        job_store.update_state(job_id, status="processing", progress_pct=5)
        state = job_store.get_state(job_id)
        raw_path = state["raw_filename"]
        is_pdf = state.get("is_pdf", False)

        # Step 1: Convert to page images
        page_images_bgr = _load_page_images(raw_path, is_pdf)
        job_store.update_state(job_id, progress_pct=20)

        # Step 2: Align each page
        aligned_pages = []
        warnings = []
        for i, img in enumerate(page_images_bgr):
            aligned, err = align_page(img)
            if err:
                warnings.append(f"Page {i + 1}: {err}")
                # Save raw page as base64 for manual alignment fallback
                _, png_buf = cv2.imencode(".png", img)
                img_b64 = base64.b64encode(png_buf.tobytes()).decode()
                job_store.update_state(job_id,
                    alignment_failed_image_b64=img_b64,
                    alignment_warnings=warnings,
                )
            aligned_pages.append(aligned)

            # Save aligned page image
            page_path = job_store.pages_dir(job_id) / f"page_{i}.png"
            cv2.imwrite(str(page_path), aligned)

        job_store.update_state(job_id, progress_pct=40, alignment_warnings=warnings)

        # Step 3: Extract glyphs
        all_glyphs = extract_all_glyphs(aligned_pages)
        job_store.update_state(job_id, progress_pct=55)

        # Step 4: Vectorize and save glyph images
        glyphs_dir = job_store.glyphs_dir(job_id)
        manifest = []
        total = len(all_glyphs)

        for idx, glyph in enumerate(all_glyphs):
            if glyph.glyph_img is None:
                # Empty cell — include in manifest as empty
                manifest.append({
                    "glyph_id": glyph.glyph_id,
                    "char": glyph.char,
                    "slot": glyph.slot,
                    "has_glyph": False,
                    "svg_paths": [],
                    "svg_paths_centerline": [],
                    "svg_width": 0,
                    "svg_height": 0,
                    "baseline_y": 0,
                })
                continue

            # Save glyph image (display version — inverted for human viewing)
            display_img = cv2.bitwise_not(glyph.glyph_img)  # white ink on white bg → black on white
            img_path = glyphs_dir / f"{glyph.glyph_id}.png"
            cv2.imwrite(str(img_path), display_img)

            # Vectorize (filled outline — dimensional font)
            vec_result = vectorize_glyph(glyph.glyph_img)
            svg_paths = []
            svg_w, svg_h, upscale_factor = 0, 0, 1.0
            if vec_result:
                svg_paths, svg_w, svg_h, upscale_factor = vec_result

            # Vectorize centerline (single-line / pen-plotter font).
            # Shares geometry (width, height, upscale_factor) with the dimensional
            # vectorizer because both run on the same upscaled glyph image.
            cl_result = vectorize_centerline(glyph.glyph_img)
            svg_paths_centerline = cl_result[0] if cl_result else []

            # Save SVG data as JSON
            import json
            svg_json_path = glyphs_dir / f"{glyph.glyph_id}.json"
            with open(svg_json_path, "w") as f:
                json.dump({
                    "svg_paths": svg_paths,
                    "svg_paths_centerline": svg_paths_centerline,
                    "svg_width": svg_w,
                    "svg_height": svg_h,
                    "baseline_y": glyph.baseline_y_in_glyph,
                    "upscale_factor": upscale_factor,
                }, f)

            manifest.append({
                "glyph_id": glyph.glyph_id,
                "char": glyph.char,
                "slot": glyph.slot,
                "has_glyph": True,
                "svg_paths": svg_paths,
                "svg_paths_centerline": svg_paths_centerline,
                "svg_width": svg_w,
                "svg_height": svg_h,
                "baseline_y": glyph.baseline_y_in_glyph,
                "upscale_factor": upscale_factor,
            })

            progress = 55 + int(40 * (idx + 1) / max(total, 1))
            job_store.update_state(job_id, progress_pct=progress)

        job_store.update_state(
            job_id,
            status="awaiting_review",
            progress_pct=100,
            glyph_manifest=manifest,
        )

    except Exception:
        job_store.update_state(
            job_id,
            status="error",
            error_message=traceback.format_exc()[:2000],
        )


def _build_font_job(job_id: str):
    """
    Font assembly pipeline: approved glyphs → two OTFs + two WOFF2s.

    Builds both a dimensional (filled outline) font and a single-line
    (centerline) companion font intended for pen-plotter use. The line
    font reuses the same family name and OpenType features (calt/ss01/ss02)
    so substitution rules behave identically.
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
        line_skipped_chars: List[str] = []

        for entry in manifest:
            glyph_id = entry["glyph_id"]
            if glyph_id not in approved_ids:
                continue
            if not entry.get("has_glyph"):
                continue

            char = entry["char"]
            slot = entry["slot"]
            glyph_name = char_to_glyph_name(char, slot)
            svg_w = entry.get("svg_width", 0)
            svg_h = entry.get("svg_height", 0)
            baseline_y = entry.get("baseline_y", 0)
            upscale_factor = entry.get("upscale_factor", 1.0)
            is_lower = char in lowercase_chars

            dimensional_glyphs.append(GlyphData(
                char=char,
                slot=slot,
                glyph_name=glyph_name,
                svg_paths=entry.get("svg_paths", []),
                svg_width=svg_w,
                svg_height=svg_h,
                baseline_y_in_svg=baseline_y,
                is_lowercase=is_lower,
                upscale_factor=upscale_factor,
            ))

            centerline_paths = entry.get("svg_paths_centerline", [])
            if centerline_paths:
                line_glyphs.append(GlyphData(
                    char=char,
                    slot=slot,
                    glyph_name=glyph_name,
                    svg_paths=centerline_paths,
                    svg_width=svg_w,
                    svg_height=svg_h,
                    baseline_y_in_svg=baseline_y,
                    is_lowercase=is_lower,
                    upscale_factor=upscale_factor,
                ))
            else:
                line_skipped_chars.append(glyph_id)

        job_store.update_state(job_id, progress_pct=25)

        if not dimensional_glyphs:
            raise ValueError("No valid glyphs were approved")

        # Build dimensional OTF + WOFF2
        otf_bytes, fea_warning = build_otf(
            dimensional_glyphs, font_name, font_style, letter_spacing, space_width,
        )
        job_store.update_state(job_id, progress_pct=50)
        woff2_bytes = otf_to_woff2(otf_bytes)
        job_store.update_state(job_id, progress_pct=65)

        # Build line OTF + WOFF2 (same family, "Line" style)
        line_otf_bytes = b""
        line_woff2_bytes = b""
        line_fea_warning: Optional[str] = None
        if line_glyphs:
            line_otf_bytes, line_fea_warning = build_otf(
                line_glyphs, font_name, "Line", letter_spacing, space_width,
            )
            job_store.update_state(job_id, progress_pct=85)
            line_woff2_bytes = otf_to_woff2(line_otf_bytes)

        job_store.update_state(job_id, progress_pct=92)

        # Save all files
        safe_name = font_name.replace(" ", "_")
        out_dir = job_store.output_dir(job_id)
        otf_path = out_dir / f"{safe_name}.otf"
        woff2_path = out_dir / f"{safe_name}.woff2"
        with open(otf_path, "wb") as f:
            f.write(otf_bytes)
        with open(woff2_path, "wb") as f:
            f.write(woff2_bytes)

        font_files = {
            "otf": str(otf_path),
            "woff2": str(woff2_path),
        }

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
        if line_skipped_chars:
            extra["line_skipped_glyphs"] = line_skipped_chars

        job_store.update_state(
            job_id,
            status="complete",
            progress_pct=100,
            font_files=font_files,
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


def _load_page_images(raw_path: str, is_pdf: bool) -> List[np.ndarray]:
    """Load raw uploaded file into a list of BGR page images."""
    if is_pdf:
        return _pdf_to_images(raw_path)
    else:
        img = load_image_from_file(raw_path)
        return [img]


def _pdf_to_images(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    """Convert each page of a PDF to a BGR numpy array at the given DPI."""
    doc = fitz.open(pdf_path)
    images = []

    for page in doc:
        zoom = dpi / 72  # fitz default is 72 dpi
        # Cap so longest side never exceeds 5000 px (prevents OOM on phone-photo PDFs)
        longest = max(page.rect.width, page.rect.height) * zoom
        if longest > 5000:
            zoom = zoom * 5000 / longest
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        images.append(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

    doc.close()
    return images
