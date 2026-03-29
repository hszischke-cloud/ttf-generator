# TTF Generator — Claude Instructions

## Project Overview
Handwriting-to-font web app. User downloads a PDF template, hand-writes characters, scans/uploads the page, reviews detected glyphs, and downloads an OTF + WOFF2 font file.

Integrates into an existing Next.js website for personalized letters.

**Stack:**
- Backend: Python + FastAPI on Railway
- Frontend: Next.js/TypeScript on Vercel
- Image processing: OpenCV + adaptive thresholding
- Vectorization: vtracer via cv2 contours (potrace fallback mentioned but not yet implemented)
- Font generation: fonttools (CFF/OTF) + feaLib for calt/ss01/ss02 OpenType features
- Template: reportlab

---

## Architecture

```
Frontend (Next.js/TypeScript)
├── pages/font-generator/
│   ├── index.tsx        — landing (instructions + template download)
│   ├── upload.tsx       — file upload + job creation
│   ├── review.tsx       — poll status, approve/reject glyphs, trigger finalize
│   └── download.tsx     — poll until complete, live font preview, download
├── components/
│   ├── UploadDropzone.tsx
│   ├── GlyphGrid.tsx
│   └── GlyphCell.tsx
└── /api/backend/*  →  proxy to backend (BACKEND_URL env var)

Backend (FastAPI/Python)
├── main.py          — endpoints + background task launcher
├── models.py        — Pydantic types + JobStatus enum
├── job_store.py     — filesystem job state (/tmp/jobs/{job_id}/)
├── worker.py        — subprocess entry point for processing
├── template.py      — PDF generation + TEMPLATE_SPEC (single source of truth)
└── processing/
    ├── alignment.py     — registration marker detection + perspective correction
    ├── extraction.py    — cell crop + adaptive-threshold glyph segmentation
    ├── vectorize.py     — contour → SVG paths
    └── font_builder.py  — SVG → OTF/WOFF2 via fonttools
```

---

## Job Lifecycle

```
POST /upload → pending → processing → awaiting_review (user reviews glyphs)
                                    ↓
                              POST /finalize → finalizing → complete
                                    (any step can → error)
```

**Job directory structure:**
```
/tmp/jobs/{job_id}/
├── raw/             # original upload
├── pages/           # page_0.png, page_1.png (300 dpi)
├── glyphs/          # {glyph_id}.png + {glyph_id}.json (SVG paths + metadata)
├── output/          # {font_name}.otf + {font_name}.woff2
└── state.json       # atomic lock-protected job state
```

State fields: `status`, `progress_pct`, `error_message`, `created_at`, `glyph_manifest`, `alignment_warnings`, `approved_glyph_ids`, `font_name`, `font_style`, `font_files`, `raw_filename`, `is_pdf`, `alignment_failed_image_b64`

Cleanup: jobs deleted after 24h (checked hourly). On startup: jobs >10 min old deleted.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | `{"status": "ok"}` |
| GET | `/template/download` | PDF binary (2-page A4) |
| POST | `/process/upload` | MultipartForm file → `{"job_id": str}` |
| GET | `/process/{job_id}/status` | `JobStatusResponse` |
| GET | `/process/{job_id}/glyphs` | `GlyphsResponse` (requires awaiting_review) |
| POST | `/process/{job_id}/finalize` | `FinalizeRequest` → `FinalizeResponse` |
| GET | `/fonts/{job_id}/{filename}` | OTF or WOFF2 binary (202 if not ready) |
| GET | `/ui` | Standalone HTML UI |

**Key models:**
```python
class JobStatus(str, Enum):
    PENDING | PROCESSING | AWAITING_REVIEW | FINALIZING | COMPLETE | ERROR

class GlyphInfo:
    glyph_id: str       # e.g. "e_0", "period_1"
    char: str
    slot: int           # 0 = primary, 1+ = alternate
    image_b64: str      # base64 PNG tight-crop
    accepted: bool = True

class FinalizeRequest:
    approved_glyph_ids: List[str]
    font_name: str
    font_style: str = "Regular"
    manual_alignment: Optional[List[List[ManualAlignmentPoint]]]  # not yet wired up
```

---

## Template Specification (`TEMPLATE_SPEC` in template.py — single source of truth)

```
Page size: A4 (210×297 mm)
Cell size: 18×24 mm, 10 columns
Margins: left 12 mm, top 20 mm

Guideline ratios (fraction of cell height from top):
  CAP line:       0.15
  x-height:       0.42
  baseline:       0.72
  descender:      0.90

Guide gutter (left, excluded from extraction): 3.5 mm
Registration markers: 8 mm diameter, 4 mm from page edge (4 corners)

Characters:
  Page 0: A-Z (cells 0-25), a-z (cells 26-51)
  Page 1: 0-9, .,;:!?"'()-/@&# (punct), then alternates section
  Alternates: e,t,a,o,i,n,s,h,r,d — 2 extra slots each
```

---

## Processing Pipeline Details

### Alignment (`alignment.py`)
1. EXIF auto-rotate (PIL)
2. Grayscale → threshold (THRESH_BINARY_INV, 127)
3. Find contours, filter by area `[r×0.4, r×1.8]` and circularity ≥ 0.75
4. Pick best 4 by circularity → sort to [TL, TR, BL, BR]
5. `findHomography` (RANSAC, 5.0) → `warpPerspective` (LANCZOS4, white border)
6. Output: 2480×3508 px (A4 at 300 dpi)
- Min input: 1200×1600 px; if alignment fails, returns raw image + error

### Extraction (`extraction.py`)
Per cell: crop to CAP→descender lines, exclude 3.5 mm left gutter + 8px inset
- Gaussian blur (3×3) → adaptive threshold (GAUSSIAN_C, blockSize=21, C=10)
- Morphological closing (3×3, 2 iterations) to connect strokes
- Find external contours, filter: min area 50px², max 80% cell area
- Merge bounding boxes, add 8px padding
- Glyph IDs: letters/digits as-is; punct mapped (period, comma, exclam, etc.)

### Vectorization (`vectorize.py`)
- Upscale if height < 400px (LANCZOS4)
- Re-threshold → `findContours(RETR_TREE, CHAIN_APPROX_TC89_KCOS)`
- Filter: min area 0.1% of image; Douglas-Peucker epsilon = 0.005 × perimeter
- Output: SVG `M L C Z` paths, width/height px, upscale_factor
- Reject if ink < 20px or no valid contours

### Font Builder (`font_builder.py`)
**UPM = 1000.** Constants derived from TEMPLATE_SPEC ratios:
```
ASCENDER = 800, CAP_HEIGHT = 700
X_HEIGHT ≈ 368, DESCENDER ≈ -221
```

Coordinate transform (SVG → font space):
```
coord_scale = CELL_SCALE / upscale_factor
fx = sx × coord_scale
fy = -(sy × coord_scale) + (baseline_y_in_svg × CELL_SCALE)
```
All glyphs use same CELL_SCALE → consistent baseline alignment.

Advance width: `int(svg_width × coord_scale) + 60` (fixed 60 UPM side bearing)

**OpenType features:**
- `calt`: distance-based cycling up to 8 glyphs look-back; each letter cycles through its forms for natural variety
- `ss01`: force all letters to first alternate
- `ss02`: force all letters to second alternate
- feaLib errors are caught/skipped — font still builds without features

---

## Incomplete / Not Yet Implemented

1. **Manual alignment UI**: `FinalizeRequest.manual_alignment` field exists, `align_page_with_manual_points()` exists in alignment.py, but frontend has no corner-selection UI and worker never calls it
2. **Cloudflare R2**: CLAUDE.md mentions R2 for output persistence — no R2 code exists yet; fonts saved to local filesystem only
3. **Potrace fallback**: mentioned in docs but not in code; current vectorizer uses only cv2 contours
4. **`_DeferredPen`**: class exists in font_builder.py but is unused

---

## Environment Variables

| Var | Default | Used by |
|-----|---------|---------|
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:3001` | Backend CORS |
| `JOBS_DIR` | `/tmp/jobs` | Backend job storage |
| `BACKEND_URL` | — | Frontend Next.js proxy |

---

## Key Thresholds (for debugging/tuning)

| Stage | Parameter | Value |
|-------|-----------|-------|
| Alignment | Min input resolution | 1200×1600 px |
| Alignment | Marker circularity | ≥ 0.75 |
| Extraction | Adaptive threshold block | 21 px |
| Extraction | Min contour area | 50 px² |
| Vectorization | Min height before upscale | 400 px |
| Vectorization | D-P epsilon | 0.005 × perimeter |
| Font | calt look-back distance | 8 glyphs |
| Font | Side bearing | +60 UPM |
| Upload | Max file size | 50 MB |
| Frontend | Poll interval | 2000 ms |
| Cleanup | Max job age | 24 h |

---

## Working Preferences

- Think in product/user-flow terms first, then implementation
- Plan thoroughly before coding — raise design questions upfront
- Look for existing open-source approaches before designing from scratch
- Keep responses concise and direct
