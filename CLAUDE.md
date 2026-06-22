# TTF Generator — Claude Instructions

## Project Overview
Handwriting-to-font web app: the user **draws** each character on a canvas
(no scanning/uploading — that flow was removed long ago), reviews the glyphs,
fine-tunes spacing and per-letter borders, and downloads an OTF — plus a
single-line OTF for pen plotters. Branded for snailmail.eco (personalized
letters).

**Stack:**
- Backend: Python + FastAPI (deployed on Render: `ttf-generator.onrender.com`)
- Persistence: Supabase — Postgres (`jobs`, `saved_fonts` tables) + public
  Storage bucket `ttf-generator`
- Font generation: fontTools (CFF/OTF), feaLib (`calt` alternates + cursive
  positional rules), COLR/CPAL for the chosen ink colour
- Frontend: two single-file pages (vanilla JS, no build step), served by the
  backend with the API base rewritten to same-origin:
  - `app.html` at `/ui` (and `/` redirects there) — the STUDIO: dashboard,
    free-form drawing, review, spacing, borders editor, saved fonts.
  - `client.html` at `/create` — the LOCKED-DOWN guided creator for clients:
    intro screen → one prompted letter at a time → build → name + spacing →
    download. No dashboard or saved-fonts access. The dashboard's "Share
    with a client" group links/copies this URL.

---

## Repository layout

```
app.html                     — studio frontend (single file, /ui)
client.html                  — guided client creator (single file, /create)
schema.sql                   — Supabase schema; paste into the SQL editor
backend/
├── main.py                  — FastAPI endpoints + font build pipeline
├── models.py                — Pydantic request/response models
├── job_store.py             — Supabase-backed job state + storage
├── font_registry.py         — saved-fonts registry (saved_fonts table)
├── supabase_client.py       — singleton client (forces HTTP/1.1)
├── tests/                   — pytest suite (in-memory Supabase stub)
└── processing/
    ├── font_builder.py      — glyph outlines → OTF (metrics, features, COLR)
    ├── pen_realistic.py     — realistic-ink stroker (pen_paths → outlines)
    ├── autospace.py         — optical side-bearing suggestions
    ├── centerline.py        — pen_paths → single-line SVG for the line OTF
    ├── perturb.py           — legacy outline jitter (permanently off)
    ├── proof_sheet.py       — OTF → SVG proof sheet
    ├── plotter_opt.py       — SVG draw-order optimizer (NN + 2-opt, pen-up travel)
    └── image_trace/         — image → SVG tracer (vendored): centerline (line)
                               OR find_contours (outline); shared RDP + Bézier
```

## Pen-plotter / Cricut tools (standalone, stateless)
Two pure-function tools share the studio shell, aimed at the plotter/Cricut
audience — no font or job state involved:
- `POST /tools/image-to-svg` — upload an image, get a single-line **(mode=line,
  centerline)** SVG for pens/plotters OR a filled **(mode=outline, even-odd)**
  SVG for cutting. Backed by `processing/image_trace/` (PIL + scikit-image:
  threshold→despeckle→skeletonize/centerline or find_contours/outline).
- `POST /tools/optimize-svg` — reorder an SVG's strokes to minimise pen-up
  travel (`processing/plotter_opt.py`). Returns original/optimized travel.

Both run in a threadpool, lazily import the image stack, cap uploads at 25 MB,
and downscale large inputs. UI: dashboard cards → `page-trace` / `page-plotter`
in app.html (friendly presets + plain-language toggles; "Send to optimizer"
hands the traced SVG straight to the optimizer).

## Pen weights (dimensional OTF)
Every glyph stores `pen_paths` (raw centerline strokes, points
`[x, y, pressure, t_ms]`; legacy `[x, y]` still supported) plus `svg_paths`
(legacy canvas brush polygons, kept only as a fallback for glyphs whose
pen_paths can't be stroked). At build time `processing/pen_realistic.py`
ALWAYS re-strokes each glyph's `pen_paths` with an ink-dynamics model:
speed/pressure width modulation, ink pooling at corners/dwell points, round
tip caps, blob starts, flick tails, Bézier liquid edges with paper-fibre
notches. The UPM-space perturb/divot pass is permanently off (the stroker
owns the edge texture). Deterministic per glyph_id.

The job-state field `pen_style` picks only the WEIGHT:
- `realistic` — fine, true to the drawn pen size (default).
- `realistic-bold` — same model with the tip scaled by `BOLD_WIDTH_SCALE`
  (1.35); the whole model scales so it pools/caps like a fatter pen.
- legacy values (`classic`, missing) are coerced to `realistic` by
  `_normalize_pen_style` in main.py.

Switching is lossless both ways (`POST /process/{job_id}/pen-style`, rebuilds
with stored settings; advances are identical across weights so layout never
shifts). `FinalizeRequest.pen_style=None` preserves the job's current weight.
UI: Fine/Bold selector on the draw page (canvas previews the full ink model
live), toggle on the download page and per saved font card.

## Auto-borders on first build
A print job's FIRST build computes the optical margin-equalizing side
bearings (`processing/autospace.py`) and persists them as `glyph_bearings`,
so every glyph starts with perceived-width-equalized spacing and later
rebuilds / the borders editor inherit the same values. Gated on "never built
+ no manual bearings + not cursive" so pre-existing fonts and user-adjusted
borders are never silently re-spaced.

## Job lifecycle

```
POST /draw/create → awaiting_review  (glyphs submitted in batches)
POST /process/{id}/finalize → finalizing → complete   (or → error)
POST /process/{id}/reopen → awaiting_review            (edit a saved font)
```

State is one JSONB blob per job in the `jobs` table. It carries the full
glyph manifest (svg_paths + pen_paths per glyph — several MB), so:
- **Never fetch the whole blob to read a couple of fields** — use
  `job_store.get_state_fields()` (PostgREST `state->key` selection).
- Writes go through the `patch_job_state` RPC (server-side shallow merge);
  falls back to read-modify-write if the RPC isn't deployed.

Cleanup: jobs older than 24 h are deleted hourly; jobs in `saved_fonts` are
kept forever.

## API endpoints (all in main.py)

| Method | Path | Notes |
|--------|------|-------|
| GET  | `/health` | |
| GET  | `/`, `/ui` | studio (app.html; `/` redirects to `/ui`) |
| GET  | `/create` | guided client creator (client.html) |
| POST | `/tools/image-to-svg` | image → single-line (`mode=line`) or outline (`mode=outline`) SVG |
| POST | `/tools/optimize-svg` | reorder SVG strokes to cut pen-plotter travel |
| POST | `/draw/create` | new empty job |
| POST | `/draw/{id}/glyph` | single glyph upsert (legacy) |
| POST | `/draw/{id}/glyphs/batch` | **preferred** — one manifest write for N glyphs |
| GET  | `/process/{id}/status` | cheap field-select; polled every 1–2 s |
| GET  | `/process/{id}/glyphs` | review list; thumbnails as public CDN URLs |
| POST | `/process/{id}/finalize` | build OTF + line OTF (BackgroundTasks) |
| POST | `/process/{id}/pen-style` | switch fine/bold weight + rebuild (lossless) |
| GET  | `/process/{id}/bearings` | border-editor payload (iso glyphs only) |
| GET  | `/process/{id}/bearings/auto?tightness=N` | optical lsb/rsb suggestions |
| GET  | `/process/{id}/pen-paths` | full glyph data to re-open for editing |
| GET  | `/process/{id}/settings` | spacing/name/pen_style needed to re-finalize |
| POST | `/process/{id}/reopen` | back to awaiting_review |
| GET  | `/process/{id}/proof?font=line\|dimensional` | SVG proof sheet |
| GET  | `/fonts/{id}/{filename}` | 302 → content-versioned public storage URL |
| POST/GET/PATCH/DELETE | `/fonts/save/{id}`, `/fonts/saved`, `/fonts/saved/{id}` | saved-fonts registry (PATCH = rename) |

## Coordinate system & font metrics (font_builder.py)

- Canvas: 300×400 logical px. Guidelines: cap y=60, x-height 168,
  **baseline 288**, descender 380 (ratio 0.95 — must match
  `GUIDELINE_BOTTOM_RATIO`). Safe band: y 30–380, x 16–284 (the UI
  hard-clamps strokes into it; cursive connecting forms keep free x).
- Glyphs are submitted post-xShift: ink starts at x = PAD (12 px);
  `svg_width` = ink width + 2·PAD.
- `CELL_SCALE` ≈ 4.35 UPM/px (UPM 1000). Real ink landmarks: cap ink
  ≈ 991 UPM, x-height ≈ 522, descender ink ≈ −400.
  **Don't change CELL_SCALE** — it would resize every saved font.
- Vertical metrics derive from the real canvas mapping: hhea/typo
  ascent 1050 / descent ≈ −450; `usWinAscent 1280 / usWinDescent 520`
  cover the whole drawable canvas so Windows never clips.
- iso glyphs: advance = lsb + ink width + rsb; default bearings are 0/0
  (edges touch at letter_spacing 0). Per-glyph overrides live in state
  `glyph_bearings` and are preserved across spacing-only re-finalizes.
- Cursive (init/medi/fina) forms derive bearings from entry_x/exit_x
  connection points; they're excluded from the border editor.

## Border auto-adjust (processing/autospace.py)

HT-Letterspacer-style optical margins: scanlines across a per-case zone
(x-height band for lowercase, cap band for caps/digits, own ink band for
punctuation), depth-capped, target = median letter margin (self-calibrating),
output clamped to [−150, 250] UPM. Applied automatically on a print job's
first build (see above); afterwards suggestions only — the borders editor
shows them as ghost ticks and nothing persists until the user applies.

## Frontend notes (app.html)

- Single file: pages = dashboard (home) / draw / review / download / borders,
  toggled by `goTo()`. `const API = 'http://localhost:8001'` is rewritten to
  '' when served by the backend.
- Drafts autosave to localStorage (`hwfont_draft_v1`, points keep pressure +
  time so a resumed draft re-inks identically) debounced after every stroke;
  resume card on the dashboard. Edit sessions of saved fonts are NOT
  autosaved (server is source of truth) — they get a beforeunload guard.
- Canvas uses a devicePixelRatio backing store; all logic stays in logical
  300×400 coords. The canvas renders exactly what the OTF will be: realistic
  ink half-width profile mirroring pen_realistic.py (`PEN_BOLD_SCALE` must
  match `BOLD_WIDTH_SCALE`), flat single-colour nonzero fill + round caps.
- Spacing preview: the previewed OTF has the last build's spacing BAKED IN;
  CSS letter/word-spacing applies only the *delta* from `lastBuiltSpacing`.
- Builds show the full-screen overlay (`buildOverlayShow/Set/Hide` +
  `waitForBuild`); backend progress is eased/crept so the bar never stalls.
- Border editor: drag lines or type values; ghost ticks show the auto
  suggestion; live word preview composes the real outlines with current
  bearings (no rebuild). Tightness slider re-applies cached suggestions
  client-side.
- Glyph thumbnails are public CDN URLs (content-versioned), rendered at half
  scale on upload. Saved-font card names render in their own font via lazy
  @font-face.

## Testing

`cd backend && pytest tests/` — no network needed: `tests/conftest.py`
installs an in-memory Supabase stub into `sys.modules` before app import.
The API test does a REAL fontTools build. CI runs this + a node syntax check
of app.html on every push/PR (`.github/workflows/ci.yml`).

## Environment variables

| Var | Used by |
|-----|---------|
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | required by the backend |
| `ALLOWED_ORIGINS` (default `*`) | CORS |

## Working preferences

- Think in product/user-flow terms first, then implementation.
- Plan thoroughly before coding — raise design questions upfront.
- Never change saved fonts' metrics/widths implicitly; bearings and spacing
  are user-owned values.
- Keep responses concise and direct.
