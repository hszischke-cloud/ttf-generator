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
- Frontend: **one** single-file app — `app.html` (vanilla JS, no build step),
  served by the backend at `/ui` and `/create` (same file; `/` redirects)

---

## Repository layout

```
app.html                     — the entire frontend (single file)
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
    ├── autospace.py         — optical side-bearing suggestions
    ├── centerline.py        — pen_paths → single-line SVG for the line OTF
    ├── perturb.py           — organic outline jitter
    └── proof_sheet.py       — OTF → SVG proof sheet
```

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
| GET  | `/`, `/ui`, `/create` | serve app.html (API base rewritten to '') |
| POST | `/draw/create` | new empty job |
| POST | `/draw/{id}/glyph` | single glyph upsert (legacy) |
| POST | `/draw/{id}/glyphs/batch` | **preferred** — one manifest write for N glyphs |
| GET  | `/process/{id}/status` | cheap field-select; polled every 1–2 s |
| GET  | `/process/{id}/glyphs` | review list; thumbnails as public CDN URLs |
| POST | `/process/{id}/finalize` | build OTF + line OTF (BackgroundTasks) |
| GET  | `/process/{id}/bearings` | border-editor payload (iso glyphs only) |
| GET  | `/process/{id}/bearings/auto?tightness=N` | optical lsb/rsb suggestions |
| GET  | `/process/{id}/pen-paths` | full glyph data to re-open for editing |
| GET  | `/process/{id}/settings` | spacing/name needed to re-finalize |
| POST | `/process/{id}/reopen` | back to awaiting_review |
| GET  | `/process/{id}/proof?font=line\|dimensional` | SVG proof sheet |
| GET  | `/fonts/{id}/{filename}` | 302 → content-versioned public storage URL |
| POST/GET/PATCH/DELETE | `/fonts/save/{id}`, `/fonts/saved`, `/fonts/saved/{id}` | saved-fonts registry (PATCH = rename) |

## Coordinate system & font metrics (font_builder.py)

- Canvas: 300×400 logical px. Guidelines: cap y=60, x-height 168,
  **baseline 288**, descender 360. Safe band: y 30–360, x 16–284 (the UI
  hard-clamps strokes into it).
- Glyphs are submitted post-xShift: ink starts at x = PAD (12 px);
  `svg_width` = ink width + 2·PAD.
- `CELL_SCALE` ≈ 4.35 UPM/px (UPM 1000). Font y = −(canvas y)·scale +
  baseline·scale. Caps reach ≈ 990 UPM; descenders ≈ −313.
- `usWinAscent 1150 / usWinDescent 340` cover the full safe band so Windows
  never clips; hhea/typo metrics stay at 800/−221 so line spacing is stable.
  **Don't change CELL_SCALE** — it would resize every saved font.
- iso glyphs: advance = lsb + ink width + rsb; default bearings are 0/0
  (edges touch at letter_spacing 0). Per-glyph overrides live in state
  `glyph_bearings` and are preserved across spacing-only re-finalizes.
- Cursive (init/medi/fina) forms derive bearings from entry_x/exit_x
  connection points; they're excluded from the border editor.

## Border auto-adjust (processing/autospace.py)

HT-Letterspacer-style optical margins: scanlines across a per-case zone
(x-height band for lowercase, cap band for caps/digits, own ink band for
punctuation), depth-capped, target = median letter margin (self-calibrating),
output clamped to [−150, 250] UPM. Suggestions only — nothing persists until
the user applies (finalize with `glyph_bearings`).

## Frontend notes (app.html)

- Single file: pages = home / draw / review / download / borders, toggled by
  `goTo()`. `const API = 'http://localhost:8001'` is rewritten to '' when
  served by the backend.
- Drafts autosave to localStorage (`hwfont_draft_v1`) debounced after every
  stroke; resume card on home. Edit sessions of saved fonts are NOT
  autosaved (server is source of truth) — they get a beforeunload guard.
- Canvas uses a devicePixelRatio backing store; all logic stays in logical
  300×400 coords.
- Spacing preview: the previewed OTF has the last build's spacing BAKED IN;
  CSS letter/word-spacing applies only the *delta* from `lastBuiltSpacing`.
- Builds show the full-screen overlay (`buildOverlayShow/Set/Hide` +
  `waitForBuild`); backend progress is eased/crept so the bar never stalls.
- Border editor: drag lines or type values; ghost ticks show the auto
  suggestion; live word preview composes the real outlines with current
  bearings (no rebuild). Tightness slider re-applies cached suggestions
  client-side.
- Glyph thumbnails are public CDN URLs (content-versioned), rendered at half
  scale on upload.

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
