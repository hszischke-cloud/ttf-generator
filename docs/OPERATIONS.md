# Operations Runbook — Supabase setup, upload limits, future tables

This is the single reference for getting the app running on a fresh Supabase
and for the guard-rails around the pen-plotter / Cricut tools. It's written so a
future session (or you) can execute it top-to-bottom with no guesswork.

---

## 0. TL;DR — do the pen-plotter / Cricut tools need Supabase?

**No.** `POST /tools/image-to-svg` and `POST /tools/optimize-svg` are **stateless
pure functions** — they take an upload, process it in a threadpool, and return
SVG. They touch **no database, no storage, and no auth**. They work the moment
the backend is deployed with the image dependencies installed
(`pillow`, `numpy`, `scipy`, `scikit-image`, `python-multipart` — already in
`backend/requirements.txt`).

The only Supabase that exists is for the **font app** (jobs + saved fonts +
font-file storage). Section 3 is the runbook for that. Sections 5–6 are
**optional, future** Supabase additions (rate-limiting, analytics, saved
exports, accounts) with ready-to-paste SQL for when the business needs them.

---

## 1. Upload limits & guard-rails (live now)

All limits live in **`backend/processing/tool_limits.py`** and are
**env-overridable** — change them on the host (e.g. Render → Environment) with
no code change. The browser fetches them from `GET /tools/limits` so the UI
helper text and client-side pre-checks always match the server.

| Env var | Default | Guards against | User-facing error |
|---|---|---|---|
| `TOOL_MAX_IMAGE_BYTES` | 15 MB | Oversized image uploads | 413 — "That image is X MB, over the 15 MB limit…" |
| `TOOL_MAX_IMAGE_MEGAPIXELS` | 40 | Decompression bombs / OOM (a tiny PNG can decode to 100s of MP) | 422 — "That image is X megapixels, over the 40 MP limit…" |
| `TOOL_MAX_TRACE_DIMENSION` | 2000 | Slow tracing — inputs are downscaled to this longest edge | (none; silent, reported in `stats.scale`) |
| `TOOL_MAX_SVG_BYTES` | 5 MB | Oversized / pathological SVGs | 413 — "That SVG is X MB, over the 5 MB limit." |

Defense in depth on the image path, in order:
1. **Byte cap** (`413`) — rejected before anything is decoded.
2. **Format allow-list** (`415`) — only PNG/JPG/GIF/BMP/WEBP/TIFF; anything else
   (incl. an SVG sent to the raster endpoint) is refused with the supported list.
3. **Megapixel cap from the header** (`422`) — checked from `image.size` *before*
   `image.load()`, so a bomb never gets decoded.
4. **PIL `MAX_IMAGE_PIXELS`** is also set per-request as a backstop
   (`DecompressionBombError` → friendly `422`).
5. **Downscale** to `TOOL_MAX_TRACE_DIMENSION` before tracing.

SVG path: byte cap (`413`) → UTF-8 decode check (`415`, catches `.svgz`/binary) →
parse (`400` "We couldn't read that SVG…").

**"No paths found" is not an error.** If tracing yields zero paths the endpoint
returns `200` with `path_count: 0`; the UI shows a friendly hint to try a
clearer image / toggle "light on dark" / raise ink detection.

To change a limit: set the env var and redeploy. Example (Render):
`TOOL_MAX_IMAGE_MEGAPIXELS=24`. No client change needed — the UI re-reads it.

---

## 2. Environment variables (complete)

| Var | Required | Used by | Notes |
|---|---|---|---|
| `SUPABASE_URL` | ✅ (font app) | `supabase_client.py` | Project URL `https://xxx.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | ✅ (font app) | `supabase_client.py` | **Service role** key — backend only, bypasses RLS. Never ship to the browser. |
| `ALLOWED_ORIGINS` | optional | CORS | Comma-separated; default `*` |
| `TOOL_MAX_IMAGE_BYTES` | optional | tools | Default 15 MB |
| `TOOL_MAX_IMAGE_MEGAPIXELS` | optional | tools | Default 40 |
| `TOOL_MAX_TRACE_DIMENSION` | optional | tools | Default 2000 |
| `TOOL_MAX_SVG_BYTES` | optional | tools | Default 5 MB |

> The backend **raises on boot** if `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY`
> are missing (see `supabase_client.py`). The tools themselves don't need them,
> but the app won't start without them today because `main.py` imports the font
> pipeline. (If you ever want a tools-only deploy, that import would need to be
> made lazy — noted in §7.)

---

## 3. Supabase setup runbook (font app)

Do this once per environment (prod, staging). Idempotent — safe to re-run.

1. **Create the project** at supabase.com → New project. Pick a region near your
   users / your Render region.
2. **Grab credentials:** Project → Settings → API → copy the **Project URL** and
   the **`service_role`** key.
3. **Set env vars** on the backend host (Render → your service → Environment):
   - `SUPABASE_URL=https://xxx.supabase.co`
   - `SUPABASE_SERVICE_ROLE_KEY=<service_role key>`
   - (optional) `ALLOWED_ORIGINS=https://yourdomain.com`
4. **Run the schema:** Supabase → SQL Editor → paste all of **`schema.sql`** →
   Run. This creates:
   - `jobs` table (JSONB blob) + the `created_at` index
   - `patch_job_state(p_job_id, p_patch)` RPC (server-side shallow merge)
   - `saved_fonts` table (FK → jobs, `ON DELETE CASCADE`)
   - the **public** `ttf-generator` storage bucket + a service-role full-access
     policy
5. **Verify the bucket:** Storage → confirm `ttf-generator` exists and **Public
   bucket = ON** (font files are served as content-versioned public URLs).
6. **Deploy & health-check:** hit `GET /health` → `{"status":"ok"}`; open `/ui`.
7. **Cleanup is automatic:** jobs older than 24 h are deleted hourly in-process
   (`_periodic_cleanup` in `main.py`); jobs referenced by `saved_fonts` are kept.

That's the whole dependency. There is no migration tool — `schema.sql` is the
source of truth and uses `IF NOT EXISTS` / `CREATE OR REPLACE` throughout.

---

## 4. Data model & security notes

- **`jobs.state`** is one JSONB blob carrying the full glyph manifest
  (`svg_paths` + `pen_paths`, several MB). Never `SELECT *` it to read a couple
  of fields — use `job_store.get_state_fields()` (PostgREST `state->key`) and
  write via the `patch_job_state` RPC.
- **RLS:** the backend uses the **service role**, which bypasses RLS, so no
  policies are required for the app to work today. The tables are not exposed to
  the browser. **If you ever add browser-direct Supabase access (anon key),
  enable RLS first** — otherwise the anon key can read every job.
- **Storage:** `ttf-generator` is public on purpose (font binaries are served via
  public CDN URLs, content-versioned by filename). Don't put anything sensitive
  in it.
- **Backups:** turn on Point-in-Time Recovery (Supabase → Database → Backups)
  once there's real saved-font data.

---

## 5. Hardening checklist

Live now:
- [x] Per-type upload byte caps (image 15 MB, SVG 5 MB)
- [x] Image megapixel cap + PIL decompression-bomb guard
- [x] Image format allow-list; SVG UTF-8/parse validation
- [x] CPU work in a threadpool; large images downscaled
- [x] Clear, specific, human-readable error messages (table in §1)
- [x] CORS via `ALLOWED_ORIGINS`

Recommended next (in rough priority):
- [ ] **Rate-limit the free tools** to stop abuse of a public CPU endpoint — §6A.
- [ ] **Error monitoring** (Sentry or Logflare) on the backend.
- [ ] **Proxy-level body-size limit** (Render/Cloudflare) as a first line before
      the app even reads the upload.
- [ ] **Usage analytics** to see which tools/modes get used — §6B.
- [ ] **Storage growth caps / lifecycle** if you start saving user outputs — §6C.

---

## 6. OPTIONAL / FUTURE Supabase additions (ready-to-paste SQL)

> None of this is needed today. Paste the block for a feature when you want it,
> then wire the backend as noted. All blocks are idempotent.

### 6A. Rate-limiting the free tools (recommended before a public launch)

```sql
-- Per-identifier daily counter. Store a HASHED id (never a raw IP).
CREATE TABLE IF NOT EXISTS tool_usage (
    id    TEXT    NOT NULL,                 -- e.g. sha256(salt + client ip)
    day   DATE    NOT NULL DEFAULT CURRENT_DATE,
    tool  TEXT    NOT NULL,                 -- 'image-to-svg' | 'optimize-svg'
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (id, day, tool)
);

-- Atomically increment and return the new daily count for (id, tool).
CREATE OR REPLACE FUNCTION bump_tool_usage(p_id TEXT, p_tool TEXT)
RETURNS INTEGER LANGUAGE plpgsql AS $$
DECLARE new_count INTEGER;
BEGIN
    INSERT INTO tool_usage (id, day, tool, count)
    VALUES (p_id, CURRENT_DATE, p_tool, 1)
    ON CONFLICT (id, day, tool)
        DO UPDATE SET count = tool_usage.count + 1
    RETURNING count INTO new_count;
    RETURN new_count;
END; $$;
```

Backend wiring: at the top of each tool endpoint, hash `request.client.host`
with a server-side salt, call `supabase.rpc("bump_tool_usage", {...})`, and if
the returned count exceeds your daily cap, raise `429` with a clear message
("You've hit today's free limit — try again tomorrow."). Add a daily cleanup of
rows where `day < CURRENT_DATE - 30`.

### 6B. Usage analytics / event log

```sql
CREATE TABLE IF NOT EXISTS tool_events (
    id          BIGSERIAL   PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tool        TEXT        NOT NULL,       -- 'image-to-svg' | 'optimize-svg'
    mode        TEXT,                       -- 'line' | 'outline' | NULL
    ok          BOOLEAN     NOT NULL DEFAULT TRUE,
    duration_ms INTEGER,
    bytes_in    INTEGER,
    path_count  INTEGER
);
CREATE INDEX IF NOT EXISTS tool_events_created_at ON tool_events (created_at);
```

Backend wiring: fire-and-forget an insert after each request (wrap in try/except
so analytics never breaks the tool). Roll up with a simple SQL query or a
Supabase scheduled function.

### 6C. Saved exports (an account feature — pairs with 6D)

```sql
-- Private bucket for user-saved SVGs (NOT public).
INSERT INTO storage.buckets (id, name, public)
VALUES ('tool-exports', 'tool-exports', false)
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS tool_exports (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID        REFERENCES auth.users(id) ON DELETE CASCADE,
    tool         TEXT        NOT NULL,
    name         TEXT,
    storage_path TEXT        NOT NULL,      -- path within the tool-exports bucket
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE tool_exports ENABLE ROW LEVEL SECURITY;
CREATE POLICY "own exports" ON tool_exports
    FOR ALL TO authenticated
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());
```

### 6D. Accounts & paid entitlements (the SaaS layer)

When you add the paid product, use **Supabase Auth** for accounts and add a
small entitlements table keyed on `auth.users`:

```sql
CREATE TABLE IF NOT EXISTS entitlements (
    user_id    UUID        PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    plan       TEXT        NOT NULL DEFAULT 'free',   -- 'free' | 'plus' | 'studio'
    status     TEXT        NOT NULL DEFAULT 'active',
    renews_at  TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE entitlements ENABLE ROW LEVEL SECURITY;
CREATE POLICY "read own entitlement" ON entitlements
    FOR SELECT TO authenticated USING (user_id = auth.uid());
-- Writes happen server-side (service role) from your payment webhook only.
```

Wiring: a payment webhook (Stripe / Lemon Squeezy / Paddle — see the business
notes) upserts `entitlements` via the service role; the backend gates premium
features (e.g. higher limits, no watermark, batch) on `plan`.

---

## 7. Local dev & tests

```bash
cd backend
pip install -r requirements.txt          # or the CI subset (see ci.yml)
pytest tests/ -q                          # 37 tests; no network (Supabase stubbed)
```

- `tests/conftest.py` installs an in-memory Supabase stub into `sys.modules`
  before the app imports, so the whole suite runs offline.
- Frontend check (matches CI): extract the `<script>` from `app.html` and run
  `node --check`.
- The tools tests (`test_image_trace.py`, `test_plotter_opt.py`,
  `test_tools_api.py`) need the image stack + `python-multipart`; CI installs
  them (`.github/workflows/ci.yml`).
- **Tools-only deploy (future):** if you ever want to run just the plotter/Cricut
  tools without Supabase, make the font-pipeline imports in `main.py` lazy and
  guard the Supabase client boot — the `/tools/*` routes have zero Supabase
  dependencies today.
