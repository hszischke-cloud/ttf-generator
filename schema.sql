-- ============================================================
-- TTF Generator — Supabase schema
-- Paste this into the Supabase SQL editor and click Run.
-- ============================================================

-- Jobs: all job state stored as a JSONB blob so it mirrors the
-- old state.json exactly, with no schema migration needed when
-- new fields are added to the pipeline.
CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT        PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    state      JSONB       NOT NULL DEFAULT '{}'
);

-- Index so the cleanup query (find old jobs by created_at) is fast.
CREATE INDEX IF NOT EXISTS jobs_state_created_at
    ON jobs (((state->>'created_at')::double precision));

-- Saved-fonts registry: fonts the user has explicitly kept.
-- ON DELETE CASCADE: if a job row is hard-deleted the saved_fonts
-- row disappears too; saved jobs are normally excluded from cleanup.
CREATE TABLE IF NOT EXISTS saved_fonts (
    job_id      TEXT        PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    font_name   TEXT        NOT NULL,
    mode        TEXT        NOT NULL DEFAULT 'print',
    glyph_count INTEGER     NOT NULL DEFAULT 0,
    saved_at    DOUBLE PRECISION NOT NULL  -- Unix timestamp (matches Python time.time())
);

-- ============================================================
-- Storage bucket
-- Run this section OR create the bucket manually in the
-- Storage tab: name = "ttf-generator", Public bucket = ON.
-- ============================================================

-- Only works if you have the storage extension enabled (it is by default):
INSERT INTO storage.buckets (id, name, public)
VALUES ('ttf-generator', 'ttf-generator', true)
ON CONFLICT (id) DO NOTHING;

-- Allow the service role (used by the backend) to read/write everything.
-- Public read is already granted by the bucket being public.
CREATE POLICY "service role full access"
    ON storage.objects
    FOR ALL
    TO service_role
    USING (bucket_id = 'ttf-generator')
    WITH CHECK (bucket_id = 'ttf-generator');
