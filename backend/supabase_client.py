"""
supabase_client.py — Singleton Supabase client used across the backend.

Required env vars:
  SUPABASE_URL               — project URL (https://xxx.supabase.co)
  SUPABASE_SERVICE_ROLE_KEY  — service role key (bypasses RLS, backend-only)
"""

import os

from supabase import Client, create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set. "
        "Get them from your Supabase project → Settings → API."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

STORAGE_BUCKET = "ttf-generator"
