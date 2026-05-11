"""
supabase_client.py — Singleton Supabase client used across the backend.

Required env vars:
  SUPABASE_URL               — project URL (https://xxx.supabase.co)
  SUPABASE_SERVICE_ROLE_KEY  — service role key (bypasses RLS, backend-only)
"""

import os

# Force HTTP/1.1 for all httpx sync clients before the supabase client is
# created.  storage3 (supabase's storage library) passes http2=True to httpx,
# which multiplexes all requests over a single HTTP/2 connection.  After ~37
# streams the Supabase server sends an HTTP/2 GOAWAY frame (error_code:1
# PROTOCOL_ERROR) and terminates the connection mid-upload, causing
# RemoteProtocolError.  HTTP/1.1 opens a fresh TCP connection per request so
# there is no stream counter to hit; the trade-off is negligible for our
# upload sizes.
import httpx as _httpx
_orig_httpx_client_init = _httpx.Client.__init__


def _http1_only(self, *args, **kwargs):
    kwargs["http2"] = False
    _orig_httpx_client_init(self, *args, **kwargs)


_httpx.Client.__init__ = _http1_only

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
