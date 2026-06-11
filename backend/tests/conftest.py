"""
conftest.py — test fixtures with an in-memory Supabase stub.

The stub is installed into sys.modules BEFORE the app modules import
`supabase_client`, so the whole API can be exercised without a network or
credentials. State lives in FakeTable.rows / FakeBucket.files per test.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_supabase_stub():
    if "supabase_client" in sys.modules:
        return sys.modules["supabase_client"]

    stub = types.ModuleType("supabase_client")
    stub.SUPABASE_URL = "https://stub.supabase.co"
    stub.STORAGE_BUCKET = "ttf-generator"

    class FakeTable:
        rows = {}        # job_id -> state dict (jobs table)
        saved = {}       # job_id -> row dict (saved_fonts table)

        def __init__(self, name):
            self.name = name
            self._sel = None
            self._eq = None
            self._op = None
            self._payload = None
            self._order = None

        def select(self, sel):
            self._op = "select"
            self._sel = sel
            return self

        def insert(self, payload):
            self._op = "insert"
            self._payload = payload
            return self

        def upsert(self, payload):
            self._op = "upsert"
            self._payload = payload
            return self

        def update(self, payload):
            self._op = "update"
            self._payload = payload
            return self

        def delete(self):
            self._op = "delete"
            return self

        def eq(self, key, value):
            self._eq = value
            return self

        def order(self, *a, **k):
            return self

        def execute(self):
            R = types.SimpleNamespace
            if self.name == "saved_fonts":
                rows = FakeTable.saved
                if self._op == "upsert":
                    rows[self._payload["job_id"]] = dict(self._payload)
                    return R(data=[self._payload])
                if self._op == "update":
                    if self._eq in rows:
                        rows[self._eq].update(self._payload)
                        return R(data=[rows[self._eq]])
                    return R(data=[])
                if self._op == "delete":
                    return R(data=[rows.pop(self._eq)] if self._eq in rows else [])
                if self._eq is not None:
                    row = rows.get(self._eq)
                    return R(data=[row] if row else [])
                return R(data=list(rows.values()))

            rows = FakeTable.rows
            if self._op == "insert":
                rows[self._payload["job_id"]] = self._payload["state"]
                return R(data=[{}])
            if self._op == "update":
                if self._eq in rows:
                    rows[self._eq] = self._payload["state"]
                return R(data=[{}])
            if self._op == "delete":
                rows.pop(self._eq, None)
                return R(data=[{}])
            if self._eq is not None and self._eq not in rows:
                return R(data=[])
            state = rows.get(self._eq, {})
            if self._sel == "state":
                return R(data=[{"state": state}])
            if self._sel and self._sel.startswith("state->"):
                out = {p.split("->")[-1]: state.get(p.split("->")[-1])
                       for p in self._sel.split(",")}
                return R(data=[out])
            return R(data=[{"job_id": self._eq}])

    class FakeBucket:
        files = {}

        def upload(self, path, data, file_options=None):
            FakeBucket.files[path] = data

        def download(self, path):
            if path not in FakeBucket.files:
                raise Exception("404")
            return FakeBucket.files[path]

        def list(self, prefix):
            return []

        def remove(self, paths):
            for p in paths:
                FakeBucket.files.pop(p, None)

    class FakeStorage:
        def from_(self, bucket):
            return FakeBucket()

    class FakeSupabase:
        storage = FakeStorage()

        def table(self, name):
            return FakeTable(name)

        def rpc(self, name, params):
            class X:
                def execute(self_inner):
                    jid, patch = params["p_job_id"], params["p_patch"]
                    if jid in FakeTable.rows:
                        FakeTable.rows[jid].update(patch)
                    return types.SimpleNamespace(data=[{}])
            if name != "patch_job_state":
                raise Exception("unknown rpc")
            return X()

    stub.supabase = FakeSupabase()
    stub._FakeTable = FakeTable
    stub._FakeBucket = FakeBucket
    sys.modules["supabase_client"] = stub
    return stub


_stub = _install_supabase_stub()


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    import main
    _stub._FakeTable.rows.clear()
    _stub._FakeTable.saved.clear()
    _stub._FakeBucket.files.clear()
    return TestClient(main.app)
