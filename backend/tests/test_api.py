"""End-to-end API tests against the in-memory Supabase stub.

Covers the whole product flow: create → batch submit → review → auto
bearings → finalize (real fontTools build) → serve → save/rename.
"""

import base64
import time

PNG_B64 = base64.b64encode(b"\x89PNG fake").decode()


def rect(x0, y0, x1, y1):
    return f"M {x0} {y0} L {x1} {y0} L {x1} {y1} L {x0} {y1} Z"


def glyph(gid, char, paths, w):
    return {
        "glyph_id": gid, "char": char, "slot": 0, "form": "iso",
        "svg_paths": paths, "pen_paths": [[[12, 168], [50, 288]]],
        "svg_width": w, "svg_height": 400, "baseline_y": 288,
        "upscale_factor": 1.0, "thumbnail_png_b64": PNG_B64, "x_shift": 0,
    }


def make_glyphs():
    return [
        glyph("n_0", "n", [rect(12, 168, 188, 288)], 200),
        glyph("i_0", "i", [rect(12, 168, 28, 288)], 40),
        glyph("T_0", "T", [rect(12, 60, 188, 85), rect(90, 85, 110, 288)], 200),
    ]


def create_job(client):
    res = client.post("/draw/create")
    assert res.status_code == 200
    return res.json()["job_id"]


def wait_complete(client, job_id, tries=120):
    for _ in range(tries):
        s = client.get(f"/process/{job_id}/status").json()
        if s["status"] in ("complete", "error"):
            return s
        time.sleep(0.2)
    raise AssertionError("build did not finish")


def finalize(client, job_id, **overrides):
    body = {
        "approved_glyph_ids": ["n_0", "i_0", "T_0"],
        "font_name": "Test Font", "letter_spacing": 60, "space_width": 600,
    }
    body.update(overrides)
    res = client.post(f"/process/{job_id}/finalize", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_unknown_job_404s(client):
    assert client.get("/process/nope/status").status_code == 404
    assert client.get("/process/nope/glyphs").status_code == 404
    assert client.get("/process/nope/bearings").status_code == 404


def test_batch_submit_and_review(client):
    job_id = create_job(client)
    res = client.post(f"/draw/{job_id}/glyphs/batch", json={"glyphs": make_glyphs()})
    assert res.status_code == 200 and res.json()["count"] == 3

    # upsert: resubmitting one glyph must not duplicate it
    res = client.post(f"/draw/{job_id}/glyph", json=make_glyphs()[0])
    assert res.status_code == 200

    data = client.get(f"/process/{job_id}/glyphs").json()
    ids = [g["glyph_id"] for g in data["glyphs"]]
    assert sorted(ids) == ["T_0", "i_0", "n_0"]
    assert all(g["image_url"].startswith("https://stub.supabase.co/storage")
               for g in data["glyphs"])


def test_status_is_field_selected(client):
    job_id = create_job(client)
    s = client.get(f"/process/{job_id}/status").json()
    assert s["status"] == "awaiting_review"
    assert s["progress_pct"] == 0


def test_auto_bearings(client):
    job_id = create_job(client)
    client.post(f"/draw/{job_id}/glyphs/batch", json={"glyphs": make_glyphs()})
    auto = client.get(f"/process/{job_id}/bearings/auto").json()["bearings"]
    assert set(auto) == {"n_0", "i_0", "T_0"}
    biased = client.get(f"/process/{job_id}/bearings/auto?tightness=20").json()["bearings"]
    assert biased["n_0"]["lsb"] == auto["n_0"]["lsb"] + 20


def test_full_build_serve_and_bearing_override(client):
    job_id = create_job(client)
    client.post(f"/draw/{job_id}/glyphs/batch", json={"glyphs": make_glyphs()})
    finalize(client, job_id, glyph_bearings={"i_0": {"lsb": 40, "rsb": 40}})
    s = wait_complete(client, job_id)
    assert s["status"] == "complete", s

    # font served via redirect to the content-versioned public URL
    res = client.get(f"/fonts/{job_id}/Test_Font.otf", follow_redirects=False)
    assert res.status_code == 302
    assert "/storage/v1/object/public/" in res.headers["location"]

    # override round-trips into the bearings editor payload
    b = client.get(f"/process/{job_id}/bearings").json()
    i_g = next(g for g in b["glyphs"] if g["glyph_id"] == "i_0")
    assert i_g["lsb"] == 40 and i_g["is_override"]

    # a plain spacing re-finalize must NOT wipe the stored override
    finalize(client, job_id, letter_spacing=80)
    assert wait_complete(client, job_id)["status"] == "complete"
    b = client.get(f"/process/{job_id}/bearings").json()
    i_g = next(g for g in b["glyphs"] if g["glyph_id"] == "i_0")
    assert i_g["lsb"] == 40

    # settings reflect the latest build
    settings = client.get(f"/process/{job_id}/settings").json()
    assert settings["letter_spacing"] == 80


def test_save_rename_delete(client):
    job_id = create_job(client)
    client.post(f"/draw/{job_id}/glyphs/batch", json={"glyphs": make_glyphs()})
    finalize(client, job_id)
    wait_complete(client, job_id)

    assert client.post(f"/fonts/save/{job_id}").status_code == 200
    fonts = client.get("/fonts/saved").json()["fonts"]
    assert fonts[0]["job_id"] == job_id and fonts[0]["job_exists"]

    res = client.patch(f"/fonts/saved/{job_id}", json={"font_name": "Renamed"})
    assert res.status_code == 200
    assert client.get("/fonts/saved").json()["fonts"][0]["font_name"] == "Renamed"
    # state name updated too, so the next rebuild bakes the new name
    assert client.get(f"/process/{job_id}/settings").json()["font_name"] == "Renamed"

    assert client.delete(f"/fonts/saved/{job_id}").status_code == 200
    assert client.get("/fonts/saved").json()["fonts"] == []


def test_friendly_build_error(client):
    job_id = create_job(client)
    client.post(f"/draw/{job_id}/glyphs/batch", json={"glyphs": make_glyphs()})
    finalize(client, job_id, approved_glyph_ids=["does_not_exist"])
    s = wait_complete(client, job_id)
    assert s["status"] == "error"
    # human sentence, not a traceback
    assert "Traceback" not in (s["error_message"] or "")


def test_reopen_and_resubmit(client):
    job_id = create_job(client)
    client.post(f"/draw/{job_id}/glyphs/batch", json={"glyphs": make_glyphs()})
    finalize(client, job_id)
    wait_complete(client, job_id)

    assert client.post(f"/process/{job_id}/reopen").status_code == 200
    res = client.post(f"/draw/{job_id}/glyph", json=make_glyphs()[1])
    assert res.status_code == 200
    pp = client.get(f"/process/{job_id}/pen-paths").json()
    assert len(pp["glyphs"]) == 3
    assert all(g.get("image_url") for g in pp["glyphs"])


def test_pen_style_switch(client):
    job_id = create_job(client)
    client.post(f"/draw/{job_id}/glyphs/batch", json={"glyphs": make_glyphs()})
    finalize(client, job_id)
    wait_complete(client, job_id)

    # switch to bold → rebuilds losslessly
    r = client.post(f"/process/{job_id}/pen-style", json={"pen_style": "realistic-bold"})
    assert r.status_code == 200 and r.json()["rebuilt"]
    assert wait_complete(client, job_id)["status"] == "complete"
    assert client.get(f"/process/{job_id}/settings").json()["pen_style"] == "realistic-bold"

    # same weight again is a no-op (no rebuild)
    r = client.post(f"/process/{job_id}/pen-style", json={"pen_style": "realistic-bold"})
    assert r.status_code == 200 and r.json()["rebuilt"] is False

    # unknown weights are rejected
    assert client.post(f"/process/{job_id}/pen-style",
                       json={"pen_style": "extra-bold"}).status_code == 422


def test_app_routes_served(client):
    # /ui = the studio (dashboard, borders editor, saved fonts)
    res = client.get("/ui")
    assert res.status_code == 200
    assert "const API = ''" in res.text
    assert "page-borders" in res.text

    # /create = the locked-down guided client flow (intro screen first)
    res = client.get("/create")
    assert res.status_code == 200
    assert "const API = ''" in res.text
    assert "screen-landing" in res.text
    assert "page-borders" not in res.text  # no studio pages for clients

    root = client.get("/", follow_redirects=False)
    assert root.status_code == 302 and root.headers["location"] == "/ui"
