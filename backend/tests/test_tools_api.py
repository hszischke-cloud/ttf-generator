"""Endpoint tests for the pen-plotter / Cricut tools."""

import io

from PIL import Image, ImageDraw

from processing import tool_limits


def _png_bytes():
    im = Image.new("L", (140, 100), 255)
    d = ImageDraw.Draw(im)
    d.line((10, 20, 130, 20), fill=0, width=4)
    d.rectangle((30, 45, 110, 85), fill=0)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_image_to_svg_line_mode(client):
    resp = client.post(
        "/tools/image-to-svg",
        files={"file": ("art.png", _png_bytes(), "image/png")},
        data={"mode": "line", "stroke_color": "#5B4134"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["path_count"] >= 1
    assert 'fill="none"' in body["svg"]
    assert body["stats"]["mode"] == "line"


def test_image_to_svg_outline_mode(client):
    resp = client.post(
        "/tools/image-to-svg",
        files={"file": ("art.png", _png_bytes(), "image/png")},
        data={"mode": "outline"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["path_count"] >= 1
    assert 'fill-rule="evenodd"' in body["svg"]
    assert body["stats"]["mode"] == "outline"


def test_image_to_svg_rejects_non_image(client):
    resp = client.post(
        "/tools/image-to-svg",
        files={"file": ("notimg.txt", b"hello world", "text/plain")},
        data={"mode": "line"},
    )
    assert resp.status_code in (400, 415)
    assert "image" in resp.json()["detail"].lower()


def test_limits_endpoint(client):
    body = client.get("/tools/limits").json()
    for key in ("max_image_mb", "max_image_megapixels", "max_svg_mb", "allowed_image_label"):
        assert key in body


def test_image_too_large_in_bytes(client, monkeypatch):
    monkeypatch.setattr(tool_limits, "MAX_IMAGE_BYTES", 10)  # 10 bytes
    resp = client.post(
        "/tools/image-to-svg",
        files={"file": ("art.png", _png_bytes(), "image/png")},
        data={"mode": "line"},
    )
    assert resp.status_code == 413
    assert "limit" in resp.json()["detail"].lower()


def test_image_too_many_pixels(client, monkeypatch):
    monkeypatch.setattr(tool_limits, "MAX_IMAGE_MEGAPIXELS", 0.001)
    resp = client.post(
        "/tools/image-to-svg",
        files={"file": ("art.png", _png_bytes(), "image/png")},
        data={"mode": "line"},
    )
    assert resp.status_code == 422
    assert "megapixel" in resp.json()["detail"].lower()


def test_svg_too_large_in_bytes(client, monkeypatch):
    monkeypatch.setattr(tool_limits, "MAX_SVG_BYTES", 10)  # 10 bytes
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M 0 0 L 9 9"/></svg>'
    resp = client.post(
        "/tools/optimize-svg",
        files={"file": ("a.svg", svg.encode("utf-8"), "image/svg+xml")},
    )
    assert resp.status_code == 413


def test_optimize_rejects_unparseable_svg(client):
    resp = client.post(
        "/tools/optimize-svg",
        files={"file": ("a.svg", b"this is not <svg at all", "image/svg+xml")},
    )
    assert resp.status_code == 400
    assert "svg" in resp.json()["detail"].lower()


def test_optimize_svg_roundtrip(client):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" '
        'viewBox="0 0 100 100">'
        '<path d="M 0 0 L 10 0" stroke="#000" fill="none"/>'
        '<path d="M 90 90 L 100 90" stroke="#000" fill="none"/>'
        '<path d="M 12 0 L 22 0" stroke="#000" fill="none"/>'
        "</svg>"
    )
    resp = client.post(
        "/tools/optimize-svg",
        files={"file": ("art.svg", svg.encode("utf-8"), "image/svg+xml")},
        data={"allow_reverse": "true", "group_by_color": "true", "two_opt": "true"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["path_count"] == 3
    assert body["optimized_travel"] <= body["original_travel"]
