"""Endpoint tests for the pen-plotter / Cricut tools."""

import io

from PIL import Image, ImageDraw


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
    assert resp.status_code == 400


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
