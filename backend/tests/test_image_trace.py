"""Tests for the image→SVG tracing engine (centerline + outline modes)."""

from PIL import Image, ImageDraw

from processing.image_trace import TraceParams, trace


def _img(draw_fn, size=(140, 100), bg=255):
    im = Image.new("L", size, bg)
    draw_fn(ImageDraw.Draw(im))
    return im


def test_default_mode_is_line():
    assert TraceParams().mode == "line"


def test_line_mode_traces_centerline_strokes():
    im = _img(lambda d: (d.line((10, 15, 130, 15), fill=0, width=4),
                         d.line((10, 50, 130, 85), fill=0, width=4)))
    r = trace(im, TraceParams(mode="line", stroke_color="#5B4134"))
    assert r.path_count >= 2
    assert 'fill="none"' in r.svg and "<path" in r.svg
    assert "#5B4134" in r.svg
    assert r.stroke_width > 0
    assert r.stats["mode"] == "line"


def test_outline_mode_is_filled_with_evenodd():
    im = _img(lambda d: d.rectangle((25, 25, 110, 75), fill=0))
    r = trace(im, TraceParams(mode="outline", stroke_color="#5B4134"))
    assert r.path_count >= 1
    assert 'fill-rule="evenodd"' in r.svg
    assert 'fill="none"' not in r.svg
    assert r.stats["mode"] == "outline"


def test_outline_preserves_holes():
    # A filled square with a hole punched out -> outer boundary + inner contour.
    def draw(d):
        d.rectangle((25, 20, 115, 80), fill=0)
        d.rectangle((50, 38, 90, 62), fill=255)

    r = trace(_img(draw), TraceParams(mode="outline"))
    assert r.path_count >= 2


def test_blank_image_yields_no_paths():
    im = Image.new("L", (60, 60), 255)
    assert trace(im, TraceParams(mode="line")).path_count == 0
    assert trace(im, TraceParams(mode="outline")).path_count == 0


def test_invert_handles_light_lines_on_dark():
    im = _img(lambda d: d.line((10, 50, 130, 50), fill=255, width=6), bg=0)
    r = trace(im, TraceParams(mode="line", invert=True))
    assert r.path_count >= 1


def test_large_image_is_downscaled():
    im = Image.new("L", (5000, 200), 255)
    ImageDraw.Draw(im).line((0, 100, 5000, 100), fill=0, width=8)
    r = trace(im, TraceParams(mode="line", max_dimension=2000))
    assert max(r.width, r.height) <= 2000
    assert r.stats["scale"] < 1.0
