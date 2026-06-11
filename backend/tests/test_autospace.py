"""Unit tests for the optical side-bearing suggestions."""

import math

from processing.autospace import (
    CLAMP_MAX, CLAMP_MIN, compute_auto_bearings, measure_glyph_margins,
)

H, W = 400, 200
PAD = 12
BASE, XH, CAP = 288, 168, 60


def rect(x0, y0, x1, y1):
    return f"M {x0} {y0} L {x1} {y0} L {x1} {y1} L {x0} {y1} Z"


def glyph(gid, char, paths, w):
    return {
        "glyph_id": gid, "char": char, "slot": 0, "has_glyph": True,
        "form": "iso", "svg_paths": paths, "svg_width": w, "svg_height": H,
        "baseline_y": BASE, "upscale_factor": 1.0,
    }


def circle_path(cx, cy, r, n=24):
    pts = [(cx + r * math.cos(2 * math.pi * i / n),
            cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts) + " Z"


def make_manifest():
    cx, cy, r = 100, (XH + BASE) / 2, (BASE - XH) / 2
    return [
        # straight-sided block (n-like): zero margins
        glyph("n_0", "n", [rect(PAD, XH, 30, BASE), rect(80, XH, 98, BASE),
                           rect(PAD, XH, 98, 200)], 110),
        # bare stem (i-like): zero margins, narrow
        glyph("i_0", "i", [rect(PAD, XH, 28, BASE)], 40),
        # T: wide cap bar with a narrow stem → huge margins under the arms
        glyph("T_0", "T", [rect(PAD, CAP, 188, 85), rect(90, 85, 110, BASE)], 200),
        # o: circle → modest margins at the top/bottom of the zone
        glyph("o_0", "o", [circle_path(cx, cy, r)], int(cx + r + PAD)),
        # v: open triangle
        glyph("v_0", "v", ["M 12 168 L 50 288 L 88 168 L 70 168 L 50 250 L 30 168 Z"], 100),
    ]


def test_measures_margins():
    m = measure_glyph_margins(make_manifest()[0])
    assert m is not None
    assert m["avg_l"] == 0 and m["avg_r"] == 0
    assert m["coord_scale"] > 0


def test_stem_gets_more_space_than_overhang():
    auto = compute_auto_bearings(make_manifest())
    assert set(auto) == {"n_0", "i_0", "T_0", "o_0", "v_0"}
    # bare stems get the most border space; overhanging arms get tucked in
    assert auto["i_0"]["lsb"] >= auto["o_0"]["rsb"] >= auto["T_0"]["rsb"]
    assert auto["T_0"]["rsb"] < 0
    assert auto["i_0"]["lsb"] > 0


def test_bias_shifts_everything():
    base = compute_auto_bearings(make_manifest())
    loose = compute_auto_bearings(make_manifest(), bias_upm=25)
    for gid in base:
        if CLAMP_MIN < base[gid]["lsb"] < CLAMP_MAX - 25:
            assert loose[gid]["lsb"] == base[gid]["lsb"] + 25


def test_clamped_to_range():
    auto = compute_auto_bearings(make_manifest(), bias_upm=10000)
    for v in auto.values():
        assert CLAMP_MIN <= v["lsb"] <= CLAMP_MAX
        assert CLAMP_MIN <= v["rsb"] <= CLAMP_MAX


def test_skips_cursive_and_empty():
    manifest = make_manifest()
    manifest[0]["form"] = "medi"           # cursive form: not adjustable
    manifest[1]["has_glyph"] = False       # never drawn
    manifest[2]["svg_paths"] = []          # no outline
    auto = compute_auto_bearings(manifest)
    assert "n_0" not in auto and "i_0" not in auto and "T_0" not in auto
    assert "o_0" in auto
