"""Tests for the pen-plotter draw-order optimizer."""

import xml.etree.ElementTree as ET

import pytest

from processing.plotter_opt import optimize_svg


def _svg(paths):
    body = "".join(f'<path d="{d}" stroke="{c}" fill="none"/>' for d, c in paths)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" '
        f'viewBox="0 0 100 100">{body}</svg>'
    )


def test_optimize_reduces_or_keeps_travel():
    # Deliberately bad order: near/far strokes interleaved so reordering helps.
    svg = _svg([
        ("M 0 0 L 10 0", "#000"),
        ("M 90 90 L 100 90", "#000"),
        ("M 12 0 L 22 0", "#000"),
        ("M 80 90 L 88 90", "#000"),
    ])
    r = optimize_svg(svg, allow_reverse=True, group_by_color=True, two_opt=True)
    assert r.path_count == 4
    assert r.optimized_travel <= r.original_travel + 1e-6
    assert r.optimized_travel < r.original_travel  # this layout is improvable
    ET.fromstring(r.svg)  # output is well-formed XML


def test_geometry_count_preserved():
    svg = _svg([("M 0 0 L 5 5", "#000"), ("M 50 50 L 55 55", "#000")])
    r = optimize_svg(svg)
    assert r.svg.count("<path") == 2


def test_group_by_color_counts_colors():
    svg = _svg([
        ("M 0 0 L 5 0", "#ff0000"),
        ("M 50 0 L 55 0", "#0000ff"),
        ("M 6 0 L 9 0", "#ff0000"),
    ])
    r = optimize_svg(svg, group_by_color=True)
    assert r.stats["colors"] == 2
    assert r.path_count == 3


def test_invalid_svg_raises_valueerror():
    with pytest.raises(ValueError):
        optimize_svg("this is not <svg")


def test_empty_svg_returns_zero_paths():
    r = optimize_svg('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>')
    assert r.path_count == 0
    assert r.original_travel == 0.0
