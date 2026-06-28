"""
Regression tests for dimensional-OTF overlap removal.

Every realistic-ink stroke is emitted as its own closed contour, so a glyph's
strokes overlap wherever the pen crosses its own path. Overlapping contours fill
solid under the non-zero winding rule but leave white slivers under an even-odd
rule (and trip up print/design tooling). The dimensional build unions each
glyph's strokes into clean, non-overlapping outlines; the single-line plotter
OTF must keep its overlaps. These tests pin both behaviours.
"""

import io

import pytest
from fontTools.ttLib import TTFont

from processing.font_builder import GlyphData, build_otf

pathops = pytest.importorskip("pathops")  # the real merge needs skia-pathops


# Two bars forming a plus. Drawn as separate, same-wound rectangles that overlap
# in the middle — exactly the topology of a stroke crossing (an 'x', a 't').
_H_BAR = "M 20 90 L 120 90 L 120 110 L 20 110 Z"
_V_BAR = "M 60 30 L 80 30 L 80 170 L 60 170 Z"


def _build(remove_overlaps):
    g = GlyphData(
        char="x", slot=0, glyph_name="x",
        svg_paths=[_H_BAR, _V_BAR],
        svg_width=140, svg_height=200, baseline_y_in_svg=170,
        is_lowercase=True,
    )
    otf, _ = build_otf([g], "T", "Regular", perturb=False,
                       remove_overlaps=remove_overlaps)
    return otf


def _glyph_path(otf, fill, glyph_name="x"):
    font = TTFont(io.BytesIO(otf))
    path = pathops.Path()
    path.fillType = fill
    font.getGlyphSet()[glyph_name].draw(path.getPen())
    return path


def _center(path):
    x0, y0, x1, y1 = path.bounds
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def test_without_removal_overlap_gaps_under_even_odd():
    """Sanity-check the fixture: the raw outline really does overlap, so its
    centre drops out under an even-odd fill (the white-sliver bug)."""
    otf = _build(remove_overlaps=False)
    nz = _glyph_path(otf, pathops.FillType.WINDING)
    eo = _glyph_path(otf, pathops.FillType.EVEN_ODD)
    c = _center(nz)
    assert nz.contains(c), "non-zero fill should cover the crossing"
    assert not eo.contains(c), "fixture must overlap (even-odd leaves a gap)"


def test_removal_makes_crossing_solid_under_both_fill_rules():
    """After overlap removal the crossing is solid under BOTH fill rules — the
    hallmark of a non-overlapping outline."""
    otf = _build(remove_overlaps=True)
    nz = _glyph_path(otf, pathops.FillType.WINDING)
    eo = _glyph_path(otf, pathops.FillType.EVEN_ODD)
    c = _center(nz)
    assert nz.contains(c)
    assert eo.contains(c), "overlap should be merged away (no even-odd gap)"


def test_removal_preserves_counters():
    """A genuine counter (the hole the pen drew around, as in 'o'/'e'/'a') must
    survive overlap removal — only stroke overlaps are merged, not real holes."""
    ring = "M 30 30 L 130 30 L 130 130 L 30 130 Z M 60 60 L 60 100 L 100 100 L 100 60 Z"
    g = GlyphData(char="o", slot=0, glyph_name="o", svg_paths=[ring],
                  svg_width=160, svg_height=160, baseline_y_in_svg=140,
                  is_lowercase=True)
    otf, _ = build_otf([g], "T", "Regular", perturb=False, remove_overlaps=True)
    path = _glyph_path(otf, pathops.FillType.WINDING, glyph_name="o")
    cx, cy = _center(path)
    assert not path.contains((cx, cy)), "the counter must stay empty"


def test_advances_unchanged_by_removal():
    """Overlap removal must not shift metrics — advances are computed
    independently of the outline geometry."""
    a0 = TTFont(io.BytesIO(_build(False)))["hmtx"]["x"][0]
    a1 = TTFont(io.BytesIO(_build(True)))["hmtx"]["x"][0]
    assert a0 == a1
