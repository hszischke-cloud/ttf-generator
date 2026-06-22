"""Assemble traced paths into an SVG document.

`build_svg` emits the centerline (single-line) form: every path is `fill="none"`
with a shared stroke, so each drawn line is one vector. `build_outline_svg` emits
the cut form: all contours are combined into one filled path with the even-odd
fill rule, so interior holes (the middle of an "O", a donut) are preserved.
"""

from __future__ import annotations

from typing import List


def build_svg(
    path_data: List[str],
    width: int,
    height: int,
    stroke_width: float,
    stroke_color: str = "#1f3d1f",
    background: str | None = None,
) -> str:
    """Combine path ``d`` strings into a single stroked (single-line) SVG.

    Every path is rendered with ``fill="none"`` and a shared stroke, so each
    drawn line becomes one centerline vector rather than a filled outline.
    """
    sw = f"{stroke_width:.2f}".rstrip("0").rstrip(".")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    ]
    if background:
        lines.append(f'<rect width="{width}" height="{height}" fill="{background}"/>')

    lines.append(
        f'<g fill="none" stroke="{stroke_color}" stroke-width="{sw}" '
        f'stroke-linecap="round" stroke-linejoin="round">'
    )
    for d in path_data:
        if d:
            lines.append(f'<path d="{d}"/>')
    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines)


def build_outline_svg(
    path_data: List[str],
    width: int,
    height: int,
    fill_color: str = "#1f3d1f",
    background: str | None = None,
) -> str:
    """Combine closed-contour ``d`` strings into one filled (cut) SVG path.

    All contours go into a single ``<path>`` with ``fill-rule="evenodd"`` so a
    contour nested inside another (a hole) is subtracted rather than filled over.
    """
    combined = " ".join(d for d in path_data if d)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    ]
    if background:
        lines.append(f'<rect width="{width}" height="{height}" fill="{background}"/>')
    if combined:
        lines.append(
            f'<path d="{combined}" fill="{fill_color}" '
            f'fill-rule="evenodd" stroke="none"/>'
        )
    lines.append("</svg>")
    return "\n".join(lines)
