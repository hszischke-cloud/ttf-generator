"""Image-to-SVG tracing engine for the pen-plotter / Cricut toolkit.

Vendored from the standalone `svgconverter` project and adapted for this app:
the pipeline now produces EITHER a centerline single-line SVG (one vector down
the middle of each stroke — for pens, plotters, scoring) OR a filled outline
SVG (closed shapes around the ink — for cutting machines). Both share the same
threshold/despeckle preprocessing and RDP + Catmull-Rom smoothing, so the two
outputs stay visually consistent.
"""

from .pipeline import TraceParams, TraceResult, trace

__all__ = ["TraceParams", "TraceResult", "trace"]
