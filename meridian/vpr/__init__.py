"""VPR: image-level global descriptors.

AnyLoc reimplementation: truncated DINOv2 value facet ->
cosine VLAD over AnyLoc's cached centers.
"""

from meridian.vpr.anyloc import AnyLocPipeline, load_centers
from meridian.vpr.vlad import VLAD

__all__ = ["VLAD", "AnyLocPipeline", "load_centers"]
