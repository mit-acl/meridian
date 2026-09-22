"""VPR: image-level global descriptors, independent of the segmenters.

Both pipelines share a frozen DINOv2 ViT-G/14 layer-31 value facet; VLAD and
the transforms come from the `vpr` package in `third_party/vpr`.
"""
from meridian.vpr.vpr import (
    AnyLocPipeline,
    MeridianVprPipeline,
    load_centers,
    resolve_vpr_checkpoint,
)

__all__ = [
    "AnyLocPipeline",
    "MeridianVprPipeline",
    "load_centers",
    "resolve_vpr_checkpoint",
]
