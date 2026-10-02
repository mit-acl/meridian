"""TensorRT versions of the models the segmenters run.

Each class takes the same weights the PyTorch path takes and caches its
exported .onnx and compiled .trt engine next to them, so nothing here needs a
separate download or conversion step.
"""

from meridian.tensorrt.dino import DinoSemanticsTRT, DinoVprTRT
from meridian.tensorrt.engine import (
    TRTEngine,
    build_engine,
    ensure_engine,
    letterbox,
)
from meridian.tensorrt.fastsam import FastSAMTRT

__all__ = [
    "DinoSemanticsTRT",
    "DinoVprTRT",
    "FastSAMTRT",
    "TRTEngine",
    "build_engine",
    "ensure_engine",
    "letterbox",
]
