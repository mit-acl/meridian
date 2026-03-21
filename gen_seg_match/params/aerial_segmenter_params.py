import numpy as np
from dataclasses import dataclass
from typing import ClassVar

from gen_seg_match.params.segmenter_params_base import SegmenterParamsBase


@dataclass
class AerialSegmenterParams(SegmenterParamsBase):
    params_key: ClassVar[str] = "aerial_segmenter"

    # Override base defaults for aerial
    model_type: str = "segment_anything"
    weights_path: str = "$ROMAN_WEIGHTS/sam_vit_l_0b3195.pth"
    conf: float = 0.1
    iou: float = 0.1
    imgsz: tuple = (1024, 1024)

    # Aerial-specific
    pixel_len_m: float = 0.01
    min_area: float = 0.01
    max_area: float = np.inf
    downsample_factor: int = 5
