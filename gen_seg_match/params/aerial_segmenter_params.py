import numpy as np
from dataclasses import dataclass
from typing import ClassVar, List, Tuple

from gen_seg_match.params.params_base import ParamsBase


@dataclass
class AerialSegmenterParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "aerial_segmenter"

    ##################

    model_type: str = "segment_anything"
    weights_path: str = "$ROMAN_WEIGHTS/sam_vit_l_0b3195.pth"
    conf: float = 0.1
    iou: float = 0.1
    imgsz: Tuple[int, int] = (1024, 1024)
    device: str = "cuda"
    pixel_len_m: float = 0.01
    min_area: float = 0.01
    max_area: float = np.inf
    semantics: str = "dino"
    semantics_dim: int = 768
    triangle_ignore_masks: List[
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    ] = None
    downsample_factor: int = 5

    def get_model_type(self):
        return self.model_type.lower()
