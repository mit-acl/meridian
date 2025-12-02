import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from gen_seg_match.params.params_base import ParamsBase


@dataclass
class RGBDPoseEstimationParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "pose_estimation"

    ##################

    sample_distance: float = 1.0
    min_fov_iou: float = 0.1
    max_fov_depth: float = 8.0
    output_directory: str = None

    viz_img_matches: bool = True
    viz_img_pixel_sep: int = 10
    viz_write_ids: bool = False
