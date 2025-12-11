import numpy as np
from dataclasses import dataclass
from typing import ClassVar, List, Tuple

from gen_seg_match.params.params_base import ParamsBase


@dataclass
class GroundSegmenterParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "ground_segmenter"

    ##################
    voxel_size: float = 0.1
    outlier_removal_std: float = 1.0
    dbscan_epsilon: float = 0.25
    dbscan_min_points: int = 10
