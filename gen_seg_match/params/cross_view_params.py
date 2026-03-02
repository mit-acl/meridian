from dataclasses import dataclass
from typing import ClassVar
from gen_seg_match.params.params_base import ParamsBase


@dataclass
class CrossViewPlaceRecognitionParams(ParamsBase):
    params_key: ClassVar[str] = "cross_view_place_recognition"

    method: str = "semantic-gem"  # "semantic-gem" or "semantic-point-line"
    ground_descriptor_dist_m: float = 5.0
    k_nearest_neighbors: int = 5
