from dataclasses import dataclass
import yaml
from typing import ClassVar
from meridian.params.params_base import ParamsBase


@dataclass
class DenseToSparseParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "dense_to_sparse"

    # general parameters ———————————————
    copy_dense_points: bool = False  # Whether to copy dense points into segments
    force_points_only: bool = False  # Whether to force only point segments

    # parameters for plane —————————————

    plane_max_e2_e1: float = 0.2  # Maximum e[2]/e[1] threshold to be considered a plane
    plane_max_e2_e0: float = 0.1  # Maximum e[2]/e[0] threshold to be considered a plane
    plane_rms_threshold: float = (
        0.5  # Minimum RMS distance from plane to still be considered a plane
    )
    # do not create a line border if both points' depths are > plane_border_max_depth
    plane_border_max_depth: float = 6.0

    # parameters for line ——————————————

    line_max_e1_e0: float = 0.1  # Maximum e[1]/e[0] threshold to be considered a line
    line_max_e2_e0: float = 0.1  # Maximum e[2]/e[0] threshold to be considered a line
    line_rms_threshold: float = (
        1.0  # Maximum RMS distance from line to still be considered a line
    )
    line_separate_endpoints: bool = True  # Create separate points for endpoints
    line_separate_center_point: bool = False  # Create separate point for center point
    line_inclusion: bool = True  # Include line objects
    line_always_infinite: bool = False  # Always make lines infinite

    # joint parameters (need to see if these work) ——————————————

    non_point_min_extent: float = 1.0
    max_minor_axis_extent: float = 1.5
    max_eigval_ratio: float = 0.3

    # occlusion params
    occlusion_dist: float = 1.0

    # ——————————————————————————————————
