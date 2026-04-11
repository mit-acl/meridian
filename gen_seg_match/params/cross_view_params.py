from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import numpy as np

from gen_seg_match.params.params_base import ParamsBase


@dataclass
class CrossViewMatchingParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "cross_view_matching"

    matching_mode: str = "vpr"  # "all", "gt", "vpr", or "max_intersection"
    max_intersection_patches_per_ground_sm: int = 4

    match_min_len_m: float = 4.0

    ground_dist_from_aerial_patch_center_m: float = 75.0

    # Use CLIPPER instead of Langevin for pass 2 (rerun with known rotation)
    clipper_pass2: bool = False

    translation_only: bool = False
    rot_bias_deg: float = 0.0
    uniform_rot_noise_bounds_deg: Tuple[float, float] = (0.0, 0.0)

    points_only: bool = False
    lines_only: bool = False

    match_trans_err_m: float = 5.0
    match_rot_err_deg: float = 10.0


@dataclass
class CrossViewVisualizationParams(ParamsBase):
    params_key: ClassVar[str] = "cross_view_visualization"

    aerial_viz_pixel_size_m: float = 0.05
    aerial_viz_line_width_m: float = 0.2
    aerial_viz_target_size_kb: int = 200
    match_viz_target_size_kb: int = 200

    # Trajectory plot colors
    estimated_trajectory_color: str = "#fa5ff7"  # light magenta
    gt_trajectory_color: str = "#89fe05"  # lime green


@dataclass
class CrossViewPlaceRecognitionParams(ParamsBase):
    params_key: ClassVar[str] = "cross_view_place_recognition"

    method: str = "anyloc"  # "semantic-gem", "anyloc", or "semantic-point-line"
    ground_descriptor_dist_m: float = 5.0
    k_nearest_neighbors: int = 25


@dataclass
class CrossViewRPGOParams(ParamsBase):
    params_key: ClassVar[str] = "cross_view_rpgo"

    # CLIPPER consistency
    rot_consistency_sigma_deg: float = 5.0
    rot_consistency_eps_deg: float = 5.0
    trans_consistency_sigma_m: float = 1.0
    trans_consistency_eps_m: float = 1.0
    added_trans_noise_m_per_m: float = 0.02
    added_rot_noise_deg_per_m: float = 0.02
    min_num_associations: int = 3
    min_num_associations_rerun: Optional[int] = None

    # Optimization method
    optimization_method: str = "pgo"  # "frame_align" or "pgo"

    # PGO noise parameters
    odom_trans_sigma_m: float = 0.1
    odom_rot_sigma_deg: float = 0.5
    prior_trans_sigma_m: float = 2.0
    prior_rot_sigma_deg: float = 5.0

    rerun_match_with_known_rot: bool = True

    # GT inlier selection (bypasses CLIPPER)
    gt_inliers: bool = False
    gt_inliers_rot_err_deg: float = 5.0
    gt_inliers_trans_err_m: float = 1.0

    @property
    def rot_consistency_sigma_rad(self) -> float:
        return np.deg2rad(self.rot_consistency_sigma_deg)

    @property
    def rot_consistency_eps_rad(self) -> float:
        return np.deg2rad(self.rot_consistency_eps_deg)

    @property
    def odom_rot_sigma_rad(self) -> float:
        return np.deg2rad(self.odom_rot_sigma_deg)

    @property
    def prior_rot_sigma_rad(self) -> float:
        return np.deg2rad(self.prior_rot_sigma_deg)
