from dataclasses import dataclass
from typing import ClassVar, Tuple

import numpy as np

from gen_seg_match.params.params_base import ParamsBase


@dataclass
class CrossViewMatchingParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "cross_view_matching"

    segment_min_points: int = 10
    matching_mode: str = "vpr"  # "all", "gt", "vpr", or "max_intersection"

    min_area_m_sq: float = 0.0
    point_max_area_m_sq: float = 5.0
    point_max_len_m: float = 5.0
    line_min_length_m: float = 0.0
    # aerial segments with minor axis length below param 0 and
    # major axis length above param 1 are rejected
    aerial_segment_line_rejection_thresh_m: Tuple[float, float] = (0.4, 3.0)
    alpha_shape_alpha: float = 0.2
    alpha_shape_grid_downsample: float = 0.25
    alpha_shape_max_n_pts: int = None
    alpha_shape_ref_size_m: float = None
    circle_point_max_area: float = 50.0
    circle_point_rad_frac_fit_err: float = 0.3

    line_len_to_infinite: float = np.inf

    line_merge_dist_thresh_m: float = 1.0
    line_merge_perp_dist_thresh_m: float = 0.5
    line_merge_ang_thresh_deg: float = 10.0
    line_merge_short_thresh_m: float = None
    line_merge_semantic_sim: float = 0.7
    line_split_length_m: float = 15.0

    aerial_img_patch_side_len_m: float = 30.0
    aerial_img_patch_overlap: float = 0.5
    aerial_min_dist_to_border_m: float = 0.5

    match_min_len_m: float = 5.0

    ground_submap_dist_m: float = 20.0
    ground_submap_rad_m: float = 20.0
    ground_submap_time_s: float = 60.0
    occluded_radius_thresh_m: float = 0.5
    occluded_grid_voxel_size_m: float = 0.25
    line_occlusion_num_samples: int = 10
    line_occlusion_req_non_occluded: float = 0.5
    line_frac_near_points: float = 0.5
    line_pt_dist_check_m: float = 1.0

    aerial_viz_downsample: int = 5
    aerial_viz_line_width_m: float = 0.2
    ground_dist_from_aerial_patch_center_m: float = 75.0

    aerial_viz_target_size_kb: int = 200
    match_viz_target_size_kb: int = 200
    dense_points_max_n: int = 5000

    translation_only: bool = False
    rot_bias_deg: float = 0.0
    uniform_rot_noise_bounds_deg: Tuple[float, float] = (0.0, 0.0)

    points_only: bool = False
    lines_only: bool = False

    match_viz_dist_thresh_m: float = 5.0
    match_viz_angle_thresh_deg: float = 10.0

    sparse_conversion_max_threads: int = 8

    @property
    def line_merge_ang_thresh_rad(self):
        return np.deg2rad(self.line_merge_ang_thresh_deg)

    @property
    def circle_point_max_rad(self):
        return np.sqrt(self.circle_point_max_area / np.pi)


@dataclass
class CrossViewPlaceRecognitionParams(ParamsBase):
    params_key: ClassVar[str] = "cross_view_place_recognition"

    method: str = "semantic-gem"  # "semantic-gem", "anyloc", or "semantic-point-line"
    ground_descriptor_dist_m: float = 5.0
    k_nearest_neighbors: int = 5


@dataclass
class CrossViewRPGOParams(ParamsBase):
    params_key: ClassVar[str] = "cross_view_rpgo"

    # CLIPPER consistency
    rot_consistency_sigma_deg: float = 10.0
    rot_consistency_eps_deg: float = 20.0
    trans_consistency_sigma_m: float = 5.0
    trans_consistency_eps_m: float = 10.0
    min_num_associations: int = 3

    # Optimization method
    optimization_method: str = "pgo"  # "frame_align" or "pgo"

    # PGO noise parameters
    odom_trans_sigma_m: float = 0.1
    odom_rot_sigma_deg: float = 0.5
    prior_trans_sigma_m: float = 5.0
    prior_rot_sigma_deg: float = 10.0

    rerun_match_with_known_rot: bool = True

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
