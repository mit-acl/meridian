from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import numpy as np

from meridian.params.params_base import ParamsBase


@dataclass
class CrossViewMatchingParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "cross_view_matching"

    segment_min_points: int = 10
    matching_mode: str = "vpr"  # "all", "gt", "vpr", or "max_intersection"

    min_area_m_sq: float = 1.0
    point_max_area_m_sq: float = 5.0
    point_max_len_m: float = 5.0
    line_min_length_m: float = 0.0
    # aerial segments with minor axis length below param 0 and
    # major axis length above param 1 are rejected
    aerial_segment_line_rejection_thresh_m: Tuple[float, float] = (0.4, 3.0)
    alpha_shape_alpha: float = 1.0
    alpha_shape_grid_downsample: float = 0.25
    alpha_shape_max_n_pts: int = None
    alpha_shape_ref_size_m: float = None
    circle_point_max_area: float = 50.0
    circle_point_rad_frac_fit_err: float = 0.3

    line_len_to_infinite: float = 15.0

    line_merge_dist_thresh_m: float = 1.0
    line_merge_perp_dist_thresh_m: float = 0.5
    line_merge_ang_thresh_deg: float = 10.0
    line_merge_short_thresh_m: float = 1.0
    line_merge_semantic_sim: float = 0.8
    line_split_length_m: float = np.inf

    aerial_img_patch_side_len_m: float = 40.0
    aerial_img_segmentation_side_len_m: float = (
        None  # None -> same as aerial_img_patch_side_len_m
    )
    aerial_img_patch_overlap: float = 0.5
    max_intersection_patches_per_ground_sm: int = 4
    aerial_min_dist_to_border_m: float = 0.5

    match_min_len_m: float = 4.0

    ground_submap_dist_m: float = 20.0
    ground_submap_rad_m: float = 40.0
    ground_submap_time_s: float = 200.0
    occluded_radius_thresh_m: float = 0.5
    occluded_grid_voxel_size_m: float = 0.25
    line_occlusion_num_samples: int = 10
    line_occlusion_req_non_occluded: float = 0.3
    line_frac_near_points: float = 0.3
    line_pt_dist_check_m: float = 0.5

    ground_dist_from_aerial_patch_center_m: float = 25.0

    dense_points_max_n: int = 5000

    # Use CLIPPER instead of Langevin for pass 2 (rerun with known rotation)
    clipper_pass2: bool = False

    translation_only: bool = False
    rot_bias_deg: float = 0.0
    uniform_rot_noise_bounds_deg: Tuple[float, float] = (0.0, 0.0)

    points_only: bool = False
    lines_only: bool = False

    match_trans_err_m: float = 5.0
    match_rot_err_deg: float = 10.0

    sparse_conversion_max_threads: int = 16

    @property
    def line_merge_ang_thresh_rad(self):
        return np.deg2rad(self.line_merge_ang_thresh_deg)

    @property
    def circle_point_max_rad(self):
        return np.sqrt(self.circle_point_max_area / np.pi)


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
    trans_consistency_sigma_m: float = 2.0
    trans_consistency_eps_m: float = 2.0
    added_trans_noise_m_per_m: float = 0.02
    added_rot_noise_deg_per_m: float = 0.02
    single_lc_per_ground_sm: bool = True
    single_lc_per_ground_aerial_pair: bool = True
    fuse_lc_score: bool = True
    lc_score_method: str = "frequency-ratio"
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

    # initialize pose only with n submap registrations - this gaurds the
    # loop closure rejection from growing too large
    outlier_rejection_max_num_lcs: Optional[int] = 100_000

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


@dataclass
class CrossViewIncrementalParams(ParamsBase):
    params_key: ClassVar[str] = "cross_view_incremental"

    consistent_loop_closure_thresh: int = 3
    rot_constrained_consistent_lc_thresh: int = 8
