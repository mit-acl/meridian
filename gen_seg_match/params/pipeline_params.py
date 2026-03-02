import numpy as np
from dataclasses import dataclass, field
from typing import ClassVar, Union, List, Tuple
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
    use_additional_adjacent_imgs: Union[int, List[int]] = (
        0  # 1 = use 1 before and 1 after, etc.
    )
    mini_maps: bool = False
    bits_per_semantic_dim: int = 8

    viz_img_matches: bool = True
    viz_img_pixel_sep: int = 10
    viz_write_ids: bool = False
    viz_observations_3d: bool = False
    viz_registration: bool = False
    viz_show_dense: bool = True
    viz_camera_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, -3.0])

    @property
    def use_multiple_imgs(self) -> bool:
        return (
            self.use_additional_adjacent_imgs != 0
            and self.use_additional_adjacent_imgs != []
        )

    @property
    def additional_adjacent_imgs_list(self) -> list:
        if not self.use_multiple_imgs:
            return [0]
        elif isinstance(self.use_additional_adjacent_imgs, int):
            n = self.use_additional_adjacent_imgs
            return list(range(-n, n + 1))
        else:
            return self.use_additional_adjacent_imgs


@dataclass
class CrossViewMatchingParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "cross_view_matching"

    segment_min_points: int = 10
    matching_mode: str = "vpr"  # "all", "gt", or "vpr"

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

    @property
    def line_merge_ang_thresh_rad(self):
        return np.deg2rad(self.line_merge_ang_thresh_deg)

    @property
    def circle_point_max_rad(self):
        return np.sqrt(self.circle_point_max_area / np.pi)


@dataclass
class CrossViewLocalizationParams(ParamsBase):
    params_key: ClassVar[str] = "cross_view_localization"

    rotation_consistency_sigma: float = 0.174533  # rad
    rotation_consistency_epsilon: float = 0.349066  # rad
    translation_consistency_sigma: float = 5.0  # meters
    translation_consistency_epsilon: float = 10.0  # meters
    min_num_associations: int = 3


@dataclass
class GroundToBEVParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "ground_to_bev"

    sample_distance: float = 0.5  # Distance between sampled images (meters)
    max_depth: float = 8.0  # Maximum depth to include in point cloud (meters)
    voxel_size: float = 0.05  # Voxel size for downsampling (meters)
    depth_scale: float = 1e-3  # Multiplier to convert depth image values to meters
    start_time: float = 0.0  # Start time relative to bag start (seconds)
    end_time: float = (
        None  # End time relative to bag start (seconds), None = end of bag
    )
    bev_resolution: float = 0.02  # Resolution of BEV image (meters per pixel)
    color_aggregation_method: str = "top-1"  # "mean", "top-1", or "top-k"
    color_aggregation_k: int = 5  # k value for "top-k" aggregation method
    hole_fill_method: str = None  # None, "inpaint", "nearest", or "dilate"
    hole_fill_radius: int = 5  # Radius for hole filling (pixels)
