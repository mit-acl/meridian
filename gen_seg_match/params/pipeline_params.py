import numpy as np
from dataclasses import dataclass
from typing import ClassVar, Union, List
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

    viz_img_matches: bool = True
    viz_img_pixel_sep: int = 10
    viz_write_ids: bool = False
    viz_observations_3d: bool = False
    viz_registration: bool = False

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
            return list(range(-n, 0)) + list(range(1, n + 1))
        else:
            return self.use_additional_adjacent_imgs


@dataclass
class CrossViewLocalizationParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "cross_view_localization"

    point_max_area_m_sq: float = 5.0
    point_max_len_m: float = 5.0
    line_min_length_m: float = 0.0
    alpha_shape_alpha: float = 0.2
    alpha_shape_grid_downsample: float = 0.25

    line_merge_dist_thresh_m = 1.0
    line_merge_perp_dist_thresh_m = 0.5
    line_merge_ang_thresh_deg = 10.0
    line_split_length_m = 15.0

    aerial_img_patch_side_len_m: float = 30.0
    aerial_img_patch_overlap: float = 0.5
    aerial_min_dist_to_border_m: float = 0.5

    match_min_len_m: float = 5.0

    aerial_viz_downsample: int = 5
    ground_dist_from_aerial_patch_center_m: float = 75.0

    @property
    def line_merge_ang_thresh_rad(self):
        return np.deg2rad(self.line_merge_ang_thresh_deg)


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
