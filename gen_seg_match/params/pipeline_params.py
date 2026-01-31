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
    viz_observations_3d: bool = False


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

    aerial_img_patch_side_len_m: float = 30.0
    aerial_img_patch_overlap: float = 0.5
    aerial_min_dist_to_border_m: float = 0.5

    match_min_len_m: float = 5.0

    aerial_viz_downsample: int = 5
    ground_dist_from_aerial_patch_center_m: float = 75.0

    @property
    def line_merge_ang_thresh_rad(self):
        return np.deg2rad(self.line_merge_ang_thresh_deg)
