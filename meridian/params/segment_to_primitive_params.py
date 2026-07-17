from dataclasses import dataclass
from typing import ClassVar, Tuple

import numpy as np

from meridian.params.params_base import ParamsBase


@dataclass
class SegmentToPrimitiveConversionParams(ParamsBase):
    params_key: ClassVar[str] = "segment_to_primitive_conversion"
    fallback_params_key: ClassVar[str] = "cross_view_matching"

    segment_min_points: int = 10
    min_area_m_sq: float = 1.0
    point_max_area_m_sq: float = 5.0
    point_max_len_m: float = 5.0
    line_min_length_m: float = 0.0
    # aerial segments with minor axis length below param 0 and
    # major axis length above param 1 are rejected
    aerial_segment_line_rejection_thresh_m: Tuple[float, float] = (0.0, np.inf)
    alpha_shape_alpha: float = 1.0
    alpha_shape_grid_downsample: float = 0.25
    alpha_shape_max_n_pts: int = None
    alpha_shape_ref_size_m: float = None
    # EXPERIMENTAL outline method: if set to a float in [0, 1], use
    # shapely.concave_hull(ratio=...) instead of the alpha shape for the segment
    # outline (much faster, different shape — under evaluation). None = alpha shape.
    alpha_shape_concave_hull_ratio: float = 0.5
    # Convert to line if minor axis variance < this param
    line_min_minor_axis_var_m2: float = 0.01
    circle_point_max_area: float = 50.0
    circle_point_rad_frac_fit_err: float = 0.3

    line_len_to_infinite: float = np.inf

    line_merge_dist_thresh_m: float = 1.0
    line_merge_perp_dist_thresh_m: float = 0.5
    line_merge_ang_thresh_deg: float = 10.0
    line_merge_short_thresh_m: float = 1.0
    line_merge_semantic_sim: float = 0.8
    line_split_length_m: float = np.inf

    concat_nearby_descriptors: bool = False
    concat_nearby_descriptors_dist_m: float = 5.0

    sparse_conversion_max_threads: int = 16

    @property
    def line_merge_ang_thresh_rad(self):
        return np.deg2rad(self.line_merge_ang_thresh_deg)

    @property
    def circle_point_max_rad(self):
        return np.sqrt(self.circle_point_max_area / np.pi)


@dataclass
class AerialPatchParams(ParamsBase):
    params_key: ClassVar[str] = "aerial_patch"
    fallback_params_key: ClassVar[str] = "cross_view_matching"

    aerial_img_patch_side_len_m: float = 60.0
    aerial_img_segmentation_side_len_m: float = (
        40.0  # None -> same as aerial_img_patch_side_len_m
    )
    aerial_img_patch_overlap: float = 0.5
    aerial_min_dist_to_border_m: float = 0.5


@dataclass
class GroundSubmapParams(ParamsBase):
    params_key: ClassVar[str] = "ground_submap"
    fallback_params_key: ClassVar[str] = "cross_view_matching"

    ground_submap_dist_m: float = 20.0
    ground_submap_rad_m: float = 40.0
    ground_submap_time_s: float = 200.0
    occluded_radius_thresh_m: float = 0.5
    occluded_grid_voxel_size_m: float = 0.25
    line_occlusion_num_samples: int = 10
    line_occlusion_req_non_occluded: float = 0.1
    line_frac_near_points: float = 0.1
    line_pt_dist_check_m: float = 0.5
    dense_points_max_n: int = 5000

    viz_show_sm_origin: bool = True
