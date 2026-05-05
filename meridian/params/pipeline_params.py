from dataclasses import dataclass, field
from typing import ClassVar, Union, List
from meridian.params.params_base import ParamsBase


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
class LandmarkPoseEstimationParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "landmark_pose_estimation"

    ##################

    sec_between_imgs: float = 50.0
    increment_sec: float = 1.0
    show_confirmation: bool = True
    aerial_display_downsample: int = 4
    method: str = "frame_align"  # "frame_align" or "pgo"

    # PGO noise params
    odometry_rot_sig_deg: float = 0.5
    odometry_tran_sig_m: float = 0.1
    landmark_tran_sig_m: float = 1.0
    landmark_z_sig_m: float = 1000.0


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


@dataclass
class SemanticMatchEvaluationParams(ParamsBase):
    params_key: ClassVar[str] = "semantic_match_evaluation"

    num_point_matches: int = 50
    num_line_matches: int = 25
    point_match_max_dist_m: float = 0.5
    point_other_min_dist_m: float = 1.0
    line_match_max_dist_m: float = 0.5
    line_match_max_ang_diff_deg: float = 10.0
    line_other_min_dist_m: float = 1.0
    line_other_min_ang_diff_deg: float = 10.0
    submap_max_dist_m: float = 30.0
