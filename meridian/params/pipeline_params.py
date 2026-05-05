from dataclasses import dataclass
from typing import ClassVar
from meridian.params.params_base import ParamsBase


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
