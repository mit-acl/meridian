import numpy as np
from dataclasses import dataclass
from typing import ClassVar
from gen_seg_match.params.params_base import ParamsBase
from gen_seg_match.utils import expandvars_recursive


@dataclass
class RGBDPoseEstimationDataParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "rgbd_pose_estimation_data"

    ##################

    img_data: dict = None
    depth_data: dict = None
    camera_est_pose_data: dict = None
    camera_gt_pose_data: dict = None
    gravity_direction: np.ndarray = (0.0, 0.0, -1.0)
    depth_scale: float = 1e-3  # Multiplier to convert depth image values to meters

    def __post_init__(self):
        if self.img_data is None:
            self.img_data = {}
        else:
            self.img_data = expandvars_recursive(self.img_data)

        if self.depth_data is None:
            self.depth_data = {}
        else:
            self.depth_data = expandvars_recursive(self.depth_data)

        if self.camera_est_pose_data is None:
            self.camera_est_pose_data = {}
        else:
            self.camera_est_pose_data = expandvars_recursive(self.camera_est_pose_data)

        if self.camera_gt_pose_data is None:
            self.camera_gt_pose_data = {}
        else:
            self.camera_gt_pose_data = expandvars_recursive(self.camera_gt_pose_data)

        for pose_data in [self.camera_est_pose_data, self.camera_gt_pose_data]:
            if "T_premultiply" in pose_data:
                pose_data["T_premultiply"] = np.array(
                    pose_data["T_premultiply"]
                ).reshape((4, 4))

            if "T_postmultiply" in pose_data:
                pose_data["T_postmultiply"] = np.array(
                    pose_data["T_postmultiply"]
                ).reshape((4, 4))

        if self.gravity_direction is not None:
            self.gravity_direction = np.array(self.gravity_direction).reshape((3, 1))


@dataclass
class CrossViewLocalizationDataParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "cross_view_localization_data"

    ##################

    aerial_img_path: str
    ground_map_path: str
    gt_pose_data: dict = None

    aerial_img_scale: float = None  # None = auto-detect from GeoTIFF
    T_camera_flu: list = None

    def __post_init__(self):
        if self.T_camera_flu is not None:
            self.T_camera_flu = np.array(self.T_camera_flu).reshape((4, 4))


@dataclass
class GroundToBEVDataParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "ground_to_bev_data"

    ##################

    img_data: dict = None
    depth_data: dict = None
    camera_pose_data: dict = None

    def __post_init__(self):
        if self.img_data is None:
            self.img_data = {}
        else:
            self.img_data = expandvars_recursive(self.img_data)

        if self.depth_data is None:
            self.depth_data = {}
        else:
            self.depth_data = expandvars_recursive(self.depth_data)

        if self.camera_pose_data is None:
            self.camera_pose_data = {}
        else:
            self.camera_pose_data = expandvars_recursive(self.camera_pose_data)

        if "T_premultiply" in self.camera_pose_data:
            self.camera_pose_data["T_premultiply"] = np.array(
                self.camera_pose_data["T_premultiply"]
            ).reshape((4, 4))

        if "T_postmultiply" in self.camera_pose_data:
            self.camera_pose_data["T_postmultiply"] = np.array(
                self.camera_pose_data["T_postmultiply"]
            ).reshape((4, 4))


@dataclass
class LandmarkPoseEstimationDataParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "landmark_pose_estimation_data"

    ##################

    aerial_img_path: str = None
    img_data: dict = None
    depth_data: dict = None
    camera_pose_data: dict = None
    depth_scale: float = 1e-3

    def __post_init__(self):
        if self.img_data is None:
            self.img_data = {}
        else:
            self.img_data = expandvars_recursive(self.img_data)

        if self.depth_data is None:
            self.depth_data = {}
        else:
            self.depth_data = expandvars_recursive(self.depth_data)

        if self.camera_pose_data is None:
            self.camera_pose_data = {}
        else:
            self.camera_pose_data = expandvars_recursive(self.camera_pose_data)

        if "T_premultiply" in self.camera_pose_data:
            self.camera_pose_data["T_premultiply"] = np.array(
                self.camera_pose_data["T_premultiply"]
            ).reshape((4, 4))

        if "T_postmultiply" in self.camera_pose_data:
            self.camera_pose_data["T_postmultiply"] = np.array(
                self.camera_pose_data["T_postmultiply"]
            ).reshape((4, 4))


@dataclass
class SegmentMappingDataParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "segment_mapping_data"

    ##################

    img_data: dict = None
    depth_data: dict = None
    camera_pose_data: dict = None
    depth_scale: float = 1e-3  # Multiplier to convert depth image values to meters
    max_time: float = None  # max seconds of bag data to load at once (None = all)

    def __post_init__(self):
        if self.img_data is None:
            self.img_data = {}
        else:
            self.img_data = expandvars_recursive(self.img_data)

        if self.depth_data is None:
            self.depth_data = {}
        else:
            self.depth_data = expandvars_recursive(self.depth_data)

        if self.camera_pose_data is None:
            self.camera_pose_data = {}
        else:
            self.camera_pose_data = expandvars_recursive(self.camera_pose_data)

        if "T_premultiply" in self.camera_pose_data:
            self.camera_pose_data["T_premultiply"] = np.array(
                self.camera_pose_data["T_premultiply"]
            ).reshape((4, 4))

        if "T_postmultiply" in self.camera_pose_data:
            self.camera_pose_data["T_postmultiply"] = np.array(
                self.camera_pose_data["T_postmultiply"]
            ).reshape((4, 4))
