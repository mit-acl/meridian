import numpy as np
from dataclasses import dataclass
from typing import Union
from robotdatapy.data import PoseData, ImgData
import cv2 as cv

from roman.map.map import ROMANMap

from gen_seg_match.params.data_params import (
    RGBDPoseEstimationDataParams,
    CrossViewLocalizationDataParams,
)


@dataclass
class RGBDPoseEstimationData:
    img_data: ImgData
    depth_data: ImgData
    camera_est_pose_data: PoseData = None
    camera_gt_pose_data: PoseData = None
    gravity_direction: np.ndarray = (0.0, 0.0, -1.0)

    @classmethod
    def from_params(cls, params: Union[str, RGBDPoseEstimationDataParams]):
        if type(params) is str:
            params_file = params
            params = RGBDPoseEstimationDataParams.from_yaml(params_file)
        img_data = ImgData.from_dict(params.img_data) if params.img_data else None
        depth_data = ImgData.from_dict(params.depth_data) if params.depth_data else None
        camera_est_pose_data = (
            PoseData.from_dict(params.camera_est_pose_data)
            if params.camera_est_pose_data
            else None
        )
        camera_gt_pose_data = (
            PoseData.from_dict(params.camera_gt_pose_data)
            if params.camera_gt_pose_data
            else None
        )

        return cls(
            img_data=img_data,
            depth_data=depth_data,
            camera_est_pose_data=camera_est_pose_data,
            camera_gt_pose_data=camera_gt_pose_data,
            gravity_direction=params.gravity_direction,
        )

    def __post_init__(self):
        if self.camera_est_pose_data is None and self.camera_est_pose_data is None:
            self.gravity_direction = None
            raise Warning(
                "No camera pose data provided; gravity direction will be set to None."
            )
        else:
            self.gravity_direction = np.array(self.gravity_direction).reshape((3, 1))


@dataclass
class CrossViewLocalizationData:
    aerial_img: np.ndarray
    ground_map: ROMANMap

    aerial_img_scale: float = 0.01

    @classmethod
    def from_params(cls, params: Union[str, CrossViewLocalizationDataParams]):
        if type(params) is str:
            params_file = params
            params = CrossViewLocalizationDataParams.from_yaml(params_file)

        return cls(
            aerial_img=cv.imread(params.aerial_img_path),
            ground_map=ROMANMap.from_pickle(params.ground_map_path),
            aerial_img_scale=params.aerial_img_scale,
        )
