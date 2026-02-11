import numpy as np
from dataclasses import dataclass
from typing import Union
from robotdatapy.data import PoseData, ImgData
import cv2 as cv
import rasterio
from rasterio.transform import xy

from roman.map.map import ROMANMap

from gen_seg_match.map3d.map import SegmentMap
from gen_seg_match.params.data_params import (
    RGBDPoseEstimationDataParams,
    CrossViewLocalizationDataParams,
    GroundToBEVDataParams,
    SegmentMappingDataParams,
)


@dataclass
class RGBDPoseEstimationData:
    img_data: ImgData
    depth_data: ImgData
    camera_est_pose_data: PoseData = None
    camera_gt_pose_data: PoseData = None
    gravity_direction: np.ndarray = (0.0, 0.0, -1.0)
    depth_scale: float = 1e-3  # Multiplier to convert depth image values to meters

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
            depth_scale=params.depth_scale,
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
    aerial_img_origin: np.ndarray
    ground_map: Union[ROMANMap, SegmentMap]
    gt_pose_data: PoseData = None

    aerial_img_scale: float = 0.01

    @classmethod
    def from_params(cls, params: Union[str, CrossViewLocalizationDataParams]):
        if type(params) is str:
            params_file = params
            params = CrossViewLocalizationDataParams.from_yaml(params_file)

        with rasterio.open(params.aerial_img_path) as ds:
            transform = ds.transform
            x_utm, y_utm = xy(transform, 0, 0)  # row, col

        gt_pose_data = (
            PoseData.from_dict(params.gt_pose_data) if params.gt_pose_data else None
        )

        ground_map = cls._load_ground_map(params.ground_map_path)

        return cls(
            aerial_img=cv.imread(params.aerial_img_path),
            aerial_img_origin=np.array([x_utm, y_utm]),
            ground_map=ground_map,
            gt_pose_data=gt_pose_data,
            aerial_img_scale=params.aerial_img_scale,
        )

    @staticmethod
    def _load_ground_map(path: str) -> Union[ROMANMap, SegmentMap]:
        import pickle
        import os

        with open(os.path.expanduser(path), "rb") as f:
            ground_map = pickle.load(f)
        if isinstance(ground_map, SegmentMap):
            return ground_map
        elif isinstance(ground_map, ROMANMap):
            return ground_map
        else:
            raise TypeError(f"Expected SegmentMap or ROMANMap, got {type(ground_map)}")


@dataclass
class GroundToBEVData:
    img_data: ImgData
    depth_data: ImgData
    camera_pose_data: PoseData

    @classmethod
    def from_params(
        cls,
        params: Union[str, GroundToBEVDataParams],
        time_range: tuple = None,
    ):
        """
        Create GroundToBEVData from params.

        Args:
            params: Path to params file or GroundToBEVDataParams object.
            time_range: Optional (t0, tf) tuple of absolute timestamps to load.
                        If provided, only data within this range will be loaded,
                        significantly reducing memory usage for large bags.
        """
        if type(params) is str:
            params_file = params
            params = GroundToBEVDataParams.from_yaml(params_file)

        # Add time_range to data dicts if provided
        img_data_dict = dict(params.img_data) if params.img_data else {}
        depth_data_dict = dict(params.depth_data) if params.depth_data else {}
        camera_pose_data_dict = (
            dict(params.camera_pose_data) if params.camera_pose_data else {}
        )

        if time_range is not None:
            img_data_dict["time_range"] = time_range
            depth_data_dict["time_range"] = time_range

        img_data = ImgData.from_dict(img_data_dict) if img_data_dict else None
        depth_data = ImgData.from_dict(depth_data_dict) if depth_data_dict else None
        camera_pose_data = (
            PoseData.from_dict(camera_pose_data_dict) if camera_pose_data_dict else None
        )

        return cls(
            img_data=img_data,
            depth_data=depth_data,
            camera_pose_data=camera_pose_data,
        )

    @staticmethod
    def get_bag_time_range(params: Union[str, GroundToBEVDataParams]) -> tuple:
        """
        Get the time range of the bag without loading all data.

        Uses the bag's chunk metadata for fast lookup rather than
        iterating through all messages.

        Args:
            params: Path to params file or GroundToBEVDataParams object.

        Returns:
            Tuple of (t0, tf) absolute timestamps.
        """
        if type(params) is str:
            params = GroundToBEVDataParams.from_yaml(params)

        # Use bag_t_range for fast lookup from chunk metadata
        bag_path = params.img_data.get("path")

        if bag_path:
            return ImgData.bag_t_range(bag_path)
        return None


@dataclass
class SegmentMappingData:
    img_data: ImgData
    depth_data: ImgData
    camera_pose_data: PoseData
    depth_scale: float = 1e-3

    @classmethod
    def from_params(
        cls,
        params: Union[str, SegmentMappingDataParams],
        time_range: tuple = None,
    ):
        if type(params) is str:
            params_file = params
            params = SegmentMappingDataParams.from_yaml(params_file)

        img_data_dict = dict(params.img_data) if params.img_data else {}
        depth_data_dict = dict(params.depth_data) if params.depth_data else {}
        camera_pose_data_dict = (
            dict(params.camera_pose_data) if params.camera_pose_data else {}
        )

        if time_range is not None:
            img_data_dict["time_range"] = time_range
            depth_data_dict["time_range"] = time_range

        img_data = ImgData.from_dict(img_data_dict) if img_data_dict else None
        depth_data = ImgData.from_dict(depth_data_dict) if depth_data_dict else None
        camera_pose_data = (
            PoseData.from_dict(camera_pose_data_dict) if camera_pose_data_dict else None
        )

        return cls(
            img_data=img_data,
            depth_data=depth_data,
            camera_pose_data=camera_pose_data,
            depth_scale=params.depth_scale,
        )

    @staticmethod
    def get_bag_time_range(params: Union[str, SegmentMappingDataParams]) -> tuple:
        if type(params) is str:
            params = SegmentMappingDataParams.from_yaml(params)

        bag_path = params.img_data.get("path")
        if bag_path:
            return ImgData.bag_t_range(bag_path)
        return None
