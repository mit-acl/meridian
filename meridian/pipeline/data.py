import logging
import numpy as np
from dataclasses import dataclass
from typing import Union
from robotdatapy.data import PoseData, ImgData, PointCloudData
import cv2 as cv
import rasterio
from rasterio.crs import CRS
from rasterio.transform import xy
from rasterio.warp import transform as warp_transform

from roman.map.map import ROMANMap

from meridian.map3d.map import SegmentMap
from meridian.map3d.align_point_cloud import AlignPointCloud
from meridian.params.data_params import (
    RGBDPoseEstimationDataParams,
    CrossViewLocalizationDataParams,
    GroundToBEVDataParams,
    SegmentMappingDataParams,
)

logger = logging.getLogger(__name__)


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
    T_camera_flu: np.ndarray = None

    # GeoTIFF metadata for accurate UTM→pixel reprojection
    geotiff_transform: object = None  # rasterio Affine transform (native CRS)
    native_crs: object = None  # native CRS of the GeoTIFF
    utm_crs: object = None  # UTM CRS (used for reprojection when != native_crs)

    def aerial_local_to_pixel(self, local_x: float, local_y: float):
        """Convert aerial-local frame coordinates to image pixel (col, row).

        local_x: meters east from aerial_img_origin[0]
        local_y: meters south from aerial_img_origin[1] (image-y direction)

        Returns (col, row) as floats.
        """
        utm_x = self.aerial_img_origin[0] + local_x
        utm_y = self.aerial_img_origin[1] - local_y

        if (
            self.geotiff_transform is not None
            and self.native_crs is not None
            and self.utm_crs is not None
            and self.native_crs != self.utm_crs
        ):
            xs, ys = warp_transform(self.utm_crs, self.native_crs, [utm_x], [utm_y])
            native_x, native_y = float(xs[0]), float(ys[0])
        else:
            native_x, native_y = utm_x, utm_y

        col = (native_x - self.geotiff_transform.c) / self.geotiff_transform.a
        row = (native_y - self.geotiff_transform.f) / self.geotiff_transform.e
        return col, row

    def aerial_pixel_to_utm(self, col: float, row: float):
        """Convert image pixel (col, row) to absolute UTM (x, y)."""
        native_x = self.geotiff_transform.c + col * self.geotiff_transform.a
        native_y = self.geotiff_transform.f + row * self.geotiff_transform.e
        if (
            self.geotiff_transform is not None
            and self.native_crs is not None
            and self.utm_crs is not None
            and self.native_crs != self.utm_crs
        ):
            xs, ys = warp_transform(
                self.native_crs, self.utm_crs, [native_x], [native_y]
            )
            return float(xs[0]), float(ys[0])
        return native_x, native_y

    def aerial_utm_to_pixel(self, utm_xy: np.ndarray) -> np.ndarray:
        """Convert absolute UTM XY coordinates to aerial image pixel (col, row).

        utm_xy: shape (N, 2) array of UTM [x, y] coordinates.
        Returns: shape (N, 2) array of [col, row] pixel coordinates.
        """
        utm_xy = np.atleast_2d(utm_xy)
        if (
            self.geotiff_transform is not None
            and self.native_crs is not None
            and self.utm_crs is not None
            and self.native_crs != self.utm_crs
        ):
            xs, ys = warp_transform(
                self.utm_crs, self.native_crs, utm_xy[:, 0], utm_xy[:, 1]
            )
            native_x = np.array(xs, dtype=float)
            native_y = np.array(ys, dtype=float)
        else:
            native_x = utm_xy[:, 0]
            native_y = utm_xy[:, 1]

        cols = (native_x - self.geotiff_transform.c) / self.geotiff_transform.a
        rows = (native_y - self.geotiff_transform.f) / self.geotiff_transform.e
        return np.column_stack([cols, rows])

    @classmethod
    def from_params(cls, params: Union[str, CrossViewLocalizationDataParams]):
        if type(params) is str:
            params_file = params
            params = CrossViewLocalizationDataParams.from_yaml(params_file)

        if params.top_left_utm is not None:
            # PNG path: user-supplied UTM origin, no GeoTIFF metadata
            if params.aerial_img_scale is None:
                raise ValueError("aerial_img_scale is required when using top_left_utm")
            aerial_img = cv.imread(params.aerial_img_path)
            aerial_img_origin = np.array(params.top_left_utm)
            aerial_img_scale = params.aerial_img_scale
            geotiff_transform = None
            native_crs = None
            utm_crs = None
            logger.info(
                "Using PNG with top_left_utm=(%.1f, %.1f), scale=%.6f m/px",
                aerial_img_origin[0],
                aerial_img_origin[1],
                aerial_img_scale,
            )
        else:
            # GeoTIFF path: extract geo-referencing from the image
            with rasterio.open(params.aerial_img_path) as ds:
                geotiff_transform = ds.transform
                native_crs = ds.crs
                x_native, y_native = xy(geotiff_transform, 0, 0)  # row, col
                geotiff_pixel_size = cls._ground_pixel_size(ds)
                # Reproject the aerial image origin to true metric UTM so that the
                # aerial submap poses and the GT trajectory (also in true UTM) share
                # the same coordinate frame.  Without this, gt-mode matching fails
                # when the GeoTIFF is in a non-metric CRS like EPSG:3857.
                utm_crs = cls._utm_crs_for_ds(ds)
                if utm_crs is not None and utm_crs != native_crs:
                    xs, ys = warp_transform(native_crs, utm_crs, [x_native], [y_native])
                    x_utm, y_utm = float(xs[0]), float(ys[0])
                    logger.info(
                        "Aerial image origin reprojected from %s to %s: (%.1f, %.1f)",
                        native_crs,
                        utm_crs,
                        x_utm,
                        y_utm,
                    )
                else:
                    x_utm, y_utm = x_native, y_native

            aerial_img = cv.imread(params.aerial_img_path)
            aerial_img_origin = np.array([x_utm, y_utm])
            aerial_img_scale = params.aerial_img_scale
            if aerial_img_scale is None:
                aerial_img_scale = geotiff_pixel_size
                logger.info(
                    f"Auto-detected aerial pixel size: {aerial_img_scale:.6f} m/px"
                )

        gt_pose_data = (
            PoseData.from_dict(params.gt_pose_data) if params.gt_pose_data else None
        )

        ground_map = cls._load_ground_map(params.ground_map_path)

        return cls(
            aerial_img=aerial_img,
            aerial_img_origin=aerial_img_origin,
            ground_map=ground_map,
            gt_pose_data=gt_pose_data,
            aerial_img_scale=aerial_img_scale,
            T_camera_flu=params.T_camera_flu,
            geotiff_transform=geotiff_transform,
            native_crs=native_crs,
            utm_crs=utm_crs,
        )

    @staticmethod
    def _utm_crs_for_ds(ds) -> "CRS | None":
        """Return the metric UTM CRS appropriate for this GeoTIFF's location.

        If the native CRS is already a UTM zone (EPSG:326xx/327xx) it is
        returned unchanged.  For non-metric CRSes (e.g. EPSG:3857) we find
        the correct UTM zone from the image centre.
        """
        native_crs = ds.crs
        if native_crs is None:
            return None
        epsg = native_crs.to_epsg()
        if epsg is not None and (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
            return native_crs  # already metric UTM
        cx = (ds.bounds.left + ds.bounds.right) / 2
        cy = (ds.bounds.bottom + ds.bounds.top) / 2
        lons, lats = warp_transform(native_crs, CRS.from_epsg(4326), [cx], [cy])
        zone = int((lons[0] + 180) / 6) + 1
        utm_epsg = 32600 + zone if lats[0] >= 0 else 32700 + zone
        return CRS.from_epsg(utm_epsg)

    @staticmethod
    def _ground_pixel_size(ds) -> float:
        """Return the ground-truth pixel size in metres.

        For UTM or other conformal metric CRSs, ``ds.res`` already gives
        metres-per-pixel.  For Web Mercator (EPSG:3857) the projected
        metre is stretched by 1/cos(latitude), so we correct for that.
        """
        pixel_size = ds.res[0]
        crs = ds.crs
        if crs is not None and crs.to_epsg() == 3857:
            # Convert image centre to WGS-84 latitude
            cx = (ds.bounds.left + ds.bounds.right) / 2
            cy = (ds.bounds.bottom + ds.bounds.top) / 2
            _, lat = warp_transform(crs, CRS.from_epsg(4326), [cx], [cy])
            pixel_size *= np.cos(np.radians(lat[0]))
        return pixel_size

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
    depth_data: ImgData = None
    camera_pose_data: PoseData = None
    depth_scale: float = 1e-3
    point_cloud_data: PointCloudData = None
    align_point_cloud: AlignPointCloud = None

    @property
    def use_point_cloud(self) -> bool:
        return self.point_cloud_data is not None

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
        pcl_dict = dict(params.point_cloud_data) if params.point_cloud_data else {}

        if time_range is not None:
            img_data_dict["time_range"] = time_range
            img_data_dict["time_range_relative"] = False
            if depth_data_dict:
                depth_data_dict["time_range"] = time_range
                depth_data_dict["time_range_relative"] = False
            if pcl_dict:
                pcl_dict["time_range"] = time_range
                pcl_dict["time_range_relative"] = False

        img_data = ImgData.from_dict(img_data_dict) if img_data_dict else None
        depth_data = ImgData.from_dict(depth_data_dict) if depth_data_dict else None
        camera_pose_data = (
            PoseData.from_dict(camera_pose_data_dict) if camera_pose_data_dict else None
        )

        # Point cloud support
        point_cloud_data = None
        align_point_cloud = None
        if pcl_dict:
            T_camera_lidar = pcl_dict.pop("T_camera_lidar", None)
            if time_range is not None:
                pcl_dict["time_range"] = time_range
            point_cloud_data = PointCloudData.from_bag(
                path=pcl_dict["path"],
                topic=pcl_dict["topic"],
                time_tol=pcl_dict.get("time_tol", 0.1),
                time_range=pcl_dict.get("time_range"),
            )
            if T_camera_lidar is None:
                T_camera_lidar = AlignPointCloud.extract_T_camera_lidar(
                    point_cloud_data=point_cloud_data,
                    img_data=img_data,
                    tf_bag_path=pcl_dict["path"],
                )
            align_point_cloud = AlignPointCloud(
                point_cloud_data=point_cloud_data,
                img_data=img_data,
                camera_pose_data=camera_pose_data,
                T_camera_lidar=T_camera_lidar,
            )

        return cls(
            img_data=img_data,
            depth_data=depth_data,
            camera_pose_data=camera_pose_data,
            depth_scale=params.depth_scale,
            point_cloud_data=point_cloud_data,
            align_point_cloud=align_point_cloud,
        )

    @staticmethod
    def get_bag_time_range(params: Union[str, SegmentMappingDataParams]) -> tuple:
        if type(params) is str:
            params = SegmentMappingDataParams.from_yaml(params)

        bag_path = params.img_data.get("path")
        if not bag_path:
            return None

        topic = params.img_data.get("topic")
        ignore_ros_time = params.img_data.get("ignore_ros_time", False)
        if ignore_ros_time and topic:
            # bag_t_range reads ROS recording timestamps from bag metadata, which
            # differ from header timestamps when ignore_ros_time=True. Use the
            # actual header timestamps from the first/last messages instead.
            t0 = ImgData.topic_t0(bag_path, topic)
            tf = ImgData.topic_tf(bag_path, topic)
            return (t0, tf)
        return ImgData.bag_t_range(bag_path)
