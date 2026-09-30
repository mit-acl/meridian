import logging
import os
import pickle
import struct
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from robotdatapy.camera import CameraParams
from robotdatapy.data import PoseData, ImgData, PointCloudData
from robotdatapy.data.robot_data import RobotData
from robotdatapy.exceptions import MsgNotFound
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
import cv2 as cv
import rasterio
from rasterio.crs import CRS
from rasterio.transform import xy
from rasterio.warp import transform as warp_transform

from meridian.map3d.map import SegmentMap
from meridian.map3d.align_point_cloud import AlignPointCloud
from meridian.params.data_params import (
    CrossViewLocalizationDataParams,
    SegmentMappingDataParams,
)

logger = logging.getLogger(__name__)


@dataclass
class CrossViewLocalizationData:
    aerial_img: np.ndarray
    aerial_img_origin: np.ndarray
    # Optional: required only by pipelines that consume the ground map
    # (offline cross_view_matching ground segmentation, offline localization).
    ground_map: SegmentMap = None
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

        ground_map = (
            cls._load_ground_map(params.ground_map_path)
            if params.ground_map_path is not None
            else None
        )

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
    def _load_ground_map(path: str) -> SegmentMap:
        with open(os.path.expanduser(path), "rb") as f:
            ground_map = pickle.load(f)
        if not isinstance(ground_map, SegmentMap):
            raise TypeError(f"Expected SegmentMap, got {type(ground_map)}")
        return ground_map


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

# ---------------------------------------------------------------------------
# Lazy, random-access image loading from ROS2 MCAP bags
# ---------------------------------------------------------------------------
#
# robotdatapy's ImgData.from_bag keeps every message of a topic in memory, which
# for long uncompressed image topics can mean tens of GB. MCAP files carry a
# time-indexed chunk index, so a narrow time-window query only reads the chunk(s)
# around that time. LazyMcapImgData exploits this: nothing is loaded up front and
# each img(t) call reads (and deserializes) a single frame from disk.
#
# Intended for interactive tools that only look at a few hundred frames, e.g.
# gt_landmark_pose_estimation. It implements the subset of the ImgData interface
# those tools use: t0, tf, img(t), camera_params.

# Keys accepted from an ImgData bag dict that are meaningless for lazy loading
# (every frame is reachable, so there is nothing to subsample).
_IGNORED_KEYS = {"stride", "causal", "t0"}

# Extra bag-time slack (s) around a query window, on top of time_tol, to absorb
# jitter in the recording-time vs. header-stamp offset.
_WINDOW_SLACK = 0.1


def is_mcap_bag(path: str) -> bool:
    """True if path is a .mcap file or a rosbag2 directory containing one."""
    p = Path(os.path.expanduser(os.path.expandvars(path)))
    if p.is_file():
        return p.suffix == ".mcap"
    return p.is_dir() and any(p.glob("*.mcap"))


def _header_stamp_from_cdr(rawdata) -> float:
    """Read std_msgs/Header.stamp from a serialized message without deserializing.

    Valid for any message whose first field is a Header (Image, CompressedImage,
    ...). CDR layout: 4-byte encapsulation header, then int32 sec, uint32 nanosec.
    """
    little_endian = rawdata[1] == 1
    sec, nsec = struct.unpack_from("<iI" if little_endian else ">iI", rawdata, 4)
    return sec + nsec * 1e-9


class LazyMcapImgData:
    """Random-access image data backed by an open MCAP bag reader."""

    def __init__(
        self,
        path,
        topic,
        camera_info_topic=None,
        time_range=None,
        time_range_relative=False,
        time_tol=0.1,
        compressed=True,
        color_space=None,
        ros_distro=None,
        cache_size=16,
        **kwargs,
    ):
        """
        Args mirror ImgData.from_bag so the same params dict can be used.

        Args:
            path (str): rosbag2 directory or .mcap file.
            topic (str): Image (or CompressedImage) topic.
            camera_info_topic (str, optional): CameraInfo topic for intrinsics.
            time_range (list, optional): [start, end] restricting t0/tf.
            time_range_relative (bool, optional): time_range is relative to bag start.
            time_tol (float, optional): Max |header stamp - t| for img(t) to return
                a frame; otherwise returns None. Defaults to 0.1.
            compressed (bool, optional): Unused beyond interface compatibility;
                message_to_cvimage handles both Image and CompressedImage.
            color_space (str, optional): Output color space, e.g. 'bgr8'.
            ros_distro (str, optional): 'foxy', 'humble' or 'jazzy'.
            cache_size (int, optional): Number of decoded frames kept in an LRU
                cache (the UI redraws the same frame many times per second).
        """
        if kwargs.pop("compressed_rvl", False):
            raise NotImplementedError(
                "LazyMcapImgData does not support compressed_rvl"
            )
        ignored = set(kwargs) & _IGNORED_KEYS
        unknown = set(kwargs) - _IGNORED_KEYS
        if unknown:
            raise TypeError(f"Unexpected LazyMcapImgData args: {sorted(unknown)}")
        if ignored:
            logger.info(f"LazyMcapImgData ignoring {sorted(ignored)} for {topic}")

        self.data_path = os.path.expanduser(os.path.expandvars(path))
        if not is_mcap_bag(self.data_path):
            raise ValueError(
                f"LazyMcapImgData requires an MCAP bag, got {self.data_path}"
            )
        self.topic = topic
        self.time_tol = time_tol
        self.color_space = color_space
        self.compressed = compressed
        self._cache_size = cache_size
        self._cache = OrderedDict()

        typestore = RobotData.distro_to_typestore(ros_distro)
        self._reader = AnyReader([Path(self.data_path)], default_typestore=typestore)
        self._reader.open()
        self._connections = [c for c in self._reader.connections if c.topic == topic]
        if not self._connections:
            self._reader.close()
            raise MsgNotFound(topic, self.data_path)

        # Offset between bag recording time and header stamp, used to translate
        # header-time queries into bag-time windows.
        first_bag_ns, first_stamp = self._first_message_after(self._reader.start_time)
        self._bag_minus_header = first_bag_ns * 1e-9 - first_stamp

        t0 = first_stamp
        tf = self._last_stamp()
        if time_range is not None:
            assert (
                time_range[0] < time_range[1]
            ), "time_range must be given in incrementing order"
            if time_range_relative:
                time_range = [self._reader.start_time * 1e-9 + t for t in time_range]
            t0, tf = max(t0, time_range[0]), min(tf, time_range[1])
        self._t0, self._tf = t0, tf

        self.camera_params = CameraParams()
        if camera_info_topic is not None:
            self.camera_params = CameraParams.from_bag(
                self.data_path, camera_info_topic, ros_distro=ros_distro
            )

    @classmethod
    def from_dict(cls, img_data_dict):
        """Create from an ImgData-style dict (the 'type' key is ignored)."""
        return cls(**{k: v for k, v in img_data_dict.items() if k != "type"})

    @property
    def t0(self):
        return self._t0

    @property
    def tf(self):
        return self._tf

    def _first_message_after(self, start_ns):
        for _, bag_ns, raw in self._reader.messages(
            connections=self._connections, start=start_ns
        ):
            return bag_ns, _header_stamp_from_cdr(raw)
        raise MsgNotFound(self.topic, self.data_path)

    def _last_stamp(self):
        # Search backwards in growing windows so a sparse topic still resolves.
        window_s = 2.0
        end_ns = self._reader.end_time
        while True:
            start_ns = max(end_ns - int(window_s * 1e9), self._reader.start_time)
            stamps = [
                _header_stamp_from_cdr(raw)
                for _, _, raw in self._reader.messages(
                    connections=self._connections, start=start_ns, stop=end_ns + 1
                )
            ]
            if stamps:
                return max(stamps)
            if start_ns == self._reader.start_time:
                raise MsgNotFound(self.topic, self.data_path)
            window_s *= 4

    def img(self, t: float):
        """Image at header time t, or None if no frame within time_tol."""
        if t in self._cache:
            self._cache.move_to_end(t)
            return self._cache[t]

        t_bag = t + self._bag_minus_header
        half = self.time_tol + _WINDOW_SLACK
        best = None  # (|dt|, connection, rawdata)
        for conn, _, raw in self._reader.messages(
            connections=self._connections,
            start=int((t_bag - half) * 1e9),
            stop=int((t_bag + half) * 1e9),
        ):
            dt = abs(_header_stamp_from_cdr(raw) - t)
            if dt <= self.time_tol and (best is None or dt < best[0]):
                best = (dt, conn, raw)

        img = None
        if best is not None:
            _, conn, raw = best
            msg = self._reader.deserialize(raw, conn.msgtype)
            img = message_to_cvimage(msg, color_space=self.color_space)

        self._cache[t] = img
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return img

    def close(self):
        self._reader.close()
