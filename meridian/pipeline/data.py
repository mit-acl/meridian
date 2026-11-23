import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, Union
from robotdatapy.data import PoseData, ImgData, PointCloudData
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

    # T_utm_image: pose of the aerial image frame in true UTM, used as the pose
    # of every aerial submap. The image frame has its origin at the top-left
    # pixel (centre), x along columns and y along rows (down), in metres
    # (pixel index * aerial_img_scale); hence the diag(1, -1, -1) flip. For a
    # non-UTM GeoTIFF (e.g. EPSG:3857) it also carries the rotation between the
    # image axes and the UTM grid (meridian convergence, ~1.4 deg at MIT), from
    # a similarity fit of image pixels to their exact UTM positions. None ->
    # built from aerial_img_origin (no rotation) in __post_init__.
    T_utm_image: np.ndarray = None

    def __post_init__(self):
        if self.T_utm_image is None:
            self.T_utm_image = self.axis_aligned_image_pose(self.aerial_img_origin)

    @staticmethod
    def axis_aligned_image_pose(origin_xy, R_2d: np.ndarray = None) -> np.ndarray:
        """Image-frame pose with image x/y-down axes mapped by R_2d (default:
        x -> east, y -> south) and the origin at `origin_xy`."""
        R = np.eye(2) if R_2d is None else np.asarray(R_2d, dtype=float)
        T = np.eye(4)
        T[:2, :2] = R @ np.diag([1.0, -1.0])
        T[2, 2] = -1.0
        if origin_xy is not None:
            T[:2, 3] = np.asarray(origin_xy, dtype=float)[:2]
        return T

    def check_aerial_submaps(
        self,
        aerial_submaps: Dict[str, object],
        trans_tol_m: float = 0.05,
        rot_tol_deg: float = 0.01,
    ) -> bool:
        """Warn if loaded aerial submaps were built with a different image pose
        than this data's T_utm_image (e.g. primitives generated before the
        image->UTM rotation fix, or from a different aerial image).

        Returns True when all poses agree within tolerance.
        """
        if not aerial_submaps:
            return True
        pose = next(iter(aerial_submaps.values())).pose
        dt = np.linalg.norm(pose[:2, 3] - self.T_utm_image[:2, 3])
        R_rel = pose[:2, :2].T @ self.T_utm_image[:2, :2]
        drot = np.rad2deg(abs(np.arctan2(R_rel[1, 0], R_rel[0, 0])))
        if dt > trans_tol_m or drot > rot_tol_deg:
            logger.warning(
                "Aerial submap pose differs from the aerial image's T_utm_image "
                "by %.3f m / %.3f deg. The aerial primitives were probably "
                "generated with an older pipeline (before the image->UTM "
                "rotation fix) or from a different aerial image; regenerate them "
                "for exact UTM poses.",
                dt,
                drot,
            )
            return False
        return True

    def aerial_local_to_pixel(self, local_x: float, local_y: float):
        """Convert aerial image-frame coordinates to image pixel (col, row).

        local_x: metres along image columns from the top-left pixel
        local_y: metres along image rows (down) from the top-left pixel

        These are the coordinates of inv(T_utm_image) @ T_utm_*, i.e. the
        aerial submap frame. T_utm_image already holds the exact image->UTM
        geometry, so this is just a scale.

        Returns (col, row) as floats.
        """
        return local_x / self.aerial_img_scale, local_y / self.aerial_img_scale

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
            params = CrossViewLocalizationDataParams.load(params_file)

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
            T_utm_image = cls.axis_aligned_image_pose(aerial_img_origin)
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

                if utm_crs is not None and utm_crs != native_crs:
                    T_utm_image, fit_scale = cls._fit_utm_image_pose(ds, utm_crs)
                    geotiff_pixel_size = fit_scale
                else:
                    T_utm_image = cls.axis_aligned_image_pose([x_utm, y_utm])

            aerial_img = cv.imread(params.aerial_img_path)
            aerial_img_origin = np.array([x_utm, y_utm])
            aerial_img_scale = params.aerial_img_scale
            if aerial_img_scale is None:
                aerial_img_scale = geotiff_pixel_size
                logger.info(
                    f"Auto-detected aerial pixel size: {aerial_img_scale:.6f} m/px"
                )
            elif abs(aerial_img_scale / geotiff_pixel_size - 1.0) > 0.005:
                logger.warning(
                    "aerial_img_scale=%.6f m/px differs from the GeoTIFF's "
                    "%.6f m/px by more than 0.5%%; aerial primitive UTM "
                    "positions will be scaled wrong.",
                    aerial_img_scale,
                    geotiff_pixel_size,
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
            T_utm_image=T_utm_image,
        )

    @staticmethod
    def _fit_utm_image_pose(ds, utm_crs, n_grid: int = 9):
        """Fit image pixels -> true UTM with a 2D similarity (rotation, uniform
        scale, translation).

        For a conformal CRS such as Web Mercator the map is locally exactly a
        similarity; over a ~1 km image the residual is centimetres. Pixel
        (col, row) is taken at its centre, matching aerial_img_origin.

        Returns (T_utm_image, metres_per_pixel).
        """
        cols = np.linspace(0, ds.width - 1, n_grid)
        rows = np.linspace(0, ds.height - 1, n_grid)
        cc, rr = [a.ravel() for a in np.meshgrid(cols, rows)]
        T = ds.transform
        native_x = T.c + (cc + 0.5) * T.a + (rr + 0.5) * T.b
        native_y = T.f + (cc + 0.5) * T.d + (rr + 0.5) * T.e
        xs, ys = warp_transform(ds.crs, utm_crs, native_x, native_y)
        dst = np.column_stack([xs, ys])
        # Image axes in a right-handed (x east-ish, y north-ish) pixel frame.
        src = np.column_stack([cc, -rr])

        # Umeyama similarity: dst ~= s * R @ src + t
        mu_s, mu_d = src.mean(0), dst.mean(0)
        S, D = src - mu_s, dst - mu_d
        U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
        E = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
        R = U @ E @ Vt
        s = np.trace(np.diag(sig) @ E) / (S**2).sum(1).mean()
        t = mu_d - s * R @ mu_s

        resid = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1)
        logger.info(
            "Aerial image -> %s similarity fit: rotation %.4f deg, %.6f m/px, "
            "residual max %.3f m",
            utm_crs,
            np.rad2deg(np.arctan2(R[1, 0], R[0, 0])),
            s,
            resid.max(),
        )
        return CrossViewLocalizationData.axis_aligned_image_pose(t, R), float(s)

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
        import pickle
        import os

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
            t0, tf = (
                ImgData.topic_t0(bag_path, topic),
                ImgData.topic_tf(bag_path, topic),
            )
        else:
            t0, tf = ImgData.bag_t_range(bag_path)

        # Clamp to the user's img_data.time_range so chunked loading keeps the crop
        user_range = params.img_data.get("time_range")
        if user_range is not None:
            user_t0, user_tf = user_range
            if params.img_data.get("time_range_relative", False):
                user_t0, user_tf = t0 + user_t0, t0 + user_tf
            t0, tf = max(t0, user_t0), min(tf, user_tf)
        return (t0, tf)
