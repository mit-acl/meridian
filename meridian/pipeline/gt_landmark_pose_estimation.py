"""
Ground-truth landmark-based pose estimation pipeline.

Semi-manual tool: the user selects aerial<->ground point correspondences
interactively, then this pipeline estimates a UTM-aligned trajectory via
either Arun's rigid frame alignment or a GTSAM pose-graph optimizer.
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2 as cv
import gtsam
import numpy as np
import rasterio
from rasterio.crs import CRS as RastCRS
from rasterio.transform import rowcol
from rasterio.transform import xy as rasterio_xy
from rasterio.warp import transform as warp_transform
from robotdatapy.camera import pixel_depth_2_xyz
from robotdatapy.data import ImgData, PoseData
from robotdatapy.transform import aruns, transform_to_gtsam
from scipy.spatial.transform import Rotation as Rot, Slerp

from meridian.cross_view.rpgo import se2_to_se3
from meridian.pipeline.data import LazyMcapImgData, is_mcap_bag
from meridian.params.data_params import LandmarkPoseEstimationDataParams
from meridian.params.pipeline_params import LandmarkPoseEstimationParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LandmarkMatch:
    utm_x: float
    utm_y: float
    time: float
    point_camera: List[float]  # shape (3,)
    aerial_pixel_full: List[int]  # (col, row) in full-res aerial image
    ground_pixel: List[int]  # (col, row) in ground image

    def to_dict(self):
        return {
            "utm_x": self.utm_x,
            "utm_y": self.utm_y,
            "time": self.time,
            "point_camera": list(self.point_camera),
            "aerial_pixel_full": list(self.aerial_pixel_full),
            "ground_pixel": list(self.ground_pixel),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            utm_x=d["utm_x"],
            utm_y=d["utm_y"],
            time=d["time"],
            point_camera=d["point_camera"],
            aerial_pixel_full=d["aerial_pixel_full"],
            ground_pixel=d["ground_pixel"],
        )


# ---------------------------------------------------------------------------
# CRS helpers: ensure landmark UTM coordinates are in true metric UTM
# ---------------------------------------------------------------------------


def _get_utm_crs(native_crs, bounds):
    """Return the metric UTM CRS appropriate for the GeoTIFF's bounding box.

    If the native CRS is already a UTM projection (EPSG:326xx/327xx), it is
    returned unchanged.  For non-metric CRSes (e.g. EPSG:3857 Web Mercator)
    we find the appropriate UTM zone from the image centre.
    """
    if native_crs is None:
        return None
    epsg = native_crs.to_epsg()
    if epsg is not None and (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760):
        return native_crs  # already metric UTM
    # Derive UTM zone from the image centre
    cx = (bounds.left + bounds.right) / 2
    cy = (bounds.bottom + bounds.top) / 2
    lons, lats = warp_transform(native_crs, RastCRS.from_epsg(4326), [cx], [cy])
    zone = int((lons[0] + 180) / 6) + 1
    utm_epsg = 32600 + zone if lats[0] >= 0 else 32700 + zone
    return RastCRS.from_epsg(utm_epsg)


def _native_xy_to_utm(x, y, native_crs, utm_crs):
    """Reproject a single point from native_crs to utm_crs."""
    if utm_crs is None or utm_crs == native_crs:
        return float(x), float(y)
    xs, ys = warp_transform(native_crs, utm_crs, [x], [y])
    return float(xs[0]), float(ys[0])


def _utm_xy_to_native(utm_x, utm_y, utm_crs, native_crs):
    """Reproject a single UTM point back to native_crs."""
    if utm_crs is None or utm_crs == native_crs:
        return float(utm_x), float(utm_y)
    xs, ys = warp_transform(utm_crs, native_crs, [utm_x], [utm_y])
    return float(xs[0]), float(ys[0])


# ---------------------------------------------------------------------------
# Interactive landmark selector
# ---------------------------------------------------------------------------


class LandmarkSelector:
    """Interactive OpenCV UI for selecting aerial<->ground point correspondences."""

    HELP_TEXT = [
        "a: add match",
        "n: step +1s",
        "b: step -1s",
        "s: skip slot",
        "d: discard clicks",
        "q: quit",
        "+/-: zoom aerial",
    ]

    # Click/landmark markers: small X's so the exact pixel stays visible
    MARKER_SIZE = 7
    MARKER_THICKNESS = 1
    COLOR_SAVED = (0, 200, 0)
    COLOR_CURRENT = (0, 0, 220)

    def __init__(
        self,
        aerial_img: np.ndarray,
        geotiff_transform,
        img_data: ImgData,
        depth_data: ImgData,
        camera_pose_data: PoseData,
        params: LandmarkPoseEstimationParams,
        depth_scale: float = 1e-3,
        native_crs=None,
        utm_crs=None,
    ):
        self.aerial_full = aerial_img  # full-res aerial (BGR)
        self.geotiff_transform = geotiff_transform
        self.native_crs = native_crs
        self.utm_crs = utm_crs
        self.img_data = img_data
        self.depth_data = depth_data
        self.camera_pose_data = camera_pose_data
        self.params = params
        self.depth_scale = depth_scale

        ds = params.aerial_display_downsample
        h, w = aerial_img.shape[:2]
        self.aerial_display = cv.resize(
            aerial_img, (w // ds, h // ds), interpolation=cv.INTER_AREA
        )
        self.display_h, self.display_w = self.aerial_display.shape[:2]
        self.full_h, self.full_w = h, w

        # State
        self.landmarks: List[LandmarkMatch] = []
        self.t_current = img_data.t0
        self.aerial_click = None  # (col, row) in full-res aerial pixels
        self.ground_click = None  # (col, row) in ground image coords
        self.zoom_level = 1  # 1 = full aerial display
        self.status_msg = ""  # shown at top of ground window

        # Aerial window view: full-res pixels shown in the window, re-centred on
        # the latest click only when the zoom level changes.
        self.view_center = (w / 2, h / 2)  # full-res (col, row)
        # (x0, y0, sx, sy): window pixel (x, y) covers full-res pixels starting at
        # (x0 + x * sx, y0 + y * sy)
        self._view = (0, 0, w / self.display_w, h / self.display_h)
        self._aerial_base_key = None
        self._aerial_base = None

    # ------------------------------------------------------------------
    # Mouse callbacks
    # ------------------------------------------------------------------

    def _aerial_mouse_cb(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            x0, y0, sx, sy = self._view
            col = int(np.clip(x0 + (x + 0.5) * sx, 0, self.full_w - 1))
            row = int(np.clip(y0 + (y + 0.5) * sy, 0, self.full_h - 1))
            self.aerial_click = (col, row)
            self.status_msg = f"Aerial click: ({col}, {row}) full-res px"

    def _ground_mouse_cb(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.ground_click = (x, y)
            self.status_msg = f"Ground click: ({x}, {y})"

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_aerial(self) -> np.ndarray:
        z = self.zoom_level

        if z <= 1:
            key = (1,)
            view = (0, 0, self.full_w / self.display_w, self.full_h / self.display_h)
        else:
            # Crop 1/z of the full-res image around view_center, so detail
            # increases with zoom instead of upsampling the downsampled display.
            crop_w = max(int(round(self.full_w / z)), 1)
            crop_h = max(int(round(self.full_h / z)), 1)
            cx, cy = self.view_center
            x0 = int(np.clip(round(cx - crop_w / 2), 0, self.full_w - crop_w))
            y0 = int(np.clip(round(cy - crop_h / 2), 0, self.full_h - crop_h))
            key = (z, x0, y0)
            view = (x0, y0, crop_w / self.display_w, crop_h / self.display_h)

        if key != self._aerial_base_key:
            if z <= 1:
                base = self.aerial_display
            else:
                crop = self.aerial_full[y0 : y0 + crop_h, x0 : x0 + crop_w]
                # Nearest-neighbour once past 1:1 so single pixels stay crisp
                interp = cv.INTER_AREA if crop_w > self.display_w else cv.INTER_NEAREST
                base = cv.resize(
                    crop, (self.display_w, self.display_h), interpolation=interp
                )
            self._aerial_base_key = key
            self._aerial_base = base
        self._view = view
        img = self._aerial_base.copy()

        # Draw saved landmarks
        for lm in self.landmarks:
            pt = self._full_to_window(*lm.aerial_pixel_full)
            if pt is not None:
                self._draw_marker(img, pt, self.COLOR_SAVED)

        # Draw current aerial click
        if self.aerial_click is not None:
            pt = self._full_to_window(*self.aerial_click)
            if pt is not None:
                self._draw_marker(img, pt, self.COLOR_CURRENT)

        return img

    def _full_to_window(self, col_full: int, row_full: int):
        """Map a full-res aerial pixel to aerial window coords.
        Returns None if the point is outside the current view."""
        x0, y0, sx, sy = self._view
        px = int(round((col_full + 0.5 - x0) / sx - 0.5))
        py = int(round((row_full + 0.5 - y0) / sy - 0.5))
        if 0 <= px < self.display_w and 0 <= py < self.display_h:
            return (px, py)
        return None

    def _draw_marker(self, img: np.ndarray, pt, color):
        cv.drawMarker(
            img,
            tuple(int(v) for v in pt),
            color,
            markerType=cv.MARKER_TILTED_CROSS,
            markerSize=self.MARKER_SIZE,
            thickness=self.MARKER_THICKNESS,
        )

    def _render_ground(self, ground_img: np.ndarray) -> np.ndarray:
        img = ground_img.copy()

        # Draw saved landmarks for this time (approximate)
        for lm in self.landmarks:
            if abs(lm.time - self.t_current) < 0.1:
                self._draw_marker(img, lm.ground_pixel, self.COLOR_SAVED)

        # Draw current ground click
        if self.ground_click is not None:
            self._draw_marker(img, self.ground_click, self.COLOR_CURRENT)

        # Help overlay
        y = 20
        for line in self.HELP_TEXT:
            cv.putText(
                img, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
            )
            y += 18

        # Status
        if self.status_msg:
            cv.putText(
                img,
                self.status_msg,
                (10, img.shape[0] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1,
            )

        return img

    # ------------------------------------------------------------------
    # Depth / 3D helpers
    # ------------------------------------------------------------------

    def _camera_params(self):
        """Return camera params from img_data."""
        return self.img_data.camera_params

    def _get_3d_point(self, gx: int, gy: int) -> Optional[np.ndarray]:
        """Return 3D point in camera frame from ground click, or None if invalid."""
        depth_img = self.depth_data.img(self.t_current)
        if depth_img is None:
            self.status_msg = "WARNING: no depth image at this time"
            return None
        h, w = depth_img.shape[:2]
        if gx < 0 or gx >= w or gy < 0 or gy >= h:
            self.status_msg = "WARNING: click out of depth image bounds"
            return None
        depth_val = float(depth_img[gy, gx])
        if not np.isfinite(depth_val) or depth_val == 0:
            self.status_msg = "WARNING: invalid depth at clicked pixel"
            return None
        depth_m = depth_val * self.depth_scale
        K = self.img_data.camera_params.K
        return pixel_depth_2_xyz(gx, gy, depth_m, K)

    # ------------------------------------------------------------------
    # UTM from aerial click
    # ------------------------------------------------------------------

    def _utm_from_aerial_pixel(self, col_full: int, row_full: int):
        """Convert a full-res aerial pixel (col, row) to true metric UTM."""
        x_native, y_native = rasterio_xy(self.geotiff_transform, row_full, col_full)
        return _native_xy_to_utm(x_native, y_native, self.native_crs, self.utm_crs)

    # ------------------------------------------------------------------
    # Try to add a match
    # ------------------------------------------------------------------

    def _try_add_match(self):
        if self.aerial_click is None:
            self.status_msg = "Need aerial click first"
            return
        if self.ground_click is None:
            self.status_msg = "Need ground click first"
            return

        # Get 3D point
        gx, gy = self.ground_click
        p_cam = self._get_3d_point(gx, gy)
        if p_cam is None:
            return

        # UTM
        col_full, row_full = self.aerial_click
        utm_x, utm_y = self._utm_from_aerial_pixel(col_full, row_full)

        lm = LandmarkMatch(
            utm_x=utm_x,
            utm_y=utm_y,
            time=self.t_current,
            point_camera=list(p_cam),
            aerial_pixel_full=[int(col_full), int(row_full)],
            ground_pixel=[int(gx), int(gy)],
        )
        self.landmarks.append(lm)
        self.status_msg = f"Added landmark #{len(self.landmarks)}"
        # Reset clicks
        self.aerial_click = None
        self.ground_click = None

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> List[LandmarkMatch]:
        cv.namedWindow("Aerial", cv.WINDOW_NORMAL)
        cv.namedWindow("Ground", cv.WINDOW_NORMAL)
        cv.setMouseCallback("Aerial", self._aerial_mouse_cb)
        cv.setMouseCallback("Ground", self._ground_mouse_cb)

        t_slot = self.img_data.t0
        self.t_current = t_slot

        while True:
            ground_img = self.img_data.img(self.t_current)
            if ground_img is None:
                ground_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv.putText(
                    ground_img,
                    "No image at this time",
                    (50, 240),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

            cv.imshow("Aerial", self._render_aerial())
            cv.imshow("Ground", self._render_ground(ground_img))

            key = cv.waitKey(50) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("a"):
                self._try_add_match()
            elif key == ord("n"):
                self.t_current = min(
                    self.t_current + self.params.increment_sec, self.img_data.tf
                )
                self.status_msg = f"t = {self.t_current:.1f}"
            elif key == ord("b"):
                self.t_current = max(
                    self.t_current - self.params.increment_sec, self.img_data.t0
                )
                self.status_msg = f"t = {self.t_current:.1f}"
            elif key == ord("s"):
                t_slot += self.params.sec_between_imgs
                self.t_current = min(t_slot, self.img_data.tf)
                self.status_msg = f"t = {self.t_current:.1f} (new slot)"
            elif key == ord("d"):
                self.aerial_click = None
                self.ground_click = None
                self.status_msg = "Clicks discarded"
            elif key == ord("+") or key == ord("="):
                self._set_zoom(min(self.zoom_level + 1, 8))
            elif key == ord("-"):
                self._set_zoom(max(self.zoom_level - 1, 1))

        cv.destroyAllWindows()
        return self.landmarks

    def _set_zoom(self, zoom_level: int):
        """Change zoom, re-centring the view on the latest aerial click."""
        self.zoom_level = zoom_level
        if self.aerial_click is not None:
            self.view_center = self.aerial_click
        self.status_msg = f"Zoom {self.zoom_level}x"

    # ------------------------------------------------------------------
    # Optional confirmation step
    # ------------------------------------------------------------------

    def confirm_landmarks(self) -> List[LandmarkMatch]:
        """Show each landmark for confirmation; return kept ones."""
        ds = self.params.aerial_display_downsample
        kept = []
        cv.namedWindow("Confirmation", cv.WINDOW_NORMAL)

        for i, lm in enumerate(self.landmarks):
            # Aerial crop
            col_full, row_full = lm.aerial_pixel_full
            col_disp = col_full // ds
            row_disp = row_full // ds
            crop_size = 200
            x0 = max(col_disp - crop_size // 2, 0)
            y0 = max(row_disp - crop_size // 2, 0)
            x1 = min(x0 + crop_size, self.aerial_display.shape[1])
            y1 = min(y0 + crop_size, self.aerial_display.shape[0])
            aerial_crop = self.aerial_display[y0:y1, x0:x1].copy()
            aerial_crop = cv.resize(aerial_crop, (crop_size, crop_size))
            dot_x = col_disp - x0
            dot_y = row_disp - y0
            self._draw_marker(
                aerial_crop,
                (dot_x * crop_size / (x1 - x0), dot_y * crop_size / (y1 - y0)),
                self.COLOR_CURRENT,
            )

            # Ground image
            ground_img = self.img_data.img(lm.time)
            if ground_img is None:
                ground_img = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            else:
                ground_img = ground_img.copy()
            self._draw_marker(ground_img, lm.ground_pixel, self.COLOR_CURRENT)

            # Resize ground to match crop height
            gh, gw = ground_img.shape[:2]
            scale = crop_size / gh
            ground_resized = cv.resize(ground_img, (int(gw * scale), crop_size))

            # Side-by-side
            panel = np.hstack([aerial_crop, ground_resized])
            cv.putText(
                panel,
                f"LM #{i + 1}  y=keep  n=discard  q=done",
                (10, crop_size - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            cv.imshow("Confirmation", panel)

            while True:
                key = cv.waitKey(0) & 0xFF
                if key == ord("y"):
                    kept.append(lm)
                    break
                elif key == ord("n"):
                    break
                elif key == ord("q"):
                    # Keep all remaining
                    kept.append(lm)
                    kept.extend(self.landmarks[i + 1 :])
                    cv.destroyWindow("Confirmation")
                    return kept

        cv.destroyWindow("Confirmation")
        return kept


# ---------------------------------------------------------------------------
# Trajectory estimation
# ---------------------------------------------------------------------------


def _landmarks_to_odom_points(
    landmarks: List[LandmarkMatch], camera_pose_data: PoseData
) -> List[np.ndarray]:
    """Project each landmark's camera-frame point into the odometry frame."""
    odom_pts = []
    for lm in landmarks:
        T_odom_cam = camera_pose_data.pose(lm.time)
        p_cam_h = np.append(lm.point_camera, 1.0)
        q_odom = T_odom_cam @ p_cam_h
        odom_pts.append(q_odom[:3])
    return odom_pts


def estimate_frame_align(
    landmarks: List[LandmarkMatch], camera_pose_data: PoseData
) -> np.ndarray:
    """Arun's method: align odom xy to UTM xy. Returns T_utm_odom (4x4)."""
    odom_pts = _landmarks_to_odom_points(landmarks, camera_pose_data)
    odom_xy = np.array([[p[0], p[1]] for p in odom_pts])
    utm_xy = np.array([[lm.utm_x, lm.utm_y] for lm in landmarks])

    T_utm_odom_2d = aruns(utm_xy, odom_xy)  # 3x3 SE(2)
    T_utm_odom = se2_to_se3(T_utm_odom_2d)  # 4x4
    return T_utm_odom


def _pgo_node_times_and_poses(
    camera_pose_data: PoseData,
    landmarks: List[LandmarkMatch],
    downsample_distance_m: float,
) -> tuple:
    """Select PGO nodes: one every downsample_distance_m of odometry travel (plus
    the first and last odometry poses), and one at every landmark observation time
    so landmark factors attach to an exact, non-interpolated pose.

    Returns (node_times, node_poses) sorted by time.
    """
    times = camera_pose_data.times
    poses = camera_pose_data.all_poses()

    if downsample_distance_m is None or downsample_distance_m <= 0:
        keep = np.arange(len(times))
    else:
        keep = [0]
        last_xyz = poses[0, :3, 3]
        for i in range(1, len(times)):
            if np.linalg.norm(poses[i, :3, 3] - last_xyz) >= downsample_distance_m:
                keep.append(i)
                last_xyz = poses[i, :3, 3]
        if keep[-1] != len(times) - 1:
            keep.append(len(times) - 1)
        keep = np.array(keep)

    node_times = times[keep]
    node_poses = poses[keep]

    lm_times = np.unique([lm.time for lm in landmarks])
    lm_times = lm_times[~np.isin(lm_times, node_times)]
    if len(lm_times):
        lm_poses = np.array([camera_pose_data.pose(t) for t in lm_times])
        node_times = np.concatenate([node_times, lm_times])
        node_poses = np.concatenate([node_poses, lm_poses])
        order = np.argsort(node_times)
        node_times, node_poses = node_times[order], node_poses[order]

    return node_times, node_poses


def _densify_trajectory(
    node_times: np.ndarray,
    node_poses_opt: np.ndarray,
    node_poses_odom: np.ndarray,
    dense_times: np.ndarray,
    dense_poses_odom: np.ndarray,
) -> np.ndarray:
    """Dense UTM trajectory from sparse optimized nodes.

    At each node, the correction C = T_utm_opt @ inv(T_odom) is known. It is
    interpolated between nodes (lerp translation, slerp rotation) and applied to
    the dense odometry: T(t) = C(t) @ T_odom(t). Exact at node times; between
    nodes it keeps the odometry's local motion instead of straight-line
    interpolating the optimized poses.
    """
    C = node_poses_opt @ np.linalg.inv(node_poses_odom)

    C_dense = np.tile(np.eye(4), (len(dense_times), 1, 1))
    for k in range(3):
        C_dense[:, k, 3] = np.interp(dense_times, node_times, C[:, k, 3])
    slerp = Slerp(node_times, Rot.from_matrix(C[:, :3, :3]))
    C_dense[:, :3, :3] = slerp(
        np.clip(dense_times, node_times[0], node_times[-1])
    ).as_matrix()

    return C_dense @ dense_poses_odom


def estimate_pgo(
    landmarks: List[LandmarkMatch],
    camera_pose_data: PoseData,
    params: LandmarkPoseEstimationParams,
) -> tuple:
    """GTSAM pose-graph optimizer over a distance-downsampled trajectory.

    Returns (dense result_pose_data at every odometry time, T_utm_odom).
    """
    # Initial alignment via frame_align
    T_utm_odom = estimate_frame_align(landmarks, camera_pose_data)

    node_times, node_poses = _pgo_node_times_and_poses(
        camera_pose_data, landmarks, params.downsample_distance_m
    )
    N = len(node_times)
    logger.info(
        f"PGO: {N} nodes ({len(camera_pose_data.times)} odometry poses, "
        f"downsample_distance_m={params.downsample_distance_m})"
    )

    # Noise models
    rot_sig = np.deg2rad(params.odometry_rot_sig_deg)
    tran_sig = params.odometry_tran_sig_m
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([rot_sig, rot_sig, rot_sig, tran_sig, tran_sig, tran_sig])
    )

    lm_tran_sig = params.landmark_tran_sig_m
    lm_z_sig = params.landmark_z_sig_m
    landmark_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([lm_tran_sig, lm_tran_sig, lm_z_sig])
    )

    # For calibration factor between pose and landmark variable
    lm_calib_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    )

    # Anchor prior: fix z=0 at P_0
    anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([1e4, 1e4, 1e-3, 1e4, 1e4, 1e4])  # loose rot/xy, tight z
    )

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    lm_idx0 = N  # landmark variable indices start after trajectory

    # Trajectory between factors
    for i in range(N):
        T_i = node_poses[i]
        if i == 0:
            # Anchor z at P_0
            T_init = T_utm_odom @ T_i
            graph.add(
                gtsam.PriorFactorPose3(0, transform_to_gtsam(T_init), anchor_noise)
            )
        else:
            T_rel = np.linalg.inv(node_poses[i - 1]) @ T_i
            graph.add(
                gtsam.BetweenFactorPose3(
                    i - 1, i, transform_to_gtsam(T_rel), odom_noise
                )
            )

        # Initial estimate
        initial_estimate.insert(i, transform_to_gtsam(T_utm_odom @ T_i))

    # Landmark factors
    for k, lm in enumerate(landmarks):
        lm_var = lm_idx0 + k

        # Every landmark time is a node (see _pgo_node_times_and_poses)
        pose_idx = int(np.searchsorted(node_times, lm.time))
        assert node_times[pose_idx] == lm.time

        # T_camera_point: translation = point_camera (rotation = identity)
        T_cam_pt = np.eye(4)
        T_cam_pt[:3, 3] = lm.point_camera

        # Between factor: P_k -> L_k
        graph.add(
            gtsam.BetweenFactorPose3(
                pose_idx, lm_var, transform_to_gtsam(T_cam_pt), lm_calib_noise
            )
        )

        # Translation prior from aerial UTM
        graph.add(
            gtsam.PoseTranslationPrior3D(
                lm_var, np.array([lm.utm_x, lm.utm_y, 0.0]), landmark_noise
            )
        )

        # Initial estimate for landmark
        T_lm_init = T_utm_odom @ node_poses[pose_idx]
        T_lm_init[:3, 3] = np.array([lm.utm_x, lm.utm_y, 0.0])
        initial_estimate.insert(lm_var, transform_to_gtsam(T_lm_init))

    lm_params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, lm_params)
    result = optimizer.optimize()

    node_poses_opt = np.array([result.atPose3(i).matrix() for i in range(N)])
    dense_poses = _densify_trajectory(
        node_times,
        node_poses_opt,
        node_poses,
        camera_pose_data.times,
        camera_pose_data.all_poses(),
    )
    result_pd = PoseData.from_times_and_poses(
        camera_pose_data.times, dense_poses, time_tol=np.inf, interp=True
    )
    return result_pd, T_utm_odom


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _draw_trajectory_on_aerial(
    aerial_display: np.ndarray,
    trajectory_pd: PoseData,
    landmarks: List[LandmarkMatch],
    geotiff_transform,
    aerial_display_downsample: int,
    native_crs=None,
    utm_crs=None,
) -> np.ndarray:
    """Draw trajectory polyline and landmark dots on downsampled aerial image."""
    img = aerial_display.copy()
    ds = aerial_display_downsample

    def utm_to_display(utm_x, utm_y):
        x_native, y_native = _utm_xy_to_native(utm_x, utm_y, utm_crs, native_crs)
        row, col = rowcol(geotiff_transform, x_native, y_native)
        return int(col // ds), int(row // ds)

    # Sample trajectory
    pts = []
    for t in trajectory_pd.times[:: max(1, len(trajectory_pd.times) // 500)]:
        T = trajectory_pd.pose(t)
        x, y = T[0, 3], T[1, 3]
        col, row = utm_to_display(x, y)
        h, w = img.shape[:2]
        if 0 <= col < w and 0 <= row < h:
            pts.append((col, row))

    for i in range(1, len(pts)):
        cv.line(img, pts[i - 1], pts[i], (255, 0, 0), 2)

    # Landmark projections
    for lm in landmarks:
        col, row = utm_to_display(lm.utm_x, lm.utm_y)
        h, w = img.shape[:2]
        if 0 <= col < w and 0 <= row < h:
            # Orange line to trajectory position at landmark observation time
            try:
                T_obs = trajectory_pd.pose(lm.time)
                col_t, row_t = utm_to_display(T_obs[0, 3], T_obs[1, 3])
                if 0 <= col_t < w and 0 <= row_t < h:
                    cv.line(img, (col, row), (col_t, row_t), (0, 165, 255), 2)
            except Exception:
                pass
            # Red landmark dot (drawn on top of line)
            cv.circle(img, (col, row), 5, (0, 0, 220), -1)
            # Pixel label
            label = f"({lm.aerial_pixel_full[0]}, {lm.aerial_pixel_full[1]})"
            cv.putText(
                img,
                label,
                (col + 7, row - 7),
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 220),
                1,
                cv.LINE_AA,
            )

    return img


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def gt_landmark_pose_estimation(
    params_path: str,
    output_dir: str,
    run: str = None,
    landmarks_path: str = None,
    append: bool = False,
):
    """
    Landmark-based ground-truth pose estimation pipeline.

    Args:
        params_path: Path to params directory or YAML file.
        output_dir: Directory to save outputs.
        run: Optional run name for YAML param lookup.
        landmarks_path: If provided, skip interactive tool and load existing JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load params
    data_params = LandmarkPoseEstimationDataParams.load(params_path, run=run)
    params = LandmarkPoseEstimationParams.load(params_path, run=run)

    # Load data
    logger.info("Loading data...")
    if data_params.lazy_mcap_loading:
        for name, d in [
            ("img_data", data_params.img_data),
            ("depth_data", data_params.depth_data),
        ]:
            if d and (d.get("type") != "bag" or not is_mcap_bag(d["path"])):
                raise ValueError(
                    f"lazy_mcap_loading requires {name} to be an MCAP bag, got "
                    f"type={d.get('type')!r}, path={d.get('path')!r}"
                )
        img_data_cls = LazyMcapImgData
    else:
        img_data_cls = ImgData
    img_data = (
        img_data_cls.from_dict(data_params.img_data) if data_params.img_data else None
    )
    depth_data = (
        img_data_cls.from_dict(data_params.depth_data)
        if data_params.depth_data
        else None
    )
    camera_pose_data = (
        PoseData.from_dict(data_params.camera_pose_data)
        if data_params.camera_pose_data
        else None
    )

    # Load aerial GeoTIFF
    with rasterio.open(os.path.expandvars(data_params.aerial_img_path)) as ds:
        geotiff_transform = ds.transform
        native_crs = ds.crs
        utm_crs = _get_utm_crs(native_crs, ds.bounds)
        if utm_crs != native_crs:
            logger.info(
                "GeoTIFF CRS is %s (non-metric); reprojecting landmarks to %s",
                native_crs,
                utm_crs,
            )
        # Read as BGR
        if ds.count >= 3:
            rgb = ds.read([1, 2, 3]).transpose(1, 2, 0)
            aerial_img = cv.cvtColor(rgb.astype(np.uint8), cv.COLOR_RGB2BGR)
        else:
            gray = ds.read(1)
            aerial_img = cv.cvtColor(gray.astype(np.uint8), cv.COLOR_GRAY2BGR)

    # --------------- Step 1: interactive landmark selection ---------------
    if landmarks_path is not None and not append:
        logger.info(f"Loading landmarks from {landmarks_path}")
        with open(landmarks_path) as f:
            raw = json.load(f)
        landmarks = [LandmarkMatch.from_dict(d) for d in raw]
    else:
        logger.info("Starting interactive landmark selector...")
        selector = LandmarkSelector(
            aerial_img=aerial_img,
            geotiff_transform=geotiff_transform,
            img_data=img_data,
            depth_data=depth_data,
            camera_pose_data=camera_pose_data,
            params=params,
            depth_scale=data_params.depth_scale,
            native_crs=native_crs,
            utm_crs=utm_crs,
        )
        if landmarks_path is not None and append:
            logger.info(f"Appending to existing landmarks from {landmarks_path}")
            with open(landmarks_path) as f:
                raw = json.load(f)
            selector.landmarks = [LandmarkMatch.from_dict(d) for d in raw]
        landmarks = selector.run()

        if params.show_confirmation and landmarks:
            logger.info("Showing confirmation dialog...")
            landmarks = selector.confirm_landmarks()

        # Save landmarks
        lm_path = os.path.join(output_dir, "landmarks.json")
        with open(lm_path, "w") as f:
            json.dump([lm.to_dict() for lm in landmarks], f, indent=2)
        logger.info(f"Saved {len(landmarks)} landmarks to {lm_path}")

    if not landmarks:
        logger.warning("No landmarks provided. Exiting.")
        return

    # --------------- Step 2: trajectory estimation -----------------------
    logger.info(f"Estimating trajectory using method='{params.method}'...")

    T_utm_odom = estimate_frame_align(landmarks, camera_pose_data)

    if params.method == "frame_align":
        trajectory_pd = PoseData.from_times_and_poses(
            camera_pose_data.times,
            T_utm_odom @ camera_pose_data.all_poses(),
            time_tol=np.inf,
            interp=True,
        )
    elif params.method == "pgo":
        trajectory_pd, T_utm_odom = estimate_pgo(landmarks, camera_pose_data, params)
    else:
        raise ValueError(
            f"Unknown method: {params.method!r}. Use 'frame_align' or 'pgo'."
        )

    # --------------- Step 3: save outputs --------------------------------
    traj_path = os.path.join(output_dir, "trajectory.csv")
    trajectory_pd.to_csv(traj_path)
    logger.info(f"Saved trajectory to {traj_path}")

    fa_path = os.path.join(output_dir, "frame_align.npy")
    np.save(fa_path, T_utm_odom)
    logger.info(f"Saved frame_align transform to {fa_path}")

    # Also save landmarks JSON if they were pre-loaded (for completeness)
    lm_out_path = os.path.join(output_dir, "landmarks.json")
    if not os.path.exists(lm_out_path):
        with open(lm_out_path, "w") as f:
            json.dump([lm.to_dict() for lm in landmarks], f, indent=2)

    # Visualization
    ds = params.aerial_display_downsample
    h, w = aerial_img.shape[:2]
    aerial_display = cv.resize(
        aerial_img, (w // ds, h // ds), interpolation=cv.INTER_AREA
    )
    viz = _draw_trajectory_on_aerial(
        aerial_display,
        trajectory_pd,
        landmarks,
        geotiff_transform,
        ds,
        native_crs=native_crs,
        utm_crs=utm_crs,
    )
    viz_path = os.path.join(output_dir, "trajectory_viz.png")
    cv.imwrite(viz_path, viz)
    logger.info(f"Saved trajectory visualization to {viz_path}")

    logger.info("Done.")
    return trajectory_pd


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Landmark-based ground-truth pose estimation"
    )
    parser.add_argument(
        "-p", "--params", required=True, help="Path to params dir or YAML file"
    )
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--run", default=None, help="Run name for YAML param lookup")
    parser.add_argument(
        "--landmarks",
        default=None,
        help="Path to existing landmarks.json (skips interactive tool)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append to existing landmarks file instead of skipping interactive tool",
    )
    args = parser.parse_args()

    gt_landmark_pose_estimation(
        params_path=args.params,
        output_dir=args.output,
        run=args.run,
        landmarks_path=args.landmarks,
        append=args.append,
    )
