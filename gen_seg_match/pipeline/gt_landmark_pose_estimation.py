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
import numpy as np
import rasterio
from rasterio.transform import xy as rasterio_xy
from robotdatapy.camera import pixel_depth_2_xyz
from robotdatapy.data import ImgData, PoseData
from robotdatapy.transform import aruns, transform_to_gtsam

from gen_seg_match.cross_view.rpgo import se2_to_se3
from gen_seg_match.params.data_params import LandmarkPoseEstimationDataParams
from gen_seg_match.params.pipeline_params import LandmarkPoseEstimationParams

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
# Interactive landmark selector
# ---------------------------------------------------------------------------


class LandmarkSelector:
    """Interactive OpenCV UI for selecting aerial<->ground point correspondences."""

    HELP_TEXT = [
        "a: add match",
        "n: step +1s",
        "s: skip slot",
        "d: discard clicks",
        "q: quit",
        "+/-: zoom aerial",
    ]

    def __init__(
        self,
        aerial_img: np.ndarray,
        geotiff_transform,
        img_data: ImgData,
        depth_data: ImgData,
        camera_pose_data: PoseData,
        params: LandmarkPoseEstimationParams,
        depth_scale: float = 1e-3,
    ):
        self.aerial_full = aerial_img  # full-res aerial (BGR)
        self.geotiff_transform = geotiff_transform
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

        # State
        self.landmarks: List[LandmarkMatch] = []
        self.t_current = img_data.t0
        self.aerial_click = None  # (col, row) in display coords
        self.ground_click = None  # (col, row) in ground image coords
        self.zoom_level = 1  # 1 = full aerial display
        self.status_msg = ""  # shown at top of ground window

    # ------------------------------------------------------------------
    # Mouse callbacks
    # ------------------------------------------------------------------

    def _aerial_mouse_cb(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.aerial_click = (x, y)
            self.status_msg = f"Aerial click: ({x}, {y})"

    def _ground_mouse_cb(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.ground_click = (x, y)
            self.status_msg = f"Ground click: ({x}, {y})"

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_aerial(self) -> np.ndarray:
        ds = self.params.aerial_display_downsample
        z = self.zoom_level

        if z <= 1:
            img = self.aerial_display.copy()
        else:
            # Crop around aerial_click (or centre if no click)
            if self.aerial_click is not None:
                cx, cy = self.aerial_click
            else:
                cx, cy = self.display_w // 2, self.display_h // 2

            crop_w = max(self.display_w // z, 1)
            crop_h = max(self.display_h // z, 1)
            x0 = int(np.clip(cx - crop_w // 2, 0, self.display_w - crop_w))
            y0 = int(np.clip(cy - crop_h // 2, 0, self.display_h - crop_h))
            crop = self.aerial_display[y0 : y0 + crop_h, x0 : x0 + crop_w]
            img = cv.resize(
                crop, (self.display_w, self.display_h), interpolation=cv.INTER_LINEAR
            )
            # Remap aerial_click to display space for dot drawing
            self._zoom_offset = (x0, y0)
            self._zoom_scale = (self.display_w / crop_w, self.display_h / crop_h)

        # Draw saved landmarks
        for lm in self.landmarks:
            col_full, row_full = lm.aerial_pixel_full
            col_disp = col_full // ds
            row_disp = row_full // ds
            pt = self._to_zoom_display(col_disp, row_disp)
            if pt is not None:
                cv.circle(img, pt, 6, (0, 200, 0), -1)

        # Draw current aerial click
        if self.aerial_click is not None:
            pt = self._to_zoom_display(*self.aerial_click)
            if pt is not None:
                cv.circle(img, pt, 6, (0, 0, 220), -1)

        return img

    def _to_zoom_display(self, col_disp: int, row_disp: int):
        """Map a display-coord point through zoom transformation.
        Returns None if point is outside zoomed crop."""
        if self.zoom_level <= 1:
            return (int(col_disp), int(row_disp))
        x0, y0 = self._zoom_offset
        sx, sy = self._zoom_scale
        px = int((col_disp - x0) * sx)
        py = int((row_disp - y0) * sy)
        if 0 <= px < self.display_w and 0 <= py < self.display_h:
            return (px, py)
        return None

    def _render_ground(self, ground_img: np.ndarray) -> np.ndarray:
        img = ground_img.copy()

        # Draw saved landmarks for this time (approximate)
        for lm in self.landmarks:
            if abs(lm.time - self.t_current) < 0.1:
                cv.circle(img, tuple(lm.ground_pixel), 6, (0, 200, 0), -1)

        # Draw current ground click
        if self.ground_click is not None:
            cv.circle(img, self.ground_click, 6, (0, 0, 220), -1)

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
    # UTM from aerial display click
    # ------------------------------------------------------------------

    def _utm_from_display_click(self, col_disp: int, row_disp: int):
        """Convert a display-space aerial click to UTM and full-res pixel."""
        ds = self.params.aerial_display_downsample
        col_full = col_disp * ds
        row_full = row_disp * ds
        utm_x, utm_y = rasterio_xy(self.geotiff_transform, row_full, col_full)
        return utm_x, utm_y, col_full, row_full

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
        utm_x, utm_y, col_full, row_full = self._utm_from_display_click(
            *self.aerial_click
        )

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

        # Initialize zoom state
        self._zoom_offset = (0, 0)
        self._zoom_scale = (1.0, 1.0)

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
            elif key == ord("s"):
                t_slot += self.params.sec_between_imgs
                self.t_current = min(t_slot, self.img_data.tf)
                self.status_msg = f"t = {self.t_current:.1f} (new slot)"
            elif key == ord("d"):
                self.aerial_click = None
                self.ground_click = None
                self.status_msg = "Clicks discarded"
            elif key == ord("+") or key == ord("="):
                self.zoom_level = min(self.zoom_level + 1, 8)
                self.status_msg = f"Zoom {self.zoom_level}x"
            elif key == ord("-"):
                self.zoom_level = max(self.zoom_level - 1, 1)
                self.status_msg = f"Zoom {self.zoom_level}x"

        cv.destroyAllWindows()
        return self.landmarks

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
            cv.circle(
                aerial_crop,
                (
                    int(dot_x * crop_size / (x1 - x0)),
                    int(dot_y * crop_size / (y1 - y0)),
                ),
                6,
                (0, 0, 220),
                -1,
            )

            # Ground image
            ground_img = self.img_data.img(lm.time)
            if ground_img is None:
                ground_img = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            else:
                ground_img = ground_img.copy()
            cv.circle(ground_img, tuple(lm.ground_pixel), 8, (0, 0, 220), 2)

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


def estimate_pgo(
    landmarks: List[LandmarkMatch],
    camera_pose_data: PoseData,
    params: LandmarkPoseEstimationParams,
) -> tuple:
    """GTSAM pose-graph optimizer. Returns (result_pose_data, T_utm_odom)."""
    import gtsam

    times = camera_pose_data.times

    # Initial alignment via frame_align
    T_utm_odom = estimate_frame_align(landmarks, camera_pose_data)

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

    N = len(times)
    lm_idx0 = N  # landmark variable indices start after trajectory

    # Trajectory between factors
    T_prev = camera_pose_data.pose(times[0])
    for i, t_i in enumerate(times):
        T_i = camera_pose_data.pose(t_i)
        if i == 0:
            # Anchor z at P_0
            T_init = T_utm_odom @ T_i
            graph.add(
                gtsam.PriorFactorPose3(0, transform_to_gtsam(T_init), anchor_noise)
            )
        else:
            T_rel = np.linalg.inv(T_prev) @ T_i
            graph.add(
                gtsam.BetweenFactorPose3(
                    i - 1, i, transform_to_gtsam(T_rel), odom_noise
                )
            )
        T_prev = T_i

        # Initial estimate
        T_utm_i = T_utm_odom @ T_i
        initial_estimate.insert(i, transform_to_gtsam(T_utm_i))

    # Landmark factors
    for k, lm in enumerate(landmarks):
        lm_var = lm_idx0 + k

        # Closest trajectory index
        pose_idx = int(np.argmin(np.abs(times - lm.time)))

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
        T_lm_init = T_utm_odom @ camera_pose_data.pose(lm.time)
        T_lm_init[:3, 3] = np.array([lm.utm_x, lm.utm_y, 0.0])
        initial_estimate.insert(lm_var, transform_to_gtsam(T_lm_init))

    lm_params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, lm_params)
    result = optimizer.optimize()

    poses = [result.atPose3(i).matrix() for i in range(N)]
    result_pd = PoseData.from_times_and_poses(
        times, poses, time_tol=np.inf, interp=True
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
) -> np.ndarray:
    """Draw trajectory polyline and landmark dots on downsampled aerial image."""
    from rasterio.transform import rowcol

    img = aerial_display.copy()
    ds = aerial_display_downsample

    def utm_to_display(utm_x, utm_y):
        row, col = rowcol(geotiff_transform, utm_x, utm_y)
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
            cv.circle(img, (col, row), 5, (0, 0, 220), -1)

    return img


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def gt_landmark_pose_estimation(
    params_path: str,
    output_dir: str,
    run: str = None,
    landmarks_path: str = None,
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
    img_data = ImgData.from_dict(data_params.img_data) if data_params.img_data else None
    depth_data = (
        ImgData.from_dict(data_params.depth_data) if data_params.depth_data else None
    )
    camera_pose_data = (
        PoseData.from_dict(data_params.camera_pose_data)
        if data_params.camera_pose_data
        else None
    )

    # Load aerial GeoTIFF
    with rasterio.open(os.path.expandvars(data_params.aerial_img_path)) as ds:
        geotiff_transform = ds.transform
        # Read as BGR
        if ds.count >= 3:
            rgb = ds.read([1, 2, 3]).transpose(1, 2, 0)
            aerial_img = cv.cvtColor(rgb.astype(np.uint8), cv.COLOR_RGB2BGR)
        else:
            gray = ds.read(1)
            aerial_img = cv.cvtColor(gray.astype(np.uint8), cv.COLOR_GRAY2BGR)

    # --------------- Step 1: interactive landmark selection ---------------
    if landmarks_path is not None:
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
        )
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
        times = camera_pose_data.times
        poses = [T_utm_odom @ camera_pose_data.pose(t) for t in times]
        trajectory_pd = PoseData.from_times_and_poses(
            times, poses, time_tol=np.inf, interp=True
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
        aerial_display, trajectory_pd, landmarks, geotiff_transform, ds
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
    args = parser.parse_args()

    gt_landmark_pose_estimation(
        params_path=args.params,
        output_dir=args.output,
        run=args.run,
        landmarks_path=args.landmarks,
    )
