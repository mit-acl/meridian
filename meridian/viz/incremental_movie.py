"""Live/MP4 visualization for cross_view_incremental."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2 as cv
import numpy as np

from meridian.primitive.primitive import LinePrimitive, PointPrimitive
from meridian.primitive.primitive_list import PrimitiveList

logger = logging.getLogger(__name__)


FRAME_W = 1440
FRAME_H = 1080
TOP_BAR_H = 50
DIVIDER = 4
# Outer black border. Reserved on left/right/bottom (top is the black bar).
BORDER = 4
LABEL_PAD = 45
FOOT_PAD = 15
INNER_W = FRAME_W - 2 * BORDER
ROW_H = (FRAME_H - TOP_BAR_H - DIVIDER - BORDER) // 2
IMG_H = ROW_H - LABEL_PAD - FOOT_PAD

TOP_AERIAL_W = 600
TOP_GROUND_W = INNER_W - TOP_AERIAL_W - DIVIDER
BOT_PANE_LEFT_W = TOP_AERIAL_W
BOT_PANE_RIGHT_W = INNER_W - BOT_PANE_LEFT_W

BG_COLOR = (255, 255, 255)
BAR_COLOR = (0, 0, 0)
BAR_TEXT_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
TEXT_OUTLINE = (255, 255, 255)
GT_COLOR = (40, 160, 40)
EST_COLOR = (180, 105, 255)  # pink (BGR)
INLIER_COLOR = (0, 0, 230)  # red (BGR)
LATEST_INLIER_COLOR = (0, 215, 255)  # gold (BGR)
PATCH_BOX_COLOR = (247, 166, 86)  # #56a6f7 (BGR)
MATCH_LINE_COLOR = (0, 255, 0)

FONT = cv.FONT_HERSHEY_SIMPLEX

SHOW_TOTAL_TIME = False


def _put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 1.1,
    thickness: int = 2,
    center: bool = False,
    stroke_extra: int = 3,
):
    """Black text with a white stroke, legible on any background."""
    (tw, _th), _ = cv.getTextSize(text, FONT, scale, thickness)
    x = org[0] - tw // 2 if center else org[0]
    y = org[1]
    cv.putText(
        img, text, (x, y), FONT, scale, TEXT_OUTLINE,
        thickness + stroke_extra, cv.LINE_AA,
    )
    cv.putText(img, text, (x, y), FONT, scale, TEXT_COLOR, thickness, cv.LINE_AA)


def _strip_label(
    frame: np.ndarray,
    text: str,
    x_left: int,
    strip_y_top: int,
    strip_h: int = LABEL_PAD,
    color: Tuple[int, int, int] = TEXT_COLOR,
    scale: float = 0.88,
    thickness: int = 2,
):
    """Left-aligned, vertically-centered label inside an existing strip."""
    (tw, th), baseline = cv.getTextSize(text, FONT, scale, thickness)
    y = strip_y_top + (strip_h + th) // 2 - 2
    cv.putText(frame, text, (x_left, y), FONT, scale, color, thickness, cv.LINE_AA)


def _white_canvas(h: int, w: int) -> np.ndarray:
    c = np.empty((h, w, 3), dtype=np.uint8)
    c[:] = BG_COLOR
    return c

def whiten_edge_black(img: np.ndarray) -> np.ndarray:
    """Flood any pure-black regions connected to the image border to pure white.

    A pixel is black iff every channel is 0. Modifies `img` in place; also returns it.
    """
    if img.size == 0:
        return img
    gray = img if img.ndim == 2 else img.max(axis=2)
    black = (gray == 0).astype(np.uint8)
    if not black.any():
        return img
    n_labels, labels = cv.connectedComponents(black, connectivity=4)
    if n_labels <= 1:
        return img
    border_labels = set()
    for arr in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
        border_labels.update(int(v) for v in np.unique(arr))
    border_labels.discard(0)
    if not border_labels:
        return img
    mask = np.isin(labels, list(border_labels))
    img[mask] = 255
    return img

def _fit_into(
    src: np.ndarray, dst_w: int, dst_h: int
) -> Tuple[np.ndarray, Tuple[int, int, float]]:
    """Letterbox src into a dst_w x dst_h white canvas. Returns (canvas, (x_off, y_off, scale))."""
    h, w = src.shape[:2]
    if h <= 0 or w <= 0:
        return _white_canvas(dst_h, dst_w), (0, 0, 1.0)
    scale = min(dst_w / w, dst_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv.resize(src, (new_w, new_h), interpolation=cv.INTER_AREA)
    if resized.ndim == 2:
        resized = cv.cvtColor(resized, cv.COLOR_GRAY2BGR)
    canvas = _white_canvas(dst_h, dst_w)
    x_off = (dst_w - new_w) // 2
    y_off = (dst_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas, (x_off, y_off, scale)


def _primitive_anchor(seg) -> Optional[np.ndarray]:
    if isinstance(seg, LinePrimitive):
        if seg.num_endpoints == 2:
            p0 = np.asarray(seg.endpoints[0]).flatten()
            p1 = np.asarray(seg.endpoints[1]).flatten()
            return (p0[:2] + p1[:2]) * 0.5
        try:
            return np.asarray(seg.get_point()).flatten()[:2]
        except Exception:
            return None
    if isinstance(seg, PointPrimitive):
        return np.asarray(seg.get_point()).flatten()[:2]
    return None


def _primitive_bbox_points(seg) -> List[np.ndarray]:
    if isinstance(seg, LinePrimitive) and seg.num_endpoints == 2:
        return [
            np.asarray(seg.endpoints[0]).flatten()[:2],
            np.asarray(seg.endpoints[1]).flatten()[:2],
        ]
    a = _primitive_anchor(seg)
    return [a] if a is not None else []


_NICE_LENGTHS_M = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def _draw_scale_bar(
    canvas: np.ndarray,
    m_per_px: float,
    right_x: int,
    bottom_y: int,
    target_px: int = 120,
    color: Tuple[int, int, int] = (40, 40, 40),
    thickness: int = 3,
):
    """Right-anchored scale bar with end ticks. Picks a nice-round length from
    `_NICE_LENGTHS_M` near `target_px * m_per_px`. Bar gets a white stroke."""
    if m_per_px <= 0:
        return
    target_m = target_px * m_per_px
    L_m = min(_NICE_LENGTHS_M, key=lambda v: abs(v - target_m))
    bar_px = int(round(L_m / m_per_px))
    x1 = right_x - bar_px
    if x1 < 4 or right_x >= canvas.shape[1]:
        return
    tick = thickness + 3
    out_t = thickness + 6
    cv.line(canvas, (x1, bottom_y), (right_x, bottom_y), TEXT_OUTLINE, out_t, cv.LINE_AA)
    cv.line(canvas, (x1, bottom_y - tick), (x1, bottom_y + tick), TEXT_OUTLINE, out_t, cv.LINE_AA)
    cv.line(canvas, (right_x, bottom_y - tick), (right_x, bottom_y + tick), TEXT_OUTLINE, out_t, cv.LINE_AA)
    cv.line(canvas, (x1, bottom_y), (right_x, bottom_y), color, thickness, cv.LINE_AA)
    cv.line(canvas, (x1, bottom_y - tick), (x1, bottom_y + tick), color, thickness, cv.LINE_AA)
    cv.line(canvas, (right_x, bottom_y - tick), (right_x, bottom_y + tick), color, thickness, cv.LINE_AA)
    label = f"{L_m} m" if L_m < 1000 else f"{L_m / 1000:.1f} km"
    _put_text(canvas, label, (x1, bottom_y - 12), scale=0.7, stroke_extra=6)


def _draw_primitive_world(
    img: np.ndarray,
    seg,
    world_to_px: Callable[[np.ndarray], Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int = 4,
) -> Optional[Tuple[int, int]]:
    anchor = _primitive_anchor(seg)
    anchor_px = world_to_px(anchor) if anchor is not None else None

    if isinstance(seg, LinePrimitive) and seg.num_endpoints == 2:
        p0 = world_to_px(np.asarray(seg.endpoints[0]).flatten()[:2])
        p1 = world_to_px(np.asarray(seg.endpoints[1]).flatten()[:2])
        cv.line(img, p0, p1, color, thickness, cv.LINE_AA)
    elif isinstance(seg, LinePrimitive):
        pt = np.asarray(seg.get_point()).flatten()[:2]
        d = np.asarray(seg.get_direction()).flatten()[:2]
        far = 1e4
        p0 = world_to_px(pt - d * far)
        p1 = world_to_px(pt + d * far)
        h, w = img.shape[:2]
        ret, p0, p1 = cv.clipLine((0, 0, w, h), p0, p1)
        if ret:
            cv.line(img, p0, p1, color, thickness, cv.LINE_AA)
    elif isinstance(seg, PointPrimitive):
        if anchor_px is not None:
            cv.circle(img, anchor_px, max(6, round((thickness + 2))), color, -1, cv.LINE_AA)
    return anchor_px


@dataclass
class LastMatch:
    ground_key: str
    aerial_key: str
    matched_aerial: PrimitiveList
    matched_ground: PrimitiveList
    ground_dense_segments: list
    # SE(2) ground -> aerial. 2x2 has negative determinant (reflects), which
    # un-mirrors the ground view to match the aerial orientation.
    T_aerial_ground_2d: Optional[np.ndarray] = None


@dataclass
class IncrementalMovieWriter:
    output_path: str
    live: bool
    aerial_img: np.ndarray
    utm_to_pixel: Callable[[np.ndarray], np.ndarray]
    gt_pose_data: Optional[object]
    T_camera_flu: Optional[np.ndarray]
    px_per_m: float
    patch_side_len_m: float
    patch_overlap: float
    fps: int = 10
    total_time_s: Optional[float] = None

    _writer: Optional[cv.VideoWriter] = field(default=None, init=False)
    _aerial_thumb: np.ndarray = field(default=None, init=False)
    _aerial_thumb_offset: Tuple[int, int, float] = field(default=None, init=False)
    _last_match: Optional[LastMatch] = field(default=None, init=False)
    _inlier_positions_utm: List[np.ndarray] = field(default_factory=list, init=False)
    _latest_inlier_position_utm: Optional[np.ndarray] = field(default=None, init=False)
    _gt_traj_px: List[Tuple[float, float]] = field(default_factory=list, init=False)
    _gt_t_cache: float = field(default=-1.0, init=False)
    _live_window: str = field(default="MERIDIAN Incremental Visualization", init=False)
    # Live-viewer GUI runs in its own thread so window resizes / repaints
    # don't stall waiting for the next rendered frame.
    _gui_thread: Optional[threading.Thread] = field(default=None, init=False)
    _gui_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _latest_frame: Optional[np.ndarray] = field(default=None, init=False)
    _gui_stop: bool = field(default=False, init=False)

    def __post_init__(self):
        self._aerial_thumb, self._aerial_thumb_offset = _fit_into(
            self.aerial_img, TOP_AERIAL_W, IMG_H
        )
        whiten_edge_black(self._aerial_thumb)

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        self._writer = cv.VideoWriter(
            self.output_path, fourcc, self.fps, (FRAME_W, FRAME_H)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter at {self.output_path}")
        if self.live:
            # Window is created inside the GUI thread; on Linux HighGUI a
            # window's events must be pumped from the same thread that called
            # namedWindow.
            self._gui_thread = threading.Thread(
                target=self._gui_loop, name="movie-gui", daemon=True
            )
            self._gui_thread.start()

    def _gui_loop(self):
        """Owns the HighGUI window: creates it and pumps events."""
        window_created = False
        while not self._gui_stop:
            with self._gui_lock:
                frame = self._latest_frame
            if frame is not None:
                if not window_created:
                    cv.namedWindow(self._live_window, cv.WINDOW_NORMAL)
                    cv.resizeWindow(self._live_window, FRAME_W, FRAME_H)
                    window_created = True
                cv.imshow(self._live_window, frame)
            cv.waitKey(15)

    def update_match(
        self,
        ground_key: str,
        aerial_key: str,
        matched_aerial: PrimitiveList,
        matched_ground: PrimitiveList,
        ground_dense_segments: list,
        T_aerial_ground_2d: Optional[np.ndarray] = None,
    ):
        if not aerial_key:
            return
        self._last_match = LastMatch(
            ground_key=ground_key,
            aerial_key=aerial_key,
            matched_aerial=matched_aerial,
            matched_ground=matched_ground,
            ground_dense_segments=ground_dense_segments,
            T_aerial_ground_2d=T_aerial_ground_2d,
        )

    def update_inliers(
        self,
        positions_utm: List[np.ndarray],
        latest_position_utm: Optional[np.ndarray] = None,
    ):
        """Refresh inlier UTM xy positions (red stars) and the latest (gold)."""
        self._inlier_positions_utm = list(positions_utm)
        self._latest_inlier_position_utm = (
            None if latest_position_utm is None else np.asarray(latest_position_utm)
        )

    def write_frame(self, t: float, ground_img: np.ndarray, instant_pose_history):
        frame = _white_canvas(FRAME_H, FRAME_W)

        top_y0 = TOP_BAR_H
        top_img_y0 = top_y0 + LABEL_PAD
        top_img_y1 = top_img_y0 + IMG_H
        top_y1 = top_y0 + ROW_H
        mid_div_y1 = top_y1 + DIVIDER
        bot_y0 = mid_div_y1
        bot_img_y0 = bot_y0 + LABEL_PAD
        bot_img_y1 = bot_img_y0 + IMG_H

        frame[0:TOP_BAR_H, :] = BAR_COLOR
        bar_scale = 1.05
        bar_thick = 2
        bar_baseline = TOP_BAR_H - 16
        cv.putText(
            frame, "MERIDIAN", (18, bar_baseline),
            FONT, bar_scale, BAR_TEXT_COLOR, bar_thick, cv.LINE_AA,
        )
        if SHOW_TOTAL_TIME and self.total_time_s is not None and self.total_time_s > 0:
            t_text = f"t = {t:.2f} / {self.total_time_s:.2f} s"
        else:
            t_text = f"t = {t:.2f} s"
        (tw, _), _ = cv.getTextSize(t_text, FONT, bar_scale, bar_thick)
        cv.putText(
            frame, t_text, (FRAME_W - tw - 18, bar_baseline),
            FONT, bar_scale, BAR_TEXT_COLOR, bar_thick, cv.LINE_AA,
        )

        x0 = BORDER
        gr_x0 = x0 + TOP_AERIAL_W + DIVIDER
        frame[top_img_y0:top_img_y1, x0:x0 + TOP_AERIAL_W] = self._render_aerial_traj(
            t, instant_pose_history
        )
        frame[top_y0:top_y1, x0 + TOP_AERIAL_W:x0 + TOP_AERIAL_W + DIVIDER] = BAR_COLOR
        frame[top_img_y0:top_img_y1, gr_x0:gr_x0 + TOP_GROUND_W] = (
            self._render_ground_rgb(ground_img)
        )
        _strip_label(frame, "Aerial View - GT (green) / Est (pink)", x0 + 10, top_y0)
        _strip_label(frame, "Ground View", gr_x0 + 10, top_y0)

        frame[top_y1:mid_div_y1, :] = BAR_COLOR

        bot_aerial, aerial_anchors = self._render_aerial_patch_pane()
        bot_ground, ground_anchors = self._render_ground_dense_pane()
        frame[bot_img_y0:bot_img_y1, x0:x0 + BOT_PANE_LEFT_W] = bot_aerial
        frame[bot_img_y0:bot_img_y1, x0 + BOT_PANE_LEFT_W:x0 + BOT_PANE_LEFT_W + BOT_PANE_RIGHT_W] = bot_ground
        if self._last_match is not None:
            _strip_label(
                frame, f"Aerial patch (ID {self._last_match.aerial_key})",
                x0 + 10, bot_y0,
            )
            _strip_label(
                frame, f"Ground submap (ID {self._last_match.ground_key})",
                x0 + BOT_PANE_LEFT_W + 10, bot_y0,
            )

        for ap, gp in zip(aerial_anchors, ground_anchors):
            if ap is None or gp is None:
                continue
            a = (ap[0] + x0, ap[1] + bot_img_y0)
            g = (gp[0] + x0 + BOT_PANE_LEFT_W, gp[1] + bot_img_y0)
            cv.line(frame, a, g, MATCH_LINE_COLOR, 3, cv.LINE_AA)

        frame[:BORDER, :] = BAR_COLOR
        frame[-BORDER:, :] = BAR_COLOR
        frame[:, :BORDER] = BAR_COLOR
        frame[:, -BORDER:] = BAR_COLOR

        self._writer.write(frame)
        if self.live:
            with self._gui_lock:
                self._latest_frame = frame

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.live and self._gui_thread is not None:
            self._gui_stop = True
            self._gui_thread.join(timeout=0.01)
            self._gui_thread = None

    def _render_aerial_traj(self, t: float, instant_pose_history) -> np.ndarray:
        canvas = self._aerial_thumb.copy()
        x_off, y_off, scale = self._aerial_thumb_offset

        def full_to_canvas(px_xy: np.ndarray) -> np.ndarray:
            px = np.atleast_2d(px_xy)
            cols = px[:, 0] * scale + x_off
            rows = px[:, 1] * scale + y_off
            return np.column_stack([cols, rows])

        if self.gt_pose_data is not None and t > self._gt_t_cache:
            try:
                gt_pose = self.gt_pose_data.pose(t)
                gt_body = (
                    gt_pose @ self.T_camera_flu
                    if self.T_camera_flu is not None
                    else gt_pose
                )
                gt_utm = gt_body[:2, 3]
                px = self.utm_to_pixel(np.atleast_2d(gt_utm))[0]
                self._gt_traj_px.append((float(px[0]), float(px[1])))
                self._gt_t_cache = t
            except Exception:
                pass
        if len(self._gt_traj_px) >= 2:
            pts_full = np.array(self._gt_traj_px)
            pts = full_to_canvas(pts_full).astype(np.int32)
            cv.polylines(canvas, [pts], False, GT_COLOR, 2, cv.LINE_AA)

        est_utm = []
        for _, T in instant_pose_history:
            if T is None or np.any(np.isnan(T)):
                est_utm.append(None)
            else:
                est_utm.append(T[:2, 3])
        run, runs = [], []
        for p in est_utm:
            if p is None:
                if run:
                    runs.append(np.array(run))
                    run = []
            else:
                run.append(p)
        if run:
            runs.append(np.array(run))
        for r in runs:
            if len(r) < 2:
                continue
            pts_full = self.utm_to_pixel(r)
            pts = full_to_canvas(pts_full).astype(np.int32)
            cv.polylines(canvas, [pts], False, EST_COLOR, 2, cv.LINE_AA)

        self._draw_aerial_patch_box(canvas, full_to_canvas)

        h, w = canvas.shape[:2]
        if self._inlier_positions_utm:
            utm_arr = np.atleast_2d(np.asarray(self._inlier_positions_utm))
            pts_full = self.utm_to_pixel(utm_arr)
            pts_cv = full_to_canvas(pts_full)
            for (cx, cy) in pts_cv:
                if 0 <= cx < w and 0 <= cy < h:
                    cv.drawMarker(
                        canvas,
                        (int(round(cx)), int(round(cy))),
                        INLIER_COLOR,
                        markerType=cv.MARKER_STAR,
                        markerSize=18,
                        thickness=2,
                        line_type=cv.LINE_AA,
                    )

        if self._latest_inlier_position_utm is not None:
            pts_full = self.utm_to_pixel(
                np.atleast_2d(self._latest_inlier_position_utm)
            )
            pts_cv = full_to_canvas(pts_full)
            cx, cy = pts_cv[0]
            if 0 <= cx < w and 0 <= cy < h:
                cv.drawMarker(
                    canvas,
                    (int(round(cx)), int(round(cy))),
                    LATEST_INLIER_COLOR,
                    markerType=cv.MARKER_STAR,
                    markerSize=18,
                    thickness=2,
                    line_type=cv.LINE_AA,
                )

        if scale > 0 and self.px_per_m > 0:
            # Pin to the bottom-right of the actual aerial image (letterboxed
            # inside the pane), not the pane bounds.
            _draw_scale_bar(
                canvas,
                m_per_px=1.0 / (scale * self.px_per_m),
                right_x=TOP_AERIAL_W - x_off - 12,
                bottom_y=IMG_H - y_off - 16,
                target_px=140,
            )

        return canvas

    def _draw_aerial_patch_box(
        self,
        canvas: np.ndarray,
        full_to_canvas: Callable[[np.ndarray], np.ndarray],
    ):
        if self._last_match is None:
            return
        try:
            i_a, j_a = (int(x) for x in self._last_match.aerial_key.split("_"))
        except Exception:
            return
        stride_m = self.patch_side_len_m * (1.0 - self.patch_overlap)
        x_min_m = i_a * stride_m
        y_min_m = j_a * stride_m
        x_max_m = x_min_m + self.patch_side_len_m
        y_max_m = y_min_m + self.patch_side_len_m
        corners_full = np.array(
            [
                [x_min_m * self.px_per_m, y_min_m * self.px_per_m],
                [x_max_m * self.px_per_m, y_max_m * self.px_per_m],
            ]
        )
        c = full_to_canvas(corners_full)
        p1 = (int(round(c[0, 0])), int(round(c[0, 1])))
        p2 = (int(round(c[1, 0])), int(round(c[1, 1])))
        cv.rectangle(canvas, p1, p2, PATCH_BOX_COLOR, 3, cv.LINE_AA)

    def _render_ground_rgb(self, ground_img: np.ndarray) -> np.ndarray:
        if ground_img is None:
            return _white_canvas(IMG_H, TOP_GROUND_W)
        img = ground_img
        if img.ndim == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        canvas, _ = _fit_into(img, TOP_GROUND_W, IMG_H)
        return canvas

    def _render_aerial_patch_pane(self) -> Tuple[np.ndarray, List[Optional[Tuple[int, int]]]]:
        canvas = _white_canvas(IMG_H, BOT_PANE_LEFT_W)
        if self._last_match is None:
            return canvas, []
        m = self._last_match

        try:
            i_a, j_a = (int(x) for x in m.aerial_key.split("_"))
        except Exception:
            return canvas, []

        patch_size_m = self.patch_side_len_m
        stride_m = patch_size_m * (1.0 - self.patch_overlap)
        x_min_m = i_a * stride_m
        y_min_m = j_a * stride_m
        x_max_m = x_min_m + patch_size_m
        y_max_m = y_min_m + patch_size_m

        H, W = self.aerial_img.shape[:2]
        px_x1 = int(round(x_min_m * self.px_per_m))
        px_y1 = int(round(y_min_m * self.px_per_m))
        px_x2 = int(round(x_max_m * self.px_per_m))
        px_y2 = int(round(y_max_m * self.px_per_m))
        px_x1c = max(0, px_x1)
        px_y1c = max(0, px_y1)
        px_x2c = min(W, px_x2)
        px_y2c = min(H, px_y2)
        if px_x2c <= px_x1c or px_y2c <= px_y1c:
            return canvas, []
        crop = self.aerial_img[px_y1c:px_y2c, px_x1c:px_x2c]
        whiten_edge_black(crop)
        pane, (x_off, y_off, scale) = _fit_into(crop, BOT_PANE_LEFT_W, IMG_H)
        crop_origin_m = (px_x1c / self.px_per_m, px_y1c / self.px_per_m)

        def world_to_px(xy_m: np.ndarray) -> Tuple[int, int]:
            col_in_crop = (xy_m[0] - crop_origin_m[0]) * self.px_per_m
            row_in_crop = (xy_m[1] - crop_origin_m[1]) * self.px_per_m
            col = col_in_crop * scale + x_off
            row = row_in_crop * scale + y_off
            return (int(round(col)), int(round(row)))

        anchors: List[Optional[Tuple[int, int]]] = []
        for i, seg in enumerate(m.matched_aerial):
            a = _draw_primitive_world(
                pane, seg, world_to_px, _color_for(i), thickness=9
            )
            anchors.append(a)

        # Pin to the patch crop's bottom-right (not the pane bounds — the
        # crop is letterboxed so x_off/y_off are non-zero on the short side).
        _draw_scale_bar(
            pane,
            m_per_px=1.0 / (scale * self.px_per_m),
            right_x=BOT_PANE_LEFT_W - x_off - 12,
            bottom_y=IMG_H - y_off - 16,
        )
        return pane, anchors

    def _render_ground_dense_pane(self) -> Tuple[np.ndarray, List[Optional[Tuple[int, int]]]]:
        canvas = _white_canvas(IMG_H, BOT_PANE_RIGHT_W)
        if self._last_match is None:
            return canvas, []
        m = self._last_match

        # Warp ground coords into the aerial-aligned frame (un-mirrors them).
        T = m.T_aerial_ground_2d
        if T is not None:
            R = T[:2, :2]
            t = T[:2, 2]

            def warp(pts: np.ndarray) -> np.ndarray:
                pts = np.atleast_2d(pts)
                return pts @ R.T + t
        else:

            def warp(pts: np.ndarray) -> np.ndarray:
                return np.atleast_2d(pts)

        # Bbox from dense points; fall back to matched-primitive anchors.
        pts_list = [
            warp(np.asarray(s.dense_points[:, :2]))
            for s in m.ground_dense_segments
            if getattr(s, "dense_points", None) is not None and len(s.dense_points) > 0
        ]
        if not pts_list:
            pts_list = [
                warp(_primitive_anchor(seg).reshape(1, 2))
                for seg in m.matched_ground
                if _primitive_anchor(seg) is not None
            ]
        if not pts_list:
            return canvas, []
        all_pts = np.concatenate(pts_list, axis=0)
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)
        pad_m = 1.0
        x_min -= pad_m
        x_max += pad_m
        y_min -= pad_m
        y_max += pad_m

        margin_px = 24
        avail_w = BOT_PANE_RIGHT_W - 2 * margin_px
        avail_h = IMG_H - 2 * margin_px
        span_x = x_max - x_min
        span_y = y_max - y_min
        scale = min(avail_w / span_x, avail_h / span_y)
        out_w = span_x * scale
        out_h = span_y * scale
        x_off = (BOT_PANE_RIGHT_W - out_w) * 0.5
        y_off = (IMG_H - out_h) * 0.5

        def world_to_px(xy_m: np.ndarray) -> Tuple[int, int]:
            pt = warp(np.asarray(xy_m).reshape(1, 2))[0]
            col = (pt[0] - x_min) * scale + x_off
            row = (pt[1] - y_min) * scale + y_off
            return (int(round(col)), int(round(row)))

        # Largest segments first so smaller ones land on top.
        pt_radius_px = 3  # roughly matches aerial primitive thickness
        segs_sorted = sorted(
            m.ground_dense_segments,
            key=lambda s: 0
            if getattr(s, "dense_points", None) is None
            else len(s.dense_points),
            reverse=True,
        )
        for s in segs_sorted:
            dp = getattr(s, "dense_points", None)
            if dp is None or len(dp) == 0:
                continue
            r_, g_, b_ = s.color_from_id(order="rgb", num_type=int)
            # Darken so the brightly colored matched primitives pop on top.
            r_, g_, b_ = (int(c) * 6 // 10 for c in (r_, g_, b_))
            color_bgr = np.array([b_, g_, r_], dtype=np.uint8)
            pts = warp(np.asarray(dp[:, :2]))
            cols = ((pts[:, 0] - x_min) * scale + x_off).astype(np.int32)
            rows = ((pts[:, 1] - y_min) * scale + y_off).astype(np.int32)
            for dy in range(-pt_radius_px, pt_radius_px + 1):
                ry = rows + dy
                for dx in range(-pt_radius_px, pt_radius_px + 1):
                    cx_ = cols + dx
                    ok = (
                        (cx_ >= 0)
                        & (cx_ < BOT_PANE_RIGHT_W)
                        & (ry >= 0)
                        & (ry < IMG_H)
                    )
                    canvas[ry[ok], cx_[ok]] = color_bgr

        # Pin to the pane edge (not the rendered bbox), so the bar sits in
        # the margin alongside the points rather than over them.
        _draw_scale_bar(
            canvas,
            m_per_px=1.0 / scale,
            right_x=BOT_PANE_RIGHT_W - 12,
            bottom_y=IMG_H - 16,
            target_px=140,
        )

        anchors: List[Optional[Tuple[int, int]]] = []
        for i, seg in enumerate(m.matched_ground):
            a = _draw_primitive_world(
                canvas, seg, world_to_px, _color_for(i), thickness=7
            )
            anchors.append(a)

        return canvas, anchors


# matplotlib tab10, brightened (each color's max channel scaled to 255), in BGR.
_MATCH_PALETTE = [
    (255, 169, 44),   # tab:blue
    (14, 127, 255),   # tab:orange
    (70, 255, 70),    # tab:green
    (48, 46, 255),    # tab:red
    (255, 139, 200),  # tab:purple
    (137, 157, 255),  # tab:brown
    (218, 134, 255),  # tab:pink
    (180, 180, 180),  # tab:gray
    (46, 255, 254),   # tab:olive
    (255, 234, 28),   # tab:cyan
]


def _color_for(i: int) -> Tuple[int, int, int]:
    return _MATCH_PALETTE[i % len(_MATCH_PALETTE)]
