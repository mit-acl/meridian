"""Live/MP4 visualization for cross_view_incremental.

Composite frame layout (fixed 1440x1080, 4:3):

  top row (h=540):
    [ aerial trajectory (540x540, square) | current ground RGB (900x540) ]
  bottom row (h=540):
    [ aerial patch + matched primitives (720x540) |
      ground dense pcd + matched primitives (720x540) ]

The aerial trajectory pane is square (aerial backdrop is letterboxed into a
540x540 square). The other panes get extra horizontal room. The aerial patch
pane shows the full configured patch (e.g. 60m x 60m) cropped from the
aerial; its pixel rectangle is derived from patch geometry in meters so it
isn't sensitive to int() rounding on stride. Green lines connect matched
aerial/ground primitives across the bottom row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2 as cv
import numpy as np

from meridian.primitive.primitive import LinePrimitive, PointPrimitive
from meridian.primitive.primitive_list import PrimitiveList

logger = logging.getLogger(__name__)


FRAME_W = 1440
FRAME_H = 1080
ROW_H = 540
# Bottom split matches top split. Aerial pane stays square (aerial backdrop
# is letterboxed into a 540x540 sub-region inside it). Ground RGB is slightly
# narrower than half-width and is stretched in (aspect not preserved) to
# fill the pane.
LEFT_PANE_W = 600
RIGHT_PANE_W = FRAME_W - LEFT_PANE_W  # 840
TOP_AERIAL_W = LEFT_PANE_W
TOP_GROUND_W = RIGHT_PANE_W
BOT_PANE_LEFT_W = LEFT_PANE_W
BOT_PANE_RIGHT_W = RIGHT_PANE_W
assert TOP_AERIAL_W + TOP_GROUND_W == FRAME_W
assert BOT_PANE_LEFT_W + BOT_PANE_RIGHT_W == FRAME_W

BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
TEXT_OUTLINE = (255, 255, 255)
GT_COLOR = (40, 160, 40)
EST_COLOR = (180, 105, 255)  # pink (BGR)
INLIER_COLOR = (230, 0, 180)  # red (BGR)
LATEST_INLIER_COLOR = (0, 215, 255)  # gold (BGR)
PATCH_BOX_COLOR = (230, 216, 173)  # light blue (BGR)
MATCH_LINE_COLOR = (0, 150, 0)

FONT = cv.FONT_HERSHEY_DUPLEX


def _put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.88,
    thickness: int = 2,
    center: bool = False,
):
    """Black text on a solid white rectangle so it's legible on any background.
    `org` follows the OpenCV convention (bottom-left of the text baseline).
    If `center` is True, the rectangle is horizontally centered around
    `org[0]` instead of starting at it."""
    (tw, th), baseline = cv.getTextSize(text, FONT, scale, thickness)
    pad_x = 6
    pad_y = 4
    x = org[0] - tw // 2 if center else org[0]
    y = org[1]
    x1 = x - pad_x
    y1 = y - th - pad_y
    x2 = x + tw + pad_x
    y2 = y + baseline + pad_y
    cv.rectangle(img, (x1, y1), (x2, y2), TEXT_OUTLINE, cv.FILLED)
    cv.putText(img, text, (x, y), FONT, scale, TEXT_COLOR, thickness, cv.LINE_AA)


def _white_canvas(h: int, w: int) -> np.ndarray:
    c = np.empty((h, w, 3), dtype=np.uint8)
    c[:] = BG_COLOR
    return c


def _fit_into(
    src: np.ndarray, dst_w: int, dst_h: int
) -> Tuple[np.ndarray, Tuple[int, int, float]]:
    """Resize src into a dst_w x dst_h white canvas preserving aspect,
    centered. Returns (canvas, (x_off, y_off, scale))."""
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
    """Horizontal scale bar with end ticks and a labeled length.

    Bar's right end sits at (right_x, bottom_y); it grows leftward to a
    nice-round length from `_NICE_LENGTHS_M` closest to `target_px * m_per_px`.
    No-op if the chosen bar would fall outside the canvas.
    """
    if m_per_px <= 0:
        return
    target_m = target_px * m_per_px
    L_m = min(_NICE_LENGTHS_M, key=lambda v: abs(v - target_m))
    bar_px = int(round(L_m / m_per_px))
    x1 = right_x - bar_px
    if x1 < 4 or right_x >= canvas.shape[1]:
        return
    tick = thickness + 3
    cv.line(canvas, (x1, bottom_y), (right_x, bottom_y), color, thickness, cv.LINE_AA)
    cv.line(canvas, (x1, bottom_y - tick), (x1, bottom_y + tick), color, thickness, cv.LINE_AA)
    cv.line(canvas, (right_x, bottom_y - tick), (right_x, bottom_y + tick), color, thickness, cv.LINE_AA)
    label = f"{L_m} m" if L_m < 1000 else f"{L_m / 1000:.1f} km"
    _put_text(canvas, label, (x1, bottom_y - 10), scale=0.55)


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
            cv.circle(img, anchor_px, max(6, thickness + 3), color, -1, cv.LINE_AA)
    return anchor_px


@dataclass
class LastMatch:
    ground_key: str
    aerial_key: str
    matched_aerial: PrimitiveList
    matched_ground: PrimitiveList
    ground_dense_segments: list
    # SE(2) transform with the repo's T_<dest>_<src> convention:
    # p_aerial = T_aerial_ground_2d @ p_ground_homog. Extracted from
    # pose_result.T_i_j_hat (i=aerial, j=ground). Its 2x2 block has negative
    # determinant (z flips between aerial top-down and ground top-down), so
    # applying it both rotates and reflects, un-mirroring the ground view.
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
    # Authoritative patch geometry in meters. Pixel bounds for the crop are
    # derived at render time via px_per_m, so a single int() truncation per
    # corner is the only quantization (vs. stacking truncations on stride and
    # patch_size_px and then multiplying by i_a/j_a).
    patch_side_len_m: float
    patch_overlap: float
    fps: int = 10

    _writer: Optional[cv.VideoWriter] = field(default=None, init=False)
    _aerial_thumb: np.ndarray = field(default=None, init=False)
    _aerial_thumb_offset: Tuple[int, int, float] = field(default=None, init=False)
    _last_match: Optional[LastMatch] = field(default=None, init=False)
    # UTM positions (x, y) of the current CLIPPER inlier loop closures.
    # Refreshed by `update_inliers` after each ground submap is processed.
    _inlier_positions_utm: List[np.ndarray] = field(default_factory=list, init=False)
    # UTM position of the most recent inlier (highlighted gold).
    _latest_inlier_position_utm: Optional[np.ndarray] = field(default=None, init=False)
    _gt_traj_px: List[Tuple[float, float]] = field(default_factory=list, init=False)
    _gt_t_cache: float = field(default=-1.0, init=False)
    _live_window: str = field(default="cross_view_incremental", init=False)

    def __post_init__(self):
        self._aerial_thumb, self._aerial_thumb_offset = _fit_into(
            self.aerial_img, TOP_AERIAL_W, ROW_H
        )

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        self._writer = cv.VideoWriter(
            self.output_path, fourcc, self.fps, (FRAME_W, FRAME_H)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter at {self.output_path}")
        if self.live:
            # WINDOW_NORMAL = user-resizable; WINDOW_KEEPRATIO = image is
            # letterboxed to the window while preserving 4:3 aspect.
            cv.namedWindow(
                self._live_window, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO
            )
            cv.resizeWindow(self._live_window, FRAME_W, FRAME_H)

    # ------------------------------------------------------------------

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
        """Refresh the set of inlier loop-closure positions (UTM xy) to draw on
        the aerial trajectory pane as red stars. The most-recent inlier
        (`latest_position_utm`) is highlighted with a gold star."""
        self._inlier_positions_utm = list(positions_utm)
        self._latest_inlier_position_utm = (
            None if latest_position_utm is None else np.asarray(latest_position_utm)
        )

    def write_frame(self, t: float, ground_img: np.ndarray, instant_pose_history):
        frame = _white_canvas(FRAME_H, FRAME_W)

        frame[0:ROW_H, 0:TOP_AERIAL_W] = self._render_aerial_traj(
            t, instant_pose_history
        )
        frame[0:ROW_H, TOP_AERIAL_W : TOP_AERIAL_W + TOP_GROUND_W] = (
            self._render_ground_rgb(ground_img)
        )

        bot_aerial, aerial_anchors = self._render_aerial_patch_pane()
        bot_ground, ground_anchors = self._render_ground_dense_pane()
        frame[ROW_H:FRAME_H, 0:BOT_PANE_LEFT_W] = bot_aerial
        frame[ROW_H:FRAME_H, BOT_PANE_LEFT_W : BOT_PANE_LEFT_W + BOT_PANE_RIGHT_W] = (
            bot_ground
        )

        for ap, gp in zip(aerial_anchors, ground_anchors):
            if ap is None or gp is None:
                continue
            a = (ap[0], ap[1] + ROW_H)
            g = (gp[0] + BOT_PANE_LEFT_W, gp[1] + ROW_H)
            cv.line(frame, a, g, MATCH_LINE_COLOR, 2, cv.LINE_AA)

        _put_text(frame, f"t = {t:.2f}s", (FRAME_W // 2, 38), center=True)

        self._writer.write(frame)
        if self.live:
            cv.imshow(self._live_window, frame)
            cv.waitKey(1)

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.live:
            cv.destroyWindow(self._live_window)

    # ------------------------------------------------------------------

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
            cv.polylines(canvas, [pts], False, GT_COLOR, 3, cv.LINE_AA)

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
            cv.polylines(canvas, [pts], False, EST_COLOR, 3, cv.LINE_AA)

        # Aerial patch box: a thin light-blue rectangle outlining the
        # currently-selected aerial patch on the full aerial. Drawn before
        # the stars so star markers sit on top of it.
        self._draw_aerial_patch_box(canvas, full_to_canvas)

        # Inlier loop closures as red stars. Drawn last so they sit on top of
        # the trajectory polylines. Skipped if outside the canvas bounds.
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

        # Latest inlier: larger gold star on top so it's distinguishable from
        # the rest of the inlier set.
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

        _put_text(
            canvas,
            "Aerial - GT (green) / Est (pink) / Inliers (red, latest gold)",
            (10, ROW_H - 14),
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
            return _white_canvas(ROW_H, TOP_GROUND_W)
        img = ground_img
        if img.ndim == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        canvas, _ = _fit_into(img, TOP_GROUND_W, ROW_H)
        _put_text(canvas, "Ground RGB", (10, ROW_H - 14))
        return canvas

    def _render_aerial_patch_pane(self) -> Tuple[np.ndarray, List[Optional[Tuple[int, int]]]]:
        canvas = _white_canvas(ROW_H, BOT_PANE_LEFT_W)
        if self._last_match is None:
            return canvas, []
        m = self._last_match

        try:
            i_a, j_a = (int(x) for x in m.aerial_key.split("_"))
        except Exception:
            return canvas, []

        # Patch bounds in WORLD meters (full-aerial frame; primitives live in
        # this same frame). Derived from key + patch_side_len_m + overlap so
        # the result doesn't drift with int(stride_px) truncation.
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
        pane, (x_off, y_off, scale) = _fit_into(crop, BOT_PANE_LEFT_W, ROW_H)
        crop_origin_m = (px_x1c / self.px_per_m, px_y1c / self.px_per_m)

        def world_to_px(xy_m: np.ndarray) -> Tuple[int, int]:
            col_in_crop = (xy_m[0] - crop_origin_m[0]) * self.px_per_m
            row_in_crop = (xy_m[1] - crop_origin_m[1]) * self.px_per_m
            col = col_in_crop * scale + x_off
            row = row_in_crop * scale + y_off
            return (int(round(col)), int(round(row)))

        # Thick lines survive the large downscale from a full 60m crop into a
        # 720x540 pane (typically ~10x reduction).
        anchors: List[Optional[Tuple[int, int]]] = []
        for i, seg in enumerate(m.matched_aerial):
            a = _draw_primitive_world(
                pane, seg, world_to_px, _color_for(i), thickness=6
            )
            anchors.append(a)

        _draw_scale_bar(
            pane,
            m_per_px=1.0 / (scale * self.px_per_m),
            right_x=BOT_PANE_LEFT_W - 12,
            bottom_y=ROW_H - 16,
        )
        _put_text(pane, f"Aerial patch ({m.aerial_key})", (10, 25))
        return pane, anchors

    def _render_ground_dense_pane(self) -> Tuple[np.ndarray, List[Optional[Tuple[int, int]]]]:
        canvas = _white_canvas(ROW_H, BOT_PANE_RIGHT_W)
        if self._last_match is None:
            return canvas, []
        m = self._last_match

        # Apply the matched SE(2) transform (ground -> aerial frame) to all
        # ground coordinates before bbox / draw. The 2x2 block of the
        # registration result has negative determinant, so this rotates *and*
        # reflects, fixing the "ground looks mirrored" appearance and aligning
        # it with the aerial orientation. Without the transform we'd be
        # rendering raw submap-local coords, which use a different axis
        # convention than the aerial image.
        T = m.T_aerial_ground_2d  # (3,3) or None
        if T is not None:
            R = T[:2, :2]
            t = T[:2, 2]

            def warp(pts: np.ndarray) -> np.ndarray:
                pts = np.atleast_2d(pts)
                return pts @ R.T + t
        else:

            def warp(pts: np.ndarray) -> np.ndarray:
                return np.atleast_2d(pts)

        # Gather all dense points to define the world bbox. Fall back to
        # matched-primitive anchors only if no dense points are available.
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

        # Uniform scale that fits the bbox in either dimension, preserving
        # aspect (no stretching). Then center within the rectangular pane.
        margin_px = 24
        avail_w = BOT_PANE_RIGHT_W - 2 * margin_px
        avail_h = ROW_H - 2 * margin_px
        span_x = x_max - x_min
        span_y = y_max - y_min
        scale = min(avail_w / span_x, avail_h / span_y)
        out_w = span_x * scale
        out_h = span_y * scale
        x_off = (BOT_PANE_RIGHT_W - out_w) * 0.5
        y_off = (ROW_H - out_h) * 0.5

        def world_to_px(xy_m: np.ndarray) -> Tuple[int, int]:
            # _draw_primitive_world passes raw submap-frame coords here; warp
            # them into the aerial-aligned frame before mapping to pane px.
            pt = warp(np.asarray(xy_m).reshape(1, 2))[0]
            col = (pt[0] - x_min) * scale + x_off
            row = (pt[1] - y_min) * scale + y_off
            return (int(round(col)), int(round(row)))

        # # Subtle bbox frame so the user can see the actual extent of the
        # # submap even when the pcd is sparse.
        # cv.rectangle(
        #     canvas,
        #     (int(round(x_off)), int(round(y_off))),
        #     (int(round(x_off + out_w)), int(round(y_off + out_h))),
        #     (220, 220, 220),
        #     1,
        #     cv.LINE_AA,
        # )

        # Draw dense pcd: largest segments first so smaller ones land on top.
        # Use a small square stamp per point (size in pane pixels) for
        # legibility — single-pixel scatter is invisible after compression.
        pt_radius_px = 1  # half-side in px; total stamp = 2*r+1
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
            # Darken light colors so they read on white.
            r_, g_, b_ = (int(c) * 7 // 10 for c in (r_, g_, b_))
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
                        & (ry < ROW_H)
                    )
                    canvas[ry[ok], cx_[ok]] = color_bgr

        _draw_scale_bar(
            canvas,
            m_per_px=1.0 / scale,
            right_x=int(round(x_off + out_w - 8)),
            bottom_y=int(round(y_off + out_h - 12)),
            target_px=int(round(out_w * 0.25)),
        )

        anchors: List[Optional[Tuple[int, int]]] = []
        for i, seg in enumerate(m.matched_ground):
            a = _draw_primitive_world(
                canvas, seg, world_to_px, _color_for(i), thickness=4
            )
            anchors.append(a)

        _put_text(canvas, f"Ground submap ({m.ground_key})", (10, 25))
        return canvas, anchors


_MATCH_PALETTE = [
    (0, 0, 200),
    (0, 140, 200),
    (180, 0, 180),
    (200, 130, 0),
    (0, 110, 0),
    (200, 0, 0),
    (140, 60, 200),
    (60, 60, 60),
]


def _color_for(i: int) -> Tuple[int, int, int]:
    return _MATCH_PALETTE[i % len(_MATCH_PALETTE)]
