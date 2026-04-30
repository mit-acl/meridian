from typing import Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from gen_seg_match.segment.segment_types import SegmentLine, SegmentPoint, SegmentList
from gen_seg_match.viz.utils import color_from_seed

color_list = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]


def plot_seg(seg, ax, custom_color=None):
    color = (
        seg.color_from_id(order="rgb", num_type="float")
        if custom_color is None
        else custom_color
    )
    if isinstance(seg, SegmentLine):
        if seg.num_endpoints == 2:
            ax.plot(
                [seg.endpoints[0][0], seg.endpoints[1][0]],
                [seg.endpoints[0][1], seg.endpoints[1][1]],
                color=color,
                linewidth=4,
            )
        else:
            pt = seg.get_point().flatten()
            d = seg.get_direction().flatten()
            ax.axline(
                (pt[0], pt[1]),
                (pt[0] + d[0], pt[1] + d[1]),
                color=color,
                linewidth=4,
            )
    elif isinstance(seg, SegmentPoint):
        ax.plot(
            seg.get_point()[0],
            seg.get_point()[1],
            "o",
            color=color,
            markersize=5,
            markeredgewidth=3,
        )


def viz_registration_alignment(
    aerial_segments: SegmentList,
    ground_segments: SegmentList,
    matches: np.ndarray,
    T_align: np.ndarray,
):
    """Plot inlier ground (brown, transformed by T_align) and aerial (sky-blue) segments
    overlaid in the aerial frame so visual alignment reflects registration quality.

    Match rows are [aerial_id, ground_id], matching the convention in
    viz_cross_view_matches.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    aerial_ids = [int(m[0]) for m in matches]
    ground_ids = [int(m[1]) for m in matches]

    inlier_aerial = SegmentList(
        [aerial_segments.get_segment_from_id(aid) for aid in aerial_ids]
    )
    inlier_ground = SegmentList(
        [ground_segments.get_segment_from_id(gid) for gid in ground_ids]
    )
    inlier_aerial = SegmentList([s for s in inlier_aerial if s is not None])
    inlier_ground = SegmentList([s for s in inlier_ground if s is not None])

    inlier_ground = inlier_ground.copy()
    inlier_ground.transform(T_align)

    for seg in inlier_aerial:
        plot_seg(seg, ax, custom_color="skyblue")
    for seg in inlier_ground:
        plot_seg(seg, ax, custom_color="saddlebrown")

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(True)
    ax.set_title(
        f"Inlier alignment ({len(inlier_aerial)} pairs): "
        "aerial=skyblue, ground=brown (T̂)"
    )
    return fig, ax


def viz_cross_view_matches(
    aerial_segments: SegmentList,
    ground_segments: SegmentList,
    matches: np.ndarray,
    aerial_crop: np.ndarray = None,
    ground_segments_all: SegmentList = None,
    dense_points_by_id: dict = None,
    px_per_m: float = None,
    aerial_origin_m: tuple = None,
):
    has_bottom = aerial_crop is not None
    color = "k"

    if has_bottom:
        fig, ax = plt.subplots(2, 2, figsize=(15, 20))
        ax_aerial = ax[0, 0]
        ax_ground = ax[0, 1]
    else:
        fig, ax = plt.subplots(1, 2, figsize=(15, 10))
        ax_aerial = ax[0]
        ax_ground = ax[1]

    # --- Top-left: Aerial segments (abstract) ---
    ax_aerial.set_title("Aerial Segments")
    for line in aerial_segments.get_lines():
        plot_seg(line, ax_aerial, custom_color=color)
    for point in aerial_segments.get_points():
        plot_seg(point, ax_aerial, custom_color=color)
    ax_aerial.axis("equal")
    ax_aerial.invert_yaxis()

    # --- Top-right: Ground segments (abstract) ---
    ax_ground.axis("equal")
    for line in ground_segments.get_lines():
        plot_seg(line, ax_ground, custom_color=color)
    for point in ground_segments.get_points():
        plot_seg(point, ax_ground, custom_color=color)
    ax_ground.set_title("Ground Segments")

    # --- Draw matches on top row ---
    for i, match in enumerate(matches):
        seg2 = aerial_segments.get_segment_from_id(match[0])
        seg1 = ground_segments.get_segment_from_id(match[1])
        random_color = color_from_seed(i, order="rgb", num_type="float")
        plot_seg(seg1, ax_ground, custom_color=random_color)
        plot_seg(seg2, ax_aerial, custom_color=random_color)

    ax_aerial.grid(True)
    ax_ground.grid(True)

    # --- Bottom panes (only when aerial_crop is provided) ---
    if has_bottom:
        # --- Bottom-left: Aerial crop with matched segments ---
        ax_crop = ax[1, 0]
        crop_rgb = cv.cvtColor(aerial_crop, cv.COLOR_BGR2RGB)
        ax_crop.imshow(crop_rgb)
        ax_crop.set_title("Aerial Crop + Matches")
        ax_crop.set_xticks([])
        ax_crop.set_yticks([])

        if px_per_m is not None and aerial_origin_m is not None:
            for i, match in enumerate(matches):
                seg = aerial_segments.get_segment_from_id(match[0])
                match_color = color_from_seed(i, order="rgb", num_type="float")
                if isinstance(seg, SegmentLine) and seg.num_endpoints == 2:
                    pts_px = [
                        (
                            (ep[0] - aerial_origin_m[0]) * px_per_m,
                            (ep[1] - aerial_origin_m[1]) * px_per_m,
                        )
                        for ep in seg.endpoints
                    ]
                    ax_crop.plot(
                        [pts_px[0][0], pts_px[1][0]],
                        [pts_px[0][1], pts_px[1][1]],
                        color=match_color,
                        linewidth=2,
                    )
                elif isinstance(seg, SegmentLine):
                    pt = seg.get_point().flatten()
                    d = seg.get_direction().flatten()
                    pt_px = (
                        (pt[0] - aerial_origin_m[0]) * px_per_m,
                        (pt[1] - aerial_origin_m[1]) * px_per_m,
                    )
                    d_px = (d[0] * px_per_m, d[1] * px_per_m)
                    ax_crop.axline(
                        pt_px,
                        (pt_px[0] + d_px[0], pt_px[1] + d_px[1]),
                        color=match_color,
                        linewidth=2,
                    )
                elif isinstance(seg, SegmentPoint):
                    pt = seg.get_point()
                    x_px = (pt[0] - aerial_origin_m[0]) * px_per_m
                    y_px = (pt[1] - aerial_origin_m[1]) * px_per_m
                    ax_crop.plot(x_px, y_px, "o", color=match_color, markersize=5)

        # --- Bottom-right: Ground dense points + matched lines ---
        ax_dense = ax[1, 1]
        ax_dense.set_title("Ground Dense Points + Matches")

        if ground_segments_all is not None and dense_points_by_id is not None:
            segments_sorted = sorted(
                ground_segments_all,
                key=lambda s: len(
                    dense_points_by_id.get(s.history[0] if s.history else -1, [])
                ),
                reverse=True,
            )
            for seg in segments_sorted:
                parent_id = seg.history[0] if seg.history else None
                if parent_id is not None and parent_id in dense_points_by_id:
                    pts = dense_points_by_id[parent_id]
                    seg_color = color_from_seed(
                        parent_id, order="rgb", num_type="float"
                    )
                    ax_dense.plot(
                        pts[:, 0],
                        pts[:, 1],
                        ".",
                        markersize=1,
                        alpha=0.5,
                        color=seg_color,
                    )

        for i, match in enumerate(matches):
            seg = ground_segments.get_segment_from_id(match[1])
            match_color = color_from_seed(i, order="rgb", num_type="float")
            plot_seg(seg, ax_dense, custom_color=match_color)

        ax_dense.set_aspect("equal")
        ax_dense.set_xlim(ax_ground.get_xlim())
        ax_dense.set_ylim(ax_ground.get_ylim())


# ---------------------------------------------------------------------------
# Standalone viz helpers (extracted from pipeline CrossViewMatching)
# ---------------------------------------------------------------------------


def viz_aerial_segments(
    img: np.ndarray,
    segments,
    crop,
    pixel_len_m: float,
    alpha_shape_alpha: float,
    alpha_shape_grid_downsample: float,
    alpha_shape_max_n_pts: int = None,
    alpha_shape_ref_size_m: float = None,
    downsample_factor: int = 5,
    line_width_m: float = 0.2,
) -> np.ndarray:
    aerial_viz = img.copy()
    line_width_px = max(1, int(line_width_m / pixel_len_m))
    img_origin_m = (
        (0.0, 0.0)
        if crop is None
        else (
            crop[0] * pixel_len_m,
            crop[1] * pixel_len_m,
        )
    )
    for seg in segments:
        alpha_shape_px = seg.get_alpha_shape_pixels(
            img_pixel_scale=pixel_len_m,
            grid_downsample=alpha_shape_grid_downsample,
            alpha=alpha_shape_alpha,
            img_origin_m=img_origin_m,
            max_n_pts=alpha_shape_max_n_pts,
            alpha_ref_size=alpha_shape_ref_size_m,
        )
        if alpha_shape_px is None:
            continue
        cv.polylines(
            aerial_viz, [alpha_shape_px], True, seg.viz_color[::-1], line_width_px
        )

    return downsample_aerial_viz(aerial_viz, downsample_factor)


def viz_general_segments_img(
    img: np.ndarray,
    segments: SegmentList,
    crop,
    px_per_m: float,
    downsample_factor: int = 5,
    line_width_m: float = 0.2,
) -> np.ndarray:
    general_viz = img.copy()
    x1, y1, x2, y2 = crop
    pixel_len_m = 1.0 / px_per_m
    line_width_px = max(1, int(line_width_m / pixel_len_m))

    # draw points
    for seg in segments.get_points():
        p = seg.get_point()
        cv.circle(
            general_viz,
            (int(p[0] * px_per_m - x1), int(p[1] * px_per_m - y1)),
            line_width_px * 2,
            seg.color_from_id(order="bgr"),
            line_width_px,
        )

    # draw lines
    for seg in segments.get_lines():
        if seg.num_endpoints == 2:
            p0 = seg.endpoints[0]
            p1 = seg.endpoints[1]
            pt0 = (int(p0[0] * px_per_m - x1), int(p0[1] * px_per_m - y1))
            pt1 = (int(p1[0] * px_per_m - x1), int(p1[1] * px_per_m - y1))
        else:
            pt = seg.get_point().flatten()
            d = seg.get_direction().flatten()
            far = 1e4
            p0_m = pt - d * far
            p1_m = pt + d * far
            pt0 = (int(p0_m[0] * px_per_m - x1), int(p0_m[1] * px_per_m - y1))
            pt1 = (int(p1_m[0] * px_per_m - x1), int(p1_m[1] * px_per_m - y1))
            h, w = general_viz.shape[:2]
            ret, pt0, pt1 = cv.clipLine((0, 0, w, h), pt0, pt1)
            if not ret:
                continue
        cv.line(
            general_viz,
            pt0,
            pt1,
            seg.color_from_id(order="bgr"),
            line_width_px,
        )
    return downsample_aerial_viz(general_viz, downsample_factor)


def viz_ground_segments(
    flattened_submap,
    aerial_segments,
    general_segments: SegmentList,
    sparse_general_segments: SegmentList,
    alpha_shape_alpha: float,
    alpha_shape_grid_downsample: float,
    alpha_shape_max_n_pts: int = None,
    alpha_shape_ref_size_m: float = None,
    show_origin: bool = False,
    origin_axis_len_m: float = 5.0,
) -> Tuple[plt.Figure, plt.Axes]:
    # Plot just segment points (largest first so smallest draw on top)
    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    segments_by_size = sorted(
        flattened_submap.segments,
        key=lambda s: len(s.dense_points),
        reverse=True,
    )
    for seg in segments_by_size:
        ax[0, 0].plot(
            seg.dense_points[:, 0],
            seg.dense_points[:, 1],
            ".",
            linewidth=1.0,
            alpha=0.5,
            color=seg.color_from_id(num_type=float),
        )
    ax[0, 0].set_aspect("equal")

    # Plot aerial segments (alpha shapes)
    for seg in aerial_segments:
        alpha_shape = seg.get_alpha_shape(
            alpha=alpha_shape_alpha,
            grid_downsample=alpha_shape_grid_downsample,
            max_n_pts=alpha_shape_max_n_pts,
            alpha_ref_size=alpha_shape_ref_size_m,
        )
        if alpha_shape is None:
            continue
        ax[0, 1].plot(
            alpha_shape[:, 0],
            alpha_shape[:, 1],
            color=seg.color_from_id(num_type=float),
            linewidth=2,
        )
    ax[0, 1].set_aspect("equal")

    # Plot general segments (points and lines)
    viz_general_segments_plt(ax[1, 0], general_segments)
    viz_general_segments_plt(ax[1, 1], sparse_general_segments)

    # Plot occluded points (largest first so smallest draw on top)
    for seg in segments_by_size:
        ax[2, 0].plot(
            seg.dense_points[:, 0],
            seg.dense_points[:, 1],
            ".",
            linewidth=1.0,
            alpha=0.5,
            color=seg.color_from_id(num_type=float),
        )
        occ = getattr(seg, "occluded_points", None)
        if occ is not None and len(occ) > 0:
            ax[2, 0].plot(
                occ[:, 0],
                occ[:, 1],
                ".",
                markersize=3,
                color="black",
                zorder=10,
            )
    ax[2, 0].set_aspect("equal")
    ax[2, 1].set_visible(False)

    visible_axes = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[2, 0]]

    origin_endpoints = []
    if show_origin:
        meta = getattr(flattened_submap, "metadata", None) or {}
        camera_pose = meta.get("camera_pose")
        if camera_pose is not None:
            origin = np.asarray(camera_pose[:2, 3]).flatten()
            R = np.asarray(camera_pose[:2, :2])
            x_end = origin + R @ np.array([origin_axis_len_m, 0.0])
            y_end = origin + R @ np.array([0.0, origin_axis_len_m])
            origin_endpoints = [origin, x_end, y_end]
            for axi in visible_axes:
                axi.plot(
                    [origin[0], x_end[0]], [origin[1], x_end[1]], "-", color="red", lw=2
                )
                axi.plot(
                    [origin[0], y_end[0]],
                    [origin[1], y_end[1]],
                    "-",
                    color="green",
                    lw=2,
                )
                axi.plot(origin[0], origin[1], "o", color="black", markersize=4)

    xlim = ax[0, 0].get_xlim()
    ylim = ax[0, 0].get_ylim()
    if origin_endpoints:
        xs = [p[0] for p in origin_endpoints]
        ys = [p[1] for p in origin_endpoints]
        xlim = (min(xlim[0], *xs), max(xlim[1], *xs))
        ylim = (min(ylim[0], *ys), max(ylim[1], *ys))
    for i in range(3):
        for j in range(2):
            if ax[i, j].get_visible():
                ax[i, j].set_xlim(xlim)
                ax[i, j].set_ylim(ylim)
    ratio = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    if ratio > 1:
        fig.set_size_inches(10, 15 / ratio)
    else:
        fig.set_size_inches(10 * ratio, 15)

    return fig, ax


def viz_general_segments_plt(ax: plt.Axes, general_segments: SegmentList) -> plt.Axes:
    for seg in general_segments.get_points():
        p = seg.get_point()
        ax.plot(
            p[0],
            p[1],
            "o",
            markersize=4,
            color=seg.color_from_id(num_type=float),
        )

    for seg in general_segments.get_lines():
        color = seg.color_from_id(num_type=float)
        if seg.num_endpoints == 2:
            p0 = seg.endpoints[0]
            p1 = seg.endpoints[1]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                "-",
                linewidth=2,
                color=color,
            )
        else:
            pt = seg.get_point().flatten()
            d = seg.get_direction().flatten()
            ax.axline(
                (pt[0], pt[1]),
                (pt[0] + d[0], pt[1] + d[1]),
                linewidth=2,
                color=color,
            )
    ax.set_aspect("equal")
    return ax


def viz_pose_on_aerial_crop(
    aerial_img: np.ndarray,
    crop_origin_px: tuple,
    T_gt: np.ndarray,
    px_per_m: float,
    patch_size_px: int,
    T_est: np.ndarray = None,
    target_size_kb: int = 200,
    line_width_px: int = 20,
) -> bytes:
    x1, y1 = crop_origin_px
    x2 = x1 + patch_size_px
    y2 = y1 + patch_size_px

    crop = aerial_img[y1:y2, x1:x2].copy()
    if len(crop.shape) == 2:
        crop = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)

    arrow_len = 0.08 * patch_size_px

    def draw_pose(img, T, color):
        pos_px = T[:2, 3] * px_per_m - np.array([x1, y1], dtype=float)
        cx, cy = int(round(pos_px[0])), int(round(pos_px[1]))
        cv.circle(img, (cx, cy), 5, color, -1)
        x_dir = T[:2, 0]
        x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-12)
        y_dir = T[:2, 1]
        y_dir = y_dir / (np.linalg.norm(y_dir) + 1e-12)
        x_end = (
            int(round(cx + arrow_len * x_dir[0])),
            int(round(cy + arrow_len * x_dir[1])),
        )
        y_end = (
            int(round(cx + arrow_len * y_dir[0])),
            int(round(cy + arrow_len * y_dir[1])),
        )
        cv.arrowedLine(img, (cx, cy), x_end, color, line_width_px, tipLength=0.3)
        cv.arrowedLine(img, (cx, cy), y_end, color, line_width_px, tipLength=0.3)

    gt_color = (0, 200, 0)
    est_color = (0, 0, 220)

    draw_pose(crop, T_gt, gt_color)
    if T_est is not None:
        draw_pose(crop, T_est, est_color)

    return downsample_to_target_size(crop, target_size_kb)


def downsample_to_target_size(img: np.ndarray, target_kb: int) -> bytes:
    target_bytes = target_kb * 1024
    lo, hi = 10, 95
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        _, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, mid])
        encoded = buf.tobytes()
        if len(encoded) <= target_bytes:
            best = encoded
            lo = mid + 1
        else:
            hi = mid - 1
    if best is not None:
        return best
    # quality 10 still too large — scale down
    _, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 10])
    encoded = buf.tobytes()
    scale = (target_bytes / len(encoded)) ** 0.5
    new_w = max(1, int(img.shape[1] * scale))
    new_h = max(1, int(img.shape[0] * scale))
    small = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
    _, buf = cv.imencode(".jpg", small, [cv.IMWRITE_JPEG_QUALITY, 10])
    return buf.tobytes()


def downsample_aerial_viz(img: np.ndarray, factor: int) -> np.ndarray:
    return cv.resize(
        img,
        (img.shape[1] // factor, img.shape[0] // factor),
        interpolation=cv.INTER_AREA,
    )
