import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

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
        assert seg.num_endpoints == 2, (
            "only supports line segments currently (no infinite lines)"
        )
        ax.plot(
            [seg.endpoints[0][0], seg.endpoints[1][0]],
            [seg.endpoints[0][1], seg.endpoints[1][1]],
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


def viz_cross_view_matches(
    aerial_segments: SegmentList,
    ground_segments: SegmentList,
    matches: np.ndarray,
    aerial_crop: np.ndarray = None,
    ground_segments_all: SegmentList = None,
    dense_points_by_id: dict = None,
    px_per_m: float = None,
    aerial_origin_m: tuple = None,
    target_size_kb: int = 200,
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
                elif isinstance(seg, SegmentPoint):
                    pt = seg.get_point()
                    x_px = (pt[0] - aerial_origin_m[0]) * px_per_m
                    y_px = (pt[1] - aerial_origin_m[1]) * px_per_m
                    ax_crop.plot(x_px, y_px, "o", color=match_color, markersize=5)

        # --- Bottom-right: Ground dense points + matched lines ---
        ax_dense = ax[1, 1]
        ax_dense.set_title("Ground Dense Points + Matches")

        if ground_segments_all is not None and dense_points_by_id is not None:
            for seg in ground_segments_all:
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
