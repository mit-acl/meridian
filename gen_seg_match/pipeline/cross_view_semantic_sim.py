"""Interactive semantic similarity visualizer for cross-view segments.

Click on any segment in either subplot to see cosine similarity
of its cos_feature to all segments in the other subplot, colored
on a viridis scale.

Usage:
    python -m gen_seg_match.pipeline.cross_view_semantic_sim \
        --aerial <output>/aerial/segments/1_4.pkl \
        --ground <output>/ground/segments/0.pkl \
        --params gsm_tools/cross_view_matching_params/pennovation_euclid.yaml \
        --sim-min 0.3 --sim-max 0.9
"""

import argparse
import pathlib
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from gen_seg_match.segment.segment_types import SegmentLine, SegmentPoint
from gen_seg_match.viz.utils import color_from_seed


NEUTRAL_COLOR = "0.6"
HIGHLIGHT_COLOR = "red"
NA_COLOR = "0.3"


def _draw_segments(ax, segments, colors=None, show_ids=True):
    """Draw all segments on an axes with given per-segment colors."""
    for i, seg in enumerate(segments):
        color = colors[i] if colors is not None else NEUTRAL_COLOR
        if isinstance(seg, SegmentLine):
            if seg.num_endpoints == 2:
                ax.plot(
                    [seg.endpoints[0][0], seg.endpoints[1][0]],
                    [seg.endpoints[0][1], seg.endpoints[1][1]],
                    color=color,
                    linewidth=3,
                )
            else:
                pt = seg.get_point().flatten()
                d = seg.get_direction().flatten()
                ax.axline(
                    (pt[0], pt[1]),
                    (pt[0] + d[0], pt[1] + d[1]),
                    color=color,
                    linewidth=3,
                )
        elif isinstance(seg, SegmentPoint):
            ax.plot(
                seg.get_point()[0],
                seg.get_point()[1],
                "o",
                color=color,
                markersize=6,
                markeredgewidth=2,
            )
        if show_ids:
            pt = seg.get_point().flatten()
            ax.annotate(str(seg.id), (pt[0], pt[1]), fontsize=7)


def _draw_segments_on_image(
    ax, segments, px_per_m, origin_m, colors=None, show_ids=True
):
    """Draw segments converted from meter coords to pixel coords on an image axis."""
    for i, seg in enumerate(segments):
        color = colors[i] if colors is not None else NEUTRAL_COLOR
        if isinstance(seg, SegmentLine):
            if seg.num_endpoints == 2:
                pts_px = [
                    (
                        (ep[0] - origin_m[0]) * px_per_m,
                        (ep[1] - origin_m[1]) * px_per_m,
                    )
                    for ep in seg.endpoints
                ]
                ax.plot(
                    [pts_px[0][0], pts_px[1][0]],
                    [pts_px[0][1], pts_px[1][1]],
                    color=color,
                    linewidth=3,
                )
            else:
                pt = seg.get_point().flatten()
                d = seg.get_direction().flatten()
                pt_px = (
                    (pt[0] - origin_m[0]) * px_per_m,
                    (pt[1] - origin_m[1]) * px_per_m,
                )
                d_px = (d[0] * px_per_m, d[1] * px_per_m)
                ax.axline(
                    pt_px,
                    (pt_px[0] + d_px[0], pt_px[1] + d_px[1]),
                    color=color,
                    linewidth=3,
                )
        elif isinstance(seg, SegmentPoint):
            pt = seg.get_point()
            x_px = (pt[0] - origin_m[0]) * px_per_m
            y_px = (pt[1] - origin_m[1]) * px_per_m
            ax.plot(x_px, y_px, "o", color=color, markersize=6, markeredgewidth=2)
        if show_ids:
            pt = seg.get_point().flatten()
            x_px = (pt[0] - origin_m[0]) * px_per_m
            y_px = (pt[1] - origin_m[1]) * px_per_m
            ax.annotate(str(seg.id), (x_px, y_px), fontsize=7)


def _draw_highlight(ax, seg):
    """Draw a highlighted segment (red, thicker)."""
    if isinstance(seg, SegmentLine):
        if seg.num_endpoints == 2:
            ax.plot(
                [seg.endpoints[0][0], seg.endpoints[1][0]],
                [seg.endpoints[0][1], seg.endpoints[1][1]],
                color=HIGHLIGHT_COLOR,
                linewidth=5,
            )
        else:
            pt = seg.get_point().flatten()
            d = seg.get_direction().flatten()
            ax.axline(
                (pt[0], pt[1]),
                (pt[0] + d[0], pt[1] + d[1]),
                color=HIGHLIGHT_COLOR,
                linewidth=5,
            )
    elif isinstance(seg, SegmentPoint):
        ax.plot(
            seg.get_point()[0],
            seg.get_point()[1],
            "o",
            color=HIGHLIGHT_COLOR,
            markersize=10,
            markeredgewidth=3,
        )


def _draw_highlight_on_image(ax, seg, px_per_m, origin_m):
    """Draw a highlighted segment on an image axis (red, thicker)."""
    if isinstance(seg, SegmentLine):
        if seg.num_endpoints == 2:
            pts_px = [
                (
                    (ep[0] - origin_m[0]) * px_per_m,
                    (ep[1] - origin_m[1]) * px_per_m,
                )
                for ep in seg.endpoints
            ]
            ax.plot(
                [pts_px[0][0], pts_px[1][0]],
                [pts_px[0][1], pts_px[1][1]],
                color=HIGHLIGHT_COLOR,
                linewidth=5,
            )
        else:
            pt = seg.get_point().flatten()
            d = seg.get_direction().flatten()
            pt_px = (
                (pt[0] - origin_m[0]) * px_per_m,
                (pt[1] - origin_m[1]) * px_per_m,
            )
            d_px = (d[0] * px_per_m, d[1] * px_per_m)
            ax.axline(
                pt_px,
                (pt_px[0] + d_px[0], pt_px[1] + d_px[1]),
                color=HIGHLIGHT_COLOR,
                linewidth=5,
            )
    elif isinstance(seg, SegmentPoint):
        pt = seg.get_point()
        x_px = (pt[0] - origin_m[0]) * px_per_m
        y_px = (pt[1] - origin_m[1]) * px_per_m
        ax.plot(
            x_px,
            y_px,
            "o",
            color=HIGHLIGHT_COLOR,
            markersize=10,
            markeredgewidth=3,
        )


def _draw_dense_points(ax, segments, dense_points_by_id):
    """Plot dense point clouds colored by parent segment."""
    segments_sorted = sorted(
        segments,
        key=lambda s: len(
            dense_points_by_id.get(s.history[0] if s.history else -1, [])
        ),
        reverse=True,
    )
    for seg in segments_sorted:
        parent_id = seg.history[0] if seg.history else None
        if parent_id is not None and parent_id in dense_points_by_id:
            pts = dense_points_by_id[parent_id]
            seg_color = color_from_seed(parent_id, order="rgb", num_type="float")
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                ".",
                markersize=1,
                alpha=0.5,
                color=seg_color,
            )


def _find_nearest_segment(click_xy, segments):
    """Return the segment closest to click_xy.

    For lines, uses closest_point_to_point for accurate distance to the
    line segment rather than just the midpoint. For points, uses get_point().
    """
    click_pt = np.array(click_xy).reshape(-1)
    best_seg = None
    best_dist = np.inf
    for seg in segments:
        if isinstance(seg, SegmentLine):
            query = np.zeros((seg.dim, 1))
            query[:2, 0] = click_pt[:2]
            closest = seg.closest_point_to_point(query).flatten()[:2]
            dist = np.linalg.norm(closest - click_pt[:2])
        else:
            pt = seg.get_point().flatten()[:2]
            dist = np.linalg.norm(pt - click_pt[:2])
        if dist < best_dist:
            best_dist = dist
            best_seg = seg
    return best_seg


def _compute_similarities(selected_seg, other_segments):
    """Return list of cosine similarities (or None for missing cos_feature)."""
    if selected_seg.cos_feature is None:
        return [None] * len(other_segments)
    sims = []
    for seg in other_segments:
        if seg.cos_feature is None:
            sims.append(None)
        else:
            sims.append(float(np.dot(selected_seg.cos_feature, seg.cos_feature)))
    return sims


def _sim_to_colors(similarities, sim_min=0.0, sim_max=1.0):
    """Convert similarity values to viridis colors. None entries get NA_COLOR."""
    cmap = cm.get_cmap("viridis")
    norm = Normalize(vmin=sim_min, vmax=sim_max)
    colors = []
    for s in similarities:
        if s is None:
            colors.append(NA_COLOR)
        else:
            colors.append(cmap(norm(s)))
    return colors


def run(
    aerial_segments,
    ground_segments,
    sim_min=0.0,
    sim_max=1.0,
    aerial_crop=None,
    px_per_m=None,
    aerial_origin_m=None,
    dense_points_by_id=None,
):
    """Launch the interactive similarity visualizer."""
    has_context = aerial_crop is not None or dense_points_by_id is not None

    # Convert aerial crop to RGB once
    crop_rgb = None
    if aerial_crop is not None:
        crop_rgb = cv2.cvtColor(aerial_crop, cv2.COLOR_BGR2RGB)

    # Create figure with GridSpec
    if has_context:
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], hspace=0.25)
        if crop_rgb is not None:
            ax_aerial_ctx = fig.add_subplot(gs[0, 0])
        else:
            ax_aerial_ctx = None
        if dense_points_by_id is not None:
            ax_ground_ctx = fig.add_subplot(gs[0, 1])
        else:
            ax_ground_ctx = None
        ax_aerial = fig.add_subplot(gs[1, 0])
        ax_ground = fig.add_subplot(gs[1, 1])
        ax_cbar = fig.add_subplot(gs[:, 2])
    else:
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])
        ax_aerial = fig.add_subplot(gs[0, 0])
        ax_ground = fig.add_subplot(gs[0, 1])
        ax_cbar = fig.add_subplot(gs[0, 2])
        ax_aerial_ctx = None
        ax_ground_ctx = None

    fig.suptitle("Semantic Similarity Visualizer (click a segment)")

    # Colorbar
    norm = Normalize(vmin=sim_min, vmax=sim_max)
    sm = ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=ax_cbar, label="Cosine Similarity")

    def _redraw(selected=None, is_aerial_selected=None):
        """Redraw all panes with optional selection highlighting."""
        # Compute similarity colors
        sims = None
        aerial_colors = None
        ground_colors = None
        if selected is not None:
            if is_aerial_selected:
                sims = _compute_similarities(selected, ground_segments)
                ground_colors = _sim_to_colors(sims, sim_min, sim_max)
            else:
                sims = _compute_similarities(selected, aerial_segments)
                aerial_colors = _sim_to_colors(sims, sim_min, sim_max)

        # --- Bottom-left: Aerial abstract ---
        ax_aerial.cla()
        ax_aerial.set_title("Aerial Segments")
        _draw_segments(ax_aerial, aerial_segments, colors=aerial_colors)
        if selected is not None and is_aerial_selected:
            _draw_highlight(ax_aerial, selected)
        ax_aerial.set_aspect("equal")
        ax_aerial.grid(True)
        ax_aerial.invert_yaxis()

        # --- Bottom-right: Ground abstract ---
        ax_ground.cla()
        ax_ground.set_title("Ground Segments")
        _draw_segments(ax_ground, ground_segments, colors=ground_colors)
        if selected is not None and not is_aerial_selected:
            _draw_highlight(ax_ground, selected)
        ax_ground.set_aspect("equal")
        ax_ground.grid(True)

        # --- Top-left: Aerial crop ---
        if ax_aerial_ctx is not None:
            ax_aerial_ctx.cla()
            ax_aerial_ctx.set_title("Aerial Crop")
            ax_aerial_ctx.imshow(crop_rgb)
            _draw_segments_on_image(
                ax_aerial_ctx,
                aerial_segments,
                px_per_m,
                aerial_origin_m,
                colors=aerial_colors,
            )
            if selected is not None and is_aerial_selected:
                _draw_highlight_on_image(
                    ax_aerial_ctx, selected, px_per_m, aerial_origin_m
                )
            ax_aerial_ctx.set_xticks([])
            ax_aerial_ctx.set_yticks([])

        # --- Top-right: Ground dense points ---
        if ax_ground_ctx is not None:
            ax_ground_ctx.cla()
            ax_ground_ctx.set_title("Ground Dense Points")
            _draw_dense_points(ax_ground_ctx, ground_segments, dense_points_by_id)
            _draw_segments(ax_ground_ctx, ground_segments, colors=ground_colors)
            if selected is not None and not is_aerial_selected:
                _draw_highlight(ax_ground_ctx, selected)
            ax_ground_ctx.set_aspect("equal")
            ax_ground_ctx.set_xlim(ax_ground.get_xlim())
            ax_ground_ctx.set_ylim(ax_ground.get_ylim())

        # --- N/A annotations ---
        if sims is not None:
            if is_aerial_selected:
                # Annotate ground segments with missing features
                na_axes_meter = [ax_ground]
                if ax_ground_ctx is not None:
                    na_axes_meter.append(ax_ground_ctx)
                for na_ax in na_axes_meter:
                    for i, seg in enumerate(ground_segments):
                        if sims[i] is None:
                            pt = seg.get_point().flatten()
                            na_ax.annotate(
                                "N/A",
                                (pt[0], pt[1]),
                                fontsize=6,
                                color="white",
                                fontweight="bold",
                                ha="center",
                                va="bottom",
                            )
            else:
                # Annotate aerial segments with missing features (meter coords)
                for i, seg in enumerate(aerial_segments):
                    if sims[i] is None:
                        pt = seg.get_point().flatten()
                        ax_aerial.annotate(
                            "N/A",
                            (pt[0], pt[1]),
                            fontsize=6,
                            color="white",
                            fontweight="bold",
                            ha="center",
                            va="bottom",
                        )
                # Annotate on aerial crop (pixel coords)
                if ax_aerial_ctx is not None:
                    for i, seg in enumerate(aerial_segments):
                        if sims[i] is None:
                            pt = seg.get_point().flatten()
                            x_px = (pt[0] - aerial_origin_m[0]) * px_per_m
                            y_px = (pt[1] - aerial_origin_m[1]) * px_per_m
                            ax_aerial_ctx.annotate(
                                "N/A",
                                (x_px, y_px),
                                fontsize=6,
                                color="white",
                                fontweight="bold",
                                ha="center",
                                va="bottom",
                            )

        # Update title
        if selected is not None:
            has_feature = "yes" if selected.cos_feature is not None else "NO"
            fig.suptitle(f"Selected: id={selected.id} (cos_feature: {has_feature})")
        else:
            fig.suptitle("Semantic Similarity Visualizer (click a segment)")

        fig.canvas.draw_idle()

    # Initial draw
    _redraw()

    # Build set of clickable axes for each side
    aerial_axes = {ax_aerial}
    ground_axes = {ax_ground}
    if ax_aerial_ctx is not None:
        aerial_axes.add(ax_aerial_ctx)
    if ax_ground_ctx is not None:
        ground_axes.add(ax_ground_ctx)

    def on_click(event):
        if event.inaxes in aerial_axes:
            is_aerial = True
        elif event.inaxes in ground_axes:
            is_aerial = False
        else:
            return

        # Convert click coordinates to meter space
        click_xy = (event.xdata, event.ydata)
        if event.inaxes is ax_aerial_ctx:
            click_xy = (
                event.xdata / px_per_m + aerial_origin_m[0],
                event.ydata / px_per_m + aerial_origin_m[1],
            )

        clicked_segments = aerial_segments if is_aerial else ground_segments
        selected = _find_nearest_segment(click_xy, clicked_segments)
        if selected is None:
            return

        _redraw(selected=selected, is_aerial_selected=is_aerial)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()


def _load_segments(filepath):
    """Load a SegmentList or Submap from a pickle file and return segments."""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    # If it's a Submap, extract segments
    if hasattr(obj, "segments"):
        return obj.segments
    return obj


if __name__ == "__main__":
    from gen_seg_match.params.data_params import CrossViewLocalizationDataParams
    from gen_seg_match.params.pipeline_params import CrossViewMatchingParams
    from gen_seg_match.params.aerial_segmenter_params import AerialSegmenterParams

    parser = argparse.ArgumentParser(
        description="Interactive semantic similarity visualizer for cross-view segments."
    )
    parser.add_argument(
        "--aerial", required=True, help="Path to aerial segments/submap pkl file"
    )
    parser.add_argument(
        "--ground", required=True, help="Path to ground segments/submap pkl file"
    )
    parser.add_argument(
        "--params",
        default=None,
        help="Path to YAML params file for context panes (aerial crop + dense points)",
    )
    parser.add_argument(
        "--sim-min",
        type=float,
        default=0.0,
        help="Minimum similarity for colorbar (default: 0.0)",
    )
    parser.add_argument(
        "--sim-max",
        type=float,
        default=1.0,
        help="Maximum similarity for colorbar (default: 1.0)",
    )
    args = parser.parse_args()

    aerial_segments = _load_segments(args.aerial)
    ground_segments = _load_segments(args.ground)

    print(f"Aerial: {len(aerial_segments)} segments")
    print(f"Ground: {len(ground_segments)} segments")

    aerial_crop = None
    px_per_m_val = None
    aerial_origin_m_val = None

    if args.params is not None:
        data_params = CrossViewLocalizationDataParams.load(args.params)
        pipeline_params = CrossViewMatchingParams.load(args.params)
        aerial_seg_params = AerialSegmenterParams.load(args.params)

        aerial_img = cv2.imread(data_params.aerial_img_path)
        if aerial_img is not None:
            pixel_len_m = aerial_seg_params.pixel_len_m
            px_per_m_val = 1.0 / pixel_len_m
            patch_size_px = int(
                pipeline_params.aerial_img_patch_side_len_m * px_per_m_val
            )
            stride = int(
                patch_size_px * (1.0 - pipeline_params.aerial_img_patch_overlap)
            )

            # Parse aerial key from filename stem (e.g., "1_4" -> i=1, j=4)
            aerial_stem = pathlib.Path(args.aerial).stem
            i_a, j_a = (int(x) for x in aerial_stem.split("_"))

            x1_a = i_a * stride
            y1_a = j_a * stride
            aerial_crop = aerial_img[
                y1_a : y1_a + patch_size_px, x1_a : x1_a + patch_size_px
            ].copy()
            aerial_origin_m_val = (x1_a / px_per_m_val, y1_a / px_per_m_val)

            print(
                f"Aerial crop: patch ({i_a}, {j_a}), "
                f"size {patch_size_px}px, origin {aerial_origin_m_val}"
            )
        else:
            print(
                f"Warning: could not load aerial image from {data_params.aerial_img_path}"
            )

    # Auto-discover dense points from ground pkl path
    dense_points_by_id = None
    ground_path = pathlib.Path(args.ground)
    dense_dir = ground_path.parent / f"{ground_path.stem}_dense"
    if dense_dir.exists():
        dense_points_by_id = {}
        for dense_file in dense_dir.glob("*.pkl"):
            seg_id = int(dense_file.stem)
            with open(dense_file, "rb") as f:
                dense_points_by_id[seg_id] = pickle.load(f)
        print(f"Loaded {len(dense_points_by_id)} dense point sets from {dense_dir}")

    run(
        aerial_segments,
        ground_segments,
        sim_min=args.sim_min,
        sim_max=args.sim_max,
        aerial_crop=aerial_crop,
        px_per_m=px_per_m_val,
        aerial_origin_m=aerial_origin_m_val,
        dense_points_by_id=dense_points_by_id,
    )
