"""Visualize multiple Langevin registration hypotheses on aerial image crops.

Adapted from MMDA's viz_lewis_result.py to work with meridian pipeline
output (PoseEstimationResultMatrix .npz files + submap .pkl files).

Usage:
    python -m meridian.viz.multi_hypothesis_viz \
        --results-dir results/mmda_test \
        --params path/to/params.yaml \
        --ground-idx 5 \
        [--aerial-key 3_1] \
        [--top-k 10] \
        [--downsample 2]
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pathlib

from scipy.spatial.transform import Rotation as Rot
from meridian.segment.segment_types import SegmentList, SegmentPoint, SegmentLine
from meridian.params import AerialPatchParams
from meridian.params.data_params import CrossViewLocalizationDataParams
from meridian.pipeline.data import CrossViewLocalizationData
from meridian.pipeline.result import PoseEstimationResultMatrix


# ---------------------------------------------------------------------------
# Drawing helpers (from viz_lewis_result.py, adapted)
# ---------------------------------------------------------------------------


def m_to_px(point_m, px_per_m, crop_origin_px):
    """Convert meter coordinates to pixel coordinates in the crop."""
    return point_m * px_per_m - crop_origin_px


def plot_segments_px(
    ax,
    segments,
    px_per_m,
    crop_origin_px,
    color="k",
    alpha=1.0,
    lw=1.5,
    ms=4,
    label=None,
):
    """Plot segments in pixel coordinates on a matplotlib axes."""
    labeled = False
    for seg in segments:
        pt_px = m_to_px(seg.point[:2], px_per_m, crop_origin_px)
        kw = dict(label=label) if not labeled and label else {}
        if isinstance(seg, SegmentLine):
            ep0, ep1 = seg.endpoints
            if ep0 is not None and ep1 is not None:
                p0_px = m_to_px(ep0[:2], px_per_m, crop_origin_px)
                p1_px = m_to_px(ep1[:2], px_per_m, crop_origin_px)
                ax.plot(
                    [p0_px[0], p1_px[0]],
                    [p0_px[1], p1_px[1]],
                    "-",
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    **kw,
                )
                labeled = True
            ax.plot(pt_px[0], pt_px[1], "o", color=color, ms=ms - 1, alpha=alpha)
        elif isinstance(seg, SegmentPoint):
            ax.plot(pt_px[0], pt_px[1], "o", color=color, ms=ms, alpha=alpha, **kw)
            labeled = True


def transform_segments_2d(segments, T):
    """Apply a 4x4 transform to 2D segments, returning new SegmentList."""
    R = T[:2, :2]
    t = T[:2, 3]
    out = []
    for seg in segments:
        pt_new = R @ seg.point[:2] + t
        if isinstance(seg, SegmentLine):
            dir_new = R @ seg.direction[:2]
            dir_new = dir_new / (np.linalg.norm(dir_new) + 1e-12)
            ep0, ep1 = seg.endpoints
            ep0_new = R @ ep0[:2] + t if ep0 is not None else None
            ep1_new = R @ ep1[:2] + t if ep1 is not None else None
            out.append(
                SegmentLine(
                    id=seg.id,
                    point=pt_new,
                    direction=dir_new,
                    endpoints=(ep0_new, ep1_new),
                    ratio_feature=seg.ratio_feature,
                    cos_feature=seg.cos_feature,
                )
            )
        elif isinstance(seg, SegmentPoint):
            out.append(
                SegmentPoint(
                    id=seg.id,
                    point=pt_new,
                    ratio_feature=seg.ratio_feature,
                    cos_feature=seg.cos_feature,
                )
            )
    return SegmentList(out)


def draw_associations_px(
    ax, associations, aerial_segs, ground_segs, T, px_per_m, crop_origin_px
):
    """Draw lines connecting associated ground (transformed) and aerial segments.

    associations: list of (ground_id, aerial_id) tuples.
    """
    R = T[:2, :2]
    t = T[:2, 3]
    pair_colors = plt.cm.Set1(np.linspace(0, 1, max(len(associations), 1)))
    for k, (g_id, a_id) in enumerate(associations):
        g_seg = ground_segs.get_segment_from_id(g_id)
        a_seg = aerial_segs.get_segment_from_id(a_id)
        if g_seg is None or a_seg is None:
            continue
        g_pt_m = R @ g_seg.point[:2] + t
        a_pt_m = a_seg.point[:2]
        g_px = m_to_px(g_pt_m, px_per_m, crop_origin_px)
        a_px = m_to_px(a_pt_m, px_per_m, crop_origin_px)
        color = pair_colors[k]
        ax.plot(
            [g_px[0], a_px[0]], [g_px[1], a_px[1]], "--", color=color, lw=1.5, alpha=0.8
        )
        ax.plot(
            g_px[0],
            g_px[1],
            "s",
            color=color,
            ms=8,
            markeredgecolor="k",
            markeredgewidth=0.5,
            zorder=5,
        )
        ax.plot(
            a_px[0],
            a_px[1],
            "o",
            color=color,
            ms=8,
            markeredgecolor="k",
            markeredgewidth=0.5,
            zorder=5,
        )


def draw_pose_px(ax, T, px_per_m, crop_origin_px, color="red", arrow_len_m=5.0, lw=3):
    """Draw a 2D coordinate frame at the pose given by T."""
    pos_px = m_to_px(T[:2, 3], px_per_m, crop_origin_px)
    arrow_len_px = arrow_len_m * px_per_m
    x_dir = T[:2, 0]
    x_dir = x_dir / (np.linalg.norm(x_dir) + 1e-12)
    y_dir = T[:2, 1]
    y_dir = y_dir / (np.linalg.norm(y_dir) + 1e-12)
    ax.annotate(
        "",
        xy=pos_px + arrow_len_px * x_dir,
        xytext=pos_px,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw),
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=pos_px + arrow_len_px * y_dir,
        xytext=pos_px,
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle="--"),
        annotation_clip=False,
    )
    ax.plot(pos_px[0], pos_px[1], "o", color=color, ms=8, zorder=5)


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------


def find_best_aerial_key(results_matrix):
    """Find the aerial patch index with the most hypotheses."""
    best_idx = None
    best_count = 0
    for idx in np.ndindex(results_matrix.shape):
        cell = results_matrix[idx]
        n = len(cell) if isinstance(cell, list) else 1
        if n > best_count:
            best_count = n
            best_idx = idx
    return best_idx, best_count


def visualize_multi_hypothesis(
    results_dir: pathlib.Path,
    params_path: str,
    ground_idx: int,
    aerial_key: str = None,
    top_k: int = 10,
    downsample: int = 1,
    output_dir: pathlib.Path = None,
    overlay_only: bool = False,
):
    """Generate multi-hypothesis visualization plots.

    Args:
        results_dir: Pipeline output directory (contains aerial/, ground/, match/).
        params_path: Path to YAML parameter file.
        ground_idx: Ground submap index.
        aerial_key: Aerial patch key "i_j". If None, picks the patch with most hypotheses.
        top_k: Number of top hypotheses to show.
        downsample: Downsample factor for aerial image display.
        output_dir: Where to save plots. Defaults to results_dir/viz_multi_hyp/.
        overlay_only: Only generate the overlay plot.
    """
    # Load pipeline data
    data_params = CrossViewLocalizationDataParams.load(params_path)
    data = CrossViewLocalizationData.from_params(data_params)
    aerial_patch_params = AerialPatchParams.load(params_path)

    pixel_len_m = data.aerial_img_scale
    px_per_m = 1.0 / pixel_len_m
    patch_size_m = aerial_patch_params.aerial_img_patch_side_len_m
    patch_size_px = int(patch_size_m * px_per_m)
    patch_overlap = aerial_patch_params.aerial_img_patch_overlap
    stride_px = int(patch_size_px * (1.0 - patch_overlap))

    print(
        f"Aerial image: {data.aerial_img.shape}, px_per_m={px_per_m:.1f}, "
        f"patch_size_m={patch_size_m}"
    )

    # Load results matrix
    match_seg_dir = results_dir / "match" / "segments"
    matrix_path = match_seg_dir / f"ground_{ground_idx}_results_matrix.pkl.npz"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Results matrix not found: {matrix_path}")
    results_matrix = PoseEstimationResultMatrix.load(str(matrix_path))
    print(f"Results matrix shape: {results_matrix.shape}")

    # Determine aerial patch
    if aerial_key is None:
        best_idx, best_count = find_best_aerial_key(results_matrix)
        if best_idx is None:
            print("No hypotheses found in results matrix.")
            return
        i_a, j_a = best_idx
        aerial_key = f"{i_a}_{j_a}"
        print(f"Auto-selected aerial patch ({i_a}, {j_a}) with {best_count} hypotheses")
    else:
        i_a, j_a = [int(x) for x in aerial_key.split("_")]

    cell = results_matrix[i_a, j_a]
    hypotheses = cell if isinstance(cell, list) else [cell]
    n_hyp = len(hypotheses)
    print(f"Aerial patch ({i_a}, {j_a}): {n_hyp} hypotheses")

    if n_hyp == 0:
        print("No hypotheses for this pair.")
        return

    # Load segments
    aerial_seg_path = results_dir / "aerial" / "segments" / f"{i_a}_{j_a}.pkl"
    ground_seg_path = results_dir / "ground" / "segments" / f"{ground_idx}.pkl"

    from meridian.map3d.submap import Submap

    aerial_submap = Submap.load(aerial_seg_path)
    ground_submap = Submap.load(ground_seg_path)
    aerial_segs = aerial_submap.segments
    ground_segs = ground_submap.segments

    # Extract aerial image crop
    x1 = i_a * stride_px
    y1 = j_a * stride_px
    crop_origin_px = np.array([x1, y1], dtype=float)
    aerial_img = data.aerial_img
    crop_img = aerial_img[y1 : y1 + patch_size_px, x1 : x1 + patch_size_px]
    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

    # Downsample
    ds = downsample
    if ds > 1:
        crop_rgb = crop_rgb[::ds, ::ds]
        display_px_per_m = px_per_m / ds
        display_crop_origin = crop_origin_px / ds
    else:
        display_px_per_m = px_per_m
        display_crop_origin = crop_origin_px
    H, W = crop_rgb.shape[:2]

    # Get GT
    T_gt = hypotheses[0].T_i_j
    has_gt = not np.any(np.isnan(T_gt))

    # Output directory
    if output_dir is None:
        output_dir = (
            results_dir / "viz_multi_hyp" / f"ground_{ground_idx}_aerial_{aerial_key}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k = min(top_k, n_hyp)

    # --- Figure 1: All top-k poses overlaid ---
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(crop_rgb, alpha=0.8)
    plot_segments_px(
        ax,
        aerial_segs,
        display_px_per_m,
        display_crop_origin,
        color="black",
        alpha=0.4,
        lw=1,
        ms=3,
    )

    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=0, vmax=max(top_k - 1, 1))
    for rank_i in range(top_k):
        T = hypotheses[rank_i].T_i_j_hat
        if np.any(np.isnan(T)):
            continue
        color = cmap(norm(rank_i))
        draw_pose_px(
            ax,
            T,
            display_px_per_m,
            display_crop_origin,
            color=color,
            arrow_len_m=5.0,
            lw=2,
        )

    if has_gt:
        draw_pose_px(
            ax,
            T_gt,
            display_px_per_m,
            display_crop_origin,
            color="lime",
            arrow_len_m=5.0,
            lw=3,
        )
        ax.plot([], [], color="lime", lw=2.5, label="Ground truth")
        ax.legend(loc="upper right", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Rank (0 = best objective)", fontsize=10)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(f"Top-{top_k} Hypotheses: ground {ground_idx}, aerial ({i_a},{j_a})")
    plt.tight_layout()
    out_path = output_dir / "overlay_top_k.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved overlay: {out_path}")

    if overlay_only:
        return

    # --- Figure 2: Individual panels ---
    ncols = min(top_k, 3)
    nrows = (top_k + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    if top_k == 1:
        axs = np.array([axs])
    axs = np.atleast_2d(axs).ravel()

    colors = plt.cm.tab10(np.linspace(0, 1, max(top_k, 1)))

    for rank_i in range(top_k):
        hyp = hypotheses[rank_i]
        T = hyp.T_i_j_hat
        n_assoc = len(hyp.associations)

        ax = axs[rank_i]
        ax.imshow(crop_rgb, alpha=0.8)
        plot_segments_px(
            ax,
            aerial_segs,
            display_px_per_m,
            display_crop_origin,
            color="black",
            alpha=0.3,
            lw=1,
            ms=2,
        )

        if not np.any(np.isnan(T)):
            # T is T_aerial_ground (maps ground -> aerial frame)
            # To overlay ground segs on aerial image, apply T^{-1} conceptually:
            # ground segs are in ground frame, T_aerial_ground maps them to aerial
            transformed = transform_segments_2d(ground_segs, T)
            plot_segments_px(
                ax,
                transformed,
                display_px_per_m,
                display_crop_origin,
                color=colors[rank_i],
                alpha=0.8,
                lw=2,
                ms=5,
            )
            draw_pose_px(
                ax,
                T,
                display_px_per_m,
                display_crop_origin,
                color="red",
                arrow_len_m=5.0,
                lw=3,
            )

            yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
            gt_str = ""
            if has_gt:
                dt = np.linalg.norm(T[:2, 3] - T_gt[:2, 3])
                yaw_gt = np.degrees(np.arctan2(T_gt[1, 0], T_gt[0, 0]))
                dyaw = abs(yaw - yaw_gt)
                dyaw = min(dyaw, 360 - dyaw)
                gt_str = f"\n[err: t={dt:.1f}m, yaw={dyaw:.1f}deg]"

            ax.set_title(
                f"Rank {rank_i + 1}: {n_assoc} assoc, "
                f"t=({T[0, 3]:.1f},{T[1, 3]:.1f}), "
                f"yaw={yaw:.1f}deg{gt_str}",
                fontsize=9,
            )
        else:
            ax.set_title(f"Rank {rank_i + 1}: {n_assoc} assoc (no valid T)", fontsize=9)

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

    for i in range(top_k, len(axs)):
        axs[i].set_visible(False)

    plt.suptitle(
        f"Individual Panels: ground {ground_idx}, aerial ({i_a},{j_a})", fontsize=14
    )
    plt.tight_layout()
    out_path = output_dir / "panels_top_k.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved panels: {out_path}")

    # --- Figure 3: Per-hypothesis association plots ---
    for rank_i in range(top_k):
        hyp = hypotheses[rank_i]
        T = hyp.T_i_j_hat
        assoc = hyp.associations
        n_assoc = len(assoc)

        if np.any(np.isnan(T)):
            continue

        yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(crop_rgb, alpha=0.8)
        plot_segments_px(
            ax,
            aerial_segs,
            display_px_per_m,
            display_crop_origin,
            color="black",
            alpha=0.5,
            lw=1.0,
            ms=3,
            label="Aerial",
        )
        transformed = transform_segments_2d(ground_segs, T)
        plot_segments_px(
            ax,
            transformed,
            display_px_per_m,
            display_crop_origin,
            color="steelblue",
            alpha=0.7,
            lw=1.5,
            ms=4,
            label="Ground (transformed)",
        )
        draw_associations_px(
            ax,
            assoc,
            aerial_segs,
            ground_segs,
            T,
            display_px_per_m,
            display_crop_origin,
        )
        draw_pose_px(
            ax,
            T,
            display_px_per_m,
            display_crop_origin,
            color="red",
            arrow_len_m=5.0,
            lw=3,
        )
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(
            f"Rank {rank_i + 1}: {n_assoc} associations, "
            f"t=({T[0, 3]:.1f},{T[1, 3]:.1f}), yaw={yaw:.1f}deg\n"
            f"square=ground (transformed), circle=aerial",
            fontsize=11,
        )
        plt.tight_layout()
        out_path = output_dir / f"assoc_rank{rank_i + 1}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved association plot: {out_path}")

    # --- Print error summary ---
    if has_gt:
        print(
            f"\nGT pose: t=({T_gt[0, 3]:.1f}, {T_gt[1, 3]:.1f}), "
            f"yaw={np.degrees(np.arctan2(T_gt[1, 0], T_gt[0, 0])):.1f}deg"
        )
        t_errs = []
        r_errs = []
        yaw_gt = np.arctan2(T_gt[1, 0], T_gt[0, 0])
        for i in range(n_hyp):
            T = hypotheses[i].T_i_j_hat
            if np.any(np.isnan(T)):
                t_errs.append(np.inf)
                r_errs.append(np.inf)
                continue
            dt = np.linalg.norm(T[:2, 3] - T_gt[:2, 3])
            yaw = np.arctan2(T[1, 0], T[0, 0])
            dyaw = abs(yaw - yaw_gt)
            dyaw = min(dyaw, 2 * np.pi - dyaw)
            t_errs.append(dt)
            r_errs.append(dyaw)

        best_idx = int(np.argmin([t + r for t, r in zip(t_errs, r_errs)]))
        print(
            f"Best by SE(2) error: rank {best_idx + 1}/{n_hyp}, "
            f"trans={t_errs[best_idx]:.2f}m, rot={np.degrees(r_errs[best_idx]):.2f}deg"
        )
        print(f"Top-1: trans={t_errs[0]:.2f}m, rot={np.degrees(r_errs[0]):.2f}deg")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize multi-hypothesis registration results"
    )
    parser.add_argument(
        "-r",
        "--results-dir",
        type=str,
        required=True,
        help="Pipeline output directory (e.g. results/mmda_test)",
    )
    parser.add_argument(
        "-p", "--params", type=str, required=True, help="Path to YAML parameter file"
    )
    parser.add_argument(
        "-g", "--ground-idx", type=int, required=True, help="Ground submap index"
    )
    parser.add_argument(
        "-a",
        "--aerial-key",
        type=str,
        default=None,
        help="Aerial patch key 'i_j' (default: auto-select most hypotheses)",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=10, help="Number of top hypotheses to show"
    )
    parser.add_argument(
        "-d",
        "--downsample",
        type=int,
        default=1,
        help="Downsample factor for aerial image display",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results_dir/viz_multi_hyp/)",
    )
    parser.add_argument(
        "--overlay-only", action="store_true", help="Only generate the overlay plot"
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir) if args.output_dir else None

    visualize_multi_hypothesis(
        results_dir=pathlib.Path(args.results_dir),
        params_path=args.params,
        ground_idx=args.ground_idx,
        aerial_key=args.aerial_key,
        top_k=args.top_k,
        downsample=args.downsample,
        output_dir=output_dir,
        overlay_only=args.overlay_only,
    )


if __name__ == "__main__":
    main()
