import argparse
import logging
import os
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from gen_seg_match.pipeline.cross_view_matching import (
    CrossViewMatching,
    CrossViewMatchingPipeline,
    cross_view_matching,
)
from gen_seg_match.params import CrossViewPlaceRecognitionParams
from gen_seg_match.cross_view.place_recognition import CrossViewPlaceRecognition

logger = logging.getLogger(__name__)


class CrossViewPlaceRecognitionPipeline:
    """Pipeline for place recognition visualization and results."""

    @staticmethod
    def compute_gt_patches(
        ground_submaps,
        aerial_submaps,
        ground_keys,
        aerial_keys,
        aerial_patch_params,
        gt_pose_data=None,
    ):
        """Determine which aerial patches contain each ground submap's GT position.

        Uses gt_pose_data (ground truth poses in the world/UTM frame) to compute
        the ground submap position in the aerial image frame, matching the
        approach used in CrossViewMatching.cross_view_match().

        Args:
            aerial_patch_params: AerialPatchParams with patch size and overlap.
            gt_pose_data: robotdatapy PoseData with GT poses in the same frame
                as the aerial image. If None, returns empty sets.

        Returns:
            dict: ground_key -> set of (i, j) aerial grid tuples
        """
        gt_patches = {gk: set() for gk in ground_keys}
        if gt_pose_data is None:
            return gt_patches

        # All aerial submaps share the same pose (T_world_aerial)
        any_aerial = next(iter(aerial_submaps.values()))
        T_aerial_inv = np.linalg.inv(any_aerial.pose)

        # Recompute stride in meters from pipeline params
        patch_size_m = aerial_patch_params.aerial_img_patch_side_len_m
        overlap = aerial_patch_params.aerial_img_patch_overlap
        stride_m = patch_size_m * (1.0 - overlap)

        aerial_tuples = {k: tuple(int(x) for x in k.split("_")) for k in aerial_keys}

        for gk in ground_keys:
            try:
                ground_pose_gt = gt_pose_data.pose(ground_submaps[gk].time)
            except Exception:
                continue

            T_aerial_camera = T_aerial_inv @ ground_pose_gt
            pos = T_aerial_camera[:2, 3]

            for ak in aerial_keys:
                i_a, j_a = aerial_tuples[ak]
                x1_m = i_a * stride_m
                y1_m = j_a * stride_m
                x2_m = x1_m + patch_size_m
                y2_m = y1_m + patch_size_m
                if x1_m <= pos[0] <= x2_m and y1_m <= pos[1] <= y2_m:
                    gt_patches[gk].add((i_a, j_a))

        return gt_patches

    @staticmethod
    def compute_top_k_patches(sim_matrix, ground_keys, aerial_keys, k):
        """Find top-k aerial patches by similarity for each ground submap.

        Returns:
            dict: ground_key -> list of (i, j) aerial grid tuples (descending similarity)
        """
        aerial_tuples = [tuple(int(x) for x in ak.split("_")) for ak in aerial_keys]
        top_k = {}
        for gi, gk in enumerate(ground_keys):
            row = sim_matrix[gi]
            valid_mask = ~np.isnan(row)
            if not np.any(valid_mask):
                top_k[gk] = []
                continue
            # Get indices sorted by descending similarity
            valid_indices = np.where(valid_mask)[0]
            sorted_indices = valid_indices[np.argsort(row[valid_indices])[::-1]]
            top_k[gk] = [aerial_tuples[idx] for idx in sorted_indices[:k]]
        return top_k

    @staticmethod
    def save_heatmaps(
        sim_matrix,
        ground_keys,
        aerial_keys,
        output_dir,
        gt_patches=None,
        top_k_patches=None,
    ):
        """Save per-ground-submap heatmaps and full similarity matrix.

        The per-ground heatmaps use the same orientation as the
        PoseEstimationResultMatrix in cross_view_matching: grid[i, j] where
        i is the aerial x-index (displayed on the y-axis) and j is the aerial
        y-index (displayed on the x-axis).

        Args:
            gt_patches: dict ground_key -> set of (i,j) tuples for green GT boxes
            top_k_patches: dict ground_key -> list of (i,j) tuples for red top-k circles
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine aerial grid shape from keys (format "i_j")
        aerial_tuples = [tuple(int(x) for x in k.split("_")) for k in aerial_keys]
        max_i = max(t[0] for t in aerial_tuples)
        max_j = max(t[1] for t in aerial_tuples)
        # Match PoseEstimationResultMatrix: shape (max_i+1, max_j+1), indexed [i, j]
        # imshow displays first axis as rows (y) and second as columns (x)
        grid_shape = (max_i + 1, max_j + 1)

        for gi, gk in enumerate(ground_keys):
            grid = np.full(grid_shape, np.nan)
            for ak_idx, ak in enumerate(aerial_keys):
                i_a, j_a = aerial_tuples[ak_idx]
                grid[i_a, j_a] = sim_matrix[gi, ak_idx]

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, origin="upper")
            ax.set_title(f"Ground submap {gk} — cosine similarity")
            ax.set_xlabel("aerial j")
            ax.set_ylabel("aerial i")

            # Draw green boxes around GT patches
            if gt_patches is not None and gk in gt_patches:
                for i_gt, j_gt in gt_patches[gk]:
                    # In grid[i, j], imshow row=i, col=j
                    rect = mpatches.Rectangle(
                        (j_gt - 0.5, i_gt - 0.5),
                        1,
                        1,
                        linewidth=2,
                        edgecolor="green",
                        facecolor="none",
                    )
                    ax.add_patch(rect)

            # Draw red circles around top-k patches
            if top_k_patches is not None and gk in top_k_patches:
                for rank, (i_tk, j_tk) in enumerate(top_k_patches[gk]):
                    # In grid[i, j], imshow row=i, col=j
                    circle = plt.Circle(
                        (j_tk, i_tk),
                        0.35,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                    )
                    ax.add_patch(circle)
                    ax.text(
                        j_tk,
                        i_tk,
                        str(rank + 1),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="red",
                        fontweight="bold",
                    )

            plt.colorbar(im, ax=ax)
            fig.savefig(output_dir / f"ground_{gk}_similarity.png", dpi=150)
            plt.close(fig)

        # Full matrix heatmap
        fig, ax = plt.subplots(
            figsize=(max(8, len(aerial_keys) * 0.3), max(4, len(ground_keys) * 0.5))
        )
        im = ax.imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xlabel("aerial submap")
        ax.set_ylabel("ground submap")
        ax.set_title("Place recognition similarity matrix")
        ax.set_xticks(range(len(aerial_keys)))
        ax.set_xticklabels(aerial_keys, rotation=90, fontsize=5)
        ax.set_yticks(range(len(ground_keys)))
        ax.set_yticklabels(ground_keys, fontsize=7)
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "similarity_matrix.png", dpi=150)
        plt.close(fig)

    @staticmethod
    def save_results(sim_matrix, ground_keys, aerial_keys, output_dir):
        """Save similarity matrix as .npy and a text summary."""
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "similarity_matrix.npy", sim_matrix)

        lines = ["Cross-View Place Recognition Results", "=" * 40, ""]
        for gi, gk in enumerate(ground_keys):
            row = sim_matrix[gi]
            valid = ~np.isnan(row)
            if not np.any(valid):
                lines.append(f"Ground {gk}: no valid similarities")
                continue
            best_idx = np.nanargmax(row)
            best_key = aerial_keys[best_idx]
            best_sim = row[best_idx]
            mean_sim = np.nanmean(row)
            lines.append(
                f"Ground {gk}: best={best_key} (sim={best_sim:.4f}), "
                f"mean={mean_sim:.4f}, valid={np.sum(valid)}/{len(row)}"
            )

        lines.append("")
        overall_valid = ~np.isnan(sim_matrix)
        if np.any(overall_valid):
            lines.append(f"Overall mean similarity: {np.nanmean(sim_matrix):.4f}")
            lines.append(f"Overall max similarity: {np.nanmax(sim_matrix):.4f}")

        results_str = "\n".join(lines)
        print(results_str)
        with open(output_dir / "results.txt", "w") as f:
            f.write(results_str + "\n")


def cross_view_place_recognition(
    params,
    output_dir,
    skip_segmentation=False,
    skip_aerial=False,
    skip_ground=False,
    aerial_dir=None,
    ground_dir=None,
):
    """Run place recognition pipeline.

    1. If not skip_segmentation: run cross_view_matching with skip_match=True
       to create submaps with descriptors.
    2. Load submaps from disk.
    3. Compute similarity matrix from descriptors already on submaps.
    4. Save results and heatmaps.
    """
    output_dir = str(output_dir)

    if not skip_segmentation:
        cross_view_matching(
            params,
            output_dir,
            skip_aerial=skip_aerial,
            skip_ground=skip_ground,
            skip_match=True,
            aerial_dir=aerial_dir,
            ground_dir=ground_dir,
        )

    # Load submaps (descriptors are persisted via pickle)
    from gen_seg_match.params import (
        CrossViewMatchingParams,
        CrossViewLocalizationDataParams,
        SegmentMatchParams,
        AerialSegmenterParams,
        AerialPatchParams,
        RegisterParams,
    )
    from gen_seg_match.match.segment_matcher import SegmentMatcher
    from gen_seg_match.register.registerer import Registerer2D
    from gen_seg_match.map2d.aerial_segmenter import AerialSegmenter

    pipeline_params = CrossViewMatchingParams.load(params)
    aerial_patch_params = AerialPatchParams.load(params)
    aerial_segmenter = AerialSegmenter(AerialSegmenterParams.load(params))
    runner = CrossViewMatchingPipeline(
        algorithm=CrossViewMatching(
            pipeline_params=pipeline_params,
            aerial_patch_params=aerial_patch_params,
            pixel_len_m=aerial_segmenter.params.pixel_len_m,
            matcher=SegmentMatcher(SegmentMatchParams.load(params)),
            registerer=Registerer2D(RegisterParams.load(params)),
        ),
    )

    aerial_seg_dir = os.path.join(
        aerial_dir or os.path.join(output_dir, "aerial"), "segments"
    )
    ground_seg_dir = os.path.join(
        ground_dir or os.path.join(output_dir, "ground"), "segments"
    )

    aerial_submaps = runner.load_submaps_from_dir(aerial_seg_dir)
    ground_submaps = runner.load_submaps_from_dir(ground_seg_dir)

    # Load GT pose data for green-box visualization
    gt_pose_data = None
    try:
        data_params = CrossViewLocalizationDataParams.load(params)
        if data_params.gt_pose_data is not None:
            from robotdatapy.data import PoseData

            gt_pose_data = PoseData.from_dict(data_params.gt_pose_data)
    except Exception:
        pass

    # Load place recognition params
    try:
        pr_params = CrossViewPlaceRecognitionParams.load(params)
    except Exception:
        pr_params = CrossViewPlaceRecognitionParams()
    k = pr_params.k_nearest_neighbors

    pr_output_dir = os.path.join(output_dir, "place_recognition")

    # Use the descriptor class for similarity computation
    pr_descriptor = CrossViewPlaceRecognition(pr_params)
    sim_matrix, ground_keys, aerial_keys = pr_descriptor.compute_similarity_matrix(
        ground_submaps, aerial_submaps
    )

    # Use the pipeline class for GT patches, top-k, viz, and results
    pr_pipeline = CrossViewPlaceRecognitionPipeline()
    gt_patches = pr_pipeline.compute_gt_patches(
        ground_submaps,
        aerial_submaps,
        ground_keys,
        aerial_keys,
        aerial_patch_params,
        gt_pose_data=gt_pose_data,
    )
    top_k_patches = pr_pipeline.compute_top_k_patches(
        sim_matrix, ground_keys, aerial_keys, k
    )

    viz_dir = os.path.join(pr_output_dir, "viz")
    pr_pipeline.save_heatmaps(
        sim_matrix,
        ground_keys,
        aerial_keys,
        viz_dir,
        gt_patches=gt_patches,
        top_k_patches=top_k_patches,
    )
    pr_pipeline.save_results(sim_matrix, ground_keys, aerial_keys, pr_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-view place recognition using DINO-GeM descriptors"
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        required=True,
        help="Path to params directory or YAML file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip submap creation; load existing submaps from output dir.",
    )
    parser.add_argument(
        "--skip-aerial",
        action="store_true",
        help="Skip aerial segmentation (forwarded to cross_view_matching).",
    )
    parser.add_argument(
        "--skip-ground",
        action="store_true",
        help="Skip ground segmentation (forwarded to cross_view_matching).",
    )
    parser.add_argument(
        "--aerial",
        type=str,
        default=None,
        help="Path to existing aerial directory (skips aerial segmentation).",
    )
    parser.add_argument(
        "--ground",
        type=str,
        default=None,
        help="Path to existing ground directory (skips ground segmentation).",
    )
    args = parser.parse_args()

    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    cross_view_place_recognition(
        args.params,
        args.output,
        skip_segmentation=args.skip_segmentation,
        skip_aerial=args.skip_aerial,
        skip_ground=args.skip_ground,
        aerial_dir=args.aerial,
        ground_dir=args.ground,
    )
