import argparse
import logging
import os
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

from meridian.pipeline.cross_view_matching import (
    CrossViewMatching,
    CrossViewMatchingPipeline,
    cross_view_matching,
)
from meridian.params import CrossViewPlaceRecognitionParams
from meridian.cross_view.place_recognition import CrossViewPlaceRecognition
from meridian.params.cross_view_params import check_frame_descriptors_match

logger = logging.getLogger(__name__)

# Recall@K operating points; the configured k_nearest_neighbors is added too.
RECALL_KS = (1, 5, 10, 15, 25, 50)


class CrossViewPlaceRecognitionPipeline:
    """Pipeline for place-recognition retrieval metrics: Recall@K, mAP, PR-AUC."""

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
    def labels(ground_keys, aerial_keys, gt_patches):
        """Boolean (num_ground, num_aerial) matrix: patch contains the GT position."""
        aerial_tuples = [tuple(int(x) for x in ak.split("_")) for ak in aerial_keys]
        y = np.zeros((len(ground_keys), len(aerial_keys)), dtype=bool)
        for gi, gk in enumerate(ground_keys):
            gt = gt_patches.get(gk, set())
            for ai, at in enumerate(aerial_tuples):
                y[gi, ai] = at in gt
        return y

    @staticmethod
    def valid_mask(sim_matrix):
        """Pairs that were actually scored; nan means a missing descriptor."""
        return ~np.isnan(sim_matrix)

    @staticmethod
    def rank(scores, y):
        """Sort descending and return the end index of each tied score group."""
        order = np.argsort(-scores, kind="stable")
        scores, y = scores[order], y[order]
        # Cut only where the score changes, so tied pairs share one curve point.
        last = np.r_[np.nonzero(np.diff(scores))[0], scores.size - 1]
        return scores, y, last

    @staticmethod
    def pr_curve(sim_matrix, y_true):
        """Precision and recall over every threshold on similarity, pooled.

        One point per (ground submap, aerial patch) pair. This scores a single
        global threshold, which matching does not use; it is a calibration
        diagnostic, not the retrieval metric.

        Returns:
            precision, recall, thresholds: parallel arrays, recall ascending.
        """
        valid = CrossViewPlaceRecognitionPipeline.valid_mask(sim_matrix)
        scores = sim_matrix[valid]
        y = y_true[valid]
        n_pos = int(y.sum())
        if n_pos == 0 or scores.size == 0:
            return np.array([]), np.array([]), np.array([])

        scores, y, last = CrossViewPlaceRecognitionPipeline.rank(scores, y)
        tp = np.cumsum(y)[last]
        return tp / (last + 1), tp / n_pos, scores[last]

    @staticmethod
    def per_query(sim_matrix, y_true):
        """Per-ground-submap average precision and rank of its first correct patch.

        Rows with no correct patch are skipped: the GT position lies outside the
        aerial map, so nothing there is recallable. Recall is normalized by the
        row's total positives, so a positive lost to a missing descriptor caps
        AP below 1 instead of being quietly dropped.

        Returns:
            aps, first_ranks: arrays over scorable rows; rank is inf if the row
            is unrecoverable (no scored pair, or every positive unscored).
        """
        aps, first_ranks = [], []
        for row, y_row in zip(sim_matrix, y_true):
            n_pos = int(y_row.sum())
            if n_pos == 0:
                continue
            valid = CrossViewPlaceRecognitionPipeline.valid_mask(row)
            scores, y = row[valid], y_row[valid]
            if scores.size == 0 or not y.any():
                aps.append(0.0)
                first_ranks.append(np.inf)
                continue
            scores, y, last = CrossViewPlaceRecognitionPipeline.rank(scores, y)
            tp = np.cumsum(y)[last]
            precision = tp / (last + 1)
            recall = tp / n_pos
            aps.append(float(np.sum(np.diff(np.r_[0.0, recall]) * precision)))
            # Pessimistic: the whole tie group holding the first hit is admitted.
            first_ranks.append(float(last[int(np.argmax(tp > 0))] + 1))
        return np.array(aps), np.array(first_ranks)

    @staticmethod
    def recall_at_k(first_ranks, ks):
        """Fraction of queries with a correct patch in the top k, for each k."""
        if first_ranks.size == 0:
            return {int(k): float("nan") for k in ks}
        return {int(k): float(np.mean(first_ranks <= k)) for k in ks}

    @staticmethod
    def mean_average_precision(aps):
        """mAP: one vote per query, so no cross-query score calibration is involved."""
        if aps.size == 0:
            return float("nan")
        return float(np.mean(aps))

    @staticmethod
    def average_precision(precision, recall):
        """Area under the curve as the recall-weighted mean precision."""
        if precision.size == 0:
            return float("nan")
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    @staticmethod
    def save_results(precision, recall, thresholds, y_true, sim_matrix, aps,
                     first_ranks, recalls, k_config, output_dir, run=None):
        """Write similarity_matrix.npy, metrics.npz, the plots and results.yaml/.txt.

        Counts come off the same scored-pair mask the pooled curve uses, so the
        reported chance line shares its denominators.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "similarity_matrix.npy", sim_matrix)

        valid = CrossViewPlaceRecognitionPipeline.valid_mask(sim_matrix)
        auc = CrossViewPlaceRecognitionPipeline.average_precision(precision, recall)
        n_scored = int(valid.sum())
        n_pos = int(y_true[valid].sum())
        n_aerial = int(y_true.shape[1])
        results = {"run": run}
        # Recall at the configured k is the number matching actually depends on.
        results.update({f"recall@{k}": v for k, v in recalls.items()})
        results.update({
            "mean_average_precision": CrossViewPlaceRecognitionPipeline.mean_average_precision(
                aps
            ),
            "k_nearest_neighbors": int(k_config),
            "candidate_fraction": k_config / float(n_aerial) if n_aerial else
            float("nan"),
            "num_queries": int(aps.size),
            "num_unrecoverable_queries": int(np.isinf(first_ranks).sum()),
            "num_ground_submaps": int(y_true.shape[0]),
            "num_aerial_patches": n_aerial,
            "num_scored_pairs": n_scored,
            "num_unscored_pairs": int(y_true.size - n_scored),
            "num_positive_pairs": n_pos,
            "pooled_pr_auc": auc,
            "chance_pr_auc": n_pos / float(n_scored) if n_scored else float("nan"),
        })
        with open(output_dir / "results.yaml", "w") as f:
            yaml.safe_dump(results, f, sort_keys=False)

        # The thresholds are the operating points, so keep them with the curve.
        np.savez(
            output_dir / "metrics.npz",
            precision=precision,
            recall=recall,
            thresholds=thresholds,
            average_precisions=aps,
            first_ranks=first_ranks,
            recall_ks=np.array(list(recalls.keys())),
            recall_values=np.array(list(recalls.values())),
        )

        lines = [f"{k}: {v}" for k, v in results.items()]
        results_str = "\n".join(lines)
        print(results_str)
        with open(output_dir / "results.txt", "w") as f:
            f.write(results_str + "\n")

        ks = np.array(list(recalls.keys()), dtype=float)
        vs = np.array(list(recalls.values()), dtype=float)
        if np.isfinite(vs).any():
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(ks, vs, marker="o", lw=1.5)
            ax.axvline(k_config, color="gray", ls="--", lw=1,
                       label=f"k = {k_config}")
            ax.set_xscale("log")
            ax.set_xlabel("k (candidates kept per query)")
            ax.set_ylabel("recall@k")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{run or 'place recognition'}  "
                         f"R@{int(k_config)} = {recalls.get(int(k_config), float('nan')):.4f}")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(output_dir / "recall_at_k.png", dpi=150)
            plt.close(fig)

        if precision.size:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(recall, precision, lw=1.5)
            ax.axhline(results["chance_pr_auc"], color="gray", ls="--", lw=1,
                       label="chance")
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{run or 'place recognition'}  pooled PR-AUC = {auc:.4f}")
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(output_dir / "pr_curve.png", dpi=150)
            plt.close(fig)

        return results


def cross_view_place_recognition(
    params,
    output_dir,
    run=None,
    skip_segmentation=False,
    skip_aerial=False,
    skip_ground=False,
    skip_viz=False,
    aerial_dir=None,
    ground_dir=None,
):
    """Run the place-recognition pipeline: retrieval metrics and heatmaps.

    1. If not skip_segmentation: run cross_view_matching with skip_match=True
       to create submaps with descriptors.
    2. Load submaps from disk.
    3. Compute the similarity matrix and the GT patch labels.
    4. Save Recall@K, mAP and the pooled PR curve with their plots.
    5. If not skip_viz: save per-submap similarity heatmaps with GT boxes and
       top-k circles.

    The dataset is whatever the params name, so a run key selects it.
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

    from meridian.params import (
        CrossViewMatchingParams,
        CrossViewLocalizationDataParams,
        PrimitiveMatchParams,
        AerialSegmenterParams,
        AerialPatchParams,
        RegisterParams,
    )
    from meridian.match.primitive_matcher import PrimitiveMatcher
    from meridian.register.registerer import Registerer2D
    from meridian.segmenter.aerial_segmenter import AerialSegmenter

    pipeline_params = CrossViewMatchingParams.load(params, run=run)
    aerial_patch_params = AerialPatchParams.load(params, run=run)
    aerial_segmenter = AerialSegmenter(AerialSegmenterParams.load(params, run=run))
    runner = CrossViewMatchingPipeline(
        algorithm=CrossViewMatching(
            pipeline_params=pipeline_params,
            aerial_patch_params=aerial_patch_params,
            pixel_len_m=aerial_segmenter.params.pixel_len_m,
            matcher=PrimitiveMatcher(PrimitiveMatchParams.load(params, run=run)),
            registerer=Registerer2D(RegisterParams.load(params, run=run)),
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

    # GT poses define the labels, so without them there is nothing to score.
    data_params = CrossViewLocalizationDataParams.load(params, run=run)
    if data_params.gt_pose_data is None:
        raise ValueError("gt_pose_data is required to label patches for a PR curve.")
    from robotdatapy.data import PoseData

    gt_pose_data = PoseData.from_dict(data_params.gt_pose_data)

    try:
        pr_params = CrossViewPlaceRecognitionParams.load(params, run=run)
    except FileNotFoundError:
        pr_params = CrossViewPlaceRecognitionParams()

    pr_descriptor = CrossViewPlaceRecognition(
        pr_params, check_frame_descriptors_match(params, pr_params.comparison, run=run)
    )
    sim_matrix, ground_keys, aerial_keys = pr_descriptor.compute_similarity_matrix(
        ground_submaps, aerial_submaps
    )

    pipeline = CrossViewPlaceRecognitionPipeline()
    gt_patches = pipeline.compute_gt_patches(
        ground_submaps,
        aerial_submaps,
        ground_keys,
        aerial_keys,
        aerial_patch_params,
        gt_pose_data=gt_pose_data,
    )
    y_true = pipeline.labels(ground_keys, aerial_keys, gt_patches)
    precision, recall, thresholds = pipeline.pr_curve(sim_matrix, y_true)
    aps, first_ranks = pipeline.per_query(sim_matrix, y_true)

    # A k past the map size is the same operating point as keeping everything.
    k_config = pr_params.k_nearest_neighbors
    ks = sorted({k for k in (*RECALL_KS, k_config) if k <= len(aerial_keys)})
    recalls = pipeline.recall_at_k(first_ranks, ks)

    pr_output_dir = os.path.join(output_dir, "place_recognition")
    if not skip_viz:
        pipeline.save_heatmaps(
            sim_matrix,
            ground_keys,
            aerial_keys,
            os.path.join(pr_output_dir, "viz"),
            gt_patches=gt_patches,
            top_k_patches=pipeline.compute_top_k_patches(
                sim_matrix, ground_keys, aerial_keys, k_config
            ),
        )

    return pipeline.save_results(
        precision,
        recall,
        thresholds,
        y_true,
        sim_matrix,
        aps,
        first_ranks,
        recalls,
        k_config,
        pr_output_dir,
        run=run,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-view place recognition: Recall@K, mAP, PR-AUC, heatmaps"
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
        "-r",
        "--run",
        type=str,
        default=None,
        help="Run key within the params file; selects the dataset.",
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
        "--skip-viz",
        action="store_true",
        help="Skip the per-submap similarity heatmaps (one figure per submap).",
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
        run=args.run,
        skip_segmentation=args.skip_segmentation,
        skip_aerial=args.skip_aerial,
        skip_ground=args.skip_ground,
        skip_viz=args.skip_viz,
        aerial_dir=args.aerial,
        ground_dir=args.ground,
    )
