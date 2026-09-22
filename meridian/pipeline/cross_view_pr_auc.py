import argparse
import logging
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import yaml

from meridian.pipeline.cross_view_matching import (
    CrossViewMatching,
    CrossViewMatchingPipeline,
    cross_view_matching,
)
from meridian.pipeline.cross_view_place_recognition import (
    CrossViewPlaceRecognitionPipeline,
)
from meridian.params import CrossViewPlaceRecognitionParams
from meridian.cross_view.place_recognition import CrossViewPlaceRecognition
from meridian.params.cross_view_params import check_frame_descriptors_match

logger = logging.getLogger(__name__)


class CrossViewPRAUCPipeline:
    """Pipeline for the place-recognition precision-recall curve and its AUC."""

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
    def pr_curve(sim_matrix, y_true):
        """Precision and recall over every threshold on similarity.

        One point per (ground submap, aerial patch) pair, which is the decision
        place recognition actually makes. Ties are broken pessimistically: a
        positive scoring equal to a negative is ranked behind it.

        Returns:
            precision, recall, thresholds: parallel arrays, recall ascending.
        """
        valid = CrossViewPRAUCPipeline.valid_mask(sim_matrix)
        scores = sim_matrix[valid]
        y = y_true[valid]
        n_pos = int(y.sum())
        if n_pos == 0 or scores.size == 0:
            return np.array([]), np.array([]), np.array([])

        order = np.argsort(-scores, kind="stable")
        scores, y = scores[order], y[order]
        # Cut only where the score changes, so tied pairs share one curve point.
        last = np.r_[np.nonzero(np.diff(scores))[0], scores.size - 1]
        tp = np.cumsum(y)[last]
        return tp / (last + 1), tp / n_pos, scores[last]

    @staticmethod
    def average_precision(precision, recall):
        """Area under the curve as the recall-weighted mean precision."""
        if precision.size == 0:
            return float("nan")
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    @staticmethod
    def save_results(precision, recall, thresholds, y_true, valid, output_dir,
                     run=None):
        """Write pr_curve.npz, pr_curve.png, results.yaml and results.txt.

        `valid` is the scored-pair mask, so the reported counts and the chance
        line share the curve's denominators.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        auc = CrossViewPRAUCPipeline.average_precision(precision, recall)
        n_scored = int(valid.sum())
        n_pos = int(y_true[valid].sum())
        results = {
            "run": run,
            "pr_auc": auc,
            "num_ground_submaps": int(y_true.shape[0]),
            "num_aerial_patches": int(y_true.shape[1]),
            "num_scored_pairs": n_scored,
            "num_unscored_pairs": int(y_true.size - n_scored),
            "num_positive_pairs": n_pos,
            "chance_pr_auc": n_pos / float(n_scored) if n_scored else float("nan"),
        }
        with open(output_dir / "results.yaml", "w") as f:
            yaml.safe_dump(results, f, sort_keys=False)

        # The thresholds are the operating points, so keep them with the curve.
        np.savez(
            output_dir / "pr_curve.npz",
            precision=precision,
            recall=recall,
            thresholds=thresholds,
        )

        lines = [f"{k}: {v}" for k, v in results.items()]
        results_str = "\n".join(lines)
        print(results_str)
        with open(output_dir / "results.txt", "w") as f:
            f.write(results_str + "\n")

        if precision.size:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(recall, precision, lw=1.5)
            ax.axhline(results["chance_pr_auc"], color="gray", ls="--", lw=1,
                       label="chance")
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{run or 'place recognition'}  PR-AUC = {auc:.4f}")
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(output_dir / "pr_curve.png", dpi=150)
            plt.close(fig)

        return results


def cross_view_pr_auc(
    params,
    output_dir,
    run=None,
    skip_segmentation=False,
    skip_aerial=False,
    skip_ground=False,
    aerial_dir=None,
    ground_dir=None,
):
    """Run the place-recognition PR-AUC pipeline.

    1. If not skip_segmentation: run cross_view_matching with skip_match=True
       to create submaps with descriptors.
    2. Load submaps from disk.
    3. Compute the similarity matrix and the GT patch labels.
    4. Sweep the similarity threshold and save the curve, its AUC and the plot.

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

    gt_patches = CrossViewPlaceRecognitionPipeline.compute_gt_patches(
        ground_submaps,
        aerial_submaps,
        ground_keys,
        aerial_keys,
        aerial_patch_params,
        gt_pose_data=gt_pose_data,
    )

    pipeline = CrossViewPRAUCPipeline()
    y_true = pipeline.labels(ground_keys, aerial_keys, gt_patches)
    precision, recall, thresholds = pipeline.pr_curve(sim_matrix, y_true)
    return pipeline.save_results(
        precision,
        recall,
        thresholds,
        y_true,
        pipeline.valid_mask(sim_matrix),
        os.path.join(output_dir, "pr_auc"),
        run=run,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precision-recall AUC for cross-view place recognition"
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

    cross_view_pr_auc(
        args.params,
        args.output,
        run=args.run,
        skip_segmentation=args.skip_segmentation,
        skip_aerial=args.skip_aerial,
        skip_ground=args.skip_ground,
        aerial_dir=args.aerial,
        ground_dir=args.ground,
    )
