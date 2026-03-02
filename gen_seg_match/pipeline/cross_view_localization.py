import argparse
import logging
import os
import pathlib
import pickle
from dataclasses import dataclass
from typing import List, Optional

import clipperpy
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from gen_seg_match.map3d.submap import Submap
from gen_seg_match.params import CrossViewLocalizationParams
from gen_seg_match.pipeline.cross_view_matching import cross_view_matching
from gen_seg_match.pipeline.data import CrossViewLocalizationData
from gen_seg_match.pipeline.result import PoseEstimationResultMatrix
from gen_seg_match.params.data_params import CrossViewLocalizationDataParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SE(2) helpers
# ---------------------------------------------------------------------------


def se3_to_se2(T: np.ndarray) -> np.ndarray:
    """Project a 4x4 SE(3) matrix to a 3x3 SE(2) matrix (x, y, yaw)."""
    yaw = np.arctan2(T[1, 0], T[0, 0])
    return se2_from_xytheta(T[0, 3], T[1, 3], yaw)


def se2_to_se3(T2: np.ndarray) -> np.ndarray:
    """Embed a 3x3 SE(2) matrix into a 4x4 SE(3) matrix (z=0, roll=pitch=0)."""
    T = np.eye(4)
    T[:2, :2] = T2[:2, :2]
    T[0, 3] = T2[0, 2]
    T[1, 3] = T2[1, 2]
    return T


def se2_from_xytheta(x: float, y: float, theta: float) -> np.ndarray:
    """Construct a 3x3 SE(2) matrix from x, y, yaw."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, x],
            [s, c, y],
            [0, 0, 1],
        ]
    )


def yaw_from_se2(T2: np.ndarray) -> float:
    """Extract yaw angle from a 3x3 SE(2) matrix."""
    return np.arctan2(T2[1, 0], T2[0, 0])


def average_se2(transforms: List[np.ndarray]) -> np.ndarray:
    """Average a list of SE(2) transforms (mean x,y; circular mean yaw)."""
    xs = [T[0, 2] for T in transforms]
    ys = [T[1, 2] for T in transforms]
    yaws = [yaw_from_se2(T) for T in transforms]
    mean_yaw = np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws)))
    return se2_from_xytheta(np.mean(xs), np.mean(ys), mean_yaw)


# ---------------------------------------------------------------------------
# CrossViewLocalization pipeline
# ---------------------------------------------------------------------------


@dataclass
class CrossViewLocalization:
    localization_params: CrossViewLocalizationParams

    def localize(
        self,
        match_output_dir: str,
        data: CrossViewLocalizationData,
        output_dir: str,
    ) -> Optional[np.ndarray]:
        """Run the full localization pipeline.

        Returns the 4x4 SE(3) T_utm_odom, or None if localization fails.
        """
        match_output_dir = pathlib.Path(match_output_dir)
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        candidates = self._load_candidates(match_output_dir, data)
        if len(candidates) == 0:
            logger.warning("No candidates passed filtering — returning None.")
            return None

        M, C = self._build_affinity_matrix(candidates)
        self._visualize_affinity_matrix(M, C, candidates, output_dir)
        inlier_indices = self._run_clipper(M, C)
        if len(inlier_indices) == 0:
            logger.warning("CLIPPER returned empty solution — returning None.")
            return None

        if len(inlier_indices) == 1:
            logger.warning("Only a single inlier — using it directly.")

        T_utm_odom = self._average_inlier_transforms(candidates, inlier_indices)
        self._save_results(T_utm_odom, candidates, inlier_indices, output_dir)
        self._visualize_and_report(
            T_utm_odom, data, candidates, inlier_indices, output_dir
        )
        return T_utm_odom

    # ------------------------------------------------------------------
    # Step 2: Load candidates
    # ------------------------------------------------------------------

    def _load_candidates(
        self,
        match_output_dir: pathlib.Path,
        data: CrossViewLocalizationData,
    ) -> List[dict]:
        """Load per-instance matching results and build candidate list."""
        segments_dir = match_output_dir / "segments"
        ground_dir = match_output_dir.parent / "ground" / "segments"
        aerial_dir = match_output_dir.parent / "aerial" / "segments"

        # Load aerial pose (same for all patches)
        aerial_files = sorted(aerial_dir.glob("*.pkl"))
        if len(aerial_files) == 0:
            logger.warning("No aerial submaps found.")
            return []
        aerial_submap = Submap.load(aerial_files[0])
        pose_flu = aerial_submap.pose  # T_utm_aerial (FLU-convention pose)

        min_assoc = self.localization_params.min_num_associations
        candidates = []

        result_files = sorted(segments_dir.glob("ground_*_results_matrix.pkl.npz"))
        for result_file in result_files:
            # Extract ground key from filename
            stem = result_file.name  # e.g. ground_3_results_matrix.pkl.npz
            ground_key = stem.replace("ground_", "").replace(
                "_results_matrix.pkl.npz", ""
            )

            # Load ground submap to get camera_pose
            ground_submap_path = ground_dir / f"{ground_key}.pkl"
            if not ground_submap_path.exists():
                logger.warning(
                    f"Ground submap {ground_submap_path} not found, skipping."
                )
                continue
            ground_submap = Submap.load(ground_submap_path)
            ground_camera_pose = ground_submap.metadata["camera_pose"]  # T_odom_camera

            # Load results matrix
            results_matrix = PoseEstimationResultMatrix.load(str(result_file))

            for idx in np.ndindex(results_matrix.shape):
                result = results_matrix[idx]
                if result.num_associations < min_assoc:
                    continue
                T_i_j_hat = result.T_i_j_hat
                if np.any(np.isnan(T_i_j_hat)):
                    continue

                # Compute T_odom_ground
                if data.T_camera_flu is not None:
                    T_odom_ground = ground_camera_pose @ data.T_camera_flu
                else:
                    T_odom_ground = ground_camera_pose

                # T_i_j_hat is T_aerialmeter_ground (maps ground to aerial-meter frame)
                # T_utm_odom = pose_flu @ T_hat @ inv(T_odom_ground)
                T_utm_odom_4x4 = pose_flu @ T_i_j_hat @ np.linalg.inv(T_odom_ground)
                T_utm_odom_se2 = se3_to_se2(T_utm_odom_4x4)

                candidates.append(
                    {
                        "T_utm_odom_se2": T_utm_odom_se2,
                        "T_i_j_hat": T_i_j_hat,
                        "aerial_pose": pose_flu,
                        "ground_camera_pose": ground_camera_pose,
                        "T_odom_ground": T_odom_ground,
                        "ground_key": ground_key,
                        "aerial_key": f"{idx[0]}_{idx[1]}",
                        "num_associations": result.num_associations,
                    }
                )

        logger.info(f"Loaded {len(candidates)} candidates.")
        return candidates

    # ------------------------------------------------------------------
    # Step 3: Build affinity matrix
    # ------------------------------------------------------------------

    def _build_affinity_matrix(self, candidates: List[dict]) -> tuple:
        """Build CLIPPER affinity (M) and constraint (C) matrices."""
        N = len(candidates)
        M = np.zeros((N, N))
        C = np.ones((N, N))

        sigma_r = self.localization_params.rotation_consistency_sigma
        sigma_t = self.localization_params.translation_consistency_sigma
        eps_r = self.localization_params.rotation_consistency_epsilon
        eps_t = self.localization_params.translation_consistency_epsilon

        for i in range(N):
            M[i, i] = 1.0
            ci = candidates[i]
            for j in range(i + 1, N):
                cj = candidates[j]

                # Relative transform via odometry
                odom_relative = se3_to_se2(
                    np.linalg.inv(ci["T_odom_ground"]) @ cj["T_odom_ground"]
                )

                # Relative transform via cross-view
                cv_relative = se3_to_se2(
                    np.linalg.inv(ci["T_i_j_hat"])
                    @ np.linalg.inv(ci["aerial_pose"])
                    @ cj["aerial_pose"]
                    @ cj["T_i_j_hat"]
                )

                # Error: should be identity if consistent
                error = np.linalg.inv(odom_relative) @ cv_relative
                rot_err = abs(yaw_from_se2(error))
                trans_err = np.linalg.norm(error[:2, 2])

                if rot_err < eps_r and trans_err < eps_t:
                    score = np.exp(-0.5 * (rot_err / sigma_r) ** 2) * np.exp(
                        -0.5 * (trans_err / sigma_t) ** 2
                    )
                    M[i, j] = score
                    M[j, i] = score
                else:
                    C[i, j] = 0
                    C[j, i] = 0

        return M, C

    # ------------------------------------------------------------------
    # Step 4: CLIPPER
    # ------------------------------------------------------------------

    @staticmethod
    def _run_clipper(M: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Run CLIPPER on the affinity/constraint matrices."""
        clipper = clipperpy.CLIPPER(
            clipperpy.invariants.PairwiseInvariant(), clipperpy.Params()
        )
        clipper.set_matrix_data(M=M, C=C)
        clipper.solve()
        return np.array(clipper.get_solution().nodes)

    # ------------------------------------------------------------------
    # Step 5: Average inlier transforms
    # ------------------------------------------------------------------

    @staticmethod
    def _average_inlier_transforms(
        candidates: List[dict], inlier_indices: np.ndarray
    ) -> np.ndarray:
        """Average inlier T_utm_odom estimates and return as 4x4 SE(3)."""
        inlier_se2s = [candidates[i]["T_utm_odom_se2"] for i in inlier_indices]
        T_avg_se2 = average_se2(inlier_se2s)
        return se2_to_se3(T_avg_se2)

    # ------------------------------------------------------------------
    # Diagnostics: affinity matrix heatmap
    # ------------------------------------------------------------------

    @staticmethod
    def _visualize_affinity_matrix(
        M: np.ndarray,
        C: np.ndarray,
        candidates: List[dict],
        output_dir: pathlib.Path,
    ):
        """Save a heatmap of the affinity matrix M for debugging."""
        N = M.shape[0]
        labels = [
            f"g{c['ground_key']}_a{c['aerial_key']}\n({c['num_associations']})"
            for c in candidates
        ]

        fig, axes = plt.subplots(1, 2, figsize=(8 + N * 0.3, 4 + N * 0.15))

        # Affinity matrix
        im0 = axes[0].imshow(M, cmap="viridis", vmin=0, vmax=1)
        axes[0].set_title(f"Affinity Matrix M  ({N}x{N})")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Constraint matrix
        im1 = axes[1].imshow(C, cmap="Reds", vmin=0, vmax=1)
        axes[1].set_title(f"Constraint Matrix C  ({N}x{N})")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.set_xlabel("Candidate")
            ax.set_ylabel("Candidate")
            if N <= 40:
                ax.set_xticks(range(N))
                ax.set_yticks(range(N))
                ax.set_xticklabels(labels, rotation=90, fontsize=max(4, 8 - N // 10))
                ax.set_yticklabels(labels, fontsize=max(4, 8 - N // 10))
            else:
                # Too many candidates for per-tick labels
                ax.set_xticks(range(0, N, max(1, N // 20)))
                ax.set_yticks(range(0, N, max(1, N // 20)))

        fig.tight_layout()
        fig.savefig(output_dir / "affinity_matrix.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Also print summary stats
        off_diag = M[np.triu_indices(N, k=1)]
        n_nonzero = np.count_nonzero(off_diag)
        n_constrained = int(C[np.triu_indices(N, k=1)].sum())
        logger.info(
            f"Affinity matrix: {N} candidates, "
            f"{n_nonzero} compatible pairs (M>0), "
            f"{n_constrained} constrained pairs (C=1), "
            f"max off-diag M={off_diag.max():.4f}"
            if len(off_diag) > 0
            else ""
        )
        print(
            f"Affinity matrix: {N} candidates, "
            f"{n_nonzero}/{N * (N - 1) // 2} compatible pairs, "
            f"{n_constrained}/{N * (N - 1) // 2} constrained pairs"
        )
        if len(off_diag) > 0 and n_nonzero > 0:
            print(
                f"  M off-diagonal: max={off_diag.max():.4f}, "
                f"mean(nonzero)={off_diag[off_diag > 0].mean():.4f}"
            )

    # ------------------------------------------------------------------
    # Step 6: Save, visualize, and report
    # ------------------------------------------------------------------

    def _save_results(
        self,
        T_utm_odom: np.ndarray,
        candidates: List[dict],
        inlier_indices: np.ndarray,
        output_dir: pathlib.Path,
    ):
        np.save(output_dir / "T_utm_odom.npy", T_utm_odom)
        with open(output_dir / "candidates.pkl", "wb") as f:
            pickle.dump(candidates, f)
        np.save(output_dir / "inlier_indices.npy", inlier_indices)
        logger.info(
            f"Saved T_utm_odom, {len(candidates)} candidates, "
            f"{len(inlier_indices)} inliers to {output_dir}"
        )

    def _visualize_and_report(
        self,
        T_utm_odom: np.ndarray,
        data: CrossViewLocalizationData,
        candidates: List[dict],
        inlier_indices: np.ndarray,
        output_dir: pathlib.Path,
    ):
        """Plot full trajectory on aerial image and compute error metrics."""
        ground_map = data.ground_map
        traj_times = np.array(ground_map.times)
        # ground_map.trajectory contains T_odom_camera poses
        traj_odom = ground_map.trajectory

        # Apply T_camera_flu if available so we track the FLU body position
        T_cam_flu = data.T_camera_flu  # may be None

        # Build estimated UTM positions & yaws for the full trajectory
        est_utm_positions = np.full((len(traj_odom), 2), np.nan)
        est_yaws = np.full(len(traj_odom), np.nan)
        for i, T_odom_cam in enumerate(traj_odom):
            if T_cam_flu is not None:
                T_utm_body = T_utm_odom @ T_odom_cam @ T_cam_flu
            else:
                T_utm_body = T_utm_odom @ T_odom_cam
            est_utm_positions[i] = T_utm_body[:2, 3]
            est_yaws[i] = np.arctan2(T_utm_body[1, 0], T_utm_body[0, 0])

        # Build GT UTM positions & yaws at the same timestamps
        gt_utm_positions = None
        gt_yaws = None
        if data.gt_pose_data is not None:
            gt_utm_positions = np.full((len(traj_times), 2), np.nan)
            gt_yaws = np.full(len(traj_times), np.nan)
            for i, t in enumerate(traj_times):
                try:
                    gt_pose = data.gt_pose_data.pose(t)
                    if T_cam_flu is not None:
                        gt_body = gt_pose @ T_cam_flu
                    else:
                        gt_body = gt_pose
                    gt_utm_positions[i] = gt_body[:2, 3]
                    gt_yaws[i] = np.arctan2(gt_body[1, 0], gt_body[0, 0])
                except Exception:
                    pass

        # UTM to pixel conversion
        origin_x, origin_y = data.aerial_img_origin
        pixel_len_m = data.aerial_img_scale

        def utm_to_pixel(xy_utm):
            x_px = (xy_utm[:, 0] - origin_x) / pixel_len_m
            y_px = (origin_y - xy_utm[:, 1]) / pixel_len_m
            return np.column_stack([x_px, y_px])

        est_px = utm_to_pixel(est_utm_positions)

        # Plot on aerial image
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        aerial_img = data.aerial_img
        ds = max(1, min(aerial_img.shape[0], aerial_img.shape[1]) // 2000)
        aerial_small = aerial_img[::ds, ::ds]
        ax.imshow(
            cv.cvtColor(aerial_small, cv.COLOR_BGR2RGB),
            extent=[0, aerial_img.shape[1], aerial_img.shape[0], 0],
        )

        ax.plot(
            est_px[:, 0],
            est_px[:, 1],
            "b-",
            linewidth=1.5,
            label="Estimated",
        )
        if gt_utm_positions is not None:
            gt_px = utm_to_pixel(gt_utm_positions)
            valid = ~np.any(np.isnan(gt_utm_positions), axis=1)
            ax.plot(
                gt_px[valid, 0],
                gt_px[valid, 1],
                "g-",
                linewidth=1.5,
                label="Ground Truth",
            )

        # Mark inlier candidate positions
        inlier_utm = []
        for idx in inlier_indices:
            c = candidates[idx]
            T_odom_cam = c["ground_camera_pose"]
            if T_cam_flu is not None:
                T_utm_body = T_utm_odom @ T_odom_cam @ T_cam_flu
            else:
                T_utm_body = T_utm_odom @ T_odom_cam
            inlier_utm.append(T_utm_body[:2, 3])
        inlier_utm = np.array(inlier_utm)
        inlier_px = utm_to_pixel(inlier_utm)
        ax.plot(
            inlier_px[:, 0],
            inlier_px[:, 1],
            "r*",
            markersize=8,
            label=f"Inliers ({len(inlier_indices)})",
        )

        ax.legend()
        ax.set_title("Cross-View Localization")
        fig.savefig(output_dir / "trajectory.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Compute error metrics
        results_lines = []
        results_lines.append(f"Number of candidates: {len(candidates)}")
        results_lines.append(f"Number of inliers: {len(inlier_indices)}")
        results_lines.append(f"T_utm_odom:\n{T_utm_odom}")

        if gt_utm_positions is not None:
            valid = ~np.any(np.isnan(gt_utm_positions), axis=1)
            if np.any(valid):
                trans_errors = np.linalg.norm(
                    est_utm_positions[valid] - gt_utm_positions[valid], axis=1
                )
                rmse_trans = np.sqrt(np.mean(trans_errors**2))
                results_lines.append(f"Translation RMSE (m): {rmse_trans:.3f}")
                results_lines.append(
                    f"Translation mean error (m): {np.mean(trans_errors):.3f}"
                )
                results_lines.append(
                    f"Translation max error (m): {np.max(trans_errors):.3f}"
                )

                # Per-pose 2D heading error
                yaw_valid = valid & ~np.isnan(est_yaws) & ~np.isnan(gt_yaws)
                if np.any(yaw_valid):
                    yaw_diff = est_yaws[yaw_valid] - gt_yaws[yaw_valid]
                    yaw_errors = np.abs(np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff)))
                    rmse_yaw = np.sqrt(np.mean(yaw_errors**2))
                    results_lines.append(
                        f"Heading RMSE (deg): {np.rad2deg(rmse_yaw):.3f}"
                    )
                    results_lines.append(
                        f"Heading mean error (deg): "
                        f"{np.rad2deg(np.mean(yaw_errors)):.3f}"
                    )
                    results_lines.append(
                        f"Heading max error (deg): {np.rad2deg(np.max(yaw_errors)):.3f}"
                    )

        results_str = "\n".join(results_lines)
        print(results_str)
        with open(output_dir / "results.txt", "w") as f:
            f.write(results_str + "\n")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def cross_view_localization(params, output_dir, skip_matching=False):
    """Run cross-view matching (optionally) then localization."""
    output_dir = str(output_dir)

    if not skip_matching:
        cross_view_matching(params, output_dir)

    match_output_dir = os.path.join(output_dir, "match")

    loc_params = CrossViewLocalizationParams.load(params)
    data_params = CrossViewLocalizationDataParams.load(params)
    data = CrossViewLocalizationData.from_params(data_params)

    runner = CrossViewLocalization(localization_params=loc_params)
    loc_output_dir = os.path.join(output_dir, "localization")
    T_utm_odom = runner.localize(match_output_dir, data, loc_output_dir)
    return T_utm_odom


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        required=True,
        help="Path to params directory or file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Skip cross-view matching (use existing results).",
    )
    parser.add_argument(
        "--skip-aerial",
        action="store_true",
        help="Skip aerial segmentation (passed to cross_view_matching).",
    )
    parser.add_argument(
        "--skip-ground",
        action="store_true",
        help="Skip ground segmentation (passed to cross_view_matching).",
    )
    parser.add_argument(
        "--skip-match",
        action="store_true",
        help="Skip segment matching (passed to cross_view_matching).",
    )
    args = parser.parse_args()

    if not args.skip_matching:
        cross_view_matching(
            args.params,
            args.output,
            skip_aerial=args.skip_aerial,
            skip_ground=args.skip_ground,
            skip_match=args.skip_match,
        )

    match_output_dir = os.path.join(args.output, "match")
    loc_params = CrossViewLocalizationParams.load(args.params)
    data_params = CrossViewLocalizationDataParams.load(args.params)
    data = CrossViewLocalizationData.from_params(data_params)

    runner = CrossViewLocalization(localization_params=loc_params)
    loc_output_dir = os.path.join(args.output, "localization")
    runner.localize(match_output_dir, data, loc_output_dir)
