import argparse
import logging
import os
import pathlib
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from meridian.cross_view.rpgo import (
    CrossViewRPGO,
    CrossViewRPGOResult,
    pose_data_from_trajectory,
    se2_from_xytheta,
    se2_to_se3,
    se3_to_se2,
    yaw_from_se2,
)
from meridian.map3d.submap import Submap
from meridian.params import CrossViewRPGOParams, CrossViewVisualizationParams
from meridian.pipeline.cross_view_matching import (
    CrossViewMatching,
    CrossViewMatchingPipeline,
    CrossViewMatchResult,
    cross_view_matching,
)
from meridian.pipeline.data import CrossViewLocalizationData
from meridian.pipeline.result import PoseEstimationResultMatrix
from meridian.params.data_params import CrossViewLocalizationDataParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate construction helper
# ---------------------------------------------------------------------------


def build_candidates_from_match_result(
    match_result: CrossViewMatchResult,
    ground_submaps: Dict[str, Submap],
    aerial_submaps: Dict[str, Submap],
    data: CrossViewLocalizationData,
    min_assoc: int,
) -> List[dict]:
    """Build candidate list from in-memory CrossViewMatchResult."""
    first_aerial_key = next(iter(aerial_submaps))
    pose_flu = aerial_submaps[first_aerial_key].pose

    candidates = []

    for ground_key, results_matrix in match_result.results.items():
        if ground_key not in ground_submaps:
            continue
        ground_submap = ground_submaps[ground_key]
        ground_camera_pose = ground_submap.metadata["camera_pose"]

        for idx in np.ndindex(results_matrix.shape):
            cell = results_matrix[idx]
            hypotheses = cell if isinstance(cell, list) else [cell]
            for result in hypotheses:
                if result.num_associations < min_assoc:
                    continue
                T_i_j_hat = result.T_i_j_hat
                if np.any(np.isnan(T_i_j_hat)):
                    continue

                if data.T_camera_flu is not None:
                    T_odom_ground = ground_camera_pose @ data.T_camera_flu
                else:
                    T_odom_ground = ground_camera_pose

                T_full = pose_flu @ T_i_j_hat

                if np.linalg.det(T_full[:2, :2]) < 0:
                    logger.debug(
                        "Rejecting candidate with reflected rotation "
                        f"(det(R_2x2)={np.linalg.det(T_full[:2, :2]):.4f})"
                    )
                    continue

                yaw = np.arctan2(T_full[1, 0], T_full[0, 0])

                pixel_len_m = data.aerial_img_scale
                body_col = T_i_j_hat[0, 3] / pixel_len_m
                body_row = T_i_j_hat[1, 3] / pixel_len_m
                if (
                    data.geotiff_transform is not None
                    and data.native_crs is not None
                    and data.utm_crs is not None
                    and data.native_crs != data.utm_crs
                ):
                    utm_x, utm_y = data.aerial_pixel_to_utm(body_col, body_row)
                else:
                    utm_x = T_full[0, 3]
                    utm_y = T_full[1, 3]

                T_utm_body_se2 = se2_from_xytheta(utm_x, utm_y, yaw)

                T_utm_odom_4x4 = se2_to_se3(T_utm_body_se2) @ np.linalg.inv(
                    T_odom_ground
                )
                T_utm_odom_se2 = se3_to_se2(T_utm_odom_4x4)

                T_utm_odom_gt_se2 = None
                if data.gt_pose_data is not None:
                    try:
                        gt_pose = data.gt_pose_data.pose(ground_submap.time)
                        if data.T_camera_flu is not None:
                            gt_body = gt_pose @ data.T_camera_flu
                        else:
                            gt_body = gt_pose
                        T_utm_odom_gt = gt_body @ np.linalg.inv(T_odom_ground)
                        T_utm_odom_gt_se2 = se3_to_se2(T_utm_odom_gt)
                    except Exception:
                        pass

                candidates.append(
                    {
                        "T_utm_odom_se2": T_utm_odom_se2,
                        "T_utm_body_se2": T_utm_body_se2,
                        "T_i_j_hat": T_i_j_hat,
                        "T_i_j": result.T_i_j,
                        "aerial_pose": pose_flu,
                        "ground_camera_pose": ground_camera_pose,
                        "T_odom_ground": T_odom_ground,
                        "ground_key": ground_key,
                        "aerial_key": f"{idx[0]}_{idx[1]}",
                        "num_associations": result.num_associations,
                        "ground_submap_time": ground_submap.time,
                        "T_utm_odom_gt_se2": T_utm_odom_gt_se2,
                        "count": getattr(result, "count", 1),
                    }
                )

    return candidates


# ---------------------------------------------------------------------------
# CrossViewLocalization pipeline
# ---------------------------------------------------------------------------


@dataclass
class CrossViewLocalization:
    rpgo_params: CrossViewRPGOParams
    viz_params: CrossViewVisualizationParams = field(
        default_factory=CrossViewVisualizationParams
    )

    def localize(
        self,
        match_output_dir: str,
        data: CrossViewLocalizationData,
        output_dir: str,
        pipeline: Optional[CrossViewMatchingPipeline] = None,
        aerial_submaps: Optional[Dict[str, Submap]] = None,
        ground_submaps: Optional[Dict[str, Submap]] = None,
        aerial_img: np.ndarray = None,
        main_output_dir: str = None,
        save_viz: bool = True,
    ) -> Optional[CrossViewRPGOResult]:
        """Run the full localization pipeline.

        Args:
            match_output_dir: Directory containing initial match results.
            data: Localization data (aerial image, ground map, GT, etc.).
            output_dir: Output directory for results.
            pipeline: Optional CrossViewMatchingPipeline (needed for rerun).
            aerial_submaps: Optional preloaded aerial submaps (needed for rerun).
            ground_submaps: Optional preloaded ground submaps (needed for rerun).
            aerial_img: Optional aerial image for visualizations during rerun.
            main_output_dir: Top-level output dir, used to find ground dense
                points when rerunning from a subdirectory.

        Returns CrossViewRPGOResult, or None if localization fails.
        """
        match_output_dir = pathlib.Path(match_output_dir)
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        candidates = self._load_candidates(match_output_dir, data)
        if len(candidates) == 0:
            logger.warning("No candidates passed filtering — returning None.")
            return None

        ground_map = data.ground_map
        trajectory = ground_map.trajectory
        times = np.array(ground_map.times)

        rpgo = CrossViewRPGO(params=self.rpgo_params)
        result = rpgo.solve(candidates, trajectory, times, data.T_camera_flu)

        # TODO - do we want to visualize affinity matrix? Slow and can crash if too big rn
        # if result.M is not None:
        #     self._visualize_affinity_matrix(result.M, result.C, candidates, output_dir)

        if not result.success:
            logger.warning("RPGO solve failed — returning None.")
            return None

        self._save_results(result, output_dir)
        self._visualize_and_report(result, data, output_dir, self.viz_params)

        # --- Rerun with known rotation ---
        if (
            self.rpgo_params.rerun_match_with_known_rot
            and pipeline is not None
            and aerial_submaps is not None
            and ground_submaps is not None
            and result.optimized_trajectory is not None
        ):
            rerun_result = self._rerun_with_known_rotation(
                result,
                pipeline,
                aerial_submaps,
                ground_submaps,
                data,
                trajectory,
                times,
                output_dir,
                aerial_img,
                main_output_dir,
                save_viz=save_viz,
            )
            if rerun_result is not None:
                return rerun_result

        return result

    # ------------------------------------------------------------------
    # Rerun matching with known rotation
    # ------------------------------------------------------------------

    def _rerun_with_known_rotation(
        self,
        initial_result: CrossViewRPGOResult,
        pipeline: CrossViewMatchingPipeline,
        aerial_submaps: Dict[str, Submap],
        ground_submaps: Dict[str, Submap],
        data: CrossViewLocalizationData,
        trajectory: List[np.ndarray],
        times: np.ndarray,
        output_dir: pathlib.Path,
        aerial_img: np.ndarray = None,
        main_output_dir: str = None,
        save_viz: bool = True,
    ) -> Optional[CrossViewRPGOResult]:
        """Re-run matching using optimized trajectory rotation, then re-run RPGO."""
        logger.info("Rerunning matching with known rotation from initial RPGO...")

        # The optimized trajectory is in body/FLU frame (T_utm_body), but the
        # matching pipeline expects camera-frame poses (T_utm_camera).  Convert
        # back so that _match_single_pair / _find_max_intersection_patch don't
        # double-apply T_camera_flu.
        if data.T_camera_flu is not None:
            T_flu_camera = np.linalg.inv(data.T_camera_flu)
            camera_trajectory = [
                T_utm_body @ T_flu_camera
                for T_utm_body in initial_result.optimized_trajectory
            ]
        else:
            camera_trajectory = initial_result.optimized_trajectory
        optimized_pose_data = pose_data_from_trajectory(camera_trajectory, times)

        rerun_match_dir = output_dir / "match_rerun"

        # Compute ground_dense_dir pointing to existing dense points
        ground_dense_dir = None
        if main_output_dir is not None:
            ground_dense_dir = pathlib.Path(main_output_dir) / "ground" / "segments"

        # Optionally switch to CLIPPER for pass 2
        use_clipper_pass2 = (
            pipeline.algorithm.pipeline_params.clipper_pass2
            and pipeline.algorithm.matcher.params.solver == "langevin"
        )
        if use_clipper_pass2:
            pipeline.algorithm.matcher.set_solver("clipper")

        # Re-run matching with full viz output via pipeline
        # reference_trajectory provides the rotation constraint from PGO,
        # gt_trajectory provides actual ground truth for T_i_j visualization/errors
        match_result = pipeline.run_match(
            aerial_submaps,
            ground_submaps,
            reference_trajectory=optimized_pose_data,
            output_dir=rerun_match_dir,
            aerial_img=aerial_img,
            T_camera_flu=data.T_camera_flu,
            matching_mode="max_intersection",
            translation_only=True,
            ground_dense_dir=ground_dense_dir,
            local_to_pixel_fn=data.aerial_local_to_pixel
            if data.geotiff_transform is not None
            else None,
            save_viz=save_viz,
            gt_trajectory=data.gt_pose_data,
        )

        # Build candidates from in-memory match result
        candidates = self._load_candidates_from_result(
            match_result, ground_submaps, aerial_submaps, data
        )

        if len(candidates) == 0:
            logger.warning(
                "Rerun: no candidates passed filtering — keeping initial result."
            )
            return None

        # Re-run RPGO
        rpgo = CrossViewRPGO(params=self.rpgo_params)
        rerun_result = rpgo.solve(candidates, trajectory, times, data.T_camera_flu)

        rerun_output_dir = output_dir / "rerun"
        rerun_output_dir.mkdir(parents=True, exist_ok=True)

        # TODO - can crash on large matrices (OOM in Tk/matplotlib)
        # if rerun_result.M is not None:
        #     self._visualize_affinity_matrix(
        #         rerun_result.M, rerun_result.C, candidates, rerun_output_dir
        #     )

        if not rerun_result.success:
            logger.warning("Rerun RPGO failed — keeping initial result.")
            return None

        self._save_results(rerun_result, rerun_output_dir)
        self._visualize_and_report(
            rerun_result, data, rerun_output_dir, self.viz_params
        )

        logger.info(
            f"Rerun complete: {len(rerun_result.inlier_indices)} inliers "
            f"(initial had {len(initial_result.inlier_indices)})"
        )
        return rerun_result

    # ------------------------------------------------------------------
    # Load candidates
    # ------------------------------------------------------------------

    def _load_candidates(
        self,
        match_output_dir: pathlib.Path,
        data: CrossViewLocalizationData,
    ) -> List[dict]:
        """Load per-instance matching results and build candidate list."""
        segments_dir = match_output_dir / "segments"

        # Resolve aerial/ground dirs, checking for redirect files written
        # when --aerial / --ground flags point to external directories.
        aerial_txt = match_output_dir.parent / "aerial.txt"
        if aerial_txt.exists():
            aerial_dir = pathlib.Path(aerial_txt.read_text().strip()) / "segments"
        else:
            aerial_dir = match_output_dir.parent / "aerial" / "segments"

        ground_txt = match_output_dir.parent / "ground.txt"
        if ground_txt.exists():
            ground_dir = pathlib.Path(ground_txt.read_text().strip()) / "segments"
        else:
            ground_dir = match_output_dir.parent / "ground" / "segments"

        # Load aerial pose (same for all patches)
        aerial_files = sorted(aerial_dir.glob("*.pkl"))
        if len(aerial_files) == 0:
            logger.warning("No aerial submaps found.")
            return []
        aerial_submap = Submap.load(aerial_files[0])
        pose_flu = aerial_submap.pose  # T_utm_aerial (FLU-convention pose)

        min_assoc = self.rpgo_params.min_num_associations
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
                cell = results_matrix[idx]
                hypotheses = cell if isinstance(cell, list) else [cell]
                for result in hypotheses:
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

                    # Compute T_utm_body with proper CRS conversion.
                    # pose_flu's linear mapping is inaccurate for non-UTM GeoTIFFs
                    # (e.g. EPSG:3857), so convert via pixel → native CRS → UTM.
                    T_full = pose_flu @ T_i_j_hat

                    # Reject improper rotations (reflections) from registration.
                    if np.linalg.det(T_full[:2, :2]) < 0:
                        logger.debug(
                            "Rejecting candidate with reflected rotation "
                            f"(det(R_2x2)={np.linalg.det(T_full[:2, :2]):.4f})"
                        )
                        continue

                    # Yaw from pose_flu @ T_i_j_hat (rotation approx correct)
                    yaw = np.arctan2(T_full[1, 0], T_full[0, 0])

                    # Translation via proper CRS pipeline
                    pixel_len_m = data.aerial_img_scale
                    body_col = T_i_j_hat[0, 3] / pixel_len_m
                    body_row = T_i_j_hat[1, 3] / pixel_len_m
                    if (
                        data.geotiff_transform is not None
                        and data.native_crs is not None
                        and data.utm_crs is not None
                        and data.native_crs != data.utm_crs
                    ):
                        utm_x, utm_y = data.aerial_pixel_to_utm(body_col, body_row)
                    else:
                        utm_x = T_full[0, 3]
                        utm_y = T_full[1, 3]

                    T_utm_body_se2 = se2_from_xytheta(utm_x, utm_y, yaw)

                    # Derive T_utm_odom from corrected T_utm_body
                    T_utm_odom_4x4 = se2_to_se3(T_utm_body_se2) @ np.linalg.inv(
                        T_odom_ground
                    )
                    T_utm_odom_se2 = se3_to_se2(T_utm_odom_4x4)

                    # Compute GT T_utm_odom for gt_inlier selection
                    T_utm_odom_gt_se2 = None
                    if data.gt_pose_data is not None:
                        try:
                            gt_pose = data.gt_pose_data.pose(ground_submap.time)
                            if data.T_camera_flu is not None:
                                gt_body = gt_pose @ data.T_camera_flu
                            else:
                                gt_body = gt_pose
                            T_utm_odom_gt = gt_body @ np.linalg.inv(T_odom_ground)
                            T_utm_odom_gt_se2 = se3_to_se2(T_utm_odom_gt)
                        except Exception:
                            pass

                    candidates.append(
                        {
                            "T_utm_odom_se2": T_utm_odom_se2,
                            "T_utm_body_se2": T_utm_body_se2,
                            "T_i_j_hat": T_i_j_hat,
                            "T_i_j": result.T_i_j,
                            "aerial_pose": pose_flu,
                            "ground_camera_pose": ground_camera_pose,
                            "T_odom_ground": T_odom_ground,
                            "ground_key": ground_key,
                            "aerial_key": f"{idx[0]}_{idx[1]}",
                            "num_associations": result.num_associations,
                            "ground_submap_time": ground_submap.time,
                            "T_utm_odom_gt_se2": T_utm_odom_gt_se2,
                            "count": getattr(result, "count", 1),
                        }
                    )

        logger.info(f"Loaded {len(candidates)} candidates.")
        return candidates

    def _load_candidates_from_result(
        self,
        match_result: CrossViewMatchResult,
        ground_submaps: Dict[str, Submap],
        aerial_submaps: Dict[str, Submap],
        data: CrossViewLocalizationData,
    ) -> List[dict]:
        """Build candidate list from in-memory CrossViewMatchResult."""
        min_assoc = (
            self.rpgo_params.min_num_associations_rerun
            if self.rpgo_params.min_num_associations_rerun is not None
            else self.rpgo_params.min_num_associations
        )
        return build_candidates_from_match_result(
            match_result, ground_submaps, aerial_submaps, data, min_assoc
        )

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
    # Save, visualize, and report
    # ------------------------------------------------------------------

    @staticmethod
    def _save_results(
        result: CrossViewRPGOResult,
        output_dir: pathlib.Path,
    ):
        np.save(output_dir / "T_utm_odom.npy", result.T_utm_odom)
        with open(output_dir / "candidates.pkl", "wb") as f:
            pickle.dump(result.candidates, f)
        np.save(output_dir / "inlier_indices.npy", result.inlier_indices)
        with open(output_dir / "optimized_trajectory.pkl", "wb") as f:
            pickle.dump(result.optimized_trajectory, f)
        logger.info(
            f"Saved T_utm_odom, {len(result.candidates)} candidates, "
            f"{len(result.inlier_indices)} inliers, "
            f"optimized_trajectory to {output_dir}"
        )

    @staticmethod
    def _visualize_and_report(
        result: CrossViewRPGOResult,
        data: CrossViewLocalizationData,
        output_dir: pathlib.Path,
        viz_params: CrossViewVisualizationParams = None,
    ):
        """Plot full trajectory on aerial image and compute error metrics."""
        if viz_params is None:
            viz_params = CrossViewVisualizationParams()
        optimized_traj = result.optimized_trajectory
        if result.times is not None:
            traj_times = np.asarray(result.times)
        else:
            traj_times = np.array(data.ground_map.times)

        # Build estimated UTM positions & yaws from optimized trajectory
        est_utm_positions = np.full((len(optimized_traj), 2), np.nan)
        est_yaws = np.full(len(optimized_traj), np.nan)
        for i, T_utm_body in enumerate(optimized_traj):
            est_utm_positions[i] = T_utm_body[:2, 3]
            est_yaws[i] = np.arctan2(T_utm_body[1, 0], T_utm_body[0, 0])

        # Build GT UTM positions & yaws at the same timestamps
        T_cam_flu = data.T_camera_flu
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

        if data.geotiff_transform is not None:

            def utm_to_pixel(xy_utm):
                return data.aerial_utm_to_pixel(np.atleast_2d(xy_utm))
        else:

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
            color=viz_params.estimated_trajectory_color,
            linestyle="-",
            linewidth=1.5,
            label="Estimated",
        )
        if gt_utm_positions is not None:
            gt_px = utm_to_pixel(gt_utm_positions)
            valid = ~np.any(np.isnan(gt_utm_positions), axis=1)
            ax.plot(
                gt_px[valid, 0],
                gt_px[valid, 1],
                color=viz_params.gt_trajectory_color,
                linestyle="-",
                linewidth=1.5,
                label="Ground Truth",
            )

        # Mark inlier candidate positions using optimized trajectory
        candidates = result.candidates
        inlier_indices = result.inlier_indices
        T_cam_flu = data.T_camera_flu

        inlier_utm = []
        raw_utm = []
        raw_yaws = []
        raw_ground_keys = []
        for idx in inlier_indices:
            c = candidates[idx]
            ground_time = c["ground_submap_time"]
            traj_idx = int(np.argmin(np.abs(traj_times - ground_time)))
            # Optimized position
            inlier_utm.append(optimized_traj[traj_idx][:2, 3])
            # Raw measurement position directly from registration (no lever arm)
            T_utm_body_se2 = c["T_utm_body_se2"]
            raw_utm.append(T_utm_body_se2[:2, 2])
            raw_yaws.append(yaw_from_se2(T_utm_body_se2))
            raw_ground_keys.append(c["ground_key"])

        inlier_utm = np.array(inlier_utm)
        raw_utm = np.array(raw_utm)
        inlier_px = utm_to_pixel(inlier_utm)
        raw_px = utm_to_pixel(raw_utm)

        # Axis length in pixels for raw measurement orientation
        axis_len_px = 12

        # Draw residual lines from optimized to raw measurement
        for i in range(len(inlier_indices)):
            ax.plot(
                [inlier_px[i, 0], raw_px[i, 0]],
                [inlier_px[i, 1], raw_px[i, 1]],
                "r-",
                linewidth=1.0,
                alpha=0.7,
            )
        # Draw stars at optimized positions (on top of lines)
        n_unique_submaps = len(set(raw_ground_keys))
        ax.plot(
            inlier_px[:, 0],
            inlier_px[:, 1],
            "m*",
            markersize=8,
            label=f"Inliers ({n_unique_submaps})",
        )
        # Draw raw measurement axes (x=red, y=green) and ground key labels
        for i in range(len(inlier_indices)):
            yaw = raw_yaws[i]
            ox, oy = raw_px[i, 0], raw_px[i, 1]
            # x-axis (red): UTM +x maps to pixel +x, UTM +y maps to pixel -y
            dx_x, dx_y = np.cos(yaw), -np.sin(yaw)
            ax.plot(
                [ox, ox + axis_len_px * dx_x],
                [oy, oy + axis_len_px * dx_y],
                "r-",
                linewidth=1.5,
            )
            # y-axis (green): 90° CCW from x in UTM, but pixel y is flipped
            dy_x, dy_y = -np.sin(yaw), -np.cos(yaw)
            ax.plot(
                [ox, ox + axis_len_px * dy_x],
                [oy, oy + axis_len_px * dy_y],
                "g-",
                linewidth=1.5,
            )
            # Ground submap label
            ax.text(
                ox + 4,
                oy - 4,
                raw_ground_keys[i],
                fontsize=4,
                color="r",
                ha="left",
                va="bottom",
            )

        ax.legend()
        ax.set_title("Cross-View Localization")
        fig.savefig(output_dir / "trajectory.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Compute error metrics
        results_lines = []
        results_lines.append(f"Number of candidates: {len(candidates)}")
        results_lines.append(f"Number of inliers: {len(inlier_indices)}")
        results_lines.append(f"T_utm_odom:\n{result.T_utm_odom}")

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


def cross_view_localization(
    params,
    output_dir,
    skip_matching=False,
    save_viz=True,
    aerial_dir=None,
    ground_dir=None,
):
    """Run cross-view matching (optionally) then localization."""
    output_dir = str(output_dir)

    if not skip_matching:
        cross_view_matching(
            params,
            output_dir,
            save_viz=save_viz,
            aerial_dir=aerial_dir,
            ground_dir=ground_dir,
        )

    match_output_dir = os.path.join(output_dir, "match")

    rpgo_params = CrossViewRPGOParams.load(params)
    data_params = CrossViewLocalizationDataParams.load(params)
    data = CrossViewLocalizationData.from_params(data_params)

    # Save localization params (merges with matching params already in params.txt)
    from meridian.utils import save_params

    save_params(output_dir, rpgo_params, data_params)

    # Build pipeline + load submaps for rerun if enabled
    pipeline = None
    aerial_submaps = None
    ground_submaps = None
    if rpgo_params.rerun_match_with_known_rot:
        from meridian.params import (
            PrimitiveMatchParams,
            CrossViewMatchingParams,
            CrossViewPlaceRecognitionParams,
            AerialPatchParams,
            RegisterParams,
        )
        from meridian.cross_view.place_recognition import CrossViewPlaceRecognition
        from meridian.match.primitive_matcher import PrimitiveMatcher
        from meridian.register.registerer import Registerer2D

        pipeline_params = CrossViewMatchingParams.load(params)
        aerial_patch_params = AerialPatchParams.load(params)
        primitive_match_params = PrimitiveMatchParams.load(params)
        primitive_match_params.dim = 2

        try:
            pr_params = CrossViewPlaceRecognitionParams.load(params)
        except Exception:
            pr_params = None
        if pr_params is None and pipeline_params.matching_mode == "vpr":
            pr_params = CrossViewPlaceRecognitionParams()
        place_recognition = CrossViewPlaceRecognition(pr_params) if pr_params else None

        algorithm = CrossViewMatching(
            pipeline_params=pipeline_params,
            aerial_patch_params=aerial_patch_params,
            pixel_len_m=data.aerial_img_scale,
            matcher=PrimitiveMatcher(primitive_match_params),
            registerer=Registerer2D(RegisterParams.load(params)),
            place_recognition=place_recognition,
        )
        pipeline = CrossViewMatchingPipeline(algorithm=algorithm)
        aerial_seg_dir = os.path.join(
            aerial_dir or os.path.join(output_dir, "aerial"), "segments"
        )
        ground_seg_dir = os.path.join(
            ground_dir or os.path.join(output_dir, "ground"), "segments"
        )
        aerial_submaps = pipeline.load_submaps_from_dir(aerial_seg_dir)
        ground_submaps = pipeline.load_submaps_from_dir(ground_seg_dir)

    viz_params = CrossViewVisualizationParams.load(params)
    runner = CrossViewLocalization(rpgo_params=rpgo_params, viz_params=viz_params)
    loc_output_dir = os.path.join(output_dir, "localization")
    result = runner.localize(
        match_output_dir,
        data,
        loc_output_dir,
        pipeline=pipeline,
        aerial_submaps=aerial_submaps,
        ground_submaps=ground_submaps,
        aerial_img=data.aerial_img,
        main_output_dir=output_dir,
        save_viz=save_viz,
    )
    return result


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
    parser.add_argument(
        "-v",
        "--viz",
        action="store_true",
        help="Save per-match viz images (default: off).",
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

    if not args.skip_matching:
        cross_view_matching(
            args.params,
            args.output,
            skip_aerial=args.skip_aerial,
            skip_ground=args.skip_ground,
            skip_match=args.skip_match,
            save_viz=args.viz,
            aerial_dir=args.aerial,
            ground_dir=args.ground,
        )

    match_output_dir = os.path.join(args.output, "match")
    rpgo_params = CrossViewRPGOParams.load(args.params)
    data_params = CrossViewLocalizationDataParams.load(args.params)
    data = CrossViewLocalizationData.from_params(data_params)

    # Build pipeline + load submaps for rerun if enabled
    pipeline = None
    aerial_submaps = None
    ground_submaps = None
    if rpgo_params.rerun_match_with_known_rot:
        from meridian.params import (
            PrimitiveMatchParams,
            CrossViewMatchingParams,
            CrossViewPlaceRecognitionParams,
            AerialPatchParams,
            RegisterParams,
        )
        from meridian.cross_view.place_recognition import CrossViewPlaceRecognition
        from meridian.match.primitive_matcher import PrimitiveMatcher
        from meridian.register.registerer import Registerer2D

        pipeline_params = CrossViewMatchingParams.load(args.params)
        aerial_patch_params = AerialPatchParams.load(args.params)
        primitive_match_params = PrimitiveMatchParams.load(args.params)
        primitive_match_params.dim = 2

        try:
            pr_params = CrossViewPlaceRecognitionParams.load(args.params)
        except Exception:
            pr_params = None
        if pr_params is None and pipeline_params.matching_mode == "vpr":
            pr_params = CrossViewPlaceRecognitionParams()
        place_recognition = CrossViewPlaceRecognition(pr_params) if pr_params else None

        algorithm = CrossViewMatching(
            pipeline_params=pipeline_params,
            aerial_patch_params=aerial_patch_params,
            pixel_len_m=data.aerial_img_scale,
            matcher=PrimitiveMatcher(primitive_match_params),
            registerer=Registerer2D(RegisterParams.load(args.params)),
            place_recognition=place_recognition,
        )
        pipeline = CrossViewMatchingPipeline(algorithm=algorithm)
        aerial_seg_dir = os.path.join(
            args.aerial or os.path.join(args.output, "aerial"), "segments"
        )
        ground_seg_dir = os.path.join(
            args.ground or os.path.join(args.output, "ground"), "segments"
        )
        aerial_submaps = pipeline.load_submaps_from_dir(aerial_seg_dir)
        ground_submaps = pipeline.load_submaps_from_dir(ground_seg_dir)

    viz_params = CrossViewVisualizationParams.load(args.params)
    runner = CrossViewLocalization(rpgo_params=rpgo_params, viz_params=viz_params)
    loc_output_dir = os.path.join(args.output, "localization")
    runner.localize(
        match_output_dir,
        data,
        loc_output_dir,
        pipeline=pipeline,
        aerial_submaps=aerial_submaps,
        ground_submaps=ground_submaps,
        aerial_img=data.aerial_img,
        main_output_dir=args.output,
        save_viz=args.viz,
    )
