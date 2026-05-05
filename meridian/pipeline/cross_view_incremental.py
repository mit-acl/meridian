"""Online cross-view localization pipeline.

Combines online ground submap creation (segment_mapping) with cross-view
matching, CLIPPER outlier rejection, and PGO. Operates as a state machine
with two gates before committing to global localization:

- PRE: configured-mode matching with free rotation, CLIPPER over accumulated
  candidates, no PGO. Once CLIPPER inliers >= consistent_loop_closure_thresh
  (gate #1), an attempt is made.
- ATTEMPT (synchronous, may be retried): PGO on PRE candidates -> rough
  trajectory -> rerun max_intersection + translation_only matching on
  submaps 1..n -> CLIPPER on rerun candidates. Gate #2: if rerun inliers
  >= rot_constrained_consistent_lc_thresh, commit (replace candidates, run
  final PGO, transition to POST). Otherwise, leave PRE state untouched and
  retry on the next new submap.
- POST: per new submap match only that submap with max_intersection +
  translation_only against latest optimized trajectory, append candidates,
  CLIPPER + PGO on full accumulated set.

Instantaneous pose is causal: NaN before successful global localization,
T_utm_odom @ pose_cam after.
"""

import argparse
import logging
import os
import pathlib
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from robotdatapy.data.robot_data import NoDataNearTimeException

from meridian.cross_view.matching import CrossViewMatching
from meridian.cross_view.place_recognition import CrossViewPlaceRecognition
from meridian.cross_view.rpgo import (
    CrossViewRPGO,
    pose_data_from_trajectory,
)
from meridian.map2d.ground_submap_primitive_mapping import (
    GroundSubmapPrimitiveMapping,
)
from meridian.map2d.segment_to_primitive import SegmentToPrimitiveConverter
from meridian.map3d.segment_mapper import SegmentMapper
from meridian.segmenter.segmenter3d import Segmenter
from meridian.map3d.submap import Submap
from meridian.match.segment_matcher import SegmentMatcher
from meridian.params import (
    AerialPatchParams,
    CrossViewIncrementalParams,
    CrossViewLocalizationDataParams,
    CrossViewMatchingParams,
    CrossViewPlaceRecognitionParams,
    CrossViewRPGOParams,
    CrossViewVisualizationParams,
    GroundSegmenterParams,
    GroundSubmapParams,
    RegisterParams,
    SegmentMappingDataParams,
    SegmentMappingParams,
    SegmentMatchParams,
    SegmentToPrimitiveConversionParams,
    SegmenterParams,
)
from meridian.pipeline.cross_view_localization import (
    CrossViewLocalization,
    build_candidates_from_match_result,
)
from meridian.pipeline.cross_view_matching import CrossViewMatchingPipeline
from meridian.pipeline.data import (
    CrossViewLocalizationData,
    SegmentMappingData,
)
from meridian.register.registerer import Registerer2D
from meridian.utils import save_commit_hash, save_params
from meridian.viz.cross_view_viz import viz_ground_segments

logger = logging.getLogger(__name__)


@dataclass
class CrossViewIncremental:
    # Mapping
    mapping_params: SegmentMappingParams
    segmenter: Segmenter
    mapper: SegmentMapper
    conversion_params: SegmentToPrimitiveConversionParams

    # Cross-view
    algorithm: CrossViewMatching
    aerial_submaps: Dict[str, Submap]
    data: CrossViewLocalizationData
    rpgo_params: CrossViewRPGOParams
    incremental_params: CrossViewIncrementalParams
    viz_params: CrossViewVisualizationParams

    output_dir: str = ""

    # State
    _state: str = field(default="PRE", init=False)
    _submap_count: int = field(default=0, init=False)
    _candidates: List[dict] = field(default_factory=list, init=False)
    _results_per_submap: Dict[str, object] = field(default_factory=dict, init=False)
    _match_details_per_submap: Dict[str, Dict[str, list]] = field(
        default_factory=dict, init=False
    )
    _last_T_utm_odom: Optional[np.ndarray] = field(default=None, init=False)
    _last_optimized_trajectory: Optional[List[np.ndarray]] = field(
        default=None, init=False
    )
    _optimized_pose_data: Optional[object] = field(default=None, init=False)
    _instant_pose_history: List[Tuple[float, np.ndarray]] = field(
        default_factory=list, init=False
    )
    _global_loc_frame_idx: Optional[int] = field(default=None, init=False)
    _global_loc_wall_time: Optional[float] = field(default=None, init=False)
    _failed_attempt_count: int = field(default=0, init=False)
    _gate1_first_submap_count: Optional[int] = field(default=None, init=False)
    _gate2_passed_submap_count: Optional[int] = field(default=None, init=False)
    _compute_wall_time: float = field(default=0.0, init=False)
    _last_descriptor_position: Optional[np.ndarray] = field(default=None, init=False)
    _timing: dict = field(
        default_factory=lambda: {
            "data": [],
            "segment": [],
            "map": [],
            "submap_2d": [],
            "match": [],
            "outlier_rej": [],
            "pgo": [],
        },
        init=False,
    )

    # ------------------------------------------------------------------
    # Per-frame loop
    # ------------------------------------------------------------------

    def run(self, data: SegmentMappingData):
        if data.use_point_cloud:
            t0 = max(
                data.img_data.t0, data.point_cloud_data.t0, data.camera_pose_data.t0
            )
            tf = min(
                data.img_data.tf, data.point_cloud_data.tf, data.camera_pose_data.tf
            )
        else:
            t0 = max(data.img_data.t0, data.depth_data.t0, data.camera_pose_data.t0)
            tf = min(data.img_data.tf, data.depth_data.tf, data.camera_pose_data.tf)

        times = np.arange(t0, tf, self.mapping_params.dt)
        print(f"Processing {len(times)} frames from t={t0:.2f} to t={tf:.2f}")

        t_loop_start = time.time()
        for t in tqdm.tqdm(times, desc="Incremental cross-view"):
            t_data_start = time.time()
            try:
                img_t = data.img_data.nearest_time(t)
                img = data.img_data.img(img_t)
                pose = data.camera_pose_data.pose(img_t)
                if data.use_point_cloud:
                    pcl = data.align_point_cloud.aligned_point_cloud(img_t)
                    pcl_proj = data.align_point_cloud.projected_point_cloud(pcl)
                    depth = data.align_point_cloud.filter_point_cloud_and_projection(
                        pcl, pcl_proj
                    )
                else:
                    depth = data.depth_data.img(img_t)
            except NoDataNearTimeException:
                continue

            position = pose[:3, 3]
            should_compute_descriptor = (
                self._last_descriptor_position is None
                or np.linalg.norm(position - self._last_descriptor_position)
                >= self.mapping_params.frame_descriptor_dist_m
            )

            t_seg_start = time.time()
            observations, frame_descriptor = self.segmenter.segment(
                img,
                img_t,
                pose,
                depth,
                compute_frame_descriptor=should_compute_descriptor,
            )
            if frame_descriptor is not None:
                self._last_descriptor_position = position.copy()

            t_map_start = time.time()
            self.mapper.update(img_t, pose, observations, frame_descriptor)

            t_submap_start = time.time()
            self.mapper.process_submaps_2d(img_t, pose)
            t_end = time.time()

            self._timing["data"].append(t_seg_start - t_data_start)
            self._timing["segment"].append(t_map_start - t_seg_start)
            self._timing["map"].append(t_submap_start - t_map_start)
            self._timing["submap_2d"].append(t_end - t_submap_start)

            # Detect new submap(s) and process each
            while len(self.mapper.submaps_2d) > self._submap_count:
                new_idx = self._submap_count
                new_submap = self.mapper.submaps_2d[new_idx]
                ground_key = str(new_idx)
                self._handle_new_submap(new_submap, ground_key)
                self._submap_count += 1

            self._record_instantaneous_pose(img_t, pose)

        self._compute_wall_time += time.time() - t_loop_start

    # ------------------------------------------------------------------
    # New-submap handler
    # ------------------------------------------------------------------

    def _handle_new_submap(self, new_submap: Submap, ground_key: str):
        ground_submaps = {ground_key: new_submap}

        # Match the single new submap.
        t_match_start = time.time()
        if self._state == "PRE":
            match_result = self.algorithm.cross_view_match(
                self.aerial_submaps,
                ground_submaps,
                reference_trajectory=None,
                T_camera_flu=self.data.T_camera_flu,
                show_progress=False,
                local_to_pixel_fn=self.data.aerial_local_to_pixel
                if self.data.geotiff_transform is not None
                else None,
                gt_trajectory=self.data.gt_pose_data,
            )
        else:
            match_result = self.algorithm.cross_view_match_max_intersection(
                self.aerial_submaps,
                ground_submaps,
                reference_trajectory=self._optimized_pose_data,
                T_camera_flu=self.data.T_camera_flu,
                translation_only=True,
                show_progress=False,
                local_to_pixel_fn=self.data.aerial_local_to_pixel
                if self.data.geotiff_transform is not None
                else None,
                gt_trajectory=self.data.gt_pose_data,
            )
        self._timing["match"].append(time.time() - t_match_start)

        # Stash for end-of-run heatmaps.
        if ground_key in match_result.results:
            self._results_per_submap[ground_key] = match_result.results[ground_key]
        if ground_key in match_result.match_details:
            self._match_details_per_submap[ground_key] = match_result.match_details[
                ground_key
            ]

        # Build candidate dicts and append.
        min_assoc = self._min_assoc_for_state()
        new_candidates = build_candidates_from_match_result(
            match_result,
            ground_submaps,
            self.aerial_submaps,
            self.data,
            min_assoc,
        )
        self._candidates.extend(new_candidates)

        # Outlier rejection on accumulated candidates.
        t_or_start = time.time()
        rpgo = CrossViewRPGO(params=self.rpgo_params)
        try:
            inlier_indices, _, _ = rpgo.solve_clipper_only(
                self._candidates,
                self.mapper.poses_cam_history,
                np.array(self.mapper.times_history),
            )
        except Exception as e:
            logger.warning(f"CLIPPER failed (state={self._state}): {e}")
            inlier_indices = np.array([], dtype=int)
        self._timing["outlier_rej"].append(time.time() - t_or_start)

        n_inliers = len(inlier_indices)
        logger.info(
            f"[submap {ground_key}] state={self._state} "
            f"candidates={len(self._candidates)} inliers={n_inliers}"
        )

        if self._state == "PRE":
            if n_inliers >= self.incremental_params.consistent_loop_closure_thresh:
                if self._gate1_first_submap_count is None:
                    self._gate1_first_submap_count = len(self.mapper.submaps_2d)
                if self._attempt_global_localization():
                    self._state = "POST"
        else:
            self._run_post_pgo(inlier_indices)

    def _min_assoc_for_state(self) -> int:
        if (
            self._state != "PRE"
            and self.rpgo_params.min_num_associations_rerun is not None
        ):
            return self.rpgo_params.min_num_associations_rerun
        return self.rpgo_params.min_num_associations

    # ------------------------------------------------------------------
    # POST PGO
    # ------------------------------------------------------------------

    def _run_post_pgo(self, inlier_indices: np.ndarray):
        if len(inlier_indices) == 0:
            return
        rpgo = CrossViewRPGO(params=self.rpgo_params)
        t_pgo_start = time.time()
        T_utm_odom = rpgo._frame_align(self._candidates, inlier_indices)
        if self.rpgo_params.optimization_method == "pgo":
            optimized_trajectory = rpgo._pgo(
                self._candidates,
                inlier_indices,
                self.mapper.poses_cam_history,
                np.array(self.mapper.times_history),
                self.data.T_camera_flu,
                T_utm_odom,
            )
        else:
            optimized_trajectory = rpgo._apply_rigid_transform(
                T_utm_odom,
                self.mapper.poses_cam_history,
                self.data.T_camera_flu,
            )
        self._timing["pgo"].append(time.time() - t_pgo_start)

        self._last_T_utm_odom = T_utm_odom
        self._last_optimized_trajectory = optimized_trajectory
        self._optimized_pose_data = self._build_camera_pose_data(
            optimized_trajectory, np.array(self.mapper.times_history)
        )

    # ------------------------------------------------------------------
    # PRE -> POST attempt (gated; may be retried)
    # ------------------------------------------------------------------

    def _attempt_global_localization(self) -> bool:
        """Attempt global localization. Returns True iff both gates pass and
        trajectory state was committed; False otherwise (caller stays in PRE).

        Invariant: must not mutate self._candidates, self._results_per_submap,
        self._match_details_per_submap, self._last_T_utm_odom,
        self._last_optimized_trajectory, or self._optimized_pose_data until
        gate #2 passes.
        """
        wc_start = time.time()
        gate2_thresh = self.incremental_params.rot_constrained_consistent_lc_thresh
        logger.info(
            f"[global-loc] gate-1 met at submap {self._submap_count}: "
            f"{len(self._candidates)} candidates"
        )

        rpgo = CrossViewRPGO(params=self.rpgo_params)

        # Step 1: initial PGO on PRE candidates.
        t_pgo_start = time.time()
        try:
            initial_result = rpgo.solve(
                self._candidates,
                self.mapper.poses_cam_history,
                np.array(self.mapper.times_history),
                self.data.T_camera_flu,
            )
        except Exception as e:
            logger.warning(f"[global-loc] initial PGO failed: {e}")
            self._timing["pgo"].append(time.time() - t_pgo_start)
            self._failed_attempt_count += 1
            return False
        self._timing["pgo"].append(time.time() - t_pgo_start)

        if not initial_result.success:
            logger.warning("[global-loc] initial PGO returned no success — aborting.")
            self._failed_attempt_count += 1
            return False

        # Step 2: rough-trajectory pose data for known-rotation matching.
        rough_pose_data = self._build_camera_pose_data(
            initial_result.optimized_trajectory,
            np.array(self.mapper.times_history),
        )

        # Step 3: rerun matching on submaps 1..n with max_intersection +
        # translation_only.
        all_ground_submaps = {str(i): sm for i, sm in enumerate(self.mapper.submaps_2d)}

        t_match_start = time.time()
        rerun_result = self.algorithm.cross_view_match_max_intersection(
            self.aerial_submaps,
            all_ground_submaps,
            reference_trajectory=rough_pose_data,
            T_camera_flu=self.data.T_camera_flu,
            translation_only=True,
            show_progress=False,
            local_to_pixel_fn=self.data.aerial_local_to_pixel
            if self.data.geotiff_transform is not None
            else None,
            gt_trajectory=self.data.gt_pose_data,
        )
        self._timing["match"].append(time.time() - t_match_start)

        # Step 4: build rerun_candidates locally — do NOT assign to self yet.
        min_assoc = (
            self.rpgo_params.min_num_associations_rerun
            if self.rpgo_params.min_num_associations_rerun is not None
            else self.rpgo_params.min_num_associations
        )
        rerun_candidates = build_candidates_from_match_result(
            rerun_result,
            all_ground_submaps,
            self.aerial_submaps,
            self.data,
            min_assoc,
        )

        # Step 5: CLIPPER on rerun candidates (still local, no commit yet).
        t_or_start = time.time()
        try:
            rerun_inliers, _, _ = rpgo.solve_clipper_only(
                rerun_candidates,
                self.mapper.poses_cam_history,
                np.array(self.mapper.times_history),
            )
        except Exception as e:
            logger.warning(f"[global-loc] rerun CLIPPER failed: {e}")
            rerun_inliers = np.array([], dtype=int)
        self._timing["outlier_rej"].append(time.time() - t_or_start)

        # Step 6: gate #2 check.
        if len(rerun_inliers) < gate2_thresh:
            logger.warning(
                f"[global-loc] gate-2 FAILED: rerun inliers={len(rerun_inliers)} "
                f"< rot_constrained_consistent_lc_thresh={gate2_thresh}. "
                f"Staying in PRE; preserving {len(self._candidates)} PRE candidates."
            )
            self._failed_attempt_count += 1
            return False

        # Step 7: commit (gate #2 passed).
        t_pgo_start = time.time()
        T_utm_odom = rpgo._frame_align(rerun_candidates, rerun_inliers)
        if self.rpgo_params.optimization_method == "pgo":
            optimized_trajectory = rpgo._pgo(
                rerun_candidates,
                rerun_inliers,
                self.mapper.poses_cam_history,
                np.array(self.mapper.times_history),
                self.data.T_camera_flu,
                T_utm_odom,
            )
        else:
            optimized_trajectory = rpgo._apply_rigid_transform(
                T_utm_odom,
                self.mapper.poses_cam_history,
                self.data.T_camera_flu,
            )
        self._timing["pgo"].append(time.time() - t_pgo_start)

        self._candidates = rerun_candidates
        self._results_per_submap = dict(rerun_result.results)
        self._match_details_per_submap = dict(rerun_result.match_details)
        self._last_T_utm_odom = T_utm_odom
        self._last_optimized_trajectory = optimized_trajectory
        self._optimized_pose_data = self._build_camera_pose_data(
            optimized_trajectory, np.array(self.mapper.times_history)
        )

        self._global_loc_frame_idx = len(self._instant_pose_history)
        self._global_loc_wall_time = time.time() - wc_start
        self._gate2_passed_submap_count = len(self.mapper.submaps_2d)
        logger.info(
            f"[global-loc] gate-2 PASSED: {len(rerun_candidates)} rerun candidates, "
            f"{len(rerun_inliers)} inliers >= {gate2_thresh}, "
            f"{self._global_loc_wall_time:.2f}s"
        )
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_camera_pose_data(
        self, body_trajectory: List[np.ndarray], times: np.ndarray
    ):
        """Convert body-frame optimized trajectory to camera-frame PoseData
        suitable as reference_trajectory for matching."""
        if self.data.T_camera_flu is not None:
            T_flu_camera = np.linalg.inv(self.data.T_camera_flu)
            cam_traj = [T @ T_flu_camera for T in body_trajectory]
        else:
            cam_traj = body_trajectory
        return pose_data_from_trajectory(cam_traj, times)

    def _record_instantaneous_pose(self, t: float, pose_cam: np.ndarray):
        if self._state == "POST" and self._last_T_utm_odom is not None:
            T_utm_camera = self._last_T_utm_odom @ pose_cam
            if self.data.T_camera_flu is not None:
                T_utm_body = T_utm_camera @ self.data.T_camera_flu
            else:
                T_utm_body = T_utm_camera
            self._instant_pose_history.append((t, T_utm_body))
        else:
            self._instant_pose_history.append((t, np.full((4, 4), np.nan)))

    # ------------------------------------------------------------------
    # End-of-run output
    # ------------------------------------------------------------------

    def write_outputs(self, save_viz: bool = True):
        out = pathlib.Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self._write_mapping_outputs()
        self._write_match_heatmaps()
        self._write_incremental_outputs()
        self._write_localization_outputs(save_viz=save_viz)

    def _write_mapping_outputs(self):
        """Mirror of segment_mapping.py end-of-run mapping outputs."""
        out = pathlib.Path(self.output_dir)

        # segment_map.pkl
        from meridian.map3d.map import SegmentMap

        descriptors = self.mapper.frame_descriptors_history
        if not any(d is not None for d in descriptors):
            descriptors = None
        segment_map = SegmentMap(
            segments=self.mapper.get_segment_map(),
            trajectory=self.mapper.poses_cam_history,
            times=self.mapper.times_history,
            descriptors=descriptors,
        )
        segment_map.save(str(out / "segment_map.pkl"))

        # Mapping-only timing.txt
        n_frames = len(self._timing["data"])
        if n_frames > 0:
            mean_data = float(np.mean(self._timing["data"]))
            mean_seg = float(np.mean(self._timing["segment"]))
            mean_map = float(np.mean(self._timing["map"]))
            mean_sm2d = float(np.mean(self._timing["submap_2d"]))
            mean_total = mean_data + mean_seg + mean_map + mean_sm2d
            with open(out / "timing.txt", "w") as f:
                f.write(f"Frames:           {n_frames}\n")
                f.write("\nPer-frame averages:\n")
                f.write(
                    f"  Data fetch:     {mean_data:.4f}s  ({mean_data / mean_total * 100:.1f}%)\n"
                )
                f.write(
                    f"  Segmenter:      {mean_seg:.4f}s  ({mean_seg / mean_total * 100:.1f}%)\n"
                )
                f.write(
                    f"  Mapper update:  {mean_map:.4f}s  ({mean_map / mean_total * 100:.1f}%)\n"
                )
                f.write(
                    f"  Submap 2D:      {mean_sm2d:.4f}s  ({mean_sm2d / mean_total * 100:.1f}%)\n"
                )
                f.write(
                    f"  Total:          {mean_total:.4f}s  ({1 / mean_total:.1f} fps)\n"
                )
                f.write(f"\n2D Submaps created: {len(self.mapper.submaps_2d)}\n")

        if not self.mapper.submaps_2d:
            return

        ground_dir = out / "ground" / "segments"
        ground_dir.mkdir(parents=True, exist_ok=True)
        viz_dir = out / "ground" / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        ground_submap_params = GroundSubmapParams.load(self._params_path)

        for k, submap_2d in enumerate(self.mapper.submaps_2d):
            submap_2d.save(str(ground_dir / f"{k}.pkl"))

            if k < len(self.mapper._submap_intermediates):
                intermediate = self.mapper._submap_intermediates[k]
                if intermediate is None:
                    continue
                if intermediate.aerial_segments:
                    dense_dir = ground_dir / f"{k}_dense"
                    dense_dir.mkdir(parents=True, exist_ok=True)
                    for aerial_seg in intermediate.aerial_segments:
                        pts = aerial_seg.points
                        max_n = ground_submap_params.dense_points_max_n
                        if max_n is not None and len(pts) > max_n:
                            idx = np.round(np.linspace(0, len(pts) - 1, max_n)).astype(
                                int
                            )
                            pts = pts[idx]
                        with open(dense_dir / f"{aerial_seg.id}.pkl", "wb") as f:
                            pickle.dump(pts, f)

                fig, _ = viz_ground_segments(
                    intermediate.flattened_submap,
                    intermediate.aerial_segments,
                    intermediate.general_segments,
                    submap_2d.segments,
                    self.conversion_params.alpha_shape_alpha,
                    self.conversion_params.alpha_shape_grid_downsample,
                    self.conversion_params.alpha_shape_max_n_pts,
                    self.conversion_params.alpha_shape_ref_size_m,
                    show_origin=ground_submap_params.viz_show_sm_origin,
                )
                fig.savefig(viz_dir / f"{k}.png", dpi=400)
                plt.close(fig)

    def _write_match_heatmaps(self):
        out = pathlib.Path(self.output_dir)
        match_seg_dir = out / "match" / "segments"
        match_viz_dir = out / "match" / "viz"
        match_seg_dir.mkdir(parents=True, exist_ok=True)
        match_viz_dir.mkdir(parents=True, exist_ok=True)

        params = self.algorithm.pipeline_params
        for ground_key, results_matrix in self._results_per_submap.items():
            results_matrix.save(
                str(match_seg_dir / f"ground_{ground_key}_results_matrix.pkl")
            )
            try:
                results_matrix.plot_cross_view(
                    dist_thresh=params.match_trans_err_m,
                    angle_thresh_deg=params.match_rot_err_deg,
                )
                plt.savefig(match_viz_dir / f"ground_{ground_key}_all.png", dpi=400)
                plt.close()
            except Exception as e:
                logger.warning(f"Heatmap render failed for ground {ground_key}: {e}")

    def _write_incremental_outputs(self):
        out = pathlib.Path(self.output_dir) / "incremental"
        out.mkdir(parents=True, exist_ok=True)

        self._write_incremental_timing(out)
        self._write_trajectory_incremental_plot(out)
        err_metrics = self._write_error_vs_time_plot(out)
        self._write_incremental_results(out, err_metrics)

    def _write_incremental_timing(self, out: pathlib.Path):
        n_frames = len(self._timing["data"])
        n_submaps = len(self.mapper.submaps_2d)
        n_match = len(self._timing["match"])
        n_or = len(self._timing["outlier_rej"])
        n_pgo = len(self._timing["pgo"])

        lines = [
            f"Frames:                 {n_frames}",
            f"Submaps:                {n_submaps}",
        ]
        if n_frames > 0:
            mean_data = float(np.mean(self._timing["data"]))
            mean_seg = float(np.mean(self._timing["segment"]))
            mean_map = float(np.mean(self._timing["map"]))
            mean_sm2d = float(np.mean(self._timing["submap_2d"]))
            lines.append("\nPer-frame averages:")
            lines.append(f"  Data fetch:     {mean_data:.4f}s")
            lines.append(f"  Segmenter:      {mean_seg:.4f}s")
            lines.append(f"  Mapper update:  {mean_map:.4f}s")
            lines.append(f"  Submap 2D:      {mean_sm2d:.4f}s")

        def _stats(name, vals):
            if not vals:
                return [f"  {name}: (no calls)"]
            arr = np.array(vals)
            return [
                f"  {name}: n={len(arr)} "
                f"mean={arr.mean():.4f}s total={arr.sum():.4f}s "
                f"min={arr.min():.4f}s max={arr.max():.4f}s"
            ]

        lines.append("\nPer-call cross-view stats:")
        lines += _stats("match", self._timing["match"])
        lines += _stats("outlier_rej", self._timing["outlier_rej"])
        lines += _stats("pgo", self._timing["pgo"])

        lines.append("\nCounts:")
        lines.append(f"  match calls:        {n_match}")
        lines.append(f"  outlier_rej calls:  {n_or}")
        lines.append(f"  pgo calls:          {n_pgo}")
        lines.append(f"  failed attempts:    {self._failed_attempt_count}")
        lines.append(f"\nCompute wall time:      {self._compute_wall_time:.2f}s")
        if self._global_loc_frame_idx is not None:
            lines.append(f"Global loc frame index: {self._global_loc_frame_idx}")
            lines.append(f"Global loc wall time:   {self._global_loc_wall_time:.2f}s")
        else:
            lines.append("Global loc: not triggered")
        lines.append(f"Final state:            {self._state}")

        with open(out / "timing.txt", "w") as f:
            f.write("\n".join(lines) + "\n")

    def _write_trajectory_incremental_plot(self, out: pathlib.Path):
        if not self._instant_pose_history:
            return

        # Identify segments split by NaN (pre-loc) and by PGO updates.
        positions = []
        for _, T in self._instant_pose_history:
            if np.any(np.isnan(T)):
                positions.append(None)
            else:
                positions.append(T[:2, 3])

        # Split into contiguous non-NaN runs.
        segments_xy = []
        cur = []
        for p in positions:
            if p is None:
                if cur:
                    segments_xy.append(np.array(cur))
                    cur = []
            else:
                cur.append(p)
        if cur:
            segments_xy.append(np.array(cur))

        # UTM -> pixel conversion.
        if self.data.geotiff_transform is not None:

            def utm_to_pixel(xy):
                return self.data.aerial_utm_to_pixel(np.atleast_2d(xy))
        else:
            origin_x, origin_y = self.data.aerial_img_origin
            pixel_len_m = self.data.aerial_img_scale

            def utm_to_pixel(xy):
                xy = np.atleast_2d(xy)
                cols = (xy[:, 0] - origin_x) / pixel_len_m
                rows = (origin_y - xy[:, 1]) / pixel_len_m
                return np.column_stack([cols, rows])

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        aerial_img = self.data.aerial_img
        ds = max(1, min(aerial_img.shape[0], aerial_img.shape[1]) // 2000)
        ax.imshow(
            cv.cvtColor(aerial_img[::ds, ::ds], cv.COLOR_BGR2RGB),
            extent=[0, aerial_img.shape[1], aerial_img.shape[0], 0],
        )

        for i, seg_xy in enumerate(segments_xy):
            if seg_xy.size == 0:
                continue
            seg_px = utm_to_pixel(seg_xy)
            label = "Estimated (incremental)" if i == 0 else None
            ax.plot(
                seg_px[:, 0],
                seg_px[:, 1],
                color=self.viz_params.estimated_trajectory_color,
                linestyle="-",
                linewidth=1.2,
                label=label,
            )

        if self.data.gt_pose_data is not None:
            gt_xy = []
            for t, _ in self._instant_pose_history:
                try:
                    gt_pose = self.data.gt_pose_data.pose(t)
                    if self.data.T_camera_flu is not None:
                        gt_body = gt_pose @ self.data.T_camera_flu
                    else:
                        gt_body = gt_pose
                    gt_xy.append(gt_body[:2, 3])
                except Exception:
                    gt_xy.append([np.nan, np.nan])
            gt_xy = np.array(gt_xy)
            valid = ~np.any(np.isnan(gt_xy), axis=1)
            if np.any(valid):
                gt_px = utm_to_pixel(gt_xy[valid])
                ax.plot(
                    gt_px[:, 0],
                    gt_px[:, 1],
                    color=self.viz_params.gt_trajectory_color,
                    linestyle="-",
                    linewidth=1.2,
                    label="Ground Truth",
                )

        ax.legend()
        ax.set_title("Incremental Cross-View Localization")
        fig.savefig(out / "trajectory_incremental.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _write_error_vs_time_plot(self, out: pathlib.Path) -> dict:
        if not self._instant_pose_history or self.data.gt_pose_data is None:
            return {}

        t0 = self._instant_pose_history[0][0]
        ts, trans_errs, yaw_errs = [], [], []
        for t, T in self._instant_pose_history:
            ts.append(t - t0)
            if np.any(np.isnan(T)):
                trans_errs.append(np.nan)
                yaw_errs.append(np.nan)
                continue
            try:
                gt_pose = self.data.gt_pose_data.pose(t)
            except Exception:
                trans_errs.append(np.nan)
                yaw_errs.append(np.nan)
                continue
            if self.data.T_camera_flu is not None:
                gt_body = gt_pose @ self.data.T_camera_flu
            else:
                gt_body = gt_pose
            trans_errs.append(float(np.linalg.norm(T[:2, 3] - gt_body[:2, 3])))
            est_yaw = np.arctan2(T[1, 0], T[0, 0])
            gt_yaw = np.arctan2(gt_body[1, 0], gt_body[0, 0])
            d = est_yaw - gt_yaw
            yaw_errs.append(float(np.abs(np.arctan2(np.sin(d), np.cos(d)))))

        ts = np.array(ts)
        trans_errs = np.array(trans_errs)
        yaw_errs = np.array(yaw_errs)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(ts, trans_errs, "-", color="tab:blue")
        axes[0].set_ylabel("Translation error (m)")
        axes[0].grid(True)
        axes[1].plot(ts, np.rad2deg(yaw_errs), "-", color="tab:orange")
        axes[1].set_ylabel("Heading error (deg)")
        axes[1].set_xlabel("t - t0 (s)")
        axes[1].grid(True)
        fig.suptitle("Incremental localization error vs. time")
        fig.savefig(out / "error_vs_time.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        valid = ~np.isnan(trans_errs) & ~np.isnan(yaw_errs)
        n_valid = int(valid.sum())
        n_total = len(ts)
        if n_valid > 0:
            return {
                "n_valid": n_valid,
                "n_total": n_total,
                "rmse_trans_m": float(np.sqrt(np.mean(trans_errs[valid] ** 2))),
                "mean_trans_m": float(np.mean(trans_errs[valid])),
                "max_trans_m": float(np.max(trans_errs[valid])),
                "rmse_yaw_deg": float(
                    np.rad2deg(np.sqrt(np.mean(yaw_errs[valid] ** 2)))
                ),
                "mean_yaw_deg": float(np.rad2deg(np.mean(yaw_errs[valid]))),
                "max_yaw_deg": float(np.rad2deg(np.max(yaw_errs[valid]))),
            }
        return {"n_valid": 0, "n_total": n_total}

    def _write_incremental_results(self, out: pathlib.Path, m: dict):
        lines = []
        lines.append(f"Final state:           {self._state}")
        lines.append(f"Total ground submaps:  {len(self.mapper.submaps_2d)}")
        lines.append(f"Compute wall time (s): {self._compute_wall_time:.2f}")
        gate1_str = (
            str(self._gate1_first_submap_count)
            if self._gate1_first_submap_count is not None
            else "none"
        )
        gate2_str = (
            str(self._gate2_passed_submap_count)
            if self._gate2_passed_submap_count is not None
            else "none"
        )
        lines.append(f"Gate-1 first submap:   {gate1_str}")
        lines.append(f"Gate-2 passed submap:  {gate2_str}")
        lines.append(f"Failed attempts:       {self._failed_attempt_count}")
        if self._global_loc_frame_idx is not None:
            lines.append(f"Global loc frame idx:  {self._global_loc_frame_idx}")
            lines.append(f"Global loc wall time:  {self._global_loc_wall_time:.2f}s")
        else:
            lines.append("Global loc:            not triggered")

        if m and m.get("n_valid", 0) > 0:
            lines.append(f"Frames localized:      {m['n_valid']} / {m['n_total']}")
            lines.append(f"Translation RMSE (m):  {m['rmse_trans_m']:.3f}")
            lines.append(f"Translation mean (m):  {m['mean_trans_m']:.3f}")
            lines.append(f"Translation max (m):   {m['max_trans_m']:.3f}")
            lines.append(f"Heading RMSE (deg):    {m['rmse_yaw_deg']:.3f}")
            lines.append(f"Heading mean (deg):    {m['mean_yaw_deg']:.3f}")
            lines.append(f"Heading max (deg):     {m['max_yaw_deg']:.3f}")
        else:
            n_total = m.get("n_total") if m else 0
            lines.append(f"Frames localized:      0 / {n_total}")

        with open(out / "results.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))

    def _write_localization_outputs(self, save_viz: bool = True):
        out = pathlib.Path(self.output_dir) / "localization"
        out.mkdir(parents=True, exist_ok=True)
        if not self._candidates:
            logger.warning("No candidates accumulated — skipping localization outputs.")
            return
        rpgo = CrossViewRPGO(params=self.rpgo_params)
        try:
            result = rpgo.solve(
                self._candidates,
                self.mapper.poses_cam_history,
                np.array(self.mapper.times_history),
                self.data.T_camera_flu,
            )
        except Exception as e:
            logger.warning(f"Final localization solve failed: {e}")
            return
        if not result.success:
            logger.warning("Final localization solve returned no success.")
            return
        CrossViewLocalization._save_results(result, out)
        CrossViewLocalization._visualize_and_report(
            result, self.data, out, self.viz_params
        )

    # ------------------------------------------------------------------
    # Setters used by entry-point glue
    # ------------------------------------------------------------------

    _params_path: str = field(default="", init=False, repr=False)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def cross_view_incremental(
    params_path: str,
    output_dir: str,
    aerial_dir: str,
    run: str = None,
    save_viz: bool = True,
):
    if not aerial_dir:
        raise ValueError(
            "--aerial is required: point to a directory containing segments/*.pkl "
            "(produced by `cross_view_matching --skip-match --skip-ground` or equivalent)."
        )

    print("Loading parameters...")
    mapping_params = SegmentMappingParams.load(params_path, run=run)
    mapping_data_params = SegmentMappingDataParams.load(params_path, run=run)
    segmenter_params = SegmenterParams.load(params_path, run=run)
    if mapping_data_params.point_cloud_data:
        segmenter_params.use_point_cloud = True
    else:
        segmenter_params.depth_scale = 1 / mapping_data_params.depth_scale

    ground_submap_params = GroundSubmapParams.load(params_path, run=run)
    conversion_params = SegmentToPrimitiveConversionParams.load(params_path, run=run)
    ground_segmenter_params = GroundSegmenterParams.load(params_path, run=run)

    pipeline_params = CrossViewMatchingParams.load(params_path, run=run)
    aerial_patch_params = AerialPatchParams.load(params_path, run=run)
    segment_match_params = SegmentMatchParams.load(params_path, run=run)
    segment_match_params.dim = 2
    register_params = RegisterParams.load(params_path, run=run)
    rpgo_params = CrossViewRPGOParams.load(params_path, run=run)
    incremental_params = CrossViewIncrementalParams.load(params_path, run=run)
    loc_data_params = CrossViewLocalizationDataParams.load(params_path, run=run)

    try:
        viz_params = CrossViewVisualizationParams.load(params_path, run=run)
    except Exception:
        viz_params = CrossViewVisualizationParams()

    try:
        pr_params = CrossViewPlaceRecognitionParams.load(params_path, run=run)
    except Exception:
        pr_params = None
    if pr_params is None and pipeline_params.matching_mode == "vpr":
        pr_params = CrossViewPlaceRecognitionParams()
    place_recognition = (
        CrossViewPlaceRecognition(pr_params) if pr_params is not None else None
    )

    os.makedirs(output_dir, exist_ok=True)

    # Record the aerial source dir for downstream tools.
    with open(os.path.join(output_dir, "aerial.txt"), "w") as f:
        f.write(os.path.abspath(aerial_dir) + "\n")

    print("Loading aerial submaps...")
    matching_pipeline = CrossViewMatchingPipeline(
        algorithm=CrossViewMatching(
            pipeline_params=pipeline_params,
            aerial_patch_params=aerial_patch_params,
            pixel_len_m=loc_data_params.aerial_img_scale or 0.01,
            matcher=SegmentMatcher(segment_match_params),
            registerer=Registerer2D(register_params),
            place_recognition=place_recognition,
        ),
    )
    aerial_seg_dir = os.path.join(aerial_dir, "segments")
    aerial_submaps = matching_pipeline.load_submaps_from_dir(aerial_seg_dir)
    if not aerial_submaps:
        raise ValueError(f"No aerial submaps found in {aerial_seg_dir}")

    print("Loading bag time range...")
    bag_t_range = SegmentMappingData.get_bag_time_range(mapping_data_params)
    if bag_t_range is not None:
        full_t0, full_tf = bag_t_range
        print(f"Bag time range: {full_t0:.2f} to {full_tf:.2f}")
        # Honor user-specified time_range in params so chunking stops at the
        # requested end instead of the bag's end.
        user_range = (mapping_data_params.img_data or {}).get("time_range")
        if user_range is not None:
            relative = (mapping_data_params.img_data or {}).get(
                "time_range_relative", False
            )
            user_t0, user_tf = user_range
            if relative:
                user_t0 = full_t0 + user_t0
                user_tf = full_t0 + user_tf
            full_t0 = max(full_t0, user_t0)
            full_tf = min(full_tf, user_tf)
            print(f"Clamped to user time_range: {full_t0:.2f} to {full_tf:.2f}")
    else:
        full_t0, full_tf = None, None

    if mapping_data_params.max_time is not None and full_t0 is not None:
        init_time_range = (
            full_t0,
            full_t0 + min(mapping_data_params.max_time, full_tf - full_t0),
        )
    else:
        init_time_range = None

    print("Loading initial mapping data...")
    init_data = SegmentMappingData.from_params(
        mapping_data_params, time_range=init_time_range
    )
    camera_params = init_data.img_data.camera_params

    # Sync algorithm pixel_len_m with localization data's aerial scale.
    print("Loading localization data...")
    loc_data = CrossViewLocalizationData.from_params(loc_data_params)
    matching_pipeline.algorithm.pixel_len_m = loc_data.aerial_img_scale

    print("Setting up segmenter and mapper...")
    segmenter = Segmenter(segmenter_params, depth_cam_params=camera_params)

    converter = SegmentToPrimitiveConverter(conversion_params)
    ground_submap_mapping = GroundSubmapPrimitiveMapping(
        ground_submap_params,
        converter,
        ground_segmenter_params,
        place_recognition,
    )
    mapper = SegmentMapper(
        mapping_params,
        camera_params,
        ground_submap_mapping=ground_submap_mapping,
        place_recognition=place_recognition,
    )

    pipeline = CrossViewIncremental(
        mapping_params=mapping_params,
        segmenter=segmenter,
        mapper=mapper,
        conversion_params=conversion_params,
        algorithm=matching_pipeline.algorithm,
        aerial_submaps=aerial_submaps,
        data=loc_data,
        rpgo_params=rpgo_params,
        incremental_params=incremental_params,
        viz_params=viz_params,
        output_dir=output_dir,
    )
    pipeline._params_path = params_path

    # Save params + commit hash (mirror cross_view_matching).
    all_params = [
        mapping_params,
        mapping_data_params,
        segmenter_params,
        ground_submap_params,
        conversion_params,
        ground_segmenter_params,
        pipeline_params,
        aerial_patch_params,
        segment_match_params,
        register_params,
        rpgo_params,
        incremental_params,
        loc_data_params,
        viz_params,
    ]
    if pr_params is not None:
        all_params.append(pr_params)
    save_params(output_dir, *all_params)
    save_commit_hash(output_dir)

    wc_t0 = time.time()
    if mapping_data_params.max_time is None or full_t0 is None:
        print("Running incremental pipeline (no chunking)...")
        pipeline.run(init_data)
    else:
        chunk_idx = 0
        print(
            f"Running chunk {chunk_idx} ({full_t0:.2f} to {init_time_range[1]:.2f})..."
        )
        pipeline.run(init_data)
        del init_data
        chunk_start = init_time_range[1]
        chunk_idx += 1
        while chunk_start < full_tf:
            chunk_end = min(chunk_start + mapping_data_params.max_time, full_tf)
            print(
                f"Running chunk {chunk_idx} ({chunk_start:.2f} to {chunk_end:.2f})..."
            )
            data = SegmentMappingData.from_params(
                mapping_data_params, time_range=(chunk_start, chunk_end)
            )
            pipeline.run(data)
            del data
            chunk_start = chunk_end
            chunk_idx += 1
    wall_time = time.time() - wc_t0
    print(f"Pipeline took {wall_time:.2f}s")

    print("Writing outputs...")
    pipeline.write_outputs(save_viz=save_viz)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online cross-view localization.")
    parser.add_argument("-p", "--params", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument(
        "--aerial",
        type=str,
        required=True,
        help="Path to aerial directory containing segments/*.pkl.",
    )
    parser.add_argument("-r", "--run", type=str, default=None)
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip per-pair match visualizations (heatmaps still rendered).",
    )
    args = parser.parse_args()

    cross_view_incremental(
        args.params,
        args.output,
        aerial_dir=args.aerial,
        run=args.run,
        save_viz=not args.no_viz,
    )
