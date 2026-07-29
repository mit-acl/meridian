"""Incremental cross-view localization algorithm.

`IncrementalLocalization` is the algorithm core extracted from the
`cross_view_incremental` pipeline. It is a state machine that ingests one new
ground submap at a time (plus the current robot trajectory) and maintains a
global-localization estimate against a fixed database of aerial submaps. It owns
no I/O, visualization, or data loading — the pipeline (or a ROS wrapper) is
responsible for those and drives this class via `update()`.

State machine (two gates before committing to global localization):

- PRE: configured-mode matching with free rotation, CLIPPER over accumulated
  candidates, no PGO. Once CLIPPER inliers >= consistent_loop_closure_thresh
  (gate #1), an attempt is made.
- ATTEMPT (synchronous, may be retried): PGO on PRE candidates -> rough
  trajectory -> rerun max_intersection + translation_only matching on
  submaps 1..n -> CLIPPER on rerun candidates. Gate #2: if rerun inliers
  >= rot_constrained_consistent_lc_thresh AND inliers-per-submap
  >= rot_constrained_consistent_lc_frac, commit (replace candidates, run final
  PGO, transition to POST). Otherwise stay in PRE and retry on the next submap.
- POST: per new submap match only that submap with max_intersection +
  translation_only against the latest optimized trajectory, append candidates,
  CLIPPER + PGO on the full accumulated set.

The instantaneous pose is causal: None before successful global localization,
T_utm_lastopt @ inv(T_odom_lastopt) @ pose_cam after.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from meridian.cross_view.candidates import (
    LocalizationContext,
    build_candidates_from_match_result,
)
from meridian.cross_view.matching import CrossViewMatching, CrossViewMatchResult
from meridian.cross_view.rpgo import (
    CrossViewRPGO,
    CrossViewRPGOResult,
    pose_data_from_trajectory,
)
from meridian.map3d.submap import Submap
from meridian.params import CrossViewIncrementalParams, CrossViewRPGOParams

logger = logging.getLogger(__name__)


@dataclass
class AttemptDiagnostics:
    """Per-attempt artifacts from a PRE->POST gate-2 rerun, returned to the
    caller so it can render the diagnostic rerun visualization. Populated once
    the rerun matching + CLIPPER have run (whether or not gate-2 passes); None
    when the attempt aborted earlier (e.g., initial PGO failed)."""

    ground_key: str
    rerun_result: CrossViewMatchResult
    rerun_candidates: List[dict]
    rerun_inliers: np.ndarray
    rerun_lc_obj: Optional[float]
    T_utm_odom_local: Optional[np.ndarray]
    optimized_trajectory_local: Optional[List[np.ndarray]]


@dataclass
class IncrementalUpdateResult:
    """Outcome of a single `IncrementalLocalization.update` call. Carries
    everything the pipeline needs for visualization, diagnostics, and timing
    aggregation without exposing algorithm internals."""

    state: str
    match_result: CrossViewMatchResult
    inlier_indices: np.ndarray
    objective: Optional[float]
    # The viz pose belief is recomputed every update (PRE refreshes the rigid
    # frame-align; POST may re-run PGO), so the caller should rebuild its viz
    # trajectory whenever this is True.
    transform_changed: bool
    gate1_met: bool
    attempted: bool
    gate2_passed: bool
    attempt_wall_time: Optional[float]
    attempt: Optional[AttemptDiagnostics]
    # Per-call durations measured during this update, e.g. {"match": [...],
    # "outlier_rej": [...], "pgo": [...]}; the caller extends its own timing
    # accumulators with these.
    timings: Dict[str, List[float]]
    n_total_submaps: int


@dataclass
class IncrementalLocalization:
    matcher: CrossViewMatching
    aerial_submaps: Dict[str, Submap]
    rpgo_params: CrossViewRPGOParams
    incremental_params: CrossViewIncrementalParams
    context: LocalizationContext

    # State
    _state: str = field(default="PRE", init=False)
    _candidates: List[dict] = field(default_factory=list, init=False)
    # Ground submaps accumulated across updates, keyed by ground_key. Needed for
    # the gate-2 rerun, which re-matches submaps 1..n.
    _ground_submaps: Dict[str, Submap] = field(default_factory=dict, init=False)
    _results_per_submap: Dict[str, object] = field(default_factory=dict, init=False)
    _match_details_per_submap: Dict[str, Dict[str, list]] = field(
        default_factory=dict, init=False
    )
    _last_optimized_trajectory: Optional[List[np.ndarray]] = field(
        default=None, init=False
    )
    # Camera-frame anchors for propagating future poses consistently with the
    # most recent PGO. Future T_utm_camera(t) = T_utm_lastopt @ inv(T_odom_lastopt) @ T_odom_cam(t).
    _T_utm_lastopt: Optional[np.ndarray] = field(default=None, init=False)
    _T_odom_lastopt: Optional[np.ndarray] = field(default=None, init=False)
    # Viz-only T_utm_odom (PRE state); refreshed via rigid frame_align on
    # current candidates. Independent of the lastopt anchors.
    _T_utm_odom_viz: Optional[np.ndarray] = field(default=None, init=False)
    # Most recent accepted CLIPPER outlier-rejection objective (for the
    # `allowable_outlier_lc_obj_drop` guard). None until first POST commit.
    _last_lc_out_rej_obj: Optional[float] = field(default=None, init=False)
    _optimized_pose_data: Optional[object] = field(default=None, init=False)
    # 2D-projected, line-filtered aerial submaps. Aerial is static across the
    # run, so we preprocess once at init and pass to every match call.
    _aerial_submaps_2d: Dict[str, Submap] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._aerial_submaps_2d = self.matcher.preprocess_aerial_submaps_2d(
            self.aerial_submaps
        )

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def candidates(self) -> List[dict]:
        return self._candidates

    @property
    def results_per_submap(self) -> Dict[str, object]:
        return self._results_per_submap

    @property
    def match_details_per_submap(self) -> Dict[str, Dict[str, list]]:
        return self._match_details_per_submap

    @property
    def last_optimized_trajectory(self) -> Optional[List[np.ndarray]]:
        return self._last_optimized_trajectory

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------
    def update(
        self,
        new_submap: Submap,
        ground_key: str,
        poses_cam_history: List[np.ndarray],
        times_history: np.ndarray,
    ) -> IncrementalUpdateResult:
        """Ingest one new ground submap and advance the localization state.

        `poses_cam_history` / `times_history` are the full robot trajectory
        (T_odom_camera per frame) and timestamps as of this submap.
        """
        times_arr = np.asarray(times_history)
        self._ground_submaps[ground_key] = new_submap
        ground_submaps = {ground_key: new_submap}
        timings: Dict[str, List[float]] = {"match": [], "outlier_rej": [], "pgo": []}

        # Match the single new submap.
        t_match_start = time.time()
        if self._state == "PRE":
            match_result = self.matcher.cross_view_match(
                self.aerial_submaps,
                ground_submaps,
                reference_trajectory=None,
                T_camera_flu=self.context.T_camera_flu,
                show_progress=False,
                local_to_pixel_fn=self.context.local_to_pixel_fn,
                gt_trajectory=self.context.gt_pose_data,
                aerial_submaps_2d=self._aerial_submaps_2d,
            )
        else:
            # Propagate the last-accepted PGO forward via odom over the full
            # trajectory so the matcher's reference reflects where the camera
            # *currently is*, not just where it was at the last accepted PGO.
            propagated_ref = self._build_propagated_reference_pose_data(
                poses_cam_history, times_arr
            )
            match_result = self.matcher.cross_view_match_max_intersection(
                self.aerial_submaps,
                ground_submaps,
                reference_trajectory=propagated_ref or self._optimized_pose_data,
                T_camera_flu=self.context.T_camera_flu,
                translation_only=True,
                show_progress=False,
                local_to_pixel_fn=self.context.local_to_pixel_fn,
                gt_trajectory=self.context.gt_pose_data,
                aerial_submaps_2d=self._aerial_submaps_2d,
            )
        timings["match"].append(time.time() - t_match_start)

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
            self.context,
            min_assoc,
        )
        self._candidates.extend(new_candidates)

        # Outlier rejection on accumulated candidates.
        rpgo = CrossViewRPGO(params=self.rpgo_params)
        t_or_start = time.time()
        try:
            inlier_indices, _M, _C, lc_obj = rpgo.solve_clipper_only(
                self._candidates,
                poses_cam_history,
                times_arr,
            )
        except Exception as e:
            logger.warning(f"CLIPPER failed (state={self._state}): {e}")
            inlier_indices = np.array([], dtype=int)
            lc_obj = None
        timings["outlier_rej"].append(time.time() - t_or_start)

        n_inliers = len(inlier_indices)
        logger.info(
            f"[submap {ground_key}] state={self._state} "
            f"candidates={len(self._candidates)} inliers={n_inliers}"
        )

        # POST uses the PGO chain, so the viz frame_align is PRE-only.
        if self._state == "PRE":
            self._refresh_viz_utm_odom(rpgo, inlier_indices)

        gate1_met = False
        attempted = False
        gate2_passed = False
        attempt_wall_time: Optional[float] = None
        attempt_diag: Optional[AttemptDiagnostics] = None
        if self._state == "PRE":
            if n_inliers >= self.incremental_params.consistent_loop_closure_thresh:
                gate1_met = True
                attempted = True
                gate2_passed, attempt_diag, attempt_wall_time = (
                    self._attempt_global_localization(
                        ground_key, poses_cam_history, times_arr, timings
                    )
                )
                if gate2_passed:
                    self._state = "POST"
        else:
            self._run_post_pgo(
                inlier_indices, lc_obj, poses_cam_history, times_arr, timings
            )

        return IncrementalUpdateResult(
            state=self._state,
            match_result=match_result,
            inlier_indices=inlier_indices,
            objective=lc_obj,
            transform_changed=True,
            gate1_met=gate1_met,
            attempted=attempted,
            gate2_passed=gate2_passed,
            attempt_wall_time=attempt_wall_time,
            attempt=attempt_diag,
            timings=timings,
            n_total_submaps=len(self._ground_submaps),
        )

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
    def _run_post_pgo(
        self,
        inlier_indices: np.ndarray,
        objective: Optional[float],
        poses_cam_history: List[np.ndarray],
        times_arr: np.ndarray,
        timings: Dict[str, List[float]],
    ):
        if len(inlier_indices) == 0:
            return

        # Guard against degenerate re-solves: if the new outlier-rejection
        # objective dropped meaningfully versus the last accepted POST step,
        # skip this update and keep the previous lastopt anchors. Skipped when
        # `objective` is None (e.g., GT-inliers path) so a missing measurement
        # doesn't silently freeze updates.
        drop = self.incremental_params.allowable_outlier_lc_obj_drop
        if (
            drop is not None
            and self._last_lc_out_rej_obj is not None
            and objective is not None
            and objective < self._last_lc_out_rej_obj - drop
        ):
            logger.info(
                f"POST PGO update rejected: objective {objective:.4f} dropped > "
                f"{drop} below previous {self._last_lc_out_rej_obj:.4f}; "
                "keeping previous lastopt anchors."
            )
            return

        rpgo = CrossViewRPGO(params=self.rpgo_params)
        t_pgo_start = time.time()
        T_utm_odom = rpgo._frame_align(self._candidates, inlier_indices)
        if self.rpgo_params.optimization_method == "pgo":
            optimized_trajectory = rpgo._pgo(
                self._candidates,
                inlier_indices,
                poses_cam_history,
                times_arr,
                self.context.T_camera_flu,
                T_utm_odom,
            )
        else:
            optimized_trajectory = rpgo._apply_rigid_transform(
                T_utm_odom,
                poses_cam_history,
                self.context.T_camera_flu,
            )
        timings["pgo"].append(time.time() - t_pgo_start)

        self._last_optimized_trajectory = optimized_trajectory
        self._set_lastopt_anchors(optimized_trajectory, poses_cam_history)
        self._optimized_pose_data = self._build_camera_pose_data(
            optimized_trajectory, times_arr
        )
        if objective is not None:
            self._last_lc_out_rej_obj = objective
        self._commit_accepted_inliers(inlier_indices)

    def _commit_accepted_inliers(self, inlier_indices: np.ndarray):
        """When `commit_accepted_inliers` is enabled, remove non-inlier
        candidates that share a ground_key with any accepted inlier. Locks the
        chosen aerial hypothesis per submap so subsequent CLIPPER runs cannot
        drift into a different inlier basin for an already-resolved submap.

        When `delay_most_recent_lc_commit_num` is > 0, defer that commit for
        the N most recent ground submaps: their inliers still feed PGO this
        cycle, but their alternative hypotheses are preserved until at least
        N newer submaps have been added.
        """
        if not self.incremental_params.commit_accepted_inliers:
            return
        if len(inlier_indices) == 0:
            return
        inlier_set = {int(i) for i in inlier_indices}
        delay = self.incremental_params.delay_most_recent_lc_commit_num
        if delay > 0:
            latest_idx = max(int(c["ground_key"]) for c in self._candidates)
            committed_ground_keys = {
                self._candidates[i]["ground_key"]
                for i in inlier_set
                if int(self._candidates[i]["ground_key"]) <= latest_idx - delay
            }
        else:
            committed_ground_keys = {
                self._candidates[i]["ground_key"] for i in inlier_set
            }
        if not committed_ground_keys:
            return
        pruned = [
            c
            for i, c in enumerate(self._candidates)
            if i in inlier_set or c["ground_key"] not in committed_ground_keys
        ]
        n_removed = len(self._candidates) - len(pruned)
        if n_removed:
            logger.info(
                f"commit_accepted_inliers: pruned {n_removed} alternative "
                f"hypotheses across {len(committed_ground_keys)} committed "
                f"submaps; candidate pool {len(self._candidates)} -> {len(pruned)}"
            )
        self._candidates = pruned

    # ------------------------------------------------------------------
    # Reference / anchor helpers
    # ------------------------------------------------------------------
    def _build_propagated_reference_pose_data(
        self, poses_cam_history: List[np.ndarray], times_arr: np.ndarray
    ):
        """Build a camera-frame PoseData covering the full trajectory by
        propagating the last-accepted PGO anchors forward via odom:
            T_utm_camera(t) = _T_utm_lastopt @ inv(_T_odom_lastopt) @ pose_cam(t).
        Used as the matcher's `reference_trajectory` so aerial-patch selection
        tracks the live camera pose instead of clamping to the last accepted
        PGO frame. Returns None if anchors are not yet set.
        """
        if self._T_utm_lastopt is None or self._T_odom_lastopt is None:
            return None
        if poses_cam_history is None or len(poses_cam_history) == 0:
            return None
        T_odom_lastopt_inv = np.linalg.inv(self._T_odom_lastopt)
        cam_traj = [
            self._T_utm_lastopt @ T_odom_lastopt_inv @ pose_cam
            for pose_cam in poses_cam_history
        ]
        return pose_data_from_trajectory(cam_traj, np.asarray(times_arr))

    def _set_lastopt_anchors(
        self,
        optimized_trajectory: List[np.ndarray],
        poses_cam_history: List[np.ndarray],
    ):
        """Capture the camera-frame T_utm and T_odom poses at the most recent
        PGO step. Used by `instantaneous_pose` to propagate future poses
        consistently with PGO via:
            T_utm_cam(t) = T_utm_lastopt @ inv(T_odom_lastopt) @ T_odom_cam(t).
        `optimized_trajectory` is in body frame (T_utm_body); convert back to
        camera frame so the chain composes directly with `pose_cam`.
        """
        if not optimized_trajectory:
            return
        idx = len(optimized_trajectory) - 1
        if self.context.T_camera_flu is not None:
            T_camera_flu_inv = np.linalg.inv(self.context.T_camera_flu)
            self._T_utm_lastopt = optimized_trajectory[idx] @ T_camera_flu_inv
        else:
            self._T_utm_lastopt = optimized_trajectory[idx]
        # poses_cam_history is already in camera frame (T_odom_camera).
        self._T_odom_lastopt = poses_cam_history[idx]

    def _build_camera_pose_data(
        self, body_trajectory: List[np.ndarray], times: np.ndarray
    ):
        """Convert a body-frame optimized trajectory to camera-frame PoseData
        suitable as a reference_trajectory for matching."""
        if self.context.T_camera_flu is not None:
            T_flu_camera = np.linalg.inv(self.context.T_camera_flu)
            cam_traj = [T @ T_flu_camera for T in body_trajectory]
        else:
            cam_traj = body_trajectory
        return pose_data_from_trajectory(cam_traj, np.asarray(times))

    # ------------------------------------------------------------------
    # PRE -> POST attempt (gated; may be retried)
    # ------------------------------------------------------------------
    def _attempt_global_localization(
        self,
        ground_key: str,
        poses_cam_history: List[np.ndarray],
        times_arr: np.ndarray,
        timings: Dict[str, List[float]],
    ) -> Tuple[bool, Optional[AttemptDiagnostics], float]:
        """Attempt global localization.

        Returns ``(gate2_passed, diagnostics, wall_time)``. ``diagnostics`` is
        populated once the rerun matching + CLIPPER have run (for the rerun viz,
        whether or not gate-2 passes) and is None for early aborts.

        Invariant: must not mutate self._candidates, self._results_per_submap,
        self._match_details_per_submap, self._last_optimized_trajectory,
        self._T_utm_lastopt, self._T_odom_lastopt, or self._optimized_pose_data
        until gate #2 passes.
        """
        wc_start = time.time()
        gate2_thresh = self.incremental_params.rot_constrained_consistent_lc_thresh
        gate2_frac = self.incremental_params.rot_constrained_consistent_lc_frac
        logger.info(f"[global-loc] gate-1 met: {len(self._candidates)} candidates")

        rpgo = CrossViewRPGO(params=self.rpgo_params)

        # Step 1: initial PGO on PRE candidates.
        t_pgo_start = time.time()
        try:
            initial_result = rpgo.solve(
                self._candidates,
                poses_cam_history,
                times_arr,
                self.context.T_camera_flu,
            )
        except Exception as e:
            logger.warning(f"[global-loc] initial PGO failed: {e}")
            timings["pgo"].append(time.time() - t_pgo_start)
            return False, None, time.time() - wc_start
        timings["pgo"].append(time.time() - t_pgo_start)

        if not initial_result.success:
            logger.warning("[global-loc] initial PGO returned no success — aborting.")
            return False, None, time.time() - wc_start

        # Step 2: rough-trajectory pose data for known-rotation matching.
        rough_pose_data = self._build_camera_pose_data(
            initial_result.optimized_trajectory, times_arr
        )

        # Step 3: rerun matching on submaps 1..n with max_intersection +
        # translation_only.
        all_ground_submaps = dict(self._ground_submaps)

        t_match_start = time.time()
        rerun_result = self.matcher.cross_view_match_max_intersection(
            self.aerial_submaps,
            all_ground_submaps,
            reference_trajectory=rough_pose_data,
            T_camera_flu=self.context.T_camera_flu,
            translation_only=True,
            show_progress=False,
            local_to_pixel_fn=self.context.local_to_pixel_fn,
            gt_trajectory=self.context.gt_pose_data,
            aerial_submaps_2d=self._aerial_submaps_2d,
        )
        timings["match"].append(time.time() - t_match_start)

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
            self.context,
            min_assoc,
        )

        # Step 5: CLIPPER on rerun candidates (still local, no commit yet).
        t_or_start = time.time()
        rerun_lc_obj: Optional[float] = None
        try:
            rerun_inliers, _M, _C, rerun_lc_obj = rpgo.solve_clipper_only(
                rerun_candidates,
                poses_cam_history,
                times_arr,
            )
        except Exception as e:
            logger.warning(f"[global-loc] rerun CLIPPER failed: {e}")
            rerun_inliers = np.array([], dtype=int)
        timings["outlier_rej"].append(time.time() - t_or_start)

        # Compute the rerun's frame_align + PGO trajectory now (local only; no
        # self.* mutation). Used for the diagnostic rerun viz regardless of
        # whether gate-2 passes, and reused at commit time if it does.
        T_utm_odom_local: Optional[np.ndarray] = None
        optimized_trajectory_local: Optional[List[np.ndarray]] = None
        if len(rerun_inliers) > 0:
            t_pgo_start = time.time()
            try:
                T_utm_odom_local = rpgo._frame_align(rerun_candidates, rerun_inliers)
                if self.rpgo_params.optimization_method == "pgo":
                    optimized_trajectory_local = rpgo._pgo(
                        rerun_candidates,
                        rerun_inliers,
                        poses_cam_history,
                        times_arr,
                        self.context.T_camera_flu,
                        T_utm_odom_local,
                    )
                else:
                    optimized_trajectory_local = rpgo._apply_rigid_transform(
                        T_utm_odom_local,
                        poses_cam_history,
                        self.context.T_camera_flu,
                    )
            except Exception as e:
                logger.warning(f"[global-loc] rerun frame_align/PGO failed: {e}")
                T_utm_odom_local = None
                optimized_trajectory_local = None
            timings["pgo"].append(time.time() - t_pgo_start)

        diag = AttemptDiagnostics(
            ground_key=ground_key,
            rerun_result=rerun_result,
            rerun_candidates=rerun_candidates,
            rerun_inliers=rerun_inliers,
            rerun_lc_obj=rerun_lc_obj,
            T_utm_odom_local=T_utm_odom_local,
            optimized_trajectory_local=optimized_trajectory_local,
        )

        # Step 6: gate #2 check. Guard against eventual inlier accumulation in
        # long aerial runs: both an absolute count and an inliers-per-submap
        # fraction must be met.
        n_inliers = len(rerun_inliers)
        n_submaps = len(all_ground_submaps)
        inlier_frac = n_inliers / n_submaps if n_submaps > 0 else 0.0
        if n_inliers < gate2_thresh or inlier_frac < gate2_frac:
            logger.warning(
                f"[global-loc] gate-2 FAILED: rerun inliers={n_inliers}/"
                f"{n_submaps} submaps ({inlier_frac:.1%}); requires >= "
                f"rot_constrained_consistent_lc_thresh={gate2_thresh} AND "
                f">= rot_constrained_consistent_lc_frac={gate2_frac:.2f}. "
                f"Staying in PRE; preserving {len(self._candidates)} PRE candidates."
            )
            return False, diag, time.time() - wc_start

        # Step 7: commit (gate #2 passed). Reuse the trajectory computed above —
        # gate-2 pass implies len(rerun_inliers) > 0, so both locals are
        # non-None unless the frame_align/PGO itself raised. In that rare case,
        # abort the commit rather than re-running here.
        if T_utm_odom_local is None or optimized_trajectory_local is None:
            logger.warning(
                "[global-loc] gate-2 passed but rerun PGO failed earlier; "
                "aborting commit and staying in PRE."
            )
            return False, diag, time.time() - wc_start
        optimized_trajectory = optimized_trajectory_local

        self._candidates = rerun_candidates
        self._results_per_submap = dict(rerun_result.results)
        self._match_details_per_submap = dict(rerun_result.match_details)
        self._last_optimized_trajectory = optimized_trajectory
        self._set_lastopt_anchors(optimized_trajectory, poses_cam_history)
        self._optimized_pose_data = self._build_camera_pose_data(
            optimized_trajectory, times_arr
        )
        # Seed the drop-guard baseline so the first POST step has something to
        # compare against.
        if rerun_lc_obj is not None:
            self._last_lc_out_rej_obj = rerun_lc_obj
        self._commit_accepted_inliers(rerun_inliers)

        wall_time = time.time() - wc_start
        logger.info(
            f"[global-loc] gate-2 PASSED: {len(rerun_candidates)} rerun candidates, "
            f"{len(rerun_inliers)} inliers >= {gate2_thresh}, {wall_time:.2f}s"
        )
        return True, diag, wall_time

    # ------------------------------------------------------------------
    # Pose-belief queries (caller stores / plots the results)
    # ------------------------------------------------------------------
    def instantaneous_pose(self, pose_cam: np.ndarray) -> Optional[np.ndarray]:
        """Causal T_utm_body for the live camera pose, or None when not yet
        globally localized.

        Propagates the last PGO estimate forward via odom-frame relative motion:
        T_utm_camera = T_utm_lastopt @ inv(T_odom_lastopt) @ pose_cam. This keeps
        the live pose consistent with the most recent PGO output instead of the
        rigid frame-align (which discards the per-pose deformation PGO produced).
        """
        if (
            self._state == "POST"
            and self._T_utm_lastopt is not None
            and self._T_odom_lastopt is not None
        ):
            T_utm_camera = (
                self._T_utm_lastopt @ np.linalg.inv(self._T_odom_lastopt) @ pose_cam
            )
            return self._apply_camera_flu(T_utm_camera)
        return None

    def estimate_T_utm_body(self, idx: int, pose_cam: np.ndarray) -> np.ndarray:
        """Best-effort T_utm_body for frame `idx` under the current estimate,
        for visualization (includes a PRE-state estimate, unlike the causal
        `instantaneous_pose`).

        POST + idx within PGO range: use the PGO-deformed trajectory directly.
        POST tail (frames after the last PGO solve): rigid-propagate from the
        lastopt anchor. PRE: rigid `_T_utm_odom_viz` (frame_align over current
        inliers). Otherwise NaN (no viz-usable estimate yet).
        """
        if (
            self._state == "POST"
            and self._last_optimized_trajectory is not None
            and self._T_utm_lastopt is not None
            and self._T_odom_lastopt is not None
        ):
            opt = self._last_optimized_trajectory
            if idx < len(opt):
                return opt[idx]
            T_odom_inv = np.linalg.inv(self._T_odom_lastopt)
            T_utm_cam = self._T_utm_lastopt @ T_odom_inv @ pose_cam
            return self._apply_camera_flu(T_utm_cam)
        if self._T_utm_odom_viz is not None:
            return self._apply_camera_flu(self._T_utm_odom_viz @ pose_cam)
        return np.full((4, 4), np.nan)

    def _apply_camera_flu(self, T_utm_camera: np.ndarray) -> np.ndarray:
        if self.context.T_camera_flu is not None:
            return T_utm_camera @ self.context.T_camera_flu
        return T_utm_camera

    def _refresh_viz_utm_odom(self, rpgo: CrossViewRPGO, inlier_indices: np.ndarray):
        if len(inlier_indices) > 0:
            indices = inlier_indices
        elif len(self._candidates) > 0:
            indices = np.array([len(self._candidates) - 1])
        else:
            return
        try:
            self._T_utm_odom_viz = rpgo._frame_align(self._candidates, indices)
        except Exception as e:
            logger.debug(f"viz frame_align failed: {e}")

    # ------------------------------------------------------------------
    # Full solve (for final-output / per-submap diagnostic visualization)
    # ------------------------------------------------------------------
    def solve_full(
        self, poses_cam_history: List[np.ndarray], times_history: np.ndarray
    ) -> Optional[CrossViewRPGOResult]:
        """Run a full CLIPPER + PGO/frame_align solve over the current candidate
        pool. Returns None on failure or unsuccessful solve. Intentionally
        heavyweight; used only for diagnostic / end-of-run visualization."""
        if not self._candidates:
            return None
        rpgo = CrossViewRPGO(params=self.rpgo_params)
        try:
            result = rpgo.solve(
                self._candidates,
                poses_cam_history,
                np.asarray(times_history),
                self.context.T_camera_flu,
            )
        except Exception as e:
            logger.warning(f"solve_full failed: {e}")
            return None
        if not result.success:
            return None
        return result
