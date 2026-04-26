import logging
import time
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from gen_seg_match.cross_view.place_recognition import CrossViewPlaceRecognition
from gen_seg_match.map3d.submap import Submap
from gen_seg_match.match.segment_matcher import SegmentMatcher
from gen_seg_match.params import CrossViewMatchingParams
from gen_seg_match.params.segment_to_primitive_params import AerialPatchParams
from gen_seg_match.pipeline.result import (
    PoseEstimationResult,
    PoseEstimationResultMatrix,
)
from gen_seg_match.register.registerer import (
    Registerer2D,
)
from gen_seg_match.segment.segment_types import SegmentList

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _aerial_key_to_tuple(key: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in key.split("_"))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SingleMatchResult:
    pose_result: PoseEstimationResult
    aerial_segs_processed: SegmentList = None
    ground_segs_processed: SegmentList = None
    matched_ground: SegmentList = None
    matched_aerial: SegmentList = None


@dataclass
class CrossViewMatchResult:
    results: Dict[str, PoseEstimationResultMatrix]  # ground_key -> matrix
    match_details: Dict[str, Dict[str, List[SingleMatchResult]]] = field(
        default_factory=dict
    )  # ground_key -> aerial_key -> list of match hypotheses


# ---------------------------------------------------------------------------
# CrossViewMatching — pure matching class (no submap creation)
# ---------------------------------------------------------------------------


@dataclass
class CrossViewMatching:
    pipeline_params: CrossViewMatchingParams
    aerial_patch_params: AerialPatchParams
    pixel_len_m: float
    matcher: SegmentMatcher
    registerer: Registerer2D
    place_recognition: CrossViewPlaceRecognition = None

    # ------------------------------------------------------------------
    # Cross-view matching (no I/O)
    # ------------------------------------------------------------------

    def cross_view_match(
        self,
        aerial_submaps: Dict[str, Submap],
        ground_submaps: Dict[str, Submap],
        reference_trajectory=None,
        T_camera_flu: np.ndarray = None,
        matching_mode: str = None,
        translation_only: bool = None,
        show_progress: bool = False,
        local_to_pixel_fn=None,
        gt_trajectory=None,
    ) -> CrossViewMatchResult:
        """Match aerial and ground submaps without any I/O.

        Args:
            aerial_submaps: Aerial submap dict (key -> Submap).
            ground_submaps: Ground submap dict (key -> Submap).
            reference_trajectory: Any PoseData providing pose(time) -> 4x4.
                Used for rotation extraction and GT-based filtering.
            T_camera_flu: Camera-to-FLU transform.
            matching_mode: Override self.pipeline_params.matching_mode.
            translation_only: Override self.pipeline_params.translation_only.

        Returns:
            CrossViewMatchResult with per-ground PoseEstimationResultMatrix
            and per-pair SingleMatchResult.
        """
        mode = matching_mode or self.pipeline_params.matching_mode
        do_translation_only = (
            translation_only
            if translation_only is not None
            else self.pipeline_params.translation_only
        )

        if not aerial_submaps:
            raise ValueError(
                "No aerial submaps provided — check the aerial directory path."
            )
        if not ground_submaps:
            raise ValueError(
                "No ground submaps provided — check the ground directory path."
            )

        aerial_submaps_2d, ground_submaps_2d = self._preprocess_submaps_2d(
            aerial_submaps, ground_submaps
        )

        aerial_key_to_tuple = _aerial_key_to_tuple
        aerial_x_max = max(aerial_key_to_tuple(key)[0] for key in aerial_submaps.keys())
        aerial_y_max = max(aerial_key_to_tuple(key)[1] for key in aerial_submaps.keys())

        # Precompute crop geometry
        px_per_m = 1.0 / self.pixel_len_m
        patch_size_px = int(
            self.aerial_patch_params.aerial_img_patch_side_len_m * px_per_m
        )
        stride = int(
            patch_size_px * (1.0 - self.aerial_patch_params.aerial_img_patch_overlap)
        )
        stride_m = stride * self.pixel_len_m
        patch_size_m_px = patch_size_px * self.pixel_len_m

        # Precompute VPR top-k patches if needed
        need_vpr = mode == "vpr" or (mode == "gt" and reference_trajectory is None)
        vpr_top_k_sets = {}
        if need_vpr:
            assert self.place_recognition is not None, (
                "VPR mode requires place_recognition to be configured"
            )
            sim_matrix, ground_keys_sorted, aerial_keys_sorted = (
                self.place_recognition.compute_similarity_matrix(
                    ground_submaps, aerial_submaps
                )
            )
            from gen_seg_match.pipeline.cross_view_place_recognition import (
                CrossViewPlaceRecognitionPipeline,
            )

            vpr_top_k = CrossViewPlaceRecognitionPipeline.compute_top_k_patches(
                sim_matrix,
                ground_keys_sorted,
                aerial_keys_sorted,
                self.place_recognition.params.k_nearest_neighbors,
            )
            vpr_top_k_sets = {gk: set(patches) for gk, patches in vpr_top_k.items()}

        all_results = {}
        all_details = {}

        iterator = ground_submaps_2d.items()
        if show_progress:
            iterator = tqdm(iterator, desc="Matching", total=len(ground_submaps_2d))
        for ground_key, ground_sm_i in iterator:
            results_matrix = PoseEstimationResultMatrix(
                (aerial_x_max + 1, aerial_y_max + 1)
            )
            details_for_ground = {}

            ground_pose_ref = None
            if reference_trajectory is not None:
                ground_pose_ref = reference_trajectory.pose(
                    ground_submaps[ground_key].time
                )
            ground_pose_gt = None
            if gt_trajectory is not None:
                try:
                    ground_pose_gt = gt_trajectory.pose(ground_submaps[ground_key].time)
                except Exception:
                    pass
            T_ground_odom_ground_robot = ground_sm_i.metadata["camera_pose"]

            for aerial_key, aerial_sm_j in aerial_submaps_2d.items():
                i_a, j_a = aerial_key_to_tuple(aerial_key)

                # --- Mode-based filtering ---
                filtered_keys = self._get_filtered_aerial_keys(
                    mode,
                    ground_key,
                    aerial_key,
                    i_a,
                    j_a,
                    aerial_sm_j,
                    ground_pose_ref,
                    stride_m,
                    patch_size_m_px,
                    vpr_top_k_sets,
                    aerial_submaps_2d,
                    ground_sm_i,
                    reference_trajectory,
                )
                if not filtered_keys:
                    continue

                single_results = self._match_single_pair(
                    aerial_sm_j,
                    ground_sm_i,
                    reference_trajectory,
                    T_camera_flu,
                    ground_pose_ref,
                    T_ground_odom_ground_robot,
                    do_translation_only,
                    local_to_pixel_fn=local_to_pixel_fn,
                    ground_pose_gt=ground_pose_gt,
                )

                results_matrix[i_a, j_a] = [r.pose_result for r in single_results]
                details_for_ground[aerial_key] = single_results

            all_results[ground_key] = results_matrix
            all_details[ground_key] = details_for_ground

        return CrossViewMatchResult(results=all_results, match_details=all_details)

    # ------------------------------------------------------------------
    # Helpers for cross_view_match
    # ------------------------------------------------------------------

    def _preprocess_submaps_2d(
        self,
        aerial_submaps: Dict[str, Submap],
        ground_submaps: Dict[str, Submap],
    ) -> Tuple[Dict[str, Submap], Dict[str, Submap]]:
        """Convert submaps to 2D and filter short lines."""
        aerial_submaps_2d: Dict[str, Submap] = {}
        ground_submaps_2d: Dict[str, Submap] = {}
        for submaps_2d_dict, original_submaps_dict in [
            (aerial_submaps_2d, aerial_submaps),
            (ground_submaps_2d, ground_submaps),
        ]:
            for key, submap in original_submaps_dict.items():
                segments_2d = submap.segments.to_dim(2)
                filtered_lines = [
                    line
                    for line in segments_2d.get_lines()
                    if line.get_length() >= self.pipeline_params.match_min_len_m
                ]
                segments_2d = segments_2d.get_points() + SegmentList(filtered_lines)
                submaps_2d_dict[key] = deepcopy(submap)
                submaps_2d_dict[key].segments = segments_2d

        # Transfer height for ground submaps
        for ground_key in ground_submaps_2d.keys():
            for segment in ground_submaps_2d[ground_key].segments:
                segment.height = (
                    ground_submaps[ground_key]
                    .segments.get_segment_from_id(segment.id)
                    .height
                )

        return aerial_submaps_2d, ground_submaps_2d

    def _get_filtered_aerial_keys(
        self,
        mode: str,
        ground_key: str,
        aerial_key: str,
        i_a: int,
        j_a: int,
        aerial_sm_j: Submap,
        ground_pose_ref,
        stride_m: float,
        patch_size_m_px: float,
        vpr_top_k_sets: dict,
        aerial_submaps_2d: Dict[str, Submap],
        ground_sm_i: Submap,
        reference_trajectory,
    ) -> bool:
        """Return True if this aerial key should be processed for the given mode."""
        if mode == "gt":
            if ground_pose_ref is None:
                # Fall back to VPR
                if (i_a, j_a) not in vpr_top_k_sets.get(ground_key, set()):
                    return False
            else:
                T_aerial_camera = np.linalg.inv(aerial_sm_j.pose) @ ground_pose_ref
                ground_pos_aerial = T_aerial_camera[:2, 3]
                x1_m = i_a * stride_m
                y1_m = j_a * stride_m
                x2_m = x1_m + patch_size_m_px
                y2_m = y1_m + patch_size_m_px
                if not (
                    x1_m <= ground_pos_aerial[0] <= x2_m
                    and y1_m <= ground_pos_aerial[1] <= y2_m
                ):
                    return False
        elif mode == "vpr":
            if (i_a, j_a) not in vpr_top_k_sets.get(ground_key, set()):
                return False
        elif mode == "max_intersection":
            pass
        # mode == "all": no filtering
        return True

    def _match_single_pair(
        self,
        aerial_sm: Submap,
        ground_sm: Submap,
        reference_trajectory,
        T_camera_flu: np.ndarray,
        ground_pose_ref,
        T_ground_odom_ground_robot: np.ndarray,
        translation_only: bool,
        local_to_pixel_fn=None,
        ground_pose_gt=None,
    ) -> List[SingleMatchResult]:
        """Match a single aerial-ground submap pair.

        Returns a list of SingleMatchResult hypotheses.

        Args:
            ground_pose_ref: Reference pose for rotation constraint (may be PGO output).
            ground_pose_gt: Actual ground truth pose for T_i_j computation.
        """
        # Compute rotation from reference trajectory if available
        R_aerial_ground_2d = None
        if ground_pose_ref is not None:
            T_aerial_camera_ref = np.linalg.inv(aerial_sm.pose) @ ground_pose_ref
            T_aerial_odom = T_aerial_camera_ref @ np.linalg.inv(
                T_ground_odom_ground_robot
            )
            R_aerial_ground_2d = T_aerial_odom[:2, :2]

        # Compute GT T_aerial_ground for T_i_j (visualization/error computation)
        gt_source = ground_pose_gt if ground_pose_gt is not None else ground_pose_ref
        if gt_source is not None:
            T_aerial_camera_gt = np.linalg.inv(aerial_sm.pose) @ gt_source
            if T_camera_flu is not None:
                T_aerial_ground = T_aerial_camera_gt @ T_camera_flu
            else:
                T_aerial_ground = T_aerial_camera_gt
            if local_to_pixel_fn is not None:
                col, row = local_to_pixel_fn(
                    float(T_aerial_ground[0, 3]), float(T_aerial_ground[1, 3])
                )
                T_aerial_ground = T_aerial_ground.copy()
                T_aerial_ground[0, 3] = col * self.pixel_len_m
                T_aerial_ground[1, 3] = row * self.pixel_len_m
            # Zero out Z — matching is 2D SE(2)
            T_aerial_ground = T_aerial_ground.copy()
            T_aerial_ground[2, 3] = 0.0
        else:
            T_aerial_ground = np.zeros((4, 4)) * np.nan

        ground_segs_i = ground_sm.segments
        aerial_segs_j = aerial_sm.segments

        if self.pipeline_params.points_only:
            ground_segs_i = ground_segs_i.get_points()
            aerial_segs_j = aerial_segs_j.get_points()
        elif self.pipeline_params.lines_only:
            ground_segs_i = ground_segs_i.get_lines()
            aerial_segs_j = aerial_segs_j.get_lines()

        match_kwargs = {}
        if translation_only and R_aerial_ground_2d is not None:
            # Apply rotation perturbation if configured
            noise_deg = self.pipeline_params.rot_bias_deg
            lo, hi = self.pipeline_params.uniform_rot_noise_bounds_deg
            if lo != 0.0 or hi != 0.0:
                noise_deg += np.random.uniform(lo, hi)
            if noise_deg != 0.0:
                noise_rad = np.deg2rad(noise_deg)
                R_noise = np.array(
                    [
                        [np.cos(noise_rad), -np.sin(noise_rad)],
                        [np.sin(noise_rad), np.cos(noise_rad)],
                    ]
                )
                R_aerial_ground_2d = R_noise @ R_aerial_ground_2d

            # Aerial is axis-aligned
            match_kwargs["global_x_dir1"] = np.array([1.0, 0.0])
            match_kwargs["global_y_dir1"] = np.array([0.0, 1.0])
            # Ground dirs from (possibly perturbed) rotation
            match_kwargs["global_x_dir2"] = R_aerial_ground_2d[:, 0]
            match_kwargs["global_y_dir2"] = R_aerial_ground_2d[:, 1]

            # Ensure the matcher uses the direction constraints
            self.matcher.params.xy_dir_constrained_2d = True

        t0 = time.time()
        match_result = self.matcher.match(
            aerial_segs_j,
            ground_segs_i,
            **match_kwargs,
        )

        # Restore default so non-translation-only calls are unaffected
        if translation_only and R_aerial_ground_2d is not None:
            self.matcher.params.xy_dir_constrained_2d = False

        all_matches = match_result.association_arrays
        n_hyps = len(all_matches)
        t_reg_total = 0.0
        t_segl_total = 0.0
        t_dim = 0.0

        raw_results = []
        for idx, (matches, score, count) in enumerate(zip(
            all_matches,
            match_result.scores,
            match_result.counts,
        )):
            t_segl0 = time.time()
            matched_ground = ground_segs_i.sublist_from_ids(matches[:, 1])
            matched_aerial = aerial_segs_j.sublist_from_ids(matches[:, 0])
            t_segl_total += time.time() - t_segl0

            t_reg0 = time.time()
            try:
                T_aerial_ground_odom_2d = self.registerer.register(matched_aerial, matched_ground).transformation
            except Exception as e:
                t_reg_total += time.time() - t_reg0
                print(f"Registration failed for idx {idx}/{len(all_matches)}: {e}")
                continue
            t_reg_total += time.time() - t_reg0

            # TODO: transform everything in SE(2) instead?
            T_aerial_ground_odom_hat = self._se2_to_se3(T_aerial_ground_odom_2d)
            T_aerial_ground_hat = T_aerial_ground_odom_hat @ T_ground_odom_ground_robot
            if T_camera_flu is not None:
                T_aerial_ground_hat = T_aerial_ground_hat @ T_camera_flu

            if np.any(np.isnan(T_aerial_ground_hat)):
                continue

            # Aerial-to-ground registration flips Z, so the 2D rotation
            # block must have negative determinant.  Reject flipped hypotheses.
            if np.linalg.det(T_aerial_ground_hat[:2, :2]) > 0:
                # print("skipping proper rotation")
                continue
            
            T_aerial_ground_hat[2, 3] = 0.0

            pose_result = PoseEstimationResult(
                T_i_j_hat=T_aerial_ground_hat,
                T_i_j=T_aerial_ground,
                associations=matches,
            )

            result = SingleMatchResult(
                pose_result=pose_result,
                aerial_segs_processed=aerial_segs_j,
                ground_segs_processed=ground_segs_i,
                matched_ground=matched_ground,
                matched_aerial=matched_aerial,
            )
            raw_results.append((result, T_aerial_ground_hat, count))

        runtime_s = time.time() - t0

        # Cluster by transformation similarity if multiple hypotheses
        if len(raw_results) > 1:
            results = self.registerer.cluster_hypotheses(raw_results)
        elif len(raw_results) == 1:
            results = [raw_results[0][0]]
        else:
            results = []

        # Set runtime on first result
        if results:
            results[0].pose_result.runtime_s = runtime_s

        # Return at least one empty result if no valid hypotheses
        if not results:
            results = [
                SingleMatchResult(
                    pose_result=PoseEstimationResult(
                        T_i_j=T_aerial_ground,
                        runtime_s=runtime_s,
                    ),
                    aerial_segs_processed=aerial_segs_j,
                    ground_segs_processed=ground_segs_i,
                )
            ]

        return results

    @staticmethod
    def _se3_to_se2(T_4x4: np.ndarray) -> np.ndarray:
        """Extract 3x3 SE(2) from a 4x4 SE(3) matrix (xy-plane projection)."""
        yaw = np.arctan2(T_4x4[1, 0], T_4x4[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([
            [c, -s, T_4x4[0, 3]],
            [s,  c, T_4x4[1, 3]],
            [0,  0,      1     ],
        ])

    @staticmethod
    def _se2_to_se3(T_3x3: np.ndarray) -> np.ndarray:
        """Embed 3x3 SE(2) into a 4x4 SE(3) matrix (z=0 plane)."""
        T_4x4 = np.eye(4)
        T_4x4[:2, :2] = T_3x3[:2, :2]
        T_4x4[:2, 3] = T_3x3[:2, 2]
        if np.linalg.det(T_4x4[:2, :2]) < 0:
            T_4x4[2, 2] = -1.0
        return T_4x4

    def _find_max_intersection_patches(
        self,
        ground_submap: Submap,
        aerial_submaps_2d: Dict[str, Submap],
        reference_trajectory,
        T_camera_flu: np.ndarray = None,
        n: int = 1,
    ) -> List[str]:
        """Find the aerial patches with most overlap with the ground submap's
        estimated position circle.

        Uses shapely to compute circle-rectangle intersection area.
        Returns up to *n* aerial keys sorted by overlap (descending).
        """
        if reference_trajectory is None:
            return []

        try:
            from shapely.geometry import Point, box
        except ImportError:
            logger.warning("shapely not installed; max_intersection mode unavailable.")
            return []

        ground_pose = reference_trajectory.pose(ground_submap.time)

        # Ground position in aerial frame (pick any aerial submap for pose)
        first_key = next(iter(aerial_submaps_2d))
        aerial_pose = aerial_submaps_2d[first_key].pose
        if T_camera_flu is not None:
            T_aerial_ground = np.linalg.inv(aerial_pose) @ ground_pose @ T_camera_flu
        else:
            T_aerial_ground = np.linalg.inv(aerial_pose) @ ground_pose
        gx, gy = T_aerial_ground[0, 3], T_aerial_ground[1, 3]

        # Need ground_submap_rad_m for circle radius — use from ground submap metadata
        # or a reasonable default
        rad_m = self.pipeline_params.ground_dist_from_aerial_patch_center_m

        ground_circle = Point(gx, gy).buffer(rad_m)

        # Compute crop geometry
        px_per_m = 1.0 / self.pixel_len_m
        patch_size_px = int(
            self.aerial_patch_params.aerial_img_patch_side_len_m * px_per_m
        )
        stride = int(
            patch_size_px * (1.0 - self.aerial_patch_params.aerial_img_patch_overlap)
        )
        stride_m = stride * self.pixel_len_m
        patch_size_m = patch_size_px * self.pixel_len_m

        overlaps = []
        aerial_key_to_tuple = _aerial_key_to_tuple

        for aerial_key in aerial_submaps_2d:
            i_a, j_a = aerial_key_to_tuple(aerial_key)
            x1_m = i_a * stride_m
            y1_m = j_a * stride_m
            x2_m = x1_m + patch_size_m
            y2_m = y1_m + patch_size_m
            patch_rect = box(x1_m, y1_m, x2_m, y2_m)
            area = ground_circle.intersection(patch_rect).area
            if area > 0:
                overlaps.append((area, aerial_key))

        overlaps.sort(reverse=True)
        return [key for _, key in overlaps[:n]]

    def cross_view_match_max_intersection(
        self,
        aerial_submaps: Dict[str, Submap],
        ground_submaps: Dict[str, Submap],
        reference_trajectory=None,
        T_camera_flu: np.ndarray = None,
        translation_only: bool = None,
        show_progress: bool = False,
        local_to_pixel_fn=None,
        gt_trajectory=None,
    ) -> CrossViewMatchResult:
        """Cross-view match using max_intersection mode.

        For each ground submap, finds the top-N best-overlapping aerial patches
        (controlled by ``max_intersection_patches_per_ground_sm``) and matches
        against each of them.
        """
        if not aerial_submaps:
            raise ValueError(
                "No aerial submaps provided — check the aerial directory path."
            )
        if not ground_submaps:
            raise ValueError(
                "No ground submaps provided — check the ground directory path."
            )

        do_translation_only = (
            translation_only
            if translation_only is not None
            else self.pipeline_params.translation_only
        )

        aerial_submaps_2d, ground_submaps_2d = self._preprocess_submaps_2d(
            aerial_submaps, ground_submaps
        )

        aerial_key_to_tuple = _aerial_key_to_tuple
        aerial_x_max = max(aerial_key_to_tuple(key)[0] for key in aerial_submaps.keys())
        aerial_y_max = max(aerial_key_to_tuple(key)[1] for key in aerial_submaps.keys())

        all_results = {}
        all_details = {}

        iterator = ground_submaps_2d.items()
        if show_progress:
            iterator = tqdm(iterator, desc="Matching", total=len(ground_submaps_2d))
        for ground_key, ground_sm_i in iterator:
            results_matrix = PoseEstimationResultMatrix(
                (aerial_x_max + 1, aerial_y_max + 1)
            )
            details_for_ground = {}

            n_patches = self.pipeline_params.max_intersection_patches_per_ground_sm
            best_aerial_keys = self._find_max_intersection_patches(
                ground_submaps[ground_key],
                aerial_submaps_2d,
                reference_trajectory,
                T_camera_flu,
                n=n_patches,
            )

            if not best_aerial_keys:
                all_results[ground_key] = results_matrix
                all_details[ground_key] = details_for_ground
                continue

            ground_pose_ref = None
            if reference_trajectory is not None:
                ground_pose_ref = reference_trajectory.pose(
                    ground_submaps[ground_key].time
                )
            ground_pose_gt = None
            if gt_trajectory is not None:
                try:
                    ground_pose_gt = gt_trajectory.pose(ground_submaps[ground_key].time)
                except Exception:
                    pass
            T_ground_odom_ground_robot = ground_sm_i.metadata["camera_pose"]

            for aerial_key in best_aerial_keys:
                aerial_sm_j = aerial_submaps_2d[aerial_key]
                i_a, j_a = aerial_key_to_tuple(aerial_key)

                single_results = self._match_single_pair(
                    aerial_sm_j,
                    ground_sm_i,
                    reference_trajectory,
                    T_camera_flu,
                    ground_pose_ref,
                    T_ground_odom_ground_robot,
                    do_translation_only,
                    local_to_pixel_fn=local_to_pixel_fn,
                    ground_pose_gt=ground_pose_gt,
                )

                results_matrix[i_a, j_a] = [r.pose_result for r in single_results]
                details_for_ground[aerial_key] = single_results

            all_results[ground_key] = results_matrix
            all_details[ground_key] = details_for_ground

        return CrossViewMatchResult(results=all_results, match_details=all_details)
