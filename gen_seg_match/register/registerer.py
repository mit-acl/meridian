import logging
import numpy as np
from typing import Any, List, Optional, Tuple
from dataclasses import dataclass
from itertools import combinations

from gen_seg_match.segment.segment_types import (
    SegmentPoint,
    GeneralSegment,
    SegmentList,
    SegmentLine,
)
from gen_seg_match.params import RegisterParams
from gen_seg_match.register.geometry import (
    extract_line_arrays,
    cross_covariance_2x2,
    svd_rotation_2x2,
    solve_translation_2x2,
    sign_align_normals,
    point_line_loss_2d,
    batch_register_core,
)

logger = logging.getLogger(__name__)

_EMPTY2 = np.empty((0, 2))
_EMPTY1 = np.empty(0)


class InsufficientAssociationsException(Exception):
    def __init__(self, map1_len, map2_len, n_associations=None, message=None):
        self.map1_len = map1_len
        self.map2_len = map2_len
        self.n_associations = n_associations
        if message is None:
            message = f"Insufficient associations. Map 1 length: {map1_len}. Map 2 length: {map2_len}. Associations: {n_associations}"
        super().__init__(message)


@dataclass
class RegistrationInput:
    source: List[GeneralSegment]
    target: List[GeneralSegment]
    correspondences: np.ndarray = None


@dataclass
class RegistrationResult:
    transformation: np.ndarray
    losses: List[float]


class Registerer2D:
    def __init__(self, params: RegisterParams):
        self.params = params

    def register(
        self,
        source: List[GeneralSegment],
        target: List[GeneralSegment],
        correspondences: np.ndarray = None,
    ):
        """
        Computes the 2D transformation that aligns target to source (T^source_target).

        Args:
            source: Segment list in source frame
            target: Segment list in target frame
            correspondences: shape=(n,2), pre-computed correspondences. If None,
                assumes source and target are already ordered 1-to-1.

        Returns:
            RegistrationResult with a 3x3 SE(2) transformation matrix.
        """
        if (
            len(source) == 0
            or len(target) == 0
            or (correspondences is not None and len(correspondences) == 0)
        ):
            raise InsufficientAssociationsException(len(source), len(target))

        if correspondences is None:
            assert len(source) == len(target)
            correspondences = np.array(
                [[seg1.id, seg2.id] for seg1, seg2 in zip(source, target)]
            )

        source = SegmentList(source)
        target = SegmentList(target)

        if self.params.only_use_points:
            correspondences = np.array(
                [
                    correspondence
                    for correspondence in correspondences
                    if isinstance(
                        source.get_segment_from_id(correspondence[0]), SegmentPoint
                    )
                ]
            )
            if len(correspondences) == 0:
                raise InsufficientAssociationsException(len(source), len(target))

        source = source.sublist_from_ids(correspondences[:, 0])
        target = target.sublist_from_ids(correspondences[:, 1])

        for i in range(len(correspondences)):
            assert type(source[i]) == type(target[i]), (
                "Corresponded segments must be of the same type. "
                + f"Got match between {type(source[i])} and {type(target[i])}."
            )

        # --- Extract numeric arrays (once) ---
        num_points = len(source.get_points())
        num_lines = len(source.get_lines())

        p = source.get_points().points  # (n_p, 2)
        q = target.get_points().points

        s_norm, s_off = extract_line_arrays(source)
        t_norm, t_off = extract_line_arrays(target)

        eps = self.params.ident_eps
        W_P = self.params.point_weight
        W_D = self.params.line_direction_weight
        W_L_M = self.params.line_moment_weight

        # --- Feasibility checks ---
        # Rotation (1 DOF): need 1 line OR 2 non-identical points
        points_determine_rotation = False
        if num_points >= 2:
            p_centered = p - p.mean(axis=0)
            points_determine_rotation = np.max(np.linalg.norm(p_centered, axis=1)) > eps

        has_rotation = num_lines >= 1 or points_determine_rotation

        # Translation (2 DOF): need 1 point OR 2 non-parallel lines
        has_translation = num_points >= 1
        if not has_translation:
            if num_lines >= 2:
                for i, j in combinations(range(num_lines), 2):
                    cross = s_norm[i, 0] * s_norm[j, 1] - s_norm[i, 1] * s_norm[j, 0]
                    if abs(cross) > eps:
                        has_translation = True
                        break

        if not has_rotation:
            raise InsufficientAssociationsException(
                len(source), len(target), num_points + num_lines,
                "Insufficient for rotation: need at least 1 line or 2 non-identical points.",
            )
        if not has_translation:
            raise InsufficientAssociationsException(
                len(source), len(target), num_points + num_lines,
                "Insufficient for translation: need at least 1 point or 2 non-parallel lines.",
            )

        # --- Determine initial rotation ---
        lin_eps = self.params.lin_eps

        if num_lines > 0 and not points_determine_rotation:
            # Sign disambiguation: try both signs for the first line normal
            best_loss = float("inf")
            second_best_loss = float("inf")
            best_R = None

            for sign_val in (-1.0, 1.0):
                first_t_norm = sign_val * t_norm[0:1]
                H = cross_covariance_2x2(p, q, s_norm[0:1], first_t_norm, W_P, W_D)
                R_cand, s0 = svd_rotation_2x2(H)
                if s0 < lin_eps:
                    continue

                # Align ALL normals, solve translation with everything
                t_norm_a, t_off_a = sign_align_normals(R_cand, s_norm, t_norm, t_off)
                t_cand = solve_translation_2x2(
                    R_cand, p, q, s_norm, s_off, t_off_a, W_P, W_L_M
                )

                loss = point_line_loss_2d(
                    R_cand, t_cand, p, q,
                    s_norm, t_norm, s_off, t_off,
                    W_P, W_D, W_L_M, True,
                )

                if loss < best_loss:
                    second_best_loss = best_loss
                    best_loss = loss
                    best_R = R_cand
                elif loss < second_best_loss:
                    second_best_loss = loss

            if best_R is None or second_best_loss - best_loss < self.params.dup_eps:
                raise InsufficientAssociationsException(
                    len(source), len(target), num_lines,
                    "Degenerate configuration: both sign choices yield similar losses.",
                )
            R = best_R
        else:
            # Points determine rotation
            H = cross_covariance_2x2(p, q, _EMPTY2, _EMPTY2, W_P, W_D)
            R, s0 = svd_rotation_2x2(H)
            if s0 < lin_eps:
                raise InsufficientAssociationsException(-1, -1)

        # --- Final registration: sign-align, re-solve R and t, invert ---
        t_norm_aligned, t_off_aligned = sign_align_normals(R, s_norm, t_norm, t_off)
        H_final = cross_covariance_2x2(p, q, s_norm, t_norm_aligned, W_P, W_D)
        R_final, s0_f = svd_rotation_2x2(H_final)
        if s0_f < lin_eps:
            raise InsufficientAssociationsException(-1, -1)

        t_final = solve_translation_2x2(
            R_final, p, q, s_norm, s_off, t_off_aligned, W_P, W_L_M
        )

        T = np.eye(3)
        T[:2, :2] = R_final
        T[:2, 2] = t_final
        T = np.linalg.inv(T)

        return RegistrationResult(transformation=T, losses=[])

    @staticmethod
    def _se2_distance(T1: np.ndarray, T2: np.ndarray):
        """Compute SE(2) translation and rotation distance between two 3x3 transforms."""
        trans_dist = np.linalg.norm(T1[:2, 2] - T2[:2, 2])
        yaw1 = np.arctan2(T1[1, 0], T1[0, 0])
        yaw2 = np.arctan2(T2[1, 0], T2[0, 0])
        rot_dist = abs(yaw1 - yaw2)
        rot_dist = min(rot_dist, 2 * np.pi - rot_dist)
        return trans_dist, rot_dist

    def cluster_hypotheses(
        self,
        items: List[Tuple[Any, np.ndarray, int]],
    ) -> List[Any]:
        """Cluster hypotheses by transformation similarity, rank by particle count.

        Greedy clustering: iterate items (assumed sorted by objective), assign each
        to the first existing cluster within thresholds, or start a new cluster.
        The representative of each cluster is the first (highest-objective) member.
        Clusters are ranked by total particle count.

        Args:
            items: List of (payload, T_hat_3x3, count) tuples.

        Returns:
            List of payload objects from the representative of each cluster.
        """
        if not items:
            return []

        trans_thresh = self.params.cluster_trans_thresh_m
        rot_thresh = np.deg2rad(self.params.cluster_rot_thresh_deg)

        # Each cluster: [representative_payload, T_rep, total_count]
        clusters = []
        for payload, T, count in items:
            assigned = False
            for cluster in clusters:
                t_dist, r_dist = self._se2_distance(T, cluster[1])
                if t_dist <= trans_thresh and r_dist <= rot_thresh:
                    cluster[2] += count
                    assigned = True
                    break
            if not assigned:
                clusters.append([payload, T, count])

        # Sort clusters by total particle count descending
        clusters.sort(key=lambda c: c[2], reverse=True)

        # Optionally keep only the top-N clusters
        max_hyp = self.params.max_hypotheses
        if max_hyp > 0:
            clusters = clusters[:max_hyp]

        logger.info(
            f"Hypothesis clustering: {len(items)} hypotheses -> "
            f"{len(clusters)} clusters, top counts: "
            f"{[c[2] for c in clusters[:5]]}"
        )

        return [c[0] for c in clusters]


class BatchRegisterer2D:
    """Batched 2D registration using numba-accelerated NumPy.

    For each problem in the batch:
      1. Take two points, solve initial rotation via SVD.
      2. Use that rotation to sign-align all line normals.
      3. Re-solve rotation with all points + aligned normals.
      4. Solve translation via normal equations.

    Problems that fail (< 2 points, degenerate, etc.) get NaN transforms.
    """

    def __init__(self, params: RegisterParams):
        self.params = params

    def register_batch(
        self,
        sources: List[List[GeneralSegment]],
        targets: List[List[GeneralSegment]],
    ) -> np.ndarray:
        """
        Batched 2D registration of pre-corresponded segment lists.

        Args:
            sources: N lists of source segments (pre-corresponded 1-to-1).
            targets: N lists of target segments.

        Returns:
            (N, 3, 3) numpy array of SE(2) transforms (target -> source).
            Failed problems are filled with NaN.
        """
        N = len(sources)
        assert N == len(targets)
        if N == 0:
            return np.zeros((0, 3, 3), dtype=np.float64)

        # -- Extract and pad --
        all_pts_s, all_pts_t = [], []
        all_norms_s, all_norms_t = [], []
        all_offs_s, all_offs_t = [], []
        n_points_list, n_lines_list = [], []

        for src, tgt in zip(sources, targets):
            src_sl = SegmentList(src)
            tgt_sl = SegmentList(tgt)

            pts_s = src_sl.get_points().points
            pts_t = tgt_sl.get_points().points
            norms_s, offs_s = extract_line_arrays(src_sl)
            norms_t, offs_t = extract_line_arrays(tgt_sl)

            all_pts_s.append(pts_s)
            all_pts_t.append(pts_t)
            all_norms_s.append(norms_s)
            all_norms_t.append(norms_t)
            all_offs_s.append(offs_s)
            all_offs_t.append(offs_t)
            n_points_list.append(len(pts_s))
            n_lines_list.append(len(norms_s))

        max_pts = max(n_points_list) if n_points_list else 0
        max_lines = max(n_lines_list) if n_lines_list else 0
        n_points_arr = np.array(n_points_list, dtype=np.int64)
        n_lines_arr = np.array(n_lines_list, dtype=np.int64)

        pts_s_pad = np.zeros((N, max(max_pts, 2), 2))
        pts_t_pad = np.zeros_like(pts_s_pad)
        norms_s_pad = np.zeros((N, max(max_lines, 1), 2))
        norms_t_pad = np.zeros_like(norms_s_pad)
        offs_s_pad = np.zeros((N, max(max_lines, 1)))
        offs_t_pad = np.zeros_like(offs_s_pad)

        for i in range(N):
            np_i, nl_i = n_points_list[i], n_lines_list[i]
            if np_i > 0:
                pts_s_pad[i, :np_i] = all_pts_s[i]
                pts_t_pad[i, :np_i] = all_pts_t[i]
            if nl_i > 0:
                norms_s_pad[i, :nl_i] = all_norms_s[i]
                norms_t_pad[i, :nl_i] = all_norms_t[i]
                offs_s_pad[i, :nl_i] = all_offs_s[i]
                offs_t_pad[i, :nl_i] = all_offs_t[i]

        # -- Numba core loop --
        T_out, n_failed = batch_register_core(
            pts_s_pad, pts_t_pad,
            norms_s_pad, norms_t_pad,
            offs_s_pad, offs_t_pad,
            n_points_arr, n_lines_arr,
            self.params.ident_eps,
            self.params.point_weight,
            self.params.line_direction_weight,
            self.params.line_moment_weight,
        )

        print(f"Batched 2D registration: {n_failed}/{N} failed")
        return T_out
