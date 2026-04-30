import logging
import numpy as np
from typing import Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from gen_seg_match.segment.segment_types import (
    SegmentPoint,
    GeneralSegment,
    SegmentList,
)
from gen_seg_match.params import RegisterParams
from gen_seg_match.register.geometry import (
    extract_line_arrays,
    register_2d_core,
    REGISTER_OK,
    REGISTER_NO_ROTATION,
    REGISTER_NO_TRANSLATION,
    REGISTER_DEGENERATE_SIGN,
    REGISTER_SINGULAR,
)

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
        if correspondences is None:
            if self.params.only_use_points:
                pairs = [
                    (s, t) for s, t in zip(source, target)
                    if isinstance(s, SegmentPoint)
                ]
                source = [s for s, _ in pairs]
                target = [t for _, t in pairs]
            if not isinstance(source, SegmentList):
                source = SegmentList(source)
            if not isinstance(target, SegmentList):
                target = SegmentList(target)
        else:
            if not isinstance(source, SegmentList):
                source = SegmentList(source)
            if not isinstance(target, SegmentList):
                target = SegmentList(target)
            if self.params.only_use_points:
                correspondences = np.array([
                    c for c in correspondences
                    if isinstance(source.get_segment_from_id(c[0]), SegmentPoint)
                ])
            source = source.sublist_from_ids(correspondences[:, 0])
            target = target.sublist_from_ids(correspondences[:, 1])

        assert len(source) == len(target)
        if len(source) == 0:
            raise InsufficientAssociationsException(0, 0, 0, "No segments to register.")
        for i in range(len(source)):
            assert type(source[i]) == type(target[i]), (
                "Corresponded segments must be of the same type. "
                + f"Got match between {type(source[i])} and {type(target[i])}."
            )

        # Pre-center inlier clouds at the origin so line moments (normal · point)
        # don't carry the lever arm of the segment frame's origin. We register
        # the centered copies and then undo the centering on the returned T.
        c_src = np.mean(
            np.array([np.asarray(seg.point).flatten()[:2] for seg in source]),
            axis=0,
        )
        c_tgt = np.mean(
            np.array([np.asarray(seg.point).flatten()[:2] for seg in target]),
            axis=0,
        )

        p = np.ascontiguousarray(source.get_points().points, dtype=np.float64)
        q = np.ascontiguousarray(target.get_points().points, dtype=np.float64)
        s_norm, s_off = extract_line_arrays(source)
        t_norm, t_off = extract_line_arrays(target)

        if p.shape[0] > 0:
            p = np.ascontiguousarray(p - c_src, dtype=np.float64)
            q = np.ascontiguousarray(q - c_tgt, dtype=np.float64)
        if s_norm.shape[0] > 0:
            s_off = s_off - s_norm @ c_src
            t_off = t_off - t_norm @ c_tgt

        status, T = register_2d_core(
            p, q, s_norm, s_off, t_norm, t_off,
            self.params.point_weight,
            self.params.line_direction_weight,
            self.params.line_moment_weight,
            self.params.eps,
            self.params.dup_eps,
        )

        if status == REGISTER_OK:
            # Undo pre-centering: T_src_tgt = translate(+c_src) @ T̃ @ translate(-c_tgt)
            T_pos_src = np.eye(3)
            T_pos_src[:2, 2] = c_src
            T_neg_tgt = np.eye(3)
            T_neg_tgt[:2, 2] = -c_tgt
            T = T_pos_src @ T @ T_neg_tgt
            return RegistrationResult(transformation=T, losses=[])

        n_pts = p.shape[0]
        n_lines = s_norm.shape[0]
        if status == REGISTER_NO_ROTATION:
            raise InsufficientAssociationsException(
                len(source), len(target), n_pts + n_lines,
                "Insufficient for rotation: need at least 1 line or 2 non-identical points.",
            )
        if status == REGISTER_NO_TRANSLATION:
            raise InsufficientAssociationsException(
                len(source), len(target), n_pts + n_lines,
                "Insufficient for translation: need at least 1 point or 2 non-parallel lines.",
            )
        if status == REGISTER_DEGENERATE_SIGN:
            raise InsufficientAssociationsException(
                len(source), len(target), n_lines,
                "Degenerate configuration: multiple sign choices yield similar losses.",
            )
        if status == REGISTER_SINGULAR:
            raise InsufficientAssociationsException(-1, -1)
        raise InsufficientAssociationsException(-1, -1, message=f"Unknown status {status}")

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
        two_pi = 2.0 * np.pi

        N = len(items)
        T_stack = np.stack([item[1] for item in items])  # (N, 4, 4)
        trans_all = T_stack[:, :2, 3]  # (N, 2)
        yaws_all = np.arctan2(T_stack[:, 1, 0], T_stack[:, 0, 0])  # (N,)
        counts_all = np.fromiter((item[2] for item in items), dtype=np.int64, count=N)

        # Indices into items[] for each cluster representative.
        rep_idxs: List[int] = []
        cluster_counts: List[int] = []

        for i in range(N):
            if not rep_idxs:
                rep_idxs.append(i)
                cluster_counts.append(int(counts_all[i]))
                continue
            rep_trans = trans_all[rep_idxs]
            rep_yaws = yaws_all[rep_idxs]
            dt = trans_all[i] - rep_trans
            t_dists = np.sqrt(dt[:, 0] * dt[:, 0] + dt[:, 1] * dt[:, 1])
            yaw_diff = np.abs(yaws_all[i] - rep_yaws)
            r_dists = np.minimum(yaw_diff, two_pi - yaw_diff)
            in_thresh = (t_dists <= trans_thresh) & (r_dists <= rot_thresh)
            if in_thresh.any():
                first_match = int(np.argmax(in_thresh))
                cluster_counts[first_match] += int(counts_all[i])
            else:
                rep_idxs.append(i)
                cluster_counts.append(int(counts_all[i]))

        order = sorted(range(len(rep_idxs)), key=lambda k: -cluster_counts[k])
        max_hyp = self.params.max_hypotheses
        if max_hyp > 0:
            order = order[:max_hyp]

        logger.info(
            f"Hypothesis clustering: {N} hypotheses -> "
            f"{len(rep_idxs)} clusters, top counts: "
            f"{[cluster_counts[k] for k in order[:5]]}"
        )

        return [items[rep_idxs[k]][0] for k in order]
