import logging
import math
import numpy as np
import torch
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
    cross_covariance,
    line_normals_2d,
    line_offsets_2d,
    PointLineLoss2D,
)

logger = logging.getLogger(__name__)


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

        num_points = len(source.get_points())
        num_lines = len(source.get_lines())
        eps = self.params.ident_eps

        p = source.get_points().points
        q = target.get_points().points

        s_norm = line_normals_2d(source)
        t_norm = line_normals_2d(target)

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

        # --- Resolve line normal sign ambiguity ---
        if num_lines > 0 and not points_determine_rotation:
            # Points alone can't determine rotation; use 1 line normal.
            # Try both signs, pick the one with lower loss.
            PLLoss = PointLineLoss2D(params=self.params, bidirectional=True)

            best_loss, second_best_loss = float("inf"), float("inf")
            best_R = None

            source_subset = source.get_points() + [source.get_lines()[0]]

            for sign in [-1, 1]:
                target_line_signed = target.get_lines()[0].copy()
                target_line_signed.direction *= sign
                target_subset_signed = target.get_points() + [target_line_signed]

                # Solve rotation from the first line (+ any points)
                R_candidate, _ = self.aruns_extended(
                    source_subset,
                    target_subset_signed,
                    rotation_only=True,
                )

                # Sign-align all lines using this rotation, then solve
                # translation with everything for a fair comparison.
                target_consistent = self.compute_consistent_directions(
                    R_candidate, source, target
                )
                t_candidate = self.solve_translation(
                    R_candidate, source, target_consistent
                )

                loss = PLLoss.compute_loss(
                    R=R_candidate, t=t_candidate,
                    source=source, target=target,
                )

                if loss < best_loss:
                    second_best_loss = best_loss
                    best_loss = loss
                    best_R = R_candidate
                elif loss < second_best_loss:
                    second_best_loss = loss

            if second_best_loss - best_loss < self.params.dup_eps:
                raise InsufficientAssociationsException(
                    len(source), len(target), num_lines,
                    "Degenerate configuration: both sign choices yield similar losses.",
                )

            R = best_R

        else:
            # Points determine rotation; no sign search needed.
            R, _ = self.aruns_extended(
                source.get_points(),
                target.get_points(),
                rotation_only=True,
            )

        # Make target directions sign-consistent with source
        target_consistent = self.compute_consistent_directions(R, source, target)

        return self.register_consistent(source, target_consistent)

    def register_consistent(
        self,
        source: SegmentList,
        target: SegmentList,
    ) -> RegistrationResult:
        R, t = self.aruns_extended(source, target, rotation_only=False)

        T = np.eye(3)
        T[:2, :2] = R
        T[:2, 2] = t
        T = np.linalg.inv(T)

        return RegistrationResult(transformation=T, losses=[])

    def compute_consistent_directions(
        self, R: np.ndarray, source: SegmentList, target: SegmentList
    ) -> SegmentList:
        """
        Create a sign-consistent version of the target by flipping line directions
        to align with the rotated source lines.

        Parameters:
        R : np.ndarray
            A 2x2 rotation matrix estimate.
        source : SegmentList
            Source segments (points, lines).
        target : SegmentList
            Target segments (points, lines).

        Returns:
        target_consistent : SegmentList
            Target with sign-consistent line directions.
        """
        s_dir = source.get_lines().directions
        t_lines = target.get_lines().copy()

        for i in range(len(s_dir)):
            s_dir_transformed = R @ s_dir[i]
            if np.dot(s_dir_transformed, t_lines[i].direction) < 0:
                t_lines[i].direction = -t_lines[i].direction

        return SegmentList(target.get_points().copy() + t_lines)

    def aruns_extended(
        self,
        source: SegmentList,
        target: SegmentList,
        rotation_only: Optional[bool] = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the 2D rigid transformation between associated point-line clouds
        using Arun's method extended to include 2D lines (normal + offset).

        Parameters:
        source : SegmentList
            Source segments (points, lines), sign-consistent.
        target : SegmentList
            Target segments (points, lines), sign-consistent.
        rotation_only : Optional[bool]
            If True, only compute rotation and set translation to zero.

        Returns:
        R : np.ndarray
            A 2x2 rotation matrix.
        t : np.ndarray
            A 2x1 translation vector.
        """

        num_points = len(source.get_points())
        num_lines = len(source.get_lines())

        assert (
            num_points == len(target.get_points())
            and num_lines == len(target.get_lines())
        ), "Source and target must have the same number of points and lines."
        assert num_lines >= 1 or num_points + num_lines >= 2, (
            "At least one directional or two total correspondences are required."
        )

        p, q = source.get_points().points, target.get_points().points
        s_norm = line_normals_2d(source)
        t_norm = line_normals_2d(target)
        weights = [self.params.line_direction_weight] * num_lines

        H = cross_covariance(
            p,
            q,
            s_norm,
            t_norm,
            W_P=self.params.point_weight,
            W_D=weights,
        )

        U, S, Vt = np.linalg.svd(H)

        if np.abs(S[0]) < self.params.lin_eps:
            raise InsufficientAssociationsException(-1, -1)

        if np.linalg.det(Vt.T @ U.T) < 0:
            Vt[1, :] *= -1

        R = Vt.T @ U.T
        t = (
            self.solve_translation(R, source, target)
            if not rotation_only
            else np.zeros(2)
        )

        return R, t

    def solve_translation(
        self, R: np.ndarray, source: SegmentList, target: SegmentList
    ) -> np.ndarray:
        """
        Solve for the 2D translation vector given rotation and associated segments
        by minimizing L2 error.

        Point constraints: t = q - R @ p  (2 equations per pair)
        Line constraints: n'^T @ t = d_t - d_s  (1 equation per pair)

        Parameters:
        R : np.ndarray
            A 2x2 rotation matrix.
        source : SegmentList
            Source segments (points, lines).
        target : SegmentList
            Target segments (points, lines).

        Returns:
        t : np.ndarray
            A length-2 translation vector.
        """
        num_points = len(source.get_points())
        num_lines = len(source.get_lines())

        assert (
            num_points == len(target.get_points())
            and num_lines == len(target.get_lines())
        ), "Source and target must have the same number of points and lines."
        assert num_points + num_lines > 0, (
            "At least one correspondence is required to solve for translation."
        )

        p, q = source.get_points().points, target.get_points().points

        s_norm = line_normals_2d(source)
        s_off = line_offsets_2d(source).ravel()
        t_off = line_offsets_2d(target).ravel()

        A = np.zeros((2 * num_points + num_lines, 2))
        b = np.zeros(2 * num_points + num_lines)

        W_P_SQ = np.sqrt(self.params.point_weight)
        W_L_OFF_SQ = np.sqrt(self.params.line_moment_weight)

        # Point constraints: t = q - R @ p
        for i in range(num_points):
            A[2 * i : 2 * i + 2] = W_P_SQ * np.eye(2)
            b[2 * i : 2 * i + 2] = W_P_SQ * (q[i] - R @ p[i])

        # Line constraints: (R @ s_norm)^T @ t = d_t - d_s
        row_offset = 2 * num_points
        for j in range(num_lines):
            A[row_offset + j] = W_L_OFF_SQ * (R @ s_norm[j]).T
            b[row_offset + j] = W_L_OFF_SQ * (t_off[j] - s_off[j])

        t, *_ = np.linalg.lstsq(A, b, rcond=None)
        return t

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
    """Batched 2D registration using PyTorch.

    For each problem in the batch:
      1. Take two points, solve initial rotation via SVD.
      2. Use that rotation to sign-align all line normals.
      3. Re-solve rotation with all points + aligned normals.
      4. Solve translation via least squares.

    Problems that fail (< 2 points, degenerate, etc.) get NaN transforms.
    """

    def __init__(self, params: RegisterParams, device: str = "cpu"):
        self.params = params
        self.device = torch.device(device)

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
        device = self.device
        eps = self.params.ident_eps

        # -- Extract and pad --
        all_pts_s, all_pts_t = [], []
        all_norms_s, all_norms_t = [], []
        all_offs_s, all_offs_t = [], []
        n_points_list, n_lines_list = [], []

        for src, tgt in zip(sources, targets):
            src_sl = SegmentList(src)
            tgt_sl = SegmentList(tgt)

            pts_s = src_sl.get_points().points  # (n_p, 2)
            pts_t = tgt_sl.get_points().points
            norms_s = line_normals_2d(src_sl)  # (n_l, 2)
            norms_t = line_normals_2d(tgt_sl)
            offs_s = line_offsets_2d(src_sl).ravel()  # (n_l,)
            offs_t = line_offsets_2d(tgt_sl).ravel()

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
        n_points = torch.tensor(n_points_list, device=device)
        n_lines = torch.tensor(n_lines_list, device=device)

        pts_s_pad = torch.zeros(N, max(max_pts, 2), 2, device=device, dtype=torch.float64)
        pts_t_pad = torch.zeros_like(pts_s_pad)
        norms_s_pad = torch.zeros(N, max(max_lines, 1), 2, device=device, dtype=torch.float64)
        norms_t_pad = torch.zeros_like(norms_s_pad)
        offs_s_pad = torch.zeros(N, max(max_lines, 1), device=device, dtype=torch.float64)
        offs_t_pad = torch.zeros_like(offs_s_pad)
        pt_mask = torch.zeros(N, max(max_pts, 2), device=device, dtype=torch.bool)
        line_mask = torch.zeros(N, max(max_lines, 1), device=device, dtype=torch.bool)

        for i in range(N):
            np_i, nl_i = n_points_list[i], n_lines_list[i]
            if np_i > 0:
                pts_s_pad[i, :np_i] = torch.from_numpy(all_pts_s[i]).to(device)
                pts_t_pad[i, :np_i] = torch.from_numpy(all_pts_t[i]).to(device)
                pt_mask[i, :np_i] = True
            if nl_i > 0:
                norms_s_pad[i, :nl_i] = torch.from_numpy(all_norms_s[i]).to(device)
                norms_t_pad[i, :nl_i] = torch.from_numpy(all_norms_t[i]).to(device)
                offs_s_pad[i, :nl_i] = torch.from_numpy(all_offs_s[i]).to(device)
                offs_t_pad[i, :nl_i] = torch.from_numpy(all_offs_t[i]).to(device)
                line_mask[i, :nl_i] = True

        # Track which problems are still valid
        valid = n_points >= 2

        # ---- Step 1: Rotation from first 2 points ----
        p2 = pts_s_pad[:, :2, :]  # (N, 2, 2)
        q2 = pts_t_pad[:, :2, :]
        p2_mean = p2.mean(dim=1, keepdim=True)
        q2_mean = q2.mean(dim=1, keepdim=True)
        p2_c = p2 - p2_mean
        q2_c = q2 - q2_mean

        H_init = p2_c.transpose(-1, -2) @ q2_c  # (N, 2, 2)
        U, S, Vt = torch.linalg.svd(H_init)

        # Degenerate if largest singular value ~ 0 (identical points)
        valid = valid & (S[:, 0] > eps)

        # Det correction
        Vt_c = Vt.clone()
        det_sign = torch.det(Vt_c.transpose(-1, -2) @ U.transpose(-1, -2))
        Vt_c[det_sign < 0, 1, :] *= -1
        R_init = Vt_c.transpose(-1, -2) @ U.transpose(-1, -2)  # (N, 2, 2)

        # ---- Step 2: Sign-align line normals using R_init ----
        # R_init @ s_norm^T → rotate source normals, compare with target
        s_norm_rotated = norms_s_pad @ R_init.transpose(-1, -2)  # (N, max_lines, 2)
        cos_sim = (s_norm_rotated * norms_t_pad).sum(-1)  # (N, max_lines)
        flip = cos_sim < 0

        norms_t_aligned = norms_t_pad.clone()
        norms_t_aligned[flip] *= -1
        offs_t_aligned = offs_t_pad.clone()
        offs_t_aligned[flip] *= -1

        # ---- Step 3: Final rotation with all points + aligned normals ----
        W_P = self.params.point_weight
        W_D = self.params.line_direction_weight

        # Point cross-covariance (masked)
        pt_mask_f = pt_mask.unsqueeze(-1).to(pts_s_pad.dtype)  # (N, max_pts, 1)
        p_sum = (pts_s_pad * pt_mask_f).sum(1)  # (N, 2)
        q_sum = (pts_t_pad * pt_mask_f).sum(1)
        n_pts_clamp = n_points.clamp(min=1).unsqueeze(-1).to(pts_s_pad.dtype)
        p_mean = p_sum / n_pts_clamp
        q_mean = q_sum / n_pts_clamp
        p_c = (pts_s_pad - p_mean.unsqueeze(1)) * pt_mask_f
        q_c = (pts_t_pad - q_mean.unsqueeze(1)) * pt_mask_f
        H_pts = W_P * p_c.transpose(-1, -2) @ q_c  # (N, 2, 2)

        # Line normal cross-covariance (masked)
        line_mask_f = line_mask.unsqueeze(-1).to(norms_s_pad.dtype)
        ns_m = norms_s_pad * line_mask_f
        nt_m = norms_t_aligned * line_mask_f
        H_lines = W_D * ns_m.transpose(-1, -2) @ nt_m  # (N, 2, 2)

        H_final = H_pts + H_lines
        U2, S2, Vt2 = torch.linalg.svd(H_final)
        valid = valid & (S2[:, 0] > eps)

        Vt2_c = Vt2.clone()
        det2 = torch.det(Vt2_c.transpose(-1, -2) @ U2.transpose(-1, -2))
        Vt2_c[det2 < 0, 1, :] *= -1
        R_final = Vt2_c.transpose(-1, -2) @ U2.transpose(-1, -2)  # (N, 2, 2)

        # ---- Step 4: Translation via least squares ----
        max_rows = 2 * max_pts + max_lines
        if max_rows == 0:
            valid[:] = False
            max_rows = 1  # dummy row so tensors are valid

        W_P_SQ = math.sqrt(W_P)
        W_L_SQ = math.sqrt(self.params.line_moment_weight)

        A = torch.zeros(N, max_rows, 2, device=device, dtype=torch.float64)
        b = torch.zeros(N, max_rows, 1, device=device, dtype=torch.float64)

        # Point constraints: I * t = q_i - R @ p_i  (2 rows per point)
        if max_pts > 0:
            p_rot = pts_s_pad @ R_final.transpose(-1, -2)  # (N, max_pts, 2)
            pt_res = pts_t_pad - p_rot  # (N, max_pts, 2)

            # Build 2-row-per-point blocks
            # Even rows (2*i): [1, 0], b = res_x
            # Odd rows (2*i+1): [0, 1], b = res_y
            idx = torch.arange(max_pts, device=device)
            pt_mask_2 = pt_mask.unsqueeze(-1).expand(-1, -1, 2).reshape(N, 2 * max_pts)

            A[:, 2 * idx, 0] = W_P_SQ
            A[:, 2 * idx + 1, 1] = W_P_SQ
            b[:, 2 * idx, 0] = W_P_SQ * pt_res[:, :, 0]
            b[:, 2 * idx + 1, 0] = W_P_SQ * pt_res[:, :, 1]

            # Zero out padded rows
            pt_mask_2_f = pt_mask_2.unsqueeze(-1).to(A.dtype)
            A[:, :2 * max_pts] *= pt_mask_2_f
            b[:, :2 * max_pts] *= pt_mask_2_f

        # Line constraints: n'^T @ t = d_t - d_s  (1 row per line)
        if max_lines > 0:
            s_norm_rot = norms_s_pad @ R_final.transpose(-1, -2)  # (N, max_lines, 2)
            line_res = offs_t_aligned - offs_s_pad  # (N, max_lines)

            row_off = 2 * max_pts
            A[:, row_off:row_off + max_lines, :] = W_L_SQ * s_norm_rot
            b[:, row_off:row_off + max_lines, 0] = W_L_SQ * line_res

            line_mask_f2 = line_mask.unsqueeze(-1).to(A.dtype)
            A[:, row_off:row_off + max_lines] *= line_mask_f2
            b[:, row_off:row_off + max_lines] *= line_mask_f2

        t_sol = torch.linalg.lstsq(A, b).solution.squeeze(-1)  # (N, 2)

        # ---- Step 5: Assemble T and invert ----
        T = torch.eye(3, device=device, dtype=torch.float64).unsqueeze(0).expand(N, -1, -1).clone()
        T[:, :2, :2] = R_final
        T[:, :2, 2] = t_sol
        T = torch.linalg.inv(T)

        # NaN out failed problems
        failed = ~valid
        T[failed] = float("nan")
        n_failed = failed.sum().item()
        print(f"Batched 2D registration: {n_failed}/{N} failed")

        return T.cpu().numpy()
