import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from itertools import combinations, product

from gen_seg_match.segment.segment_types import (
    SegmentPoint,
    GeneralSegment,
    SegmentList,
    SegmentLine,
    SegmentPlane,
)
from gen_seg_match.params import RegisterParams
from gen_seg_match.register.geometry import (
    cross_covariance,
    rank,
    skew,
    PointLinePlaneLoss,
)
from gen_seg_match.utils import vstack_opt
# from gen_seg_match.register.optimization import PointLineLoss, PointLineLoss_numpy


class InsufficientAssociationsException(Exception):
    def __init__(self, map1_len, map2_len, n_associations=None, message=None):
        self.map1_len = map1_len
        self.map2_len = map2_len
        self.n_associations = n_associations
        if message is None:
            message = f"Insufficient associations. Map 1 length: {map1_len}. Map 2 length: {map2_len}. Associations: {n_associations}"
        super().__init__(message)


@dataclass
class RegistrationResult:
    transformation: np.ndarray
    losses: List[float]


class Registerer:
    def __init__(self, params: RegisterParams):
        self.params = params

    def register(
        self,
        source: List[GeneralSegment],
        target: List[GeneralSegment],
        gravity_src: Optional[np.ndarray] = None,
        gravity_tgt: Optional[np.ndarray] = None,
        correspondences: np.array = None,
    ):
        """
        Computes the transformation that aligns target to source (T^source_target).

        Args:
            source (List[GeneralSegment]): Segment list in source frame
            target (List[GeneralSegment]): Segment list in target frame
            gravity_src (Optional[np.ndarray]): Gravity direction in source frame. Defaults to None.
            gravity_tgt (Optional[np.ndarray]): Gravity direction in target frame. Defaults to None.
            correspondences (np.array, shape=(n,2), optional): If correspondences have already
                been found, set to None. Otherwise, performs register before aligning. Defaults to None.

        Returns:
            np.array: Transformation matrix that aligns target to source (T^source_target).
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

        use_gravity = (
            self.params.use_gravity
            and gravity_src is not None
            and gravity_tgt is not None
        )

        if use_gravity:
            gravity_src = gravity_src.reshape(1, 3)
            gravity_tgt = gravity_tgt.reshape(1, 3)

        num_points = len(source.get_points())
        num_lines = len(source.get_lines())
        num_planes = len(source.get_planes())

        p = source.get_points().points
        q = target.get_points().points

        s_dir, s_mom = source.get_lines().directions, source.get_lines().moments
        t_dir, t_mom = target.get_lines().directions, target.get_lines().moments

        s_norm, s_off = source.get_planes().normals, source.get_planes().offsets
        t_norm, t_off = target.get_planes().normals, target.get_planes().offsets

        H = cross_covariance(
            p,
            q,
            gravity_src if use_gravity else [],
            gravity_tgt if use_gravity else [],
            W_P=self.params.point_weight,
            W_D=self.params.gravity_weight if use_gravity else 1.0,
        )

        H_init_rank = rank(H, tol=self.params.lin_eps)
        dirs_needed = (
            max(2 - H_init_rank, 0) if num_points > 0 else (3 - int(use_gravity))
        )

        if num_lines + num_planes < dirs_needed:
            raise InsufficientAssociationsException(
                len(source), len(target), num_lines + num_planes
            )

        dir_idxs = []
        if dirs_needed > 0:
            s_comb_dir = vstack_opt((s_dir, s_norm))
            t_comb_dir = vstack_opt((t_dir, t_norm))
            comb_weights = np.array(
                [self.params.line_direction_weight] * num_lines
                + [self.params.plane_normal_weight] * num_planes
            )

            for idx_comb in combinations(range(num_lines + num_planes), dirs_needed):
                idx_comb = list(idx_comb)
                s_dir_subset = s_comb_dir[idx_comb]
                t_dir_subset = t_comb_dir[idx_comb]
                subset_weights = comb_weights[idx_comb]

                H_temp = H + cross_covariance(
                    [],
                    [],
                    s_dir_subset,
                    t_dir_subset,
                    self.params.point_weight,
                    subset_weights,
                )

                if rank(H_temp, tol=self.params.lin_eps) >= 2:
                    dir_idxs = idx_comb
                    break

        # TODO: more specific error
        if len(dir_idxs) < dirs_needed:
            raise InsufficientAssociationsException(
                len(source), len(target), num_lines + num_planes + int(use_gravity)
            )

        if dirs_needed > 0:
            PLPLoss = PointLinePlaneLoss(
                params=self.params,
                bidirectional=True,
            )

            best_loss, second_best_loss = float("inf"), float("inf")
            best_R = None

            # TODO: make this cleaner (need to edit above indexing too)
            source_subset = (
                source.get_points()
                + [source.get_lines()[i] for i in dir_idxs if i < num_lines]
                + [
                    source.get_planes()[i - num_lines]
                    for i in dir_idxs
                    if i >= num_lines
                ]
            )

            for signs in product([-1, 1], repeat=dirs_needed):
                target_subset_signed = [
                    target.get_lines()[i].copy() for i in dir_idxs if i < num_lines
                ] + [
                    target.get_planes()[i - num_lines].copy()
                    for i in dir_idxs
                    if i >= num_lines
                ]

                for idx, segment in enumerate(target_subset_signed):
                    if isinstance(segment, SegmentLine):
                        segment.direction *= signs[idx]
                    elif isinstance(segment, SegmentPlane):
                        segment.normal *= signs[idx]

                target_subset_signed = target.get_points() + target_subset_signed

                R_candidate, t_candidate = self.aruns_extended(
                    source_subset,
                    target_subset_signed,
                    gravity_src,
                    gravity_tgt,
                    rotation_only=False,
                )

                loss = PLPLoss.compute_loss(
                    R=R_candidate,
                    t=t_candidate,
                    source=source,
                    target=target,
                    gravity_src=gravity_src,
                    gravity_tgt=gravity_tgt,
                )

                if loss < best_loss:
                    second_best_loss = best_loss
                    best_loss = loss
                    best_R = R_candidate
                elif loss < second_best_loss:
                    second_best_loss = loss

            # print(second_best_loss - best_loss)
            if second_best_loss - best_loss < self.params.dup_eps:
                raise InsufficientAssociationsException(
                    len(source),
                    len(target),
                    num_lines,
                    "Degenerate configuration: multiple sign combinations yield similar losses.",
                )

            R = best_R

        else:
            R, _ = self.aruns_extended(
                source.get_points(),
                target.get_points(),
                gravity_src,
                gravity_tgt,
                rotation_only=True,
            )

        # Make target directions sign-consistent with source
        target_consistent = self.compute_consistent_directions(R, source, target)

        return self.register_consistent(
            source, target_consistent, gravity_src, gravity_tgt
        )

    def register_consistent(
        self,
        source: SegmentList,
        target: SegmentList,
        gravity_src: Optional[np.ndarray] = None,
        gravity_tgt: Optional[np.ndarray] = None,
    ) -> RegistrationResult:
        R, t = self.aruns_extended(
            source, target, gravity_src, gravity_tgt, rotation_only=False
        )

        if self.params.run_gd:
            assert False, "Refinement not yet supported"
            # R, t, losses = refine_transform(
            #     R,
            #     t,
            #     source,
            #     target,
            #     device=self.params.device,
            #     **self.params.refine_transforms_kwargs,
            # )
        else:
            losses = []

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        T = np.linalg.inv(T)

        return RegistrationResult(transformation=T, losses=losses)

    def compute_consistent_directions(
        self, R: np.ndarray, source: SegmentList, target: SegmentList
    ) -> SegmentList:
        """
        Create a sign-consistent version of the target cloud by flipping line directions
        and moments together to align with the rotated source lines.

        Parameters:
        R : np.ndarray
            A 3x3 rotation matrix estimate.
        source : SegmentList
            Source segments (points, lines, planes), not sign-consistent.
        target : SegmentList
            Target segments (points, lines, planes), not sign-consistent.

        Returns:
        target_consistent : SegmentList
            Target cloud with sign-consistent line representations.
        """

        # TODO: avoid duplicate code
        s_dir = source.get_lines().directions
        t_lines = target.get_lines().copy()

        for i in range(len(s_dir)):
            s_dir_transformed = R @ s_dir[i]
            if np.dot(s_dir_transformed, t_lines[i].direction) < 0:
                # Flip direction
                t_lines[i].direction = -t_lines[i].direction

        s_norm = source.get_planes().normals
        t_planes = target.get_planes().copy()

        for i in range(len(s_norm)):
            s_norm_transformed = R @ s_norm[i]
            if np.dot(s_norm_transformed, t_planes[i].normal) < 0:
                # Flip normal
                t_planes[i].normal = -t_planes[i].normal

        return SegmentList(target.get_points().copy() + t_lines + t_planes)

    # TODO: add a 2D (SE(2)) registration mode for cross-view matching where
    # the problem is planar. The current 3D SVD can produce improper rotations
    # (det(R_2x2) = -1) when segments are coplanar. Currently handled by
    # rejecting reflections downstream in cross_view_localization.py.
    def aruns_extended(
        self,
        source: SegmentList,
        target: SegmentList,
        gravity_src: Optional[np.ndarray] = None,
        gravity_tgt: Optional[np.ndarray] = None,
        rotation_only: Optional[bool] = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the transformation between associated point-line clouds
        using Arun's method extended to include lines, planes, and optionally gravity.

        Parameters:
        source : SegmentList
            Source segments (points, lines, planes), sign-consistent.
        target : SegmentList
            Target segments (points, lines, planes), sign-consistent.
        gravity_src : Optional[np.ndarray]
            Gravity direction in source frame.
        gravity_tgt : Optional[np.ndarray]
            Gravity direction in target frame.
        rotation_only : Optional[bool]
            If True, only compute rotation and set translation to zero.

        Returns:
        R : np.ndarray
            A 3x3 rotation matrix.
        t : np.ndarray
            A 3x1 translation vector.
        """

        num_points, num_lines, num_planes = (
            len(source.get_points()),
            len(source.get_lines()),
            len(source.get_planes()),
        )
        use_gravity = (
            self.params.use_gravity
            and gravity_src is not None
            and gravity_tgt is not None
        )
        assert (
            num_points == len(target.get_points())
            and num_lines == len(target.get_lines())
            and num_planes == len(target.get_planes())
        ), "Source and target must have the same number of points and lines."
        assert (
            num_lines + num_planes + int(use_gravity) >= 2
            or num_points + num_lines + num_planes + int(use_gravity) >= 3
        ), "At least two directional or three total correspondences are required."

        p, q = source.get_points().points, target.get_points().points
        s_dir, t_dir = (
            vstack_opt((source.get_lines().directions, source.get_planes().normals)),
            vstack_opt((target.get_lines().directions, target.get_planes().normals)),
        )
        weights = [self.params.line_direction_weight] * num_lines + [
            self.params.plane_normal_weight
        ] * num_planes

        if use_gravity:
            s_dir = vstack_opt((s_dir, gravity_src))
            t_dir = vstack_opt((t_dir, gravity_tgt))
            weights.append(self.params.gravity_weight)

        H = cross_covariance(
            p,
            q,
            s_dir,
            t_dir,
            W_P=self.params.point_weight,
            W_D=weights,
        )

        U, S, Vt = np.linalg.svd(H)

        if np.abs(S[1]) < self.params.lin_eps:
            # TODO: more specific error
            raise InsufficientAssociationsException(-1, -1)

        if np.linalg.det(Vt.T @ U.T) < 0:
            Vt[2, :] *= -1

        R = Vt.T @ U.T
        t = (
            self.solve_translation(R, source, target)
            if not rotation_only
            else np.zeros(3)
        )

        return R, t

    def solve_translation(
        self, R: np.ndarray, source: SegmentList, target: SegmentList
    ) -> np.ndarray:
        """
        Solve for the translation vector given rotation and associated point-line clouds
        by minimizing L2 error.

        Parameters:
        R : np.ndarray
            A 3x3 rotation matrix.
        source : PointLineCloud
            Source point-line cloud with points (Nx3) and lines (Mx6 (d, m) Plücker coordinates).
        target : PointLineCloud
            Target point-line cloud with points (Nx3) and lines (Mx6 (d, m) Plücker coordinates).

        Returns:
        t : np.ndarray
            A 3x1 translation vector.
        """
        num_points, num_lines, num_planes = (
            len(source.get_points()),
            len(source.get_lines()),
            len(source.get_planes()),
        )
        assert (
            num_points == len(target.get_points())
            and num_lines == len(target.get_lines())
            and num_planes == len(target.get_planes())
        ), "Source and target must have the same number of points and lines."
        assert num_points + num_lines + num_planes > 0, (
            "At least one correspondence is required to solve for translation."
        )

        p, q = source.get_points().points, target.get_points().points
        s_dir, s_mom = source.get_lines().directions, source.get_lines().moments
        t_dir, t_mom = target.get_lines().directions, target.get_lines().moments

        s_norm, s_off = source.get_planes().normals, source.get_planes().offsets
        t_norm, t_off = target.get_planes().normals, target.get_planes().offsets

        A = np.zeros((3 * (num_points + num_lines) + num_planes, 3))
        b = np.zeros(3 * (num_points + num_lines) + num_planes)

        W_P_SQ = np.sqrt(self.params.point_weight)
        W_L_M_SQ = np.sqrt(self.params.line_moment_weight)
        W_F_C_SQ = np.sqrt(self.params.plane_offset_weight)

        # TODO: optimize with vectorization

        # Point constraints: t = q - R @ p
        for i in range(num_points):
            A[3 * i : 3 * i + 3] = W_P_SQ * np.eye(3)
            b[3 * i : 3 * i + 3] = W_P_SQ * (q[i] - R @ p[i])

        # Line constraints: -[t_dir]_x @ t = t_mom - R @ s_mom
        for j in range(num_lines):
            A[3 * (num_points + j) : 3 * (num_points + j) + 3] = W_L_M_SQ * -skew(
                R @ s_dir[j]
            )  # TODO: should be R @ s_dir[j]?
            b[3 * (num_points + j) : 3 * (num_points + j) + 3] = W_L_M_SQ * (
                t_mom[j] - R @ s_mom[j]
            )

        # Plane constraints: (R @ s_norm)^T @ t = t_off - s_off
        for k in range(num_planes):
            A[3 * (num_points + num_lines) + k] = W_F_C_SQ * (R @ s_norm[k]).T
            b[3 * (num_points + num_lines) + k] = W_F_C_SQ * (t_off[k] - s_off[k])

        t, *_ = np.linalg.lstsq(A, b, rcond=None)
        return t
