import numpy as np
from typing import Optional, List, Union, Iterable
from gen_seg_match.params import RegisterParams
from gen_seg_match.segment.segment_types import (
    SegmentPoint,
    GeneralSegment,
    SegmentList,
    SegmentLine,
    SegmentPlane,
)

# import torch


def cross_covariance(
    s_p: np.ndarray,
    t_p: np.ndarray,
    s_dir: np.ndarray,
    t_dir: np.ndarray,
    W_P: Optional[float] = 1.0,
    W_D: Optional[Union[float, List[float]]] = 1.0,
) -> np.ndarray:
    """
    Compute the cross-covariance matrix between two sets of points and line/plane/gravity directions.

    Parameters:
    s_p : np.ndarray
        An Nx3 array of 3D points from the source cloud.
    t_p : np.ndarray
        An Nx3 array of 3D points from the target cloud.
    s_dir : np.ndarray
        An Mx3 array of 3D line/plane/gravity direction vectors from the source cloud.
    t_dir : np.ndarray
        An Mx3 array of 3D line/plane/gravity direction vectors from the target cloud.
    W_P : Optional[float]
        Weight for point correspondences.
    W_D : Optional[Optional[Union[float, List[float]]]]
        Weight(s) for direction correspondences.

    Returns:
    H : np.ndarray
        A 3x3 cross-covariance matrix.
    """
    assert len(s_p) == len(t_p), "Number of source and target points must be the same."
    assert len(s_dir) == len(t_dir), (
        "Number of source and target direction vectors must be the same."
    )

    H = np.zeros((3, 3))

    if len(s_p) > 0:
        s_p_centr = s_p - s_p.mean(axis=0)
        t_p_centr = t_p - t_p.mean(axis=0)
        H += W_P * (s_p_centr.T @ t_p_centr)

    if len(s_dir) > 0:
        if isinstance(W_D, (int, float)):
            W_D = [W_D] * len(s_dir)
        else:
            assert len(W_D) == len(s_dir), (
                "Length of W_D must match number of direction vectors."
            )

        W_D = np.array(W_D).reshape(-1, 1)
        H += (W_D * s_dir).T @ t_dir

    return H


def rank(A: np.ndarray, tol: float = 1e-9) -> int:
    """Compute the rank of matrix A using SVD."""
    U, S, Vt = np.linalg.svd(A)
    return np.sum(S > tol)


# def transform_plucker_torch(
#     R: torch.Tensor, t: torch.Tensor, plucker_coords: torch.Tensor
# ) -> torch.Tensor:
#     """
#     Transform Plücker coordinates of lines using a rigid transformation (PyTorch).

#     Parameters:
#     R : torch.Tensor
#         A 3x3 rotation matrix.
#     t : torch.Tensor
#         A 3x1 translation vector.
#     plucker_coords : torch.Tensor
#         An Nx6 array of (d, m) Plücker coordinates for the lines.

#     Returns:
#     transformed_plucker : torch.Tensor
#         An Nx6 array of transformed (d, m) Plücker coordinates.
#     """
#     l_dir = plucker_coords[:, :3]
#     l_mom = plucker_coords[:, 3:]

#     # d' = R @ d
#     transformed_dir = (R @ l_dir.T).T
#     # m' = R @ m + t x d'
#     transformed_mom = (R @ l_mom.T).T + torch.cross(
#         t.unsqueeze(0), transformed_dir, dim=1
#     )

#     transformed_plucker = torch.cat([transformed_dir, transformed_mom], dim=1)
#     return transformed_plucker


def skew(v: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric cross-product matrix [v]x."""
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


class PointLinePlaneLoss:
    def __init__(
        self,
        params: RegisterParams,
        bidirectional: bool = False,
    ):
        self.params = params
        self.bidirectional = bidirectional

    def compute_loss(
        self,
        R: np.ndarray,
        t: np.ndarray,
        source: List[GeneralSegment],
        target: List[GeneralSegment],
        gravity_src: Optional[np.ndarray] = None,
        gravity_tgt: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute the combined point, line, plane, and gravity loss.

        Parameters:
        R : np.ndarray
            A 3x3 rotation matrix.
        t : np.ndarray
            A 3x1 translation vector.
        source : List[GeneralSegment]
            List of source segments (points, lines, planes).
        target : List[GeneralSegment]
            List of target segments (points, lines, planes).
        gravity_src : Optional[np.ndarray]
            Gravity direction in source frame.
        gravity_tgt : Optional[np.ndarray]
            Gravity direction in target frame.

        Returns:
        loss : float
            The computed loss value.
        """
        source = SegmentList(source)
        target = SegmentList(target)

        num_points = len(source.get_points())
        num_lines = len(source.get_lines())
        num_planes = len(source.get_planes())
        use_gravity = (
            self.params.use_gravity
            and gravity_src is not None
            and gravity_tgt is not None
        )

        s_p = source.get_points().points  # (N, 3)
        t_p = target.get_points().points

        s_dirs, s_mom = (
            source.get_lines().directions,
            source.get_lines().moments,
        )  # (M, 3), (M, 3)
        t_dirs, t_mom = target.get_lines().directions, target.get_lines().moments

        s_norm, s_off = (
            source.get_planes().normals,
            source.get_planes().offsets,
        )  # (O, 3), (O, 1)
        t_norm, t_off = target.get_planes().normals, target.get_planes().offsets

        loss = 0.0

        # -----------------------
        # Point correspondences
        # -----------------------
        if num_points > 0:
            # Point loss
            s_p_transformed = (R @ s_p.T).T + t
            loss += (
                0.5 * self.params.point_weight * np.sum((s_p_transformed - t_p) ** 2)
            )

        # -----------------------
        # Line correspondences (Plücker)
        # -----------------------
        if num_lines > 0:
            # d' = R @ d
            s_dirs_transformed = (R @ s_dirs.T).T
            # m' = R @ m + t x d'
            s_mom_transformed = (R @ s_mom.T).T + np.cross(
                t.ravel(), s_dirs_transformed
            )

            # Line direction loss
            dir_cosine_sim = np.sum(s_dirs_transformed * t_dirs, axis=1)

            if self.bidirectional:
                # also aligns moments for consistency
                reverse_mask = dir_cosine_sim < 0
                dir_cosine_sim[reverse_mask] *= -1
                s_mom_transformed[reverse_mask] *= -1

            loss += self.params.line_direction_weight * np.sum(-dir_cosine_sim)

            # Line moment loss
            loss += (
                0.5
                * self.params.line_moment_weight
                * np.sum((s_mom_transformed - t_mom) ** 2)
            )

        # -----------------------
        # Plane correspondences
        # -----------------------
        if num_planes > 0:
            # n' = R @ n
            s_norm_transformed = (R @ s_norm.T).T
            # c' = c + n' * t
            s_off_transformed = s_off + np.sum(
                s_norm_transformed * t.ravel(), axis=1, keepdims=True
            )

            # Plane normal loss
            normal_cosine_sim = np.sum(s_norm_transformed * t_norm, axis=1)

            if self.bidirectional:
                # also aligns offsets for consistency
                reverse_mask = normal_cosine_sim < 0
                normal_cosine_sim[reverse_mask] *= -1
                s_off_transformed[reverse_mask] *= -1

            loss += self.params.plane_normal_weight * np.sum(-normal_cosine_sim)

            # Plane offset loss
            loss += (
                0.5
                * self.params.plane_offset_weight
                * np.sum((s_off_transformed - t_off) ** 2)
            )

        # -----------------------
        # Gravity correspondence
        # -----------------------
        if use_gravity:
            loss += self.params.gravity_weight * (
                -np.dot(R @ gravity_src.ravel(), gravity_tgt.ravel())
            )

        return loss
