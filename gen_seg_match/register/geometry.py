import numpy as np
from typing import Optional, List, Union
from gen_seg_match.params import RegisterParams
from gen_seg_match.segment.segment_types import (
    SegmentPoint,
    GeneralSegment,
    SegmentList,
    SegmentLine,
)


def line_normals_2d(lines: SegmentList) -> np.ndarray:
    """Get unit normals for 2D lines: rotate direction by +90 degrees."""
    dirs = lines.get_lines().directions  # (N, 2)
    if len(dirs) == 0:
        return dirs
    return np.column_stack([-dirs[:, 1], dirs[:, 0]])


def line_offsets_2d(lines: SegmentList) -> np.ndarray:
    """Get signed distances from origin for 2D lines: d = dot(normal, point)."""
    normals = line_normals_2d(lines)
    if len(normals) == 0:
        return np.empty((0, 1))
    points = np.array([seg.point for seg in lines.get_lines()]).reshape(-1, 2)
    return np.sum(normals * points, axis=1, keepdims=True)


def cross_covariance(
    s_p: np.ndarray,
    t_p: np.ndarray,
    s_dir: np.ndarray,
    t_dir: np.ndarray,
    W_P: Optional[float] = 1.0,
    W_D: Optional[Union[float, List[float]]] = 1.0,
) -> np.ndarray:
    """
    Compute the 2x2 cross-covariance matrix between two sets of 2D points and
    line normal directions.

    Parameters:
    s_p : np.ndarray
        An Nx2 array of 2D points from the source cloud.
    t_p : np.ndarray
        An Nx2 array of 2D points from the target cloud.
    s_dir : np.ndarray
        An Mx2 array of 2D direction vectors (line normals) from the source cloud.
    t_dir : np.ndarray
        An Mx2 array of 2D direction vectors (line normals) from the target cloud.
    W_P : Optional[float]
        Weight for point correspondences.
    W_D : Optional[Union[float, List[float]]]
        Weight(s) for direction correspondences.

    Returns:
    H : np.ndarray
        A 2x2 cross-covariance matrix.
    """
    assert len(s_p) == len(t_p), "Number of source and target points must be the same."
    assert len(s_dir) == len(t_dir), (
        "Number of source and target direction vectors must be the same."
    )

    H = np.zeros((2, 2))

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


class PointLineLoss2D:
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
    ) -> float:
        """
        Compute the combined 2D point and line loss.

        Parameters:
        R : np.ndarray
            A 2x2 rotation matrix.
        t : np.ndarray
            A 2x1 translation vector.
        source : List[GeneralSegment]
            List of source segments (points, lines).
        target : List[GeneralSegment]
            List of target segments (points, lines).

        Returns:
        loss : float
            The computed loss value.
        """
        source = SegmentList(source)
        target = SegmentList(target)

        num_points = len(source.get_points())
        num_lines = len(source.get_lines())

        s_p = source.get_points().points  # (N, 2)
        t_p = target.get_points().points

        s_norm = line_normals_2d(source)  # (M, 2)
        s_off = line_offsets_2d(source)  # (M, 1)
        t_norm = line_normals_2d(target)
        t_off = line_offsets_2d(target)

        loss = 0.0

        # -----------------------
        # Point correspondences
        # -----------------------
        if num_points > 0:
            s_p_transformed = (R @ s_p.T).T + t
            loss += (
                0.5 * self.params.point_weight * np.sum((s_p_transformed - t_p) ** 2)
            )

        # -----------------------
        # Line correspondences (2D Plücker: normal + offset)
        # -----------------------
        if num_lines > 0:
            # n' = R @ n
            s_norm_transformed = (R @ s_norm.T).T
            # d' = d + n'^T @ t
            s_off_transformed = s_off + np.sum(
                s_norm_transformed * t.ravel(), axis=1, keepdims=True
            )

            # Line normal loss
            normal_cosine_sim = np.sum(s_norm_transformed * t_norm, axis=1)

            if self.bidirectional:
                reverse_mask = normal_cosine_sim < 0
                normal_cosine_sim[reverse_mask] *= -1
                s_off_transformed[reverse_mask] *= -1

            loss += self.params.line_direction_weight * np.sum(-normal_cosine_sim)

            # Line offset loss
            loss += (
                0.5
                * self.params.line_moment_weight
                * np.sum((s_off_transformed - t_off) ** 2)
            )

        return loss
