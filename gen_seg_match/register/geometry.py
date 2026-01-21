import numpy as np
from typing import Optional
import torch


def cross_covariance(
    p: np.ndarray,
    q: np.ndarray,
    l: np.ndarray,
    j: np.ndarray,
    W_P: Optional[float] = 1.0,
    W_L_D: Optional[float] = 1.0,
) -> np.ndarray:
    """
    Compute the cross-covariance matrix between two sets of points and line directions.

    Parameters:
    p : np.ndarray
        An Nx3 array of 3D points from the source cloud.
    q : np.ndarray
        An Nx3 array of 3D points from the target cloud.
    l : np.ndarray
        An Mx3 array of 3D line direction vectors from the source cloud.
    j : np.ndarray
        An Mx3 array of 3D line direction vectors from the target cloud.
    W_P : Optional[float]
        Weight for point correspondences.
    W_L_D : Optional[float]
        Weight for line direction correspondences.

    Returns:
    H : np.ndarray
        A 3x3 cross-covariance matrix.
    """
    assert len(p) == len(q), "Number of source and target points must be the same."
    assert len(l) == len(j), "Number of source and target lines must be the same."

    H = np.zeros((3, 3))

    if len(p) > 0:
        p_centered = p - p.mean(axis=0)
        q_centered = q - q.mean(axis=0)
        H += W_P * (p_centered.T @ q_centered)

    if len(l) > 0:
        H += W_L_D * (l.T @ j)

    return H


def rank(A: np.ndarray, tol: float = 1e-9) -> int:
    """Compute the rank of matrix A using SVD."""
    U, S, Vt = np.linalg.svd(A)
    return np.sum(S > tol)


def transform_plucker_torch(
    R: torch.Tensor, t: torch.Tensor, plucker_coords: torch.Tensor
) -> torch.Tensor:
    """
    Transform Plücker coordinates of lines using a rigid transformation (PyTorch).

    Parameters:
    R : torch.Tensor
        A 3x3 rotation matrix.
    t : torch.Tensor
        A 3x1 translation vector.
    plucker_coords : torch.Tensor
        An Nx6 array of (d, m) Plücker coordinates for the lines.

    Returns:
    transformed_plucker : torch.Tensor
        An Nx6 array of transformed (d, m) Plücker coordinates.
    """
    l_dir = plucker_coords[:, :3]
    l_mom = plucker_coords[:, 3:]

    # d' = R @ d
    transformed_dir = (R @ l_dir.T).T
    # m' = R @ m + t x d'
    transformed_mom = (R @ l_mom.T).T + torch.cross(
        t.unsqueeze(0), transformed_dir, dim=1
    )

    transformed_plucker = torch.cat([transformed_dir, transformed_mom], dim=1)
    return transformed_plucker


def skew(v: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric cross-product matrix [v]x."""
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
