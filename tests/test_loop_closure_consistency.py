"""Tests for LoopClosureConsistency C++ invariant.

Verifies that the C++ implementation produces the same affinity matrix as the
reference Python implementation in CrossViewRPGO._build_affinity_matrix_python.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

import clipperpy
from meridian.cross_view.rpgo import (
    CrossViewRPGO,
    CrossViewRPGOParams,
    _build_clipper_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_se3(rng: np.random.Generator, trans_scale: float = 20.0) -> np.ndarray:
    """Return a random 4x4 SE(3) matrix."""
    T = np.eye(4)
    T[:3, :3] = Rot.random(random_state=rng).as_matrix()
    T[:3, 3] = rng.uniform(-trans_scale, trans_scale, 3)
    return T


def make_consistent_T_hat(
    T_odom_ground_i: np.ndarray,
    T_odom_ground_j: np.ndarray,
    aerial_i: np.ndarray,
    aerial_j: np.ndarray,
    noise_rot_rad: float = 0.0,
    noise_trans_m: float = 0.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Construct T_hat_j such that candidates i and j are perfectly consistent.

    From the consistency equation:
        odom_relative = inv(T_odom_ground_i) @ T_odom_ground_j
        cv_relative   = inv(T_hat_i) @ inv(aerial_i) @ aerial_j @ T_hat_j
        error         = inv(odom_relative) @ cv_relative  ≈ I

    Setting T_hat_i = I and solving for T_hat_j:
        T_hat_j = aerial_j^{-1} @ aerial_i @ odom_relative
    """
    odom_relative = np.linalg.inv(T_odom_ground_i) @ T_odom_ground_j
    T_hat_j = np.linalg.inv(aerial_j) @ aerial_i @ odom_relative
    if noise_rot_rad > 0 or noise_trans_m > 0:
        # Add small noise
        dR = Rot.from_rotvec(
            [0, 0, rng.uniform(-noise_rot_rad, noise_rot_rad)]
        ).as_matrix()
        noise = np.eye(4)
        noise[:3, :3] = dR
        noise[:2, 3] = rng.uniform(-noise_trans_m, noise_trans_m, 2)
        T_hat_j = T_hat_j @ noise
    return T_hat_j


def build_candidates(
    n_consistent: int = 5,
    n_outliers: int = 3,
    seed: int = 42,
) -> tuple:
    """Build synthetic loop closure candidates with n_consistent inliers + n_outliers.

    Returns (candidates, trajectory, times) where trajectory and times are
    simple identity poses at integer timestamps matching ground_submap_time.
    """
    rng = np.random.default_rng(seed)

    # Shared aerial patch
    aerial_pose = random_se3(rng)

    # Ground trajectory
    ground_poses = [random_se3(rng) for _ in range(n_consistent + n_outliers)]

    # Reference T_hat for candidate 0 (identity)
    T_hat_ref = np.eye(4)

    candidates = []

    # Consistent cluster: all share the same "true" cross-view transform
    for k in range(n_consistent):
        T_odom_ground = ground_poses[k]
        T_hat = make_consistent_T_hat(
            ground_poses[0],
            T_odom_ground,
            aerial_pose,
            aerial_pose,
            noise_rot_rad=0.02,  # ~1 deg
            noise_trans_m=0.5,
            rng=rng,
        )
        # For k==0, T_hat should be T_hat_ref (approximately)
        if k == 0:
            T_hat = T_hat_ref.copy()
        candidates.append(
            {
                "aerial_pose": aerial_pose.copy(),
                "T_odom_ground": T_odom_ground.copy(),
                "T_i_j_hat": T_hat.copy(),
                "T_utm_odom_se2": np.eye(3),
                "ground_submap_time": float(k),
                "ground_key": str(k),
                "aerial_key": "0_0",
                "count": 1,
            }
        )

    # Outliers: random T_hat (inconsistent)
    for k in range(n_outliers):
        T_odom_ground = ground_poses[n_consistent + k]
        T_hat = random_se3(rng)
        candidates.append(
            {
                "aerial_pose": aerial_pose.copy(),
                "T_odom_ground": T_odom_ground.copy(),
                "T_i_j_hat": T_hat.copy(),
                "T_utm_odom_se2": np.eye(3),
                "ground_submap_time": float(n_consistent + k),
                "ground_key": str(n_consistent + k),
                "aerial_key": "0_0",
                "count": 1,
            }
        )

    # Build a simple trajectory with identity poses at integer timestamps
    n_total = n_consistent + n_outliers
    trajectory = [np.eye(4) for _ in range(n_total)]
    times = np.arange(n_total, dtype=float)

    return candidates, trajectory, times


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_affinity_matrix_agrees():
    """C++ invariant affinity matrix must match the Python reference.

    Uses added_*_noise = 0 so that the C++ and Python results are comparable
    (the Python reference does not implement distance-dependent noise).
    """
    params = CrossViewRPGOParams(
        added_trans_noise_m_per_m=0.0,
        added_rot_noise_deg_per_m=0.0,
    )
    rpgo = CrossViewRPGO(params=params)

    candidates, trajectory, times = build_candidates(
        n_consistent=5, n_outliers=3, seed=0
    )
    N = len(candidates)

    # Python reference
    M_py, _ = rpgo._build_affinity_matrix_python(candidates)

    # C++ via CLIPPERPairwiseAndSingle
    lc_scores = np.ones(len(candidates), dtype=np.float64)
    D, aerial_poses, ground_poses, ground_distances = _build_clipper_data(
        candidates, trajectory, times, lc_scores
    )
    iparams = clipperpy.invariants.LoopClosureConsistencyParams()
    iparams.rot_sigma_rad = params.rot_consistency_sigma_rad
    iparams.rot_eps_rad = params.rot_consistency_eps_rad
    iparams.trans_sigma_m = params.trans_consistency_sigma_m
    iparams.trans_eps_m = params.trans_consistency_eps_m
    iparams.added_trans_noise_m_per_m = 0.0
    iparams.added_rot_noise_deg_per_m = 0.0
    iparams.fuse_lc_score = False
    invariant = clipperpy.invariants.LoopClosureConsistency(
        aerial_poses, ground_poses, ground_distances, iparams
    )
    clipper = clipperpy.CLIPPERPairwiseAndSingle(invariant, clipperpy.Params())
    A = np.stack([np.arange(N), np.arange(N)], axis=1).astype(np.int32)
    clipper.score_pairwise_and_single_consistency(D, D, A)
    M_cpp = clipper.get_affinity_matrix()

    assert M_cpp.shape == (N, N), f"Expected ({N},{N}), got {M_cpp.shape}"
    np.testing.assert_allclose(
        M_cpp,
        M_py,
        atol=1e-6,
        err_msg="C++ and Python affinity matrices differ",
    )


def test_diagonal_ones():
    """Diagonal of the affinity matrix should be 1.0."""
    params = CrossViewRPGOParams()
    candidates, trajectory, times = build_candidates(
        n_consistent=4, n_outliers=2, seed=1
    )
    N = len(candidates)

    lc_scores = np.ones(len(candidates), dtype=np.float64)
    D, aerial_poses, ground_poses, ground_distances = _build_clipper_data(
        candidates, trajectory, times, lc_scores
    )
    iparams = clipperpy.invariants.LoopClosureConsistencyParams()
    iparams.rot_sigma_rad = params.rot_consistency_sigma_rad
    iparams.rot_eps_rad = params.rot_consistency_eps_rad
    iparams.trans_sigma_m = params.trans_consistency_sigma_m
    iparams.trans_eps_m = params.trans_consistency_eps_m
    invariant = clipperpy.invariants.LoopClosureConsistency(
        aerial_poses, ground_poses, ground_distances, iparams
    )
    clipper = clipperpy.CLIPPERPairwiseAndSingle(invariant, clipperpy.Params())
    A = np.stack([np.arange(N), np.arange(N)], axis=1).astype(np.int32)
    clipper.score_pairwise_and_single_consistency(D, D, A)
    M = clipper.get_affinity_matrix()

    np.testing.assert_allclose(np.diag(M), np.ones(N), atol=1e-9)


def test_run_clipper_cpp_finds_inliers():
    """run_clipper_cpp should return at least the consistent cluster."""
    params = CrossViewRPGOParams()
    rpgo = CrossViewRPGO(params=params)
    candidates, trajectory, times = build_candidates(
        n_consistent=5, n_outliers=3, seed=2
    )

    inlier_indices, M, C = rpgo.run_clipper_cpp(candidates, trajectory, times)

    # All returned indices should be within the consistent cluster (indices 0..4)
    assert len(inlier_indices) > 0, "No inliers found"
    assert all(idx < 5 for idx in inlier_indices), f"Outlier selected: {inlier_indices}"
