"""Tests for CrossViewRPGO.lift_to_3d and its numpy Levenberg-Marquardt _refine_3d.

_refine_3d is checked against the same factor graph solved with GTSAM (BetweenFactorPose3
plus CustomFactors), and on synthetic trajectories with known roll/pitch and height drift.
"""

import gtsam
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from meridian.cross_view.rpgo import (
    _LIFT_ANCHOR_SIGMA,
    CrossViewRPGO,
    CrossViewRPGOParams,
    cross_path_pairs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def figure_eight(n: int, a: float = 20.0, yaw_offset: float = 0.0) -> np.ndarray:
    """(n, 4, 4) T_world_body along a lemniscate on a sloped, bumpy surface.

    The path crosses itself at the origin; height is a function of (x, y), so both
    passes have the same height there.
    """
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = a * np.sin(s), a * np.sin(s) * np.cos(s)
    z = 0.05 * x + 0.5 * np.sin(0.2 * y)
    p = np.stack([x, y, z], 1)
    d = np.gradient(p, axis=0)
    yaw = np.arctan2(d[:, 1], d[:, 0]) + yaw_offset
    pitch = -np.arctan2(d[:, 2], np.linalg.norm(d[:, :2], axis=1))
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rot.from_euler("ZY", np.stack([yaw, pitch], 1)).as_matrix()
    T[:, :3, 3] = p
    return T


def drifting_odometry(
    T_true: np.ndarray, rng, rot_drift_deg: float = 0.05
) -> np.ndarray:
    """Odometry that integrates the true relative motion with a random-walk attitude error,
    expressed in a frame offset from the world (to check the lift does not need it aligned)."""
    n = len(T_true)
    walk = np.cumsum(rng.normal(0, np.radians(rot_drift_deg), (n, 3)) * [1, 1, 0.2], 0)
    T_odom = np.empty_like(T_true)
    T_odom[0] = np.eye(4)
    T_odom[0][:3, :3] = Rot.from_euler("z", 0.7).as_matrix()
    T_odom[0][:3, 3] = [5.0, -3.0, 1.0]
    T_odom[0] = T_odom[0] @ T_true[0]
    for k in range(1, n):
        step = np.linalg.inv(T_true[k - 1]) @ T_true[k]
        step[:3, :3] = step[:3, :3] @ Rot.from_rotvec(walk[k] - walk[k - 1]).as_matrix()
        T_odom[k] = T_odom[k - 1] @ step
    return T_odom


def solution_2d(T_true: np.ndarray) -> list:
    """The 2D solution lift_to_3d starts from: true x, y, yaw."""
    yaw = np.arctan2(T_true[:, 1, 0], T_true[:, 0, 0])
    T2 = np.tile(np.eye(4), (len(T_true), 1, 1))
    T2[:, :3, :3] = Rot.from_euler("z", yaw[:, None]).as_matrix()
    T2[:, :2, 3] = T_true[:, :2, 3]
    return list(T2)


def roll_pitch_err_deg(T: np.ndarray, T_true: np.ndarray) -> np.ndarray:
    dR = np.swapaxes(T_true[:, :3, :3], 1, 2) @ T[:, :3, :3]
    return np.degrees(np.abs(Rot.from_matrix(dR).as_euler("xyz")[:, :2]))


def heights_only_lift(rpgo: CrossViewRPGO, T2: list, T_odom: np.ndarray):
    """lift_to_3d's initial guess (before _refine_3d)."""
    rpgo.params.lift_3d_refine_rotation = False
    lifted = np.array(rpgo.lift_to_3d(T2, list(T_odom), None))
    rpgo.params.lift_3d_refine_rotation = True
    p = rpgo.params
    xy = np.array(T2)[:, :2, 3]
    pairs = cross_path_pairs(
        xy, p.cross_paths_3d_constraint_dist_m, p.cross_paths_3d_constraint_path_dist_m
    )
    return lifted, xy, pairs


def refine_gtsam(p: CrossViewRPGOParams, lifted, T_odom, xy, pairs) -> np.ndarray:
    """The _refine_3d factor graph, solved with GTSAM (reference implementation)."""
    n = len(lifted)
    graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
    for k in range(n):
        values.insert(k, gtsam.Pose3(lifted[k]))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.r_[[p.odom_rot_sigma_rad] * 3, [p.odom_trans_sigma_m] * 3]
    )
    for k in range(n - 1):
        T_rel = np.linalg.inv(T_odom[k]) @ T_odom[k + 1]
        graph.add(gtsam.BetweenFactorPose3(k, k + 1, gtsam.Pose3(T_rel), odom_noise))
    yaw_prior = np.arctan2(lifted[:, 1, 0], lifted[:, 0, 0])

    def xy_factor(k):
        def f(this, v, H):
            P = v.atPose3(this.keys()[0])
            if H is not None:
                H[0] = np.hstack([np.zeros((2, 3)), P.rotation().matrix()[:2, :]])
            return P.translation()[:2] - xy[k]

        return f

    def yaw_factor(k):
        def f(this, v, H):
            R = v.atPose3(this.keys()[0]).rotation().matrix()
            if H is not None:
                d = R[0, 0] ** 2 + R[1, 0] ** 2
                J = np.zeros((1, 6))
                J[0, 1] = (R[1, 0] * R[0, 2] - R[0, 0] * R[1, 2]) / d
                J[0, 2] = (R[0, 0] * R[1, 1] - R[1, 0] * R[0, 1]) / d
                H[0] = J
            e = np.arctan2(R[1, 0], R[0, 0]) - yaw_prior[k]
            return np.array([(e + np.pi) % (2 * np.pi) - np.pi])

        return f

    def height_factor(this, v, H):
        Pi, Pj = v.atPose3(this.keys()[0]), v.atPose3(this.keys()[1])
        if H is not None:
            H[0] = np.hstack([np.zeros((1, 3)), Pi.rotation().matrix()[2:3, :]])
            H[1] = np.hstack([np.zeros((1, 3)), -Pj.rotation().matrix()[2:3, :]])
        return np.array([Pi.translation()[2] - Pj.translation()[2]])

    R0, z0 = lifted[0, :3, :3].copy(), lifted[0, 2, 3]

    def anchor_factor(this, v, H):
        P = v.atPose3(0)
        R = P.rotation().matrix()
        if H is not None:
            J = np.zeros((3, 6))
            J[0, 0] = J[1, 1] = 1.0
            J[2, 3:] = R[2, :]
            H[0] = J
        return np.r_[Rot.from_matrix(R0.T @ R).as_rotvec()[:2], P.translation()[2] - z0]

    iso = gtsam.noiseModel.Isotropic.Sigma
    for k in range(n):
        graph.add(
            gtsam.CustomFactor(iso(2, p.lift_3d_prior_xy_sigma_m), [k], xy_factor(k))
        )
        graph.add(
            gtsam.CustomFactor(
                iso(1, p.lift_3d_prior_yaw_sigma_rad), [k], yaw_factor(k)
            )
        )
    for i, j in pairs:
        graph.add(
            gtsam.CustomFactor(
                iso(1, p.cross_paths_3d_constraint_sigma_m),
                [int(i), int(j)],
                height_factor,
            )
        )
    graph.add(gtsam.CustomFactor(iso(3, _LIFT_ANCHOR_SIGMA), [0], anchor_factor))
    lm = gtsam.LevenbergMarquardtParams()
    lm.setMaxIterations(50)
    result = gtsam.LevenbergMarquardtOptimizer(graph, values, lm).optimize()
    return np.array([result.atPose3(k).matrix() for k in range(n)])


@pytest.fixture
def rpgo():
    return CrossViewRPGO(CrossViewRPGOParams())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_refine_matches_gtsam(rpgo, seed):
    rng = np.random.default_rng(seed)
    T_true = figure_eight(400)
    T_odom = drifting_odometry(T_true, rng)
    lifted, xy, pairs = heights_only_lift(rpgo, solution_2d(T_true), T_odom)
    assert len(pairs) > 0

    ours = rpgo._refine_3d(lifted.copy(), T_odom, xy, pairs)
    ref = refine_gtsam(rpgo.params, lifted.copy(), T_odom, xy, pairs)

    np.testing.assert_allclose(ours[:, :3, 3], ref[:, :3, 3], atol=5e-3)
    dR = np.swapaxes(ref[:, :3, :3], 1, 2) @ ours[:, :3, :3]
    assert (
        np.degrees(np.linalg.norm(Rot.from_matrix(dR).as_rotvec(), axis=1)).max() < 5e-3
    )


def test_exact_inputs_are_nearly_a_fixed_point(rpgo):
    """Drift-free odometry and an exact 2D solution should come back (almost) unchanged.

    Not exact: cross-path pairs up to cross_paths_3d_constraint_dist_m apart are pulled to
    equal height, and on this sloped surface their true heights differ by a few cm.
    """
    T_true = figure_eight(300)
    T3 = np.array(rpgo.lift_to_3d(solution_2d(T_true), list(T_true), None))
    np.testing.assert_allclose(T3[:, :2, 3], T_true[:, :2, 3], atol=1e-4)
    np.testing.assert_allclose(T3[:, 2, 3], T_true[:, 2, 3], atol=0.06)
    np.testing.assert_allclose(T3[:, :3, :3], T_true[:, :3, :3], atol=1e-3)


def test_refine_corrects_attitude_and_height_drift(rpgo):
    # Three laps, so every point is revisited: roll/pitch drift is only observable through
    # the cross-path height constraints.
    rng = np.random.default_rng(3)
    T_true = np.concatenate([figure_eight(300)] * 3)
    T_odom = drifting_odometry(T_true, rng, rot_drift_deg=0.1)
    T2 = solution_2d(T_true)

    lifted, *_ = heights_only_lift(rpgo, T2, T_odom)
    T3 = np.array(rpgo.lift_to_3d(T2, list(T_odom), None))

    # first node is the anchor, so compare heights relative to it
    def dz(T):
        return np.abs((T[:, 2, 3] - T[0, 2, 3]) - (T_true[:, 2, 3] - T_true[0, 2, 3]))

    rp_before, rp_after = (
        roll_pitch_err_deg(lifted, T_true),
        roll_pitch_err_deg(T3, T_true),
    )
    assert np.median(rp_after) < 0.6 * np.median(rp_before)
    assert rp_after.max() < 0.7 * rp_before.max()
    assert dz(T3).max() < 0.6 * dz(lifted).max()
    # x, y, yaw still follow the 2D solution
    np.testing.assert_allclose(T3[:, :2, 3], T_true[:, :2, 3], atol=0.05)
    yaw_err = np.arctan2(T3[:, 1, 0], T3[:, 0, 0]) - np.arctan2(
        T_true[:, 1, 0], T_true[:, 0, 0]
    )
    assert np.degrees(np.abs((yaw_err + np.pi) % (2 * np.pi) - np.pi)).max() < 0.5


def test_yaw_wrap(rpgo):
    """Headings crossing +/-pi must not produce 2*pi yaw-prior residuals."""
    rng = np.random.default_rng(4)
    T_true = figure_eight(400, yaw_offset=np.pi)
    T_odom = drifting_odometry(T_true, rng)
    T3 = np.array(rpgo.lift_to_3d(solution_2d(T_true), list(T_odom), None))
    yaw_err = np.arctan2(T3[:, 1, 0], T3[:, 0, 0]) - np.arctan2(
        T_true[:, 1, 0], T_true[:, 0, 0]
    )
    assert np.degrees(np.abs((yaw_err + np.pi) % (2 * np.pi) - np.pi)).max() < 0.5


def test_no_cross_path_pairs(rpgo):
    """A path that never revisits: only the anchor fixes the gauge; result must be finite and
    keep the 2D x, y."""
    rng = np.random.default_rng(5)
    n = 200
    T_true = np.tile(np.eye(4), (n, 1, 1))
    T_true[:, 0, 3] = np.linspace(0, 40, n)
    T_true[:, 2, 3] = 0.1 * T_true[:, 0, 3]
    T_odom = drifting_odometry(T_true, rng)
    lifted, xy, pairs = heights_only_lift(rpgo, solution_2d(T_true), T_odom)
    assert len(pairs) == 0
    T3 = rpgo._refine_3d(lifted, T_odom, xy, pairs)
    assert np.all(np.isfinite(T3))
    np.testing.assert_allclose(T3[:, :2, 3], xy, atol=0.05)


@pytest.mark.parametrize("n", [1, 2])
def test_tiny_trajectories(rpgo, n):
    T_true = figure_eight(50)[:n]
    T3 = np.array(rpgo.lift_to_3d(solution_2d(T_true), list(T_true), None))
    assert T3.shape == (n, 4, 4)
    np.testing.assert_allclose(T3[:, :3, 3], T_true[:, :3, 3], atol=1e-6)
