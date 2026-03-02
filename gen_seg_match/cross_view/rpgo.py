import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import clipperpy
import gtsam
import numpy as np

from gen_seg_match.params.cross_view_params import CrossViewRPGOParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SE(2) helpers
# ---------------------------------------------------------------------------


def se3_to_se2(T: np.ndarray) -> np.ndarray:
    """Project a 4x4 SE(3) matrix to a 3x3 SE(2) matrix (x, y, yaw)."""
    yaw = np.arctan2(T[1, 0], T[0, 0])
    return se2_from_xytheta(T[0, 3], T[1, 3], yaw)


def se2_to_se3(T2: np.ndarray) -> np.ndarray:
    """Embed a 3x3 SE(2) matrix into a 4x4 SE(3) matrix (z=0, roll=pitch=0)."""
    T = np.eye(4)
    T[:2, :2] = T2[:2, :2]
    T[0, 3] = T2[0, 2]
    T[1, 3] = T2[1, 2]
    return T


def se2_from_xytheta(x: float, y: float, theta: float) -> np.ndarray:
    """Construct a 3x3 SE(2) matrix from x, y, yaw."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, x],
            [s, c, y],
            [0, 0, 1],
        ]
    )


def yaw_from_se2(T2: np.ndarray) -> float:
    """Extract yaw angle from a 3x3 SE(2) matrix."""
    return np.arctan2(T2[1, 0], T2[0, 0])


def average_se2(transforms: List[np.ndarray]) -> np.ndarray:
    """Average a list of SE(2) transforms (mean x,y; circular mean yaw)."""
    xs = [T[0, 2] for T in transforms]
    ys = [T[1, 2] for T in transforms]
    yaws = [yaw_from_se2(T) for T in transforms]
    mean_yaw = np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws)))
    return se2_from_xytheta(np.mean(xs), np.mean(ys), mean_yaw)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CrossViewRPGOResult:
    success: bool
    T_utm_odom: Optional[np.ndarray] = None
    optimized_trajectory: Optional[List[np.ndarray]] = None
    inlier_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    M: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    candidates: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CrossViewRPGO
# ---------------------------------------------------------------------------

# TODO: candidates right now are just a list of dicts - should define a dataclass for them


@dataclass
class CrossViewRPGO:
    params: CrossViewRPGOParams

    def solve(
        self,
        candidates: List[dict],
        trajectory: List[np.ndarray],
        times: np.ndarray,
        T_camera_flu: Optional[np.ndarray] = None,
    ) -> CrossViewRPGOResult:
        """Main entry: CLIPPER outlier rejection then frame_align or PGO.

        Args:
            candidates: List of candidate dicts from _load_candidates.
            trajectory: List of 4x4 T_odom_camera poses.
            times: Array of timestamps corresponding to trajectory.
            T_camera_flu: Optional 4x4 transform from camera to FLU body frame.

        Returns:
            CrossViewRPGOResult with T_utm_odom and optimized_trajectory.
        """
        M, C = self.build_affinity_matrix(candidates)
        inlier_indices = self.run_clipper(M, C)

        if len(inlier_indices) == 0:
            logger.warning("CLIPPER returned empty solution.")
            return CrossViewRPGOResult(success=False, M=M, C=C, candidates=candidates)

        if len(inlier_indices) == 1:
            logger.warning("Only a single inlier — using it directly.")

        T_utm_odom = self._frame_align(candidates, inlier_indices)

        if self.params.optimization_method == "pgo":
            optimized_trajectory = self._pgo(
                candidates,
                inlier_indices,
                trajectory,
                times,
                T_camera_flu,
                T_utm_odom,
            )
        else:
            optimized_trajectory = self._apply_rigid_transform(
                T_utm_odom, trajectory, T_camera_flu
            )

        return CrossViewRPGOResult(
            success=True,
            T_utm_odom=T_utm_odom,
            optimized_trajectory=optimized_trajectory,
            inlier_indices=inlier_indices,
            M=M,
            C=C,
            candidates=candidates,
        )

    def build_affinity_matrix(
        self, candidates: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build CLIPPER affinity (M) and constraint (C) matrices."""
        N = len(candidates)
        M = np.zeros((N, N))
        C = np.ones((N, N))

        sigma_r = self.params.rot_consistency_sigma_rad
        sigma_t = self.params.trans_consistency_sigma_m
        eps_r = self.params.rot_consistency_eps_rad
        eps_t = self.params.trans_consistency_eps_m

        for i in range(N):
            M[i, i] = 1.0
            ci = candidates[i]
            for j in range(i + 1, N):
                cj = candidates[j]

                # Relative transform via odometry
                odom_relative = se3_to_se2(
                    np.linalg.inv(ci["T_odom_ground"]) @ cj["T_odom_ground"]
                )

                # Relative transform via cross-view
                cv_relative = se3_to_se2(
                    np.linalg.inv(ci["T_i_j_hat"])
                    @ np.linalg.inv(ci["aerial_pose"])
                    @ cj["aerial_pose"]
                    @ cj["T_i_j_hat"]
                )

                # Error: should be identity if consistent
                error = np.linalg.inv(odom_relative) @ cv_relative
                rot_err = abs(yaw_from_se2(error))
                trans_err = np.linalg.norm(error[:2, 2])

                if rot_err < eps_r and trans_err < eps_t:
                    score = np.exp(-0.5 * (rot_err / sigma_r) ** 2) * np.exp(
                        -0.5 * (trans_err / sigma_t) ** 2
                    )
                    M[i, j] = score
                    M[j, i] = score
                else:
                    C[i, j] = 0
                    C[j, i] = 0

        return M, C

    @staticmethod
    def run_clipper(M: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Run CLIPPER on the affinity/constraint matrices."""
        clipper = clipperpy.CLIPPER(
            clipperpy.invariants.PairwiseInvariant(), clipperpy.Params()
        )
        clipper.set_matrix_data(M=M, C=C)
        clipper.solve()
        return np.array(clipper.get_solution().nodes)

    @staticmethod
    def _frame_align(candidates: List[dict], inlier_indices: np.ndarray) -> np.ndarray:
        """Average inlier T_utm_odom estimates and return as 4x4 SE(3)."""
        inlier_se2s = [candidates[i]["T_utm_odom_se2"] for i in inlier_indices]
        T_avg_se2 = average_se2(inlier_se2s)
        return se2_to_se3(T_avg_se2)

    @staticmethod
    def _apply_rigid_transform(
        T_utm_odom: np.ndarray,
        trajectory: List[np.ndarray],
        T_camera_flu: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        """Apply single rigid T_utm_odom to full trajectory.

        Returns list of 4x4 T_utm_body for each timestep.
        """
        result = []
        for T_odom_cam in trajectory:
            if T_camera_flu is not None:
                T_utm_body = T_utm_odom @ T_odom_cam @ T_camera_flu
            else:
                T_utm_body = T_utm_odom @ T_odom_cam
            result.append(T_utm_body)
        return result

    def _pgo(
        self,
        candidates: List[dict],
        inlier_indices: np.ndarray,
        trajectory: List[np.ndarray],
        times: np.ndarray,
        T_camera_flu: Optional[np.ndarray],
        T_utm_odom_init: np.ndarray,
    ) -> List[np.ndarray]:
        """GTSAM SE(2) pose graph optimization.

        Variables: one Pose2 per trajectory timestep representing T_utm_body.
        Factors:
          - BetweenFactorPose2 for odometry between consecutive poses.
          - PriorFactorPose2 for each CLIPPER inlier at the closest timestep.

        Returns list of 4x4 T_utm_body for each timestep.
        """
        n = len(trajectory)
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        # Noise models
        odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [
                    self.params.odom_trans_sigma_m,
                    self.params.odom_trans_sigma_m,
                    self.params.odom_rot_sigma_rad,
                ]
            )
        )
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [
                    self.params.prior_trans_sigma_m,
                    self.params.prior_trans_sigma_m,
                    self.params.prior_rot_sigma_rad,
                ]
            )
        )

        # Compute T_odom_body for each timestep
        T_odom_body_list = []
        for T_odom_cam in trajectory:
            if T_camera_flu is not None:
                T_odom_body_list.append(T_odom_cam @ T_camera_flu)
            else:
                T_odom_body_list.append(T_odom_cam)

        # Initial values from frame-align applied to trajectory
        for i in range(n):
            T_utm_body_init = T_utm_odom_init @ T_odom_body_list[i]
            se2 = se3_to_se2(T_utm_body_init)
            pose2 = gtsam.Pose2(se2[0, 2], se2[1, 2], yaw_from_se2(se2))
            initial.insert(i, pose2)

        # Odometry between factors
        for i in range(n - 1):
            T_rel = np.linalg.inv(T_odom_body_list[i]) @ T_odom_body_list[i + 1]
            rel_se2 = se3_to_se2(T_rel)
            odom_pose2 = gtsam.Pose2(
                rel_se2[0, 2], rel_se2[1, 2], yaw_from_se2(rel_se2)
            )
            graph.add(gtsam.BetweenFactorPose2(i, i + 1, odom_pose2, odom_noise))

        # Cross-view prior factors
        times_arr = np.asarray(times)
        for idx in inlier_indices:
            c = candidates[idx]
            ground_time = c["ground_submap_time"]
            traj_idx = int(np.argmin(np.abs(times_arr - ground_time)))

            # Prior: T_utm_body at this timestep from this candidate
            T_utm_odom_c = se2_to_se3(c["T_utm_odom_se2"])
            T_utm_body_c = T_utm_odom_c @ T_odom_body_list[traj_idx]
            se2_prior = se3_to_se2(T_utm_body_c)
            prior_pose2 = gtsam.Pose2(
                se2_prior[0, 2], se2_prior[1, 2], yaw_from_se2(se2_prior)
            )
            graph.add(gtsam.PriorFactorPose2(traj_idx, prior_pose2, prior_noise))

        # Optimize
        lm_params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, lm_params)
        result = optimizer.optimize()

        # Extract optimized trajectory as 4x4 SE(3)
        optimized = []
        for i in range(n):
            pose2 = result.atPose2(i)
            T_utm_body = se2_to_se3(
                se2_from_xytheta(pose2.x(), pose2.y(), pose2.theta())
            )
            optimized.append(T_utm_body)

        return optimized
