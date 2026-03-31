import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import clipperpy
import gtsam
import numpy as np
from scipy.spatial.transform import Rotation as Rot

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


def pose_data_from_trajectory(trajectory: List[np.ndarray], times: np.ndarray):
    """Create a PoseData from a list of 4x4 T_utm_body poses and timestamps.

    Returns a PoseData with interpolation enabled and infinite time tolerance
    so it can be queried at any time.
    """
    from robotdatapy.data import PoseData

    positions = np.array([T[:3, 3] for T in trajectory])
    quats = Rot.from_matrix([T[:3, :3] for T in trajectory]).as_quat()  # xyzw
    return PoseData(
        times=times,
        positions=positions,
        orientations=quats,
        interp=True,
        time_tol=np.inf,
    )


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
# CLIPPER data builder
# ---------------------------------------------------------------------------


def _build_clipper_data(
    candidates: List[dict],
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Build CLIPPER data matrix and unique pose lists from candidates.

    Each column of D encodes one candidate:
      D[0, k]     = aerial_idx  (int index into aerial_poses list)
      D[1, k]     = ground_idx  (int index into ground_poses list)
      D[2:18, k]  = T_i_j_hat.flatten() row-major (16 doubles)

    Returns:
        D: 18×N data matrix
        aerial_poses: list of unique 4×4 aerial pose matrices
        ground_poses: list of unique 4×4 ground pose matrices
    """
    aerial_key_to_idx: dict = {}
    aerial_poses: List[np.ndarray] = []
    ground_key_to_idx: dict = {}
    ground_poses: List[np.ndarray] = []

    N = len(candidates)
    # Build D as (N, 18) row-major, then transpose to (18, N) Fortran-order
    # so that Eigen receives a column-major matrix (each column = one datum).
    D_rows = np.zeros((N, 18))

    for k, c in enumerate(candidates):
        # Deduplicate aerial poses by matrix content
        aerial_key = tuple(c["aerial_pose"].flatten())
        if aerial_key not in aerial_key_to_idx:
            aerial_key_to_idx[aerial_key] = len(aerial_poses)
            aerial_poses.append(c["aerial_pose"].copy())
        aerial_idx = aerial_key_to_idx[aerial_key]

        # Deduplicate ground poses by matrix content
        ground_key = tuple(c["T_odom_ground"].flatten())
        if ground_key not in ground_key_to_idx:
            ground_key_to_idx[ground_key] = len(ground_poses)
            ground_poses.append(c["T_odom_ground"].copy())
        ground_idx = ground_key_to_idx[ground_key]

        D_rows[k, 0] = aerial_idx
        D_rows[k, 1] = ground_idx
        D_rows[k, 2:] = c["T_i_j_hat"].flatten()  # row-major (C order)

    # Transpose to (18, N) Fortran-order — pybind11 passes this to Eigen as
    # a column-major MatrixXd where each column is one datum.
    D = D_rows.T  # shape (18, N), Fortran-order (C-contiguous rows → F-contiguous cols)
    return D, aerial_poses, ground_poses


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
        if self.params.gt_inliers:
            inlier_indices = self._gt_inlier_selection(candidates)
            M, C = None, None
        else:
            inlier_indices, M, C = self.run_clipper_cpp(candidates)

        if len(inlier_indices) == 0:
            logger.warning(
                "GT inlier selection returned empty solution."
                if self.params.gt_inliers
                else "CLIPPER returned empty solution."
            )
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

    def _gt_inlier_selection(self, candidates: List[dict]) -> np.ndarray:
        """Select inliers by comparing each candidate's T_i_j_hat to GT T_i_j.

        Uses the same error metrics as matching visualization (direct pose
        comparison in the aerial-ground frame) to avoid lever-arm effects
        from comparing in the UTM-odom frame.
        """
        rot_thresh = np.deg2rad(self.params.gt_inliers_rot_err_deg)
        trans_thresh = self.params.gt_inliers_trans_err_m
        inlier_indices = []
        for i, c in enumerate(candidates):
            T_i_j = c.get("T_i_j")
            T_i_j_hat = c.get("T_i_j_hat")
            if T_i_j is None or np.any(np.isnan(T_i_j)):
                continue
            trans_err = np.linalg.norm((T_i_j - T_i_j_hat)[:3, 3])
            T_error = np.linalg.inv(T_i_j_hat) @ T_i_j
            rot_err = Rot.from_matrix(T_error[:3, :3]).magnitude()
            if rot_err < rot_thresh and trans_err < trans_thresh:
                inlier_indices.append(i)
        logger.info(
            f"GT inlier selection: {len(inlier_indices)}/{len(candidates)} candidates "
            f"within {self.params.gt_inliers_rot_err_deg}° / {self.params.gt_inliers_trans_err_m}m"
        )
        return np.array(inlier_indices, dtype=int)

    def _build_affinity_matrix_python(
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

    def run_clipper_cpp(
        self, candidates: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run CLIPPER using the C++ LoopClosureConsistency invariant.

        Returns:
            Tuple of (inlier_indices, M, C).
        """
        D, aerial_poses, ground_poses = _build_clipper_data(candidates)
        iparams = clipperpy.invariants.LoopClosureConsistencyParams()
        iparams.rot_sigma_rad = self.params.rot_consistency_sigma_rad
        iparams.rot_eps_rad = self.params.rot_consistency_eps_rad
        iparams.trans_sigma_m = self.params.trans_consistency_sigma_m
        iparams.trans_eps_m = self.params.trans_consistency_eps_m
        invariant = clipperpy.invariants.LoopClosureConsistency(
            aerial_poses, ground_poses, iparams
        )
        clipper = clipperpy.CLIPPERPairwiseAndSingle(invariant, clipperpy.Params())
        N = D.shape[1]
        A = np.stack([np.arange(N), np.arange(N)], axis=1).astype(np.int32)
        clipper.score_pairwise_and_single_consistency(D, D, A)
        clipper.solve()
        inlier_indices = np.array(clipper.get_solution().nodes)
        M = clipper.get_affinity_matrix()
        C = clipper.get_constraint_matrix()
        return inlier_indices, M, C

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

            # Prior: T_utm_body directly from registration (avoids SE2 lever arm)
            se2_prior = c["T_utm_body_se2"]
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
