import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import clipperpy
import gtsam
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from robotdatapy.data import PoseData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from meridian.params.cross_view_params import CrossViewRPGOParams

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
# 3D lift and densification helpers
# ---------------------------------------------------------------------------

# Anchor sigma (rad / m) fixing the first node's roll, pitch and height
_LIFT_ANCHOR_SIGMA = 1e-3


def cross_path_pairs(
    xy: np.ndarray, dist_m: float, min_path_dist_m: float
) -> np.ndarray:
    """Index pairs (i, j), i < j, of trajectory nodes within dist_m of each other
    in 2D that are at least min_path_dist_m apart along the path (revisits, not
    neighbours along the same pass). Each node j keeps only its nearest such
    earlier node i, so there are at most len(xy) pairs. Returns shape (k, 2)."""
    if len(xy) < 2:
        return np.zeros((0, 2), dtype=int)
    pairs = cKDTree(xy).query_pairs(dist_m, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=int)
    path_s = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    )
    pairs = np.sort(pairs, axis=1)
    pairs = pairs[path_s[pairs[:, 1]] - path_s[pairs[:, 0]] >= min_path_dist_m]
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=int)
    dist = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
    pairs = pairs[np.argsort(dist, kind="stable")]
    _, first = np.unique(pairs[:, 1], return_index=True)
    return pairs[np.sort(first)]


def densify_trajectory(
    node_times: np.ndarray,
    node_T_utm_body: List[np.ndarray],
    node_T_odom_cam: List[np.ndarray],
    dense_times: np.ndarray,
    dense_T_odom_cam: np.ndarray,
    T_camera_flu: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dense T_utm_body for high-rate odometry, from an optimized node trajectory.

    At each node the correction C = T_utm_body @ inv(T_odom_body) is known. It
    is interpolated between nodes (lerp translation, slerp rotation; held
    constant outside the node time range) and applied to the dense odometry:
    T_utm_body(t) = C(t) @ T_odom_cam(t) @ T_camera_flu. Exact at node times;
    between nodes it keeps the odometry's high-rate motion.

    Args:
        node_times: (n,) times of the optimized nodes.
        node_T_utm_body: n optimized 4x4 T_utm_body poses.
        node_T_odom_cam: n 4x4 T_odom_cam odometry poses at the nodes.
        dense_times: (m,) times of the dense odometry.
        dense_T_odom_cam: (m, 4, 4) dense T_odom_cam odometry poses.
        T_camera_flu: Optional 4x4 transform from camera to FLU body frame.

    Returns:
        (m, 4, 4) dense T_utm_body poses.
    """
    T_cb = np.eye(4) if T_camera_flu is None else T_camera_flu
    node_times = np.asarray(node_times)
    node_T_odom_body = np.asarray(node_T_odom_cam) @ T_cb
    C = np.asarray(node_T_utm_body) @ np.linalg.inv(node_T_odom_body)

    order = np.argsort(node_times)
    node_times, C = node_times[order], C[order]
    C_dense = np.tile(np.eye(4), (len(dense_times), 1, 1))
    for k in range(3):
        C_dense[:, k, 3] = np.interp(dense_times, node_times, C[:, k, 3])
    if len(node_times) > 1:
        slerp = Slerp(node_times, Rot.from_matrix(C[:, :3, :3]))
        C_dense[:, :3, :3] = slerp(
            np.clip(dense_times, node_times[0], node_times[-1])
        ).as_matrix()
    else:
        C_dense[:, :3, :3] = C[0, :3, :3]
    return C_dense @ np.asarray(dense_T_odom_cam) @ T_cb


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CrossViewRPGOResult:
    success: bool
    T_utm_odom: Optional[np.ndarray] = None
    optimized_trajectory: Optional[List[np.ndarray]] = None
    # optimized_trajectory lifted to 3D (heights, roll/pitch); see lift_to_3d
    optimized_trajectory_3d: Optional[List[np.ndarray]] = None
    times: Optional[np.ndarray] = None
    inlier_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    M: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None
    # Outlier-rejection objective u^T M u / u^T u where u is the binary
    # inlier indicator. None when CLIPPER didn't run (gt_inliers path) or
    # there were no inliers.
    objective_value: Optional[float] = None
    candidates: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CLIPPER data builder
# ---------------------------------------------------------------------------


def _compute_cumulative_path_lengths(
    trajectory: List[np.ndarray],
) -> np.ndarray:
    """Compute cumulative path length at each trajectory pose.

    Returns an array of length len(trajectory) where entry i is the total
    distance traveled from pose 0 to pose i.
    """
    cum = np.zeros(len(trajectory))
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0, 3] - trajectory[i - 1][0, 3]
        dy = trajectory[i][1, 3] - trajectory[i - 1][1, 3]
        dz = trajectory[i][2, 3] - trajectory[i - 1][2, 3]
        cum[i] = cum[i - 1] + np.sqrt(dx * dx + dy * dy + dz * dz)
    return cum


def _build_clipper_data(
    candidates: List[dict],
    trajectory: List[np.ndarray],
    times: np.ndarray,
    lc_scores: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[float]]:
    """Build CLIPPER data matrix and unique pose lists from candidates.

    Each column of D encodes one candidate:
      D[0, k]     = aerial_idx  (int index into aerial_poses list)
      D[1, k]     = ground_idx  (int index into ground_poses list)
      D[2:18, k]  = T_i_j_hat.flatten() row-major (16 doubles)
      D[18, k]    = lc_score in [0, 1] (per-LC quality)

    Returns:
        D: 19×N data matrix
        aerial_poses: list of unique 4×4 aerial pose matrices
        ground_poses: list of unique 4×4 ground pose matrices
        ground_distances: list of cumulative path lengths for each unique ground pose
    """
    cum_path = _compute_cumulative_path_lengths(trajectory)
    times_arr = np.asarray(times)

    aerial_key_to_idx: dict = {}
    aerial_poses: List[np.ndarray] = []
    ground_key_to_idx: dict = {}
    ground_poses: List[np.ndarray] = []
    ground_distances: List[float] = []

    N = len(candidates)
    # Build D as (N, 19) row-major, then transpose to (19, N) Fortran-order
    # so that Eigen receives a column-major matrix (each column = one datum).
    D_rows = np.zeros((N, 19))

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
            # Look up cumulative path length for this ground pose
            ground_time = c["ground_submap_time"]
            traj_idx = int(np.argmin(np.abs(times_arr - ground_time)))
            ground_distances.append(float(cum_path[traj_idx]))
        ground_idx = ground_key_to_idx[ground_key]

        D_rows[k, 0] = aerial_idx
        D_rows[k, 1] = ground_idx
        D_rows[k, 2:18] = c["T_i_j_hat"].flatten()  # row-major (C order)
        D_rows[k, 18] = lc_scores[k]

    # Transpose to (19, N) Fortran-order — pybind11 passes this to Eigen as
    # a column-major MatrixXd where each column is one datum.
    D = D_rows.T  # shape (19, N), Fortran-order (C-contiguous rows → F-contiguous cols)
    return D, aerial_poses, ground_poses, ground_distances


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
            M, C, objective = None, None, None
        else:
            inlier_indices, M, C, objective = self.run_clipper_cpp(
                candidates, trajectory, times
            )

        if len(inlier_indices) == 0:
            logger.warning(
                "GT inlier selection returned empty solution."
                if self.params.gt_inliers
                else "CLIPPER returned empty solution."
            )
            return CrossViewRPGOResult(
                success=False,
                M=M,
                C=C,
                objective_value=objective,
                candidates=candidates,
            )

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

        optimized_trajectory_3d = self.lift_to_3d(
            optimized_trajectory, trajectory, T_camera_flu
        )

        return CrossViewRPGOResult(
            success=True,
            T_utm_odom=T_utm_odom,
            optimized_trajectory=optimized_trajectory,
            optimized_trajectory_3d=optimized_trajectory_3d,
            times=np.asarray(times),
            inlier_indices=inlier_indices,
            M=M,
            C=C,
            objective_value=objective,
            candidates=candidates,
        )

    def solve_clipper_only(
        self,
        candidates: List[dict],
        trajectory: List[np.ndarray],
        times: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """Run only the outlier-rejection step (CLIPPER or GT) without PGO.

        Returns (inlier_indices, M, C, objective). M, C, objective are None when
        gt_inliers is set.
        """
        if self.params.gt_inliers:
            return self._gt_inlier_selection(candidates), None, None, None
        return self.run_clipper_cpp(candidates, trajectory, times)

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
            # Compare only yaw — both transforms live in SE(2) but may
            # carry residual pitch/roll from 3D camera extrinsics.
            yaw_gt = float(np.arctan2(T_i_j[1, 0], T_i_j[0, 0]))
            yaw_hat = float(np.arctan2(T_i_j_hat[1, 0], T_i_j_hat[0, 0]))
            rot_err = abs(
                np.arctan2(np.sin(yaw_gt - yaw_hat), np.cos(yaw_gt - yaw_hat))
            )
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

    def _compute_lc_scores(self, candidates: List[dict]) -> np.ndarray:
        """Per-LC quality score in [0, 1].

        Method "frequency-ratio" normalizes each candidate's particle count by
        the max count among candidates that share the same (ground_key,
        aerial_key) pair: top hypothesis per pair is 1.0, runners-up scale down.
        """
        method = self.params.lc_score_method
        if method == "frequency-ratio":
            pair_max: dict = {}
            for c in candidates:
                key = (c["ground_key"], c["aerial_key"])
                pair_max[key] = max(pair_max.get(key, 0), c.get("count", 1))
            return np.array(
                [
                    c.get("count", 1)
                    / max(pair_max[(c["ground_key"], c["aerial_key"])], 1)
                    for c in candidates
                ],
                dtype=np.float64,
            )
        raise ValueError(f"Unknown lc_score_method: {method}")

    def run_clipper_cpp(
        self,
        candidates: List[dict],
        trajectory: List[np.ndarray],
        times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
        """Run CLIPPER using the C++ LoopClosureConsistency invariant.

        When `len(candidates) > params.outlier_rejection_max_num_lcs` (and the
        threshold is set / positive), the candidate list is split into the
        smallest number of chunks such that each chunk has fewer candidates than
        the threshold; CLIPPER runs on each chunk independently, the inliers are
        pooled, and a final CLIPPER pass runs on the pool. This trades a small
        amount of recall for a memory-bounded run.

        Returns:
            Tuple of (inlier_indices, M, C, objective). `inlier_indices` indexes
            into the original `candidates` list. In the chunked path, M and C
            come from the final pool-only pass, so their dimensions equal the
            pool size. `objective` is the binary-indicator objective
            u^T M u / u^T u for the (final) pass.
        """
        n = len(candidates)
        max_n = self.params.outlier_rejection_max_num_lcs
        if max_n is None or max_n <= 0 or n <= max_n:
            return self._run_clipper_cpp_single(candidates, trajectory, times)

        n_chunks = (n // max_n) + 1
        logger.info(
            f"Chunked outlier rejection: {n} candidates > max={max_n}, "
            f"splitting into {n_chunks} chunks."
        )

        # Even-as-possible split into n_chunks groups of indices.
        chunks = np.array_split(np.arange(n), n_chunks)

        pooled_orig_idx: List[int] = []
        for ci, chunk_idx in enumerate(chunks):
            chunk_candidates = [candidates[i] for i in chunk_idx]
            sub_inliers, *_ = self._run_clipper_cpp_single(
                chunk_candidates, trajectory, times
            )
            for j in sub_inliers:
                pooled_orig_idx.append(int(chunk_idx[j]))
            logger.info(
                f"  chunk {ci + 1}/{n_chunks}: {len(chunk_candidates)} candidates "
                f"-> {len(sub_inliers)} inliers"
            )

        # Final pass over the pooled inliers.
        pooled_candidates = [candidates[i] for i in pooled_orig_idx]
        final_sub_inliers, M, C, objective = self._run_clipper_cpp_single(
            pooled_candidates, trajectory, times
        )
        final_orig_idx = np.array(
            [pooled_orig_idx[j] for j in final_sub_inliers], dtype=np.int64
        )
        logger.info(
            f"  final pass: {len(pooled_candidates)} pooled candidates "
            f"-> {len(final_orig_idx)} inliers"
        )
        return final_orig_idx, M, C, objective

    @staticmethod
    def _binary_objective(M, inlier_indices) -> Optional[float]:
        """u^T M u / u^T u with u the binary indicator over inlier_indices."""
        if M is None or len(inlier_indices) == 0:
            return None
        n = M.shape[0]
        u = np.zeros(n, dtype=float)
        u[np.asarray(inlier_indices, dtype=int)] = 1.0
        denom = float(u @ u)
        if denom <= 0:
            return None
        # M may be sparse; M @ u handles both dense and scipy.sparse.
        Mu = M @ u
        if hasattr(Mu, "A1"):  # np.matrix from sparse
            Mu = Mu.A1
        Mu = np.asarray(Mu).flatten()
        return float(u @ Mu) / denom

    def _run_clipper_cpp_single(
        self,
        candidates: List[dict],
        trajectory: List[np.ndarray],
        times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
        """Single-pass CLIPPER over the given candidates. See `run_clipper_cpp`
        for the chunked wrapper. Returns (inlier_indices, M, C, objective)."""
        lc_scores = self._compute_lc_scores(candidates)
        D, aerial_poses, ground_poses, ground_distances = _build_clipper_data(
            candidates, trajectory, times, lc_scores
        )
        iparams = clipperpy.invariants.LoopClosureConsistencyParams()
        iparams.rot_sigma_rad = self.params.rot_consistency_sigma_rad
        iparams.rot_eps_rad = self.params.rot_consistency_eps_rad
        iparams.trans_sigma_m = self.params.trans_consistency_sigma_m
        iparams.trans_eps_m = self.params.trans_consistency_eps_m
        iparams.added_trans_noise_m_per_m = self.params.added_trans_noise_m_per_m
        iparams.added_rot_noise_deg_per_m = self.params.added_rot_noise_deg_per_m
        iparams.single_lc_per_ground_sm = self.params.single_lc_per_ground_sm
        iparams.single_lc_per_ground_aerial_pair = (
            self.params.single_lc_per_ground_aerial_pair
        )
        iparams.fuse_lc_score = self.params.fuse_lc_score
        invariant = clipperpy.invariants.LoopClosureConsistency(
            aerial_poses, ground_poses, ground_distances, iparams
        )
        clipper = clipperpy.CLIPPERPairwiseAndSingle(invariant, clipperpy.Params())
        N = D.shape[1]
        A = np.stack([np.arange(N), np.arange(N)], axis=1).astype(np.int32)
        clipper.score_pairwise_and_single_consistency(D, D, A)
        clipper.solve()
        inlier_indices = np.array(clipper.get_solution().nodes)
        M = clipper.get_affinity_matrix()
        C = clipper.get_constraint_matrix()
        objective = self._binary_objective(M, inlier_indices)
        return inlier_indices, M, C, objective

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

    def lift_to_3d(
        self,
        optimized_trajectory: List[np.ndarray],
        trajectory: List[np.ndarray],
        T_camera_flu: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        """Lift the 2D solution to full 3D T_utm_body poses.

        x, y and yaw come from the 2D solution; roll and pitch start from the
        odometry. Heights are solved first as a 1D least-squares problem
        (odometry height increments, equal heights at cross-path revisits, and
        the first node's odometry height as anchor). If lift_3d_refine_rotation
        is set, a Pose3 graph then refines all poses (correcting roll/pitch
        drift in the odometry) with full 3D odometry between factors, x/y/yaw
        priors from the 2D solution, and the cross-path height constraints.

        Args:
            optimized_trajectory: 2D solution as 4x4 T_utm_body (x, y, yaw used).
            trajectory: 4x4 T_odom_camera odometry poses at the same nodes.
            T_camera_flu: Optional 4x4 transform from camera to FLU body frame.

        Returns:
            List of 4x4 T_utm_body poses with heights and roll/pitch.
        """
        n = len(trajectory)
        T_cb = np.eye(4) if T_camera_flu is None else T_camera_flu
        T_odom_body = np.array([T @ T_cb for T in trajectory])
        T_2d = np.array(optimized_trajectory)
        xy = T_2d[:, :2, 3]
        yaw_2d = np.arctan2(T_2d[:, 1, 0], T_2d[:, 0, 0])
        yaw_odom = np.arctan2(T_odom_body[:, 1, 0], T_odom_body[:, 0, 0])

        pairs = cross_path_pairs(
            xy,
            self.params.cross_paths_3d_constraint_dist_m,
            self.params.cross_paths_3d_constraint_path_dist_m,
        )
        logger.info(f"3D lift: {n} nodes, {len(pairs)} cross-path height constraints")

        z = self._solve_heights(T_odom_body[:, 2, 3], pairs)

        # Initial 3D poses: rotate odometry attitude about world z to the 2D yaw
        # (keeps the odometry's roll/pitch), 2D x/y, solved heights.
        lifted = np.tile(np.eye(4), (n, 1, 1))
        lifted[:, :3, :3] = (
            Rot.from_euler("z", (yaw_2d - yaw_odom)[:, None]).as_matrix()
            @ T_odom_body[:, :3, :3]
        )
        lifted[:, :2, 3] = xy
        lifted[:, 2, 3] = z

        if self.params.lift_3d_refine_rotation and n > 1:
            lifted = self._refine_3d(lifted, T_odom_body, xy, pairs)
        return list(lifted)

    def _solve_heights(self, z_odom: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        """Weighted linear least squares for node heights."""
        n = len(z_odom)
        if n == 1:
            return z_odom.copy()
        w_odom = 1.0 / self.params.odom_trans_sigma_m
        w_pair = 1.0 / self.params.cross_paths_3d_constraint_sigma_m
        w_anchor = 1e3

        idx = np.arange(n - 1)
        rows = np.concatenate([[0], 1 + np.repeat(idx, 2)])
        cols = np.concatenate([[0], np.stack([idx, idx + 1], 1).ravel()])
        vals = np.concatenate([[w_anchor], np.tile([-w_odom, w_odom], n - 1)])
        b = np.concatenate([[w_anchor * z_odom[0]], w_odom * np.diff(z_odom)])
        if len(pairs):
            r0 = n
            prow = r0 + np.repeat(np.arange(len(pairs)), 2)
            rows = np.concatenate([rows, prow])
            cols = np.concatenate([cols, pairs.ravel()])
            vals = np.concatenate([vals, np.tile([w_pair, -w_pair], len(pairs))])
            b = np.concatenate([b, np.zeros(len(pairs))])
        A = sp.csr_matrix((vals, (rows, cols)), shape=(len(b), n))
        return spla.spsolve((A.T @ A).tocsc(), A.T @ b)

    def _refine_3d(
        self,
        lifted: np.ndarray,
        T_odom_body: np.ndarray,
        xy: np.ndarray,
        pairs: np.ndarray,
        max_iterations: int = 50,
    ) -> np.ndarray:
        """Pose3 graph refining the lifted poses (see lift_to_3d).

        Factors: 3D odometry between consecutive poses; x, y, yaw priors from the
        2D solution; equal world-frame height for cross-path pairs; and an anchor
        on the first pose's roll, pitch and height (the gauge the other factors
        leave free).

        Solved with Levenberg-Marquardt on SE(3) in numpy/scipy rather than
        GTSAM: the height and yaw factors need Python CustomFactors, which GTSAM
        evaluates under the GIL from its worker threads (~100x slower here), and
        built-in-only reformulations were poorly conditioned.
        """
        n = len(lifted)
        R = lifted[:, :3, :3].copy()
        t = lifted[:, :3, 3].copy()
        yaw_prior = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        R0_anchor, z0_anchor = R[0].copy(), t[0, 2]
        T_rel = np.linalg.inv(T_odom_body[:-1]) @ T_odom_body[1:]
        Rm, tm = T_rel[:, :3, :3], T_rel[:, :3, 3]
        w_rot = 1.0 / self.params.odom_rot_sigma_rad
        w_tran = 1.0 / self.params.odom_trans_sigma_m
        w_xy = 1.0 / self.params.lift_3d_prior_xy_sigma_m
        w_yaw = 1.0 / self.params.lift_3d_prior_yaw_sigma_rad
        w_h = 1.0 / self.params.cross_paths_3d_constraint_sigma_m
        w_anchor = 1.0 / _LIFT_ANCHOR_SIGMA
        i0, i1 = np.arange(n - 1), np.arange(1, n)
        eye3 = np.broadcast_to(np.eye(3), (n - 1, 3, 3))

        def residuals(R, t):
            E = np.swapaxes(Rm, 1, 2) @ np.swapaxes(R[i0], 1, 2) @ R[i1]
            dt = t[i1] - t[i0]
            yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
            r_anchor = Rot.from_matrix(R0_anchor.T @ R[0]).as_rotvec()[:2]
            return np.concatenate(
                [
                    w_rot * Rot.from_matrix(E).as_rotvec().ravel(),
                    w_tran * (np.einsum("nji,nj->ni", R[i0], dt) - tm).ravel(),
                    w_xy * (t[:, :2] - xy).ravel(),
                    w_yaw * ((yaw - yaw_prior + np.pi) % (2 * np.pi) - np.pi),
                    w_h * (t[pairs[:, 0], 2] - t[pairs[:, 1], 2]),
                    w_anchor * np.r_[r_anchor, t[0, 2] - z0_anchor],
                ]
            )

        def jacobian(R, t):
            # Right perturbations: R <- R Exp(dtheta), t <- t + R dt; node k's
            # tangent occupies columns 6k:6k+3 (dtheta) and 6k+3:6k+6 (dt).
            rows, cols, vals = [], [], []

            def block(r0, node, c_off, M):
                # M: (m, a, b) blocks placed at rows r0 + a*m, cols 6*node + c_off
                m, a, b = M.shape
                rr = r0[:, None, None] + np.arange(a)[None, :, None]
                cc = 6 * node[:, None, None] + c_off + np.arange(b)[None, None, :]
                rows.append(np.broadcast_to(rr, M.shape).ravel())
                cols.append(np.broadcast_to(cc, M.shape).ravel())
                vals.append(M.ravel())

            r = 0
            # odometry rotation residual Log(Rm^T Ri^T Rj)
            rr = r + 3 * i0
            block(rr, i0, 0, -w_rot * np.swapaxes(R[i1], 1, 2) @ R[i0])
            block(rr, i1, 0, w_rot * eye3)
            r += 3 * (n - 1)
            # odometry translation residual Ri^T (tj - ti) - tm
            rr = r + 3 * i0
            dt_body = np.einsum("nji,nj->ni", R[i0], t[i1] - t[i0])
            skew = np.zeros((n - 1, 3, 3))
            skew[:, 0, 1], skew[:, 0, 2] = -dt_body[:, 2], dt_body[:, 1]
            skew[:, 1, 0], skew[:, 1, 2] = dt_body[:, 2], -dt_body[:, 0]
            skew[:, 2, 0], skew[:, 2, 1] = -dt_body[:, 1], dt_body[:, 0]
            block(rr, i0, 0, w_tran * skew)
            block(rr, i0, 3, -w_tran * eye3)
            block(rr, i1, 3, w_tran * np.swapaxes(R[i0], 1, 2) @ R[i1])
            r += 3 * (n - 1)
            # x, y priors
            nodes = np.arange(n)
            block(r + 2 * nodes, nodes, 3, w_xy * R[:, :2, :])
            r += 2 * n
            # yaw prior: d atan2(R10, R00) / d dtheta
            d = R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2
            J_yaw = np.zeros((n, 1, 3))
            J_yaw[:, 0, 1] = (R[:, 1, 0] * R[:, 0, 2] - R[:, 0, 0] * R[:, 1, 2]) / d
            J_yaw[:, 0, 2] = (R[:, 0, 0] * R[:, 1, 1] - R[:, 1, 0] * R[:, 0, 1]) / d
            block(r + nodes, nodes, 0, w_yaw * J_yaw)
            r += n
            # cross-path equal heights
            if len(pairs):
                pr = r + np.arange(len(pairs))
                block(pr, pairs[:, 0], 3, w_h * R[pairs[:, 0], 2:3, :])
                block(pr, pairs[:, 1], 3, -w_h * R[pairs[:, 1], 2:3, :])
                r += len(pairs)
            # anchor: first node roll/pitch (local rotation x, y) and height
            block(np.array([r]), np.array([0]), 0, w_anchor * np.eye(3)[None, :2, :])
            block(np.array([r + 2]), np.array([0]), 3, w_anchor * R[0:1, 2:3, :])
            r += 3
            return sp.csr_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                shape=(r, 6 * n),
            )

        res = residuals(R, t)
        cost = res @ res
        lam = 1e-4
        for _ in range(max_iterations):
            J = jacobian(R, t)
            H = (J.T @ J).tocsc()
            g = J.T @ res
            while True:
                damp = sp.diags(lam * (H.diagonal() + 1e-9))
                delta = spla.spsolve(H + damp, -g).reshape(n, 6)
                R_new = R @ Rot.from_rotvec(delta[:, :3]).as_matrix()
                t_new = t + np.einsum("nij,nj->ni", R, delta[:, 3:])
                res_new = residuals(R_new, t_new)
                cost_new = res_new @ res_new
                if cost_new < cost or lam > 1e8:
                    break
                lam *= 10
            if cost_new >= cost:
                break
            converged = cost - cost_new < 1e-9 * cost
            R, t, res, cost = R_new, t_new, res_new, cost_new
            lam = max(lam / 10, 1e-12)
            if converged:
                break

        out = np.tile(np.eye(4), (n, 1, 1))
        out[:, :3, :3], out[:, :3, 3] = R, t
        return out
