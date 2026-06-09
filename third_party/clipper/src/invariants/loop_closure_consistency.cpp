/**
 * @file loop_closure_consistency.cpp
 * @brief Loop closure consistency invariant for cross-view localization
 */

#include <cmath>

#include "clipper/invariants/loop_closure_consistency.h"

namespace clipper {
namespace invariants {

LoopClosureConsistency::LoopClosureConsistency(
    const std::vector<Eigen::MatrixXd>& aerial_poses,
    const std::vector<Eigen::MatrixXd>& ground_poses,
    const std::vector<double>& ground_distances,
    const Params& params)
  : aerial_poses_(aerial_poses), ground_poses_(ground_poses),
    ground_distances_(ground_distances), params_(params)
{}

double LoopClosureConsistency::pairwise_similarity(
    const Datum& ai, const Datum& aj,
    const Datum& bi, const Datum& bj)
{
  // Decode candidate i
  const int aerial_idx_i = static_cast<int>(ai[0]);
  const int ground_idx_i = static_cast<int>(ai[1]);
  // Map T_hat from datum[2..17] stored row-major (matches numpy .flatten())
  const Eigen::MatrixXd T_hat_i =
      Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(ai.data() + 2);

  // Decode candidate j
  const int aerial_idx_j = static_cast<int>(aj[0]);
  const int ground_idx_j = static_cast<int>(aj[1]);
  const Eigen::MatrixXd T_hat_j =
      Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(aj.data() + 2);

  // Enforce at most one loop closure per ground submap / (ground,aerial) pair.
  if (params_.single_lc_per_ground_sm && ground_idx_i == ground_idx_j) {
    return 0.0;
  }
  if (params_.single_lc_per_ground_aerial_pair &&
      ground_idx_i == ground_idx_j &&
      aerial_idx_i == aerial_idx_j) {
    return 0.0;
  }

  // Look up poses
  const Eigen::MatrixXd& T_odom_ground_i = ground_poses_[ground_idx_i];
  const Eigen::MatrixXd& T_odom_ground_j = ground_poses_[ground_idx_j];
  const Eigen::MatrixXd& aerial_i = aerial_poses_[aerial_idx_i];
  const Eigen::MatrixXd& aerial_j = aerial_poses_[aerial_idx_j];

  // Compute distance-dependent noise inflation
  const double path_dist = std::abs(
      ground_distances_[ground_idx_j] - ground_distances_[ground_idx_i]);
  const double added_rot_rad =
      path_dist * params_.added_rot_noise_deg_per_m * M_PI / 180.0;
  const double added_trans_m =
      path_dist * params_.added_trans_noise_m_per_m;

  const double rot_sigma = params_.rot_sigma_rad + added_rot_rad;
  const double rot_eps   = params_.rot_eps_rad   + added_rot_rad;
  const double trans_sigma = params_.trans_sigma_m + added_trans_m;
  const double trans_eps   = params_.trans_eps_m   + added_trans_m;

  // Relative transform via odometry: T_odom_ground_i^{-1} * T_odom_ground_j
  const Eigen::MatrixXd odom_rel_3d = T_odom_ground_i.inverse() * T_odom_ground_j;

  // Relative transform via cross-view:
  //   T_hat_i^{-1} * aerial_i^{-1} * aerial_j * T_hat_j
  const Eigen::MatrixXd cv_rel_3d =
      T_hat_i.inverse() * aerial_i.inverse() * aerial_j * T_hat_j;

  // Project each 4x4 SE(3) transform to a 3x3 SE(2) matrix (x, y, yaw).
  // Mirrors Python's se3_to_se2: yaw = atan2(T[1,0], T[0,0]).
  auto se3_to_se2 = [](const Eigen::MatrixXd& T) -> Eigen::Matrix3d {
    const double yaw = std::atan2(T(1, 0), T(0, 0));
    const double c = std::cos(yaw), s = std::sin(yaw);
    Eigen::Matrix3d T2;
    T2 << c, -s, T(0, 3),
          s,  c, T(1, 3),
          0,  0, 1;
    return T2;
  };

  const Eigen::Matrix3d odom_relative = se3_to_se2(odom_rel_3d);
  const Eigen::Matrix3d cv_relative   = se3_to_se2(cv_rel_3d);

  // Error in SE(2): should be identity if consistent
  const Eigen::Matrix3d error = odom_relative.inverse() * cv_relative;

  // Extract yaw and 2D translation from the 3x3 SE(2) error matrix
  const double rot_err = std::abs(std::atan2(error(1, 0), error(0, 0)));
  const double trans_err = std::sqrt(error(0, 2) * error(0, 2) +
                                     error(1, 2) * error(1, 2));

  if (rot_err < rot_eps && trans_err < trans_eps) {
    return std::sqrt(std::exp(-0.5 * rot_err * rot_err /
                    (rot_sigma * rot_sigma)) *
           std::exp(-0.5 * trans_err * trans_err /
                    (trans_sigma * trans_sigma)));
  }
  return 0.0;
}

double LoopClosureConsistency::single_similarity(
    const Datum& ai, const Datum& /*bi*/)
{
  return ai[18];
}

double LoopClosureConsistency::pairwise_single_fusion(
    const double& pair_ij, const double& single_i, const double& single_j)
{
  if (params_.fuse_lc_score) {
    return std::cbrt(pair_ij * single_i * single_j);
  }
  return pair_ij;
}

} // ns invariants
} // ns clipper
