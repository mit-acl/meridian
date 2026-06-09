/**
 * @file pose_consistency.cpp
 * @brief Pose consistency invariant for pairwise SE(2)/SE(3) pose agreement
 */

#include <algorithm>
#include <cmath>

#include "clipper/invariants/pose_consistency.h"

namespace clipper {
namespace invariants {

PoseConsistency::PoseConsistency(const Params& params)
  : params_(params)
{}

double PoseConsistency::pairwise_similarity(
    const Datum& ai, const Datum& aj,
    const Datum& /*bi*/, const Datum& /*bj*/)
{
  // Decode poses from flattened row-major 4x4 matrices
  const Eigen::Matrix4d T_i =
      Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(ai.data());
  const Eigen::Matrix4d T_j =
      Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(aj.data());

  // Relative transform
  const Eigen::Matrix4d T_rel = T_i.inverse() * T_j;

  double rot_err, trans_err;

  if (params_.dim == 2) {
    // SE(2): extract yaw and 2D translation
    rot_err = std::abs(std::atan2(T_rel(1, 0), T_rel(0, 0)));
    trans_err = std::sqrt(T_rel(0, 3) * T_rel(0, 3) +
                          T_rel(1, 3) * T_rel(1, 3));
  } else {
    // SE(3): rotation angle from trace and 3D translation norm
    const double cos_angle = std::min(1.0, std::max(-1.0,
        (T_rel.block<3, 3>(0, 0).trace() - 1.0) / 2.0));
    rot_err = std::acos(cos_angle);
    trans_err = T_rel.block<3, 1>(0, 3).norm();
  }

  if (rot_err < params_.rot_eps_rad && trans_err < params_.trans_eps_m) {
    return std::exp(-0.5 * rot_err * rot_err /
                    (params_.rot_sigma_rad * params_.rot_sigma_rad)) *
           std::exp(-0.5 * trans_err * trans_err /
                    (params_.trans_sigma_m * params_.trans_sigma_m));
  }
  return 0.0;
}

double PoseConsistency::single_similarity(
    const Datum& /*ai*/, const Datum& /*bi*/)
{
  return 1.0;
}

double PoseConsistency::pairwise_single_fusion(
    const double& pair_ij, const double& /*single_i*/, const double& /*single_j*/)
{
  return pair_ij;
}

} // ns invariants
} // ns clipper
