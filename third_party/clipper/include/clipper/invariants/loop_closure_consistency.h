/**
 * @file loop_closure_consistency.h
 * @brief Loop closure consistency invariant for cross-view localization
 */

#pragma once

#include <vector>

#include <Eigen/Dense>

#include "clipper/invariants/abstract.h"

namespace clipper {
namespace invariants {

/**
 * @brief Pairwise consistency invariant for loop closure candidates.
 *
 * Each datum encodes one loop closure candidate:
 *   datum[0]     = aerial_idx (index into aerial_poses_)
 *   datum[1]     = ground_idx (index into ground_poses_)
 *   datum[2..17] = T_i_j_hat flattened row-major (16 doubles)
 *
 * Two candidates are consistent if their relative transforms (estimated via
 * odometry and via cross-view measurements) agree within rotation and
 * translation thresholds. The score is an exponential kernel of the errors.
 */
class LoopClosureConsistency : public PairwiseAndSingleInvariant
{
public:
  struct Params
  {
    double rot_sigma_rad = 0.1745;  ///< ~10 deg rotation kernel spread
    double rot_eps_rad   = 0.3491;  ///< ~20 deg rotation inlier threshold
    double trans_sigma_m = 5.0;     ///< translation kernel spread (m)
    double trans_eps_m   = 10.0;    ///< translation inlier threshold (m)
    double added_trans_noise_m_per_m   = 0.005; ///< extra trans noise per meter of path
    double added_rot_noise_deg_per_m   = 0.005; ///< extra rot noise (deg) per meter of path
    bool single_lc_per_ground_sm         = true; ///< LCs sharing a ground idx are inconsistent
    bool single_lc_per_ground_aerial_pair = true; ///< LCs sharing (ground,aerial) pair are inconsistent
    bool fuse_lc_score = true; ///< LCs incorporate a single quality score \in [0, 1]
    // add dim param for operating in 3D
  };

public:
  LoopClosureConsistency(
    const std::vector<Eigen::MatrixXd>& aerial_poses,
    const std::vector<Eigen::MatrixXd>& ground_poses,
    const std::vector<double>& ground_distances,
    const Params& params);

  ~LoopClosureConsistency() = default;

  /**
   * @brief Score pairwise consistency between two loop closure candidates.
   *
   * Since D1=D2=D and A[k]=(k,k), ai==bi and aj==bj. Only ai and aj are used.
   */
  double pairwise_similarity(const Datum& ai, const Datum& aj,
                             const Datum& bi, const Datum& bj) override;

  /// Returns 1.0 — no per-candidate feature score.
  double single_similarity(const Datum& ai, const Datum& bi) override;

  /// Returns pair_ij directly — single scores are neutral.
  double pairwise_single_fusion(const double& pair_ij,
                                const double& single_i,
                                const double& single_j) override;

private:
  std::vector<Eigen::MatrixXd> aerial_poses_;
  std::vector<Eigen::MatrixXd> ground_poses_;
  std::vector<double> ground_distances_;
  Params params_;
};

using LoopClosureConsistencyPtr = std::shared_ptr<LoopClosureConsistency>;

} // ns invariants
} // ns clipper
