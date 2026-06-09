/**
 * @file pose_consistency.h
 * @brief Pose consistency invariant for pairwise SE(2)/SE(3) pose agreement
 */

#pragma once

#include <Eigen/Dense>

#include "clipper/invariants/abstract.h"

namespace clipper {
namespace invariants {

/**
 * @brief Pairwise consistency invariant for pose estimates.
 *
 * Each datum encodes one pose as a flattened row-major 4x4 SE(3) matrix
 * (16 doubles). Two poses are consistent if their relative transform has
 * rotation and translation within the specified thresholds.
 *
 * Intended usage: D1 = D2 = D (n×16 matrix of poses), associations A = {(k,k)}.
 * pairwise_similarity only uses ai and aj.
 */
class PoseConsistency : public PairwiseAndSingleInvariant
{
public:
  struct Params
  {
    int dim = 2;                    ///< 2 for SE(2), 3 for SE(3)
    double rot_sigma_rad = 0.1745;  ///< rotation kernel spread (rad)
    double rot_eps_rad   = 0.3491;  ///< rotation inlier threshold (rad)
    double trans_sigma_m = 5.0;     ///< translation kernel spread (m)
    double trans_eps_m   = 10.0;    ///< translation inlier threshold (m)
  };

public:
  PoseConsistency(const Params& params);

  ~PoseConsistency() = default;

  /**
   * @brief Score pairwise consistency between two poses.
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
  Params params_;
};

using PoseConsistencyPtr = std::shared_ptr<PoseConsistency>;

} // ns invariants
} // ns clipper
