/**
 * @file general_segment_distance.h
 * @brief General Segment Distance invariant
 * @author Mason Peterson, <masonbp@mit.edu>
 * @date 16 September 2025
 */

#pragma once

#include "clipper/invariants/abstract.h"
#include <cmath>
#include <tuple>

namespace clipper {
namespace invariants {

  /**
   * @brief      Specialization of PairwiseAndSingleInvariant to GravityConstrained distance in
   *             the real numbers using the 2-norm as the invariant.
   */
  class GeneralSegmentDistance : public PairwiseAndSingleInvariant
  {
  public:
    enum SegmentType {
      POINT,
      LINE,
      PLANE
    };

    struct Point {
      Eigen::VectorXd point;
      Eigen::VectorXd ratio_feature; // e.g., volume, area, length
      Eigen::VectorXd cos_feature; // e.g., FPFH, CLIP, Dino
    };

    struct Line {
      Eigen::VectorXd point; // point on line
      Eigen::VectorXd direction; // unit direction vector
      uint32_t num_endpoints;
      Eigen::VectorXd endpoint1; // first endpoint if num_endpoints >= 1
      Eigen::VectorXd endpoint2; // second endpoint if num_endpoints == 2
      Eigen::VectorXd ratio_feature; // e.g., length
      Eigen::VectorXd cos_feature; // e.g., FPFH, CLIP, Dino
    };

    struct Plane {
      Eigen::VectorXd point; // point on plane
      Eigen::VectorXd normal; // unit normal vector
      Eigen::VectorXd ratio_feature; // e.g., area
      Eigen::VectorXd cos_feature; // e.g., FPFH, CLIP, Dino
    };
    struct Params
    {
      uint32_t dim = 3; ///< dimension of points (2 or 3)
      uint32_t ratio_feature_dim = 0; ///< number of ratio features (e.g., volume)
      uint32_t cos_feature_dim = 0; ///< number of features used for cosine similarity

      double sigma_dist = 0.4; ///< spread / "variance" of exponential kernel
      double epsilon_dist = 0.6; ///< bound on consistency score, determines if inlier/outlier
      double min_dist = 0.0; ///< minimum allowable distance between inlier points in the same dataset
      double sigma_angle_rad = 10.0 * M_PI / 180.0; ///< spread / "variance" of exponential kernel
      double epsilon_angle_rad = 20.0 * M_PI / 180.0; ///< bound on consistency score, determines if inlier/outlier
      double min_angle_rad = 0.0; ///< minimum allowable angle (in radians) between inlier segments in the same dataset

      double distance_weight = 1.0; ///< weight of pairwise similarity in single/pairwise fusion
      double ratio_weight = 1.0; ///< weight of cosine similarity in single similarity fusion
      double cosine_weight = 1.0; ///< weight of cosine similarity in single similarity fusion
      
      Eigen::VectorXd ratio_epsilon =  Eigen::VectorXd::Zero(ratio_feature_dim); ///< bound on feature ratio score, determines if inlier/outlier
      double cosine_min = 0.5; ///< cosine similarity scaled so that cosine_min maps to 0.0 similarity score
      double cosine_max = 0.7; ///< cosine similarity scaled so that cosine_max maps to 1.0 similarity score
      // bool cosine_normalized = false; ///< option to speed up by sending in pre-normalized features
      
      bool z_dir_constrained = false; ///< whether to use gravity-guided prior
      bool xyz_dir_constrained = false; ///< whether to use rotation prior in all directions (only applicable for 3D)
      bool xy_dir_constrained_2d = false; ///< whether to use rotation prior in all directions (only applicable for 2D)
      double rot_unc_ang_rad = 0.0; ///< uncertainty adjustment for known rotation direction in radians
      double parallel_eps = 0.1 * M_PI / 180.0; ///< threshold for determining if two lines are parallel
      double nearby_eps = 1e-3; ///< threshold for determining if a point is on a line or plane

      double min_angle_for_line_to_line_dist = 30.0 * M_PI / 180.0; ///< minimum angle between two lines to use distance in line-to-line similarity
      bool bidirectional = true; ///< whether to treat lines (non-rays) and planes as potentially traveling in flipped direction
      bool point_noise_from_angle = true; ///< whether to model additional point noise from angular uncertainty for line and plane features
      
    };
  public:
    GeneralSegmentDistance(const Params& params)
    : params_(params) 
    {
      rot_unc_ang_cos_ = std::cos(params_.rot_unc_ang_rad);
      rot_unc_ang_sin_ = std::sin(params_.rot_unc_ang_rad);
    }
    ~GeneralSegmentDistance() = default;

    /**
     * @brief      Functor for pairwise invariant scoring function
     *
     * @param[in]  ai    Element i from dataset 1
     * @param[in]  aj    Element j from dataset 1
     * @param[in]  bi    Element i from dataset 2
     * @param[in]  bj    Element j from dataset 2
     *
     * @return     The consistency score for the association of (ai,bi) and (aj,bj)
     */
    double pairwise_similarity(const Datum& ai, const Datum& aj, const Datum& bi, const Datum& bj) override;

    /**
     * @brief      Functor for fusing the pairwise and single scores
     *
     * @param[in]  pair_ij    Score for pair of associations
     * @param[in]  single_i   Single-association score for i
     * @param[in]  single_j   Single-association score for j
     *
     * @return     The consistency score for the fused pairwise and single scores
     */
    virtual double pairwise_single_fusion(const double& pair_ij, const double& single_i, const double& single_j) override;

    /**
     * @brief      Functor for the scoring of a single association
     *
     * @param[in]  ai    Element i from dataset 1
     * @param[in]  bi    Element i from dataset 2
     *
     * @return     The consistency score for the association of (ai,bi)
     */
    virtual double single_similarity(const Datum& ai, const Datum& bi) override;

  private:
    Params params_;
    double rot_unc_ang_cos_;
    double rot_unc_ang_sin_;

    /**
     * @brief      Functor for pairwise invariant scoring function
     *
     * @param[in]  point_ai    Point i from dataset 1
     * @param[in]  point_aj    Point j from dataset 1
     * @param[in]  point_bi    Point i from dataset 2
     * @param[in]  point_bj    Point j from dataset 2
     *
     * @return     The consistency score for the association of (ai,bi) and (aj,bj)
     */
    double pairwise_point_to_point_sim(const Point& point_ai, const Point& point_aj, 
        const Point& point_bi, const Point& point_bj, bool enforce_min_dist = true,
        double noise_inflate = 0.0);

    double pairwise_single_dim_sim(const double& val_ai, const double& val_aj, 
      const double& val_bi, const double& val_bj, bool ignore_sign = true, double noise_inflate = 0.0);

    /**
     * @brief      Functor for pairwise invariant scoring function
     *
     * @param[in]  point_ai   Point i from dataset 1
     * @param[in]  line_aj    Line j from dataset 1
     * @param[in]  point_bi   Point i from dataset 2
     * @param[in]  line_bj    Line j from dataset 2
     *
     * @return     The consistency score for the association of (ai,bi) and (aj,bj)
     */
    double pairwise_point_to_line_sim(const Point& point_ai, const Line& line_aj, 
        const Point& point_bi, const Line& line_bj);

    /**
      * @brief      Functor for pairwise invariant scoring function
      *
      * @param[in]  point_ai   Point i from dataset 1
      * @param[in]  plane_aj    Plane j from dataset 1
      * @param[in]  point_bi   Point i from dataset 2
      * @param[in]  plane_bj    Plane j from dataset 2
      *
      * @return     The consistency score for the association of (ai,bi) and (aj,bj)
      */
    double pairwise_point_to_plane_sim(const Point& point_ai, const Plane& plane_aj, 
        const Point& point_bi, const Plane& plane_bj);
      
    /**
     * @brief      Functor for pairwise invariant scoring function
     *
     * @param[in]  line_ai    Line i from dataset 1
     * @param[in]  line_aj    Line j from dataset 1
     * @param[in]  line_bi    Line i from dataset 2
     * @param[in]  line_bj    Line j from dataset 2
     *
     * @return     The consistency score for the association of (ai,bi) and (aj,bj)
     */
    double pairwise_line_to_line_sim(const Line& line_ai, const Line& line_aj, 
        const Line& line_bi, const Line& line_bj);

    /**
      * @brief      Functor for pairwise invariant scoring function
      * @param[in]  line_ai    Line i from dataset 1
      * @param[in]  plane_aj    Plane j from dataset 1
      * @param[in]  line_bi    Line i from dataset 2
      * @param[in]  plane_bj    Plane j from dataset 2
      * @return     The consistency score for the association of (ai,bi) and (aj,bj)
      */
    double pairwise_line_to_plane_sim(const Line& line_ai, const Plane& plane_aj, 
        const Line& line_bi, const Plane& plane_bj);

    /**
     * @brief      Functor for pairwise invariant scoring function
     * @param[in]  plane_ai    Plane i from dataset 1
     * @param[in]  plane_aj    Plane j from dataset 1
     * @param[in]  plane_bi    Plane i from dataset 2
     * @param[in]  plane_bj    Plane j from dataset 2
     * @return     The consistency score for the association of (ai,bi) and (aj,bj)
     */
    double pairwise_plane_to_plane_sim(const Plane& plane_ai, const Plane& plane_aj,
        const Plane& plane_bi, const Plane& plane_bj);

    /**
     * @brief      Computes angle similarity between two direction vectors
     * @param[in]  dir_ai    Direction vector i from dataset 1
     * @param[in]  dir_aj    Direction vector j from dataset 1
     * @param[in]  dir_bi    Direction vector i from dataset 2
     * @param[in]  dir_bj    Direction vector j from dataset 2
     * @return     The angle consistency score for the association of (ai,bi) and (aj,bj)
     */
    double pairwise_angle_sim(
        const Eigen::VectorXd& dir_ai, 
        const Eigen::VectorXd& dir_aj, 
        const Eigen::VectorXd& dir_bi, 
        const Eigen::VectorXd& dir_bj);

    /**
     * @brief      Conversion from Datum to Point struct
     *
     * @param[in]  datum    Datum to convert
     *
     * @return     The Point struct
     */
    Point datum_to_point(const Datum& datum);

    /**
     * @brief      Conversion from Datum to Line struct
     *
     * @param[in]  datum    Datum to convert
     *
     * @return     The Line struct
     */
    Line datum_to_line(const Datum& datum);

    /**
     * @brief      Conversion from Datum to Plane struct
     *
     * @param[in]  datum    Datum to convert
     *
     * @return     The Plane struct
     */
    Plane datum_to_plane(const Datum& datum);
    
    /**
     * @brief      Extract the cosine feature from a Datum
     *
     * @param[in]  datum    Datum to convert
     *
     * @return     The Datum struct
     */
    Datum cos_feature_from_datum(const Datum& datum);

    /**
     * @brief      Extract the ratio feature from a Datum
     *
     * @param[in]  datum    Datum to convert
     *
     * @return     The Datum struct
     */
    Datum ratio_feature_from_datum(const Datum& datum);

    /**
     * @brief      Conversion from Point struct to Datum
     *
     * @param[in]  point    Point to convert
     *
     * @return     The Datum struct
     */
    Datum point_to_datum(const Point& point);

    /**
     * @brief      Conversion from Line struct to Datum
     *
     * @param[in]  line    Line to convert
     *
     * @return     The Datum struct
     */
    Datum line_to_datum(const Line& line);

    /**
     * @brief      Conversion from Plane struct to Datum
     *
     * @param[in]  plane    Plane to convert
     *
     * @return     The Datum struct
     */
    Datum plane_to_datum(const Plane& plane);

    /**
     * @brief      Find the nearest point on a line to a given point
     *
     * @param[in]  point    The point
     * @param[in]  line     The line
     *
     * @return     The nearest point on the line to the given point
     */
    Point nearest_point_to_line(const Point& point, const Line& line, const bool on_infinite_line = false);

    /**
      * @brief      Find the nearest point on a plane to a given point
      * 
      * @param[in]  point    The point
      * @param[in]  plane     The plane
      *
      * @return     The nearest point on the plane to the given point
     */
    Point nearest_point_to_plane(const Point& point, const Plane& plane);

    /**
     * @brief      Find the nearest points between two lines
     *
     * @param[in]  line1    The first line
     * @param[in]  line2    The second line
     *
     * @return     A tuple containing the nearest point on line1 and the nearest point on line2
     */
    std::tuple<Point, Point> nearest_points_on_lines(const Line& line1, const Line& line2);

    /**
     * @brief      Check if a point lies on a line within a threshold
     *
     * @param[in]  point    The point
     * @param[in]  line     The line
     *
     * @return     True if the point is on the line, False otherwise
     */
    bool point_is_on_line(const Point& point, const Line& line);

    /**
     * @brief      Check if two direction vectors are parallel within a threshold
     *
     * @param[in]  dir1    First direction vector
     * @param[in]  dir2    Second direction vector
     *
     * @return     True if the vectors are parallel, False otherwise
     */
    bool is_parallel(const Eigen::VectorXd& dir1, const Eigen::VectorXd& dir2) {
      double cos_angle = std::abs(dir1.transpose() * dir2) / (dir1.norm() * dir2.norm());
      return cos_angle >= std::cos(params_.parallel_eps);
    }

    bool use_angle_similarity(Datum datum) {
      if (!params_.z_dir_constrained && !params_.xyz_dir_constrained && !params_.xy_dir_constrained_2d) {
        return false;
      }
      SegmentType datum_type = static_cast<SegmentType>(static_cast<int>(datum(0)));
      return datum_type == SegmentType::PLANE || datum_type == SegmentType::LINE;
    }

    double single_angle_similarity(const Datum& ai, const Datum& bi);

    // double maximum_line_distance(const Line& line_i, const Line& line_j);
    bool violates_line_min_dist(const Line& line_i, const Line& line_j, 
      const Point& nearest_point_i, const Point& nearest_point_j);

  };

  using GeneralSegmentDistancePtr = std::shared_ptr<GeneralSegmentDistance>;

} // ns invariants
} // ns clipper