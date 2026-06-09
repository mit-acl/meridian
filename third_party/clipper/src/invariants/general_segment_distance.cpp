/**
 * @file general_segment_distance.cpp
 * @brief General Segment Distance invariant
 * @author Mason Peterson, <masonbp@mit.edu>
 * @date 16 September 2025
 */

#include "clipper/invariants/general_segment_distance.h"
#include <iostream>
#include <cmath>
#include <stdexcept>

#define SQRT_TWO_THIRDS 0.81649658092
#define SQRT_ONE_THIRD 0.57735026919
#define SQRT_ONE_HALF 0.70710678118

namespace clipper {
namespace invariants {

double GeneralSegmentDistance::pairwise_similarity(const Datum& ai, const Datum& aj,
                                     const Datum& bi, const Datum& bj)
{
  const GeneralSegmentDistance::SegmentType type_ai = static_cast<GeneralSegmentDistance::SegmentType>(static_cast<int>(ai(0)));
  const GeneralSegmentDistance::SegmentType type_aj = static_cast<GeneralSegmentDistance::SegmentType>(static_cast<int>(aj(0)));

  // assert (type_ai == type_aj && type_bi == type_bj && "Segment types must match within each dataset");
  switch (type_ai)
  {
  case GeneralSegmentDistance::POINT: {
    const GeneralSegmentDistance::Point point_ai = datum_to_point(ai);
    const GeneralSegmentDistance::Point point_bi = datum_to_point(bi);

    if (type_aj == GeneralSegmentDistance::POINT) {
      const GeneralSegmentDistance::Point point_aj = datum_to_point(aj);
      const GeneralSegmentDistance::Point point_bj = datum_to_point(bj);
      return GeneralSegmentDistance::pairwise_point_to_point_sim(point_ai, point_aj, point_bi, point_bj);
    } else if (type_aj == GeneralSegmentDistance::LINE) {
      const GeneralSegmentDistance::Line line_aj = datum_to_line(aj);
      const GeneralSegmentDistance::Line line_bj = datum_to_line(bj);
      return GeneralSegmentDistance::pairwise_point_to_line_sim(point_ai, line_aj, point_bi, line_bj);
    } else if (type_aj == GeneralSegmentDistance::PLANE) {
      const GeneralSegmentDistance::Plane plane_aj = datum_to_plane(aj);
      const GeneralSegmentDistance::Plane plane_bj = datum_to_plane(bj);
      return GeneralSegmentDistance::pairwise_point_to_plane_sim(point_ai, plane_aj, point_bi, plane_bj);
    } else {
      // Should not reach here!
      return 0.0;
    }
    break;
  }
  case GeneralSegmentDistance::LINE: {
    const GeneralSegmentDistance::Line line_ai = datum_to_line(ai);
    const GeneralSegmentDistance::Line line_bi = datum_to_line(bi);
    if (type_aj == GeneralSegmentDistance::POINT) {
      const GeneralSegmentDistance::Point point_aj = datum_to_point(aj);
      const GeneralSegmentDistance::Point point_bj = datum_to_point(bj);

      return GeneralSegmentDistance::pairwise_point_to_line_sim(point_aj, line_ai, point_bj, line_bi);
    } else if (type_aj == GeneralSegmentDistance::LINE) {
      const GeneralSegmentDistance::Line line_aj = datum_to_line(aj);
      const GeneralSegmentDistance::Line line_bj = datum_to_line(bj);
      return GeneralSegmentDistance::pairwise_line_to_line_sim(line_ai, line_aj, line_bi, line_bj);
    } else if (type_aj == GeneralSegmentDistance::PLANE) {
      const GeneralSegmentDistance::Plane plane_aj = datum_to_plane(aj);
      const GeneralSegmentDistance::Plane plane_bj = datum_to_plane(bj);
      return GeneralSegmentDistance::pairwise_line_to_plane_sim(line_ai, plane_aj, line_bi, plane_bj);
    } else {
      // Should not reach here!
      return 0.0;
    }
    break;
  }
  case GeneralSegmentDistance::PLANE: {
    const GeneralSegmentDistance::Plane plane_ai = datum_to_plane(ai);
    const GeneralSegmentDistance::Plane plane_bi = datum_to_plane(bi);
    if (type_aj == GeneralSegmentDistance::POINT) {
      const GeneralSegmentDistance::Point point_aj = datum_to_point(aj);
      const GeneralSegmentDistance::Point point_bj = datum_to_point(bj);
      return GeneralSegmentDistance::pairwise_point_to_plane_sim(point_aj, plane_ai, point_bj, plane_bi);
    } else if (type_aj == GeneralSegmentDistance::LINE) {
      const GeneralSegmentDistance::Line line_aj = datum_to_line(aj);
      const GeneralSegmentDistance::Line line_bj = datum_to_line(bj);
      return GeneralSegmentDistance::pairwise_line_to_plane_sim(line_aj, plane_ai, line_bj, plane_bi);
    } else if (type_aj == GeneralSegmentDistance::PLANE) {
      const GeneralSegmentDistance::Plane plane_aj = datum_to_plane(aj);
      const GeneralSegmentDistance::Plane plane_bj = datum_to_plane(bj);
      return GeneralSegmentDistance::pairwise_plane_to_plane_sim(plane_ai, plane_aj, plane_bi, plane_bj);
    } else {
      // Should not reach here!
      return 0.0;
    }
    break;
  }
  default:
    break;
  }
  return 0.0;

}

double GeneralSegmentDistance::pairwise_single_fusion(
    const double& pair_ij, const double& single_i, const double& single_j)
{
  // return pair_ij;
  if (params_.ratio_feature_dim > 0 || params_.cos_feature_dim > 0) {
    double dist_score_pow = std::pow(pair_ij, params_.distance_weight);
    return std::pow(dist_score_pow * single_i * single_j, 1.0/(params_.distance_weight + 2.0));
  } else {
    return pair_ij;
  }
}

double GeneralSegmentDistance::single_similarity(const Datum& ai, const Datum& bi)
{
  const bool use_angle = use_angle_similarity(ai);
  const bool use_cosine = (params_.cos_feature_dim > 0);
  const bool use_ratio = (params_.ratio_feature_dim > 0);
  
  double cosine_score_scaled = 0.0;
  double ratio_score = 0.0;
  double angle_score = 0.0;

  if (!use_angle && !use_cosine && !use_ratio) {
    return 1.0; // no features, so return 1.0
  }

  if (use_cosine) {
    const Datum ai_feat = cos_feature_from_datum(ai);
    const Datum bi_feat = cos_feature_from_datum(bi);
    const double cosine_score  = (ai_feat.transpose() * bi_feat)(0) / (ai_feat.norm() * bi_feat.norm());

    if (cosine_score >= params_.cosine_max)
      cosine_score_scaled = 1.0;
    else if (cosine_score <= params_.cosine_min)
      return 0.0; // geometric mean fusion means the fused score will be 0
    else
      cosine_score_scaled = (cosine_score - params_.cosine_min) / (params_.cosine_max - params_.cosine_min);
  }

  if (use_ratio) {
    // compute ratio feature similarity scores
    Eigen::VectorXd ratio_scores = Eigen::VectorXd::Zero(params_.ratio_feature_dim);
    const Datum ai_feat = ratio_feature_from_datum(ai);
    const Datum bi_feat = ratio_feature_from_datum(bi);

    // for each feature score, similarity score is the ratio of the smaller to the larger
    for (int i=0; i<(int) params_.ratio_feature_dim; i++) {
      ratio_scores(i) = ai_feat(i) < bi_feat(i) ? 
        ai_feat(i) / bi_feat(i) : 
        bi_feat(i) / ai_feat(i);
    }

    if ((ratio_scores.array() < params_.ratio_epsilon.array()).any()) {
      return 0.0;
    }

    ratio_score = std::pow(ratio_scores.prod(), 1.0 / params_.ratio_feature_dim);
  }

  if (use_angle) {
    angle_score = single_angle_similarity(ai, bi);
    if (angle_score == 0.0) {
      return 0.0;
    }
  }

  double score = 1.0;
  double root = 1.0 / ((use_angle ? 1.0 : 0.0) + (use_ratio ? params_.ratio_weight : 0.0) + (use_cosine ? params_.cosine_weight : 0.0));
  if (use_angle) {
    score *= angle_score;
  }
  if (use_ratio) {
    score *= std::pow(ratio_score, params_.ratio_weight);
  }
  if (use_cosine) {
    score *= std::pow(cosine_score_scaled, params_.cosine_weight);
  }

  return std::pow(score, root);
}

double GeneralSegmentDistance::single_angle_similarity(const Datum& ai, const Datum& bi)
{
  Eigen::VectorXd dir_ai;
  Eigen::VectorXd dir_bi;

  const GeneralSegmentDistance::SegmentType type_ai = static_cast<GeneralSegmentDistance::SegmentType>(static_cast<int>(ai(0)));

  if (type_ai == GeneralSegmentDistance::LINE) {
    const GeneralSegmentDistance::Line line_ai = datum_to_line(ai);
    const GeneralSegmentDistance::Line line_bi = datum_to_line(bi);
    dir_ai = line_ai.direction;
    dir_bi = line_bi.direction;
  } else if (type_ai == GeneralSegmentDistance::PLANE) {
    const GeneralSegmentDistance::Plane plane_ai = datum_to_plane(ai);
    const GeneralSegmentDistance::Plane plane_bi = datum_to_plane(bi);
    dir_ai = plane_ai.normal;
    dir_bi = plane_bi.normal;
  } else {
    throw std::invalid_argument("single_angle_similarity called on non-line/plane segment");
  }

  double diff_angle;

  if (params_.xy_dir_constrained_2d || params_.xyz_dir_constrained) {
    // Full rotation known: compare directions directly
    double dot = dir_ai.dot(dir_bi) / (dir_ai.norm() * dir_bi.norm());
    if (params_.bidirectional) dot = std::abs(dot);
    dot = std::max(-1.0, std::min(1.0, dot));
    diff_angle = std::acos(dot);
  } else {
    // z_dir_constrained: compare elevation angles above horizon
    double z_norm_ai = dir_ai(2) / dir_ai.norm();
    double elev_ai = std::asin(std::max(-1.0, std::min(1.0, z_norm_ai)));
    double z_norm_bi = dir_bi(2) / dir_bi.norm();
    double elev_bi = std::asin(std::max(-1.0, std::min(1.0, z_norm_bi)));

    if (params_.bidirectional) {
      diff_angle = std::abs(std::abs(elev_ai) - std::abs(elev_bi));
    } else {
      diff_angle = std::abs(elev_ai - elev_bi);
    }
  }

  // Score with uncertainty
  double sigma = params_.sigma_angle_rad + params_.rot_unc_ang_rad;
  double epsilon = params_.epsilon_angle_rad + params_.rot_unc_ang_rad;
  if (diff_angle > epsilon) {
    return 0.0;
  }
  return std::exp(-0.5 * diff_angle * diff_angle / (sigma * sigma));
}

  double GeneralSegmentDistance::pairwise_single_dim_sim(const double& val_ai, const double& val_aj, 
    const double& val_bi, const double& val_bj, bool ignore_sign, double noise_inflate) 
  {
    double diff1 = (val_ai - val_aj);
    double diff2 = (val_bi - val_bj);

    if (ignore_sign) {
      diff1 = std::abs(diff1);
      diff2 = std::abs(diff2);
    }

    const double c = std::abs(diff1 - diff2);
    double sigma = SQRT_ONE_HALF*params_.sigma_dist + noise_inflate;
    double epsilon = SQRT_ONE_HALF*params_.epsilon_dist + noise_inflate;

    if (c > epsilon) {
      return 0.0;
    } else {
      return std::exp(-0.5 * c * c / (sigma * sigma));
    }
  }

  double GeneralSegmentDistance::pairwise_point_to_point_sim(const Point& point_ai, const Point& point_aj, 
      const Point& point_bi, const Point& point_bj, bool enforce_min_dist, double noise_inflate) 
  {
    // distance between two points in the same cloud
    const double l1 = (point_ai.point - point_aj.point).norm();
    const double l2 = (point_bi.point - point_bj.point).norm();
    // enforce minimum distance criterion -- if points in the same dataset
    // are too close, then this pair of associations cannot be selected
    if (enforce_min_dist && params_.min_dist > 0 && (l1 < params_.min_dist || l2 < params_.min_dist)) {
      return 0.0;
    }


    // standard distance similarity
    const double c = std::abs(l1 - l2);
    double sigma = params_.sigma_dist + noise_inflate;
    double score = 0.0;
    if (c > params_.epsilon_dist) {
      return 0.0;
    } else {
      score = std::exp(-0.5*c*c/(sigma*sigma));
    }

    // distance similarity score (including gravity-guidance)
    double distance_score = 0.0;
    if (params_.xy_dir_constrained_2d) {
      // per-axis distance differences (2D)
      const double x_diff1 = point_ai.point(0) - point_aj.point(0);
      const double x_diff2 = point_bi.point(0) - point_bj.point(0);
      const double y_diff1 = point_ai.point(1) - point_aj.point(1);
      const double y_diff2 = point_bi.point(1) - point_bj.point(1);

      const double c_x = std::abs(x_diff1 - x_diff2);
      const double c_y = std::abs(y_diff1 - y_diff2);

      double sigma_x = params_.sigma_dist + noise_inflate;
      double sigma_y = params_.sigma_dist + noise_inflate;
      double epsilon_x = params_.epsilon_dist + noise_inflate;
      double epsilon_y = params_.epsilon_dist + noise_inflate;

      if (params_.rot_unc_ang_rad > 0.0) {
        const double x_dist_mean = 0.5 * (std::abs(x_diff1) + std::abs(x_diff2));
        const double y_dist_mean = 0.5 * (std::abs(y_diff1) + std::abs(y_diff2));

        sigma_x += std::abs(x_dist_mean * rot_unc_ang_sin_);
        sigma_y += std::abs(y_dist_mean * rot_unc_ang_sin_);
        epsilon_x += std::abs(x_dist_mean * rot_unc_ang_sin_);
        epsilon_y += std::abs(y_dist_mean * rot_unc_ang_sin_);
      }

      if (c_x > SQRT_ONE_HALF * epsilon_x ||
          c_y > SQRT_ONE_HALF * epsilon_y) {
        return 0.0;
      }

      return std::min(score, std::exp(-0.5 * (c_x*c_x / (sigma_x*sigma_x / 2.0) +
                              c_y*c_y / (sigma_y*sigma_y / 2.0))));

    } else if (params_.xyz_dir_constrained) {
      // per-axis distance differences
      const double x_diff1 = point_ai.point(0) - point_aj.point(0);
      const double x_diff2 = point_bi.point(0) - point_bj.point(0);
      const double y_diff1 = point_ai.point(1) - point_aj.point(1);
      const double y_diff2 = point_bi.point(1) - point_bj.point(1);
      const double z_diff1 = point_ai.point(2) - point_aj.point(2);
      const double z_diff2 = point_bi.point(2) - point_bj.point(2);

      const double c_x = std::abs(x_diff1 - x_diff2);
      const double c_y = std::abs(y_diff1 - y_diff2);
      const double c_z = std::abs(z_diff1 - z_diff2);

      double sigma_x = params_.sigma_dist + noise_inflate;
      double sigma_y = params_.sigma_dist + noise_inflate;
      double sigma_z = params_.sigma_dist + noise_inflate;
      double epsilon_x = params_.epsilon_dist + noise_inflate;
      double epsilon_y = params_.epsilon_dist + noise_inflate;
      double epsilon_z = params_.epsilon_dist + noise_inflate;

      if (params_.rot_unc_ang_rad > 0.0) {
        const double x_dist_mean = 0.5 * (std::abs(x_diff1) + std::abs(x_diff2));
        const double y_dist_mean = 0.5 * (std::abs(y_diff1) + std::abs(y_diff2));
        const double z_dist_mean = 0.5 * (std::abs(z_diff1) + std::abs(z_diff2));

        sigma_x += std::abs(x_dist_mean * rot_unc_ang_sin_);
        sigma_y += std::abs(y_dist_mean * rot_unc_ang_sin_);
        sigma_z += std::abs(z_dist_mean * rot_unc_ang_sin_);
        epsilon_x += std::abs(x_dist_mean * rot_unc_ang_sin_);
        epsilon_y += std::abs(y_dist_mean * rot_unc_ang_sin_);
        epsilon_z += std::abs(z_dist_mean * rot_unc_ang_sin_);
      }

      if (c_x > SQRT_ONE_THIRD * epsilon_x ||
          c_y > SQRT_ONE_THIRD * epsilon_y ||
          c_z > SQRT_ONE_THIRD * epsilon_z) {
        return 0.0;
      }

      return std::exp(-0.5 * (c_x*c_x / (sigma_x*sigma_x / 3.0) +
                              c_y*c_y / (sigma_y*sigma_y / 3.0) +
                              c_z*c_z / (sigma_z*sigma_z / 3.0)));

    } else if (params_.z_dir_constrained) {
      // gravity-guided distance similarity
      const double xy_dist1 = (point_ai.point.head(2) - point_aj.point.head(2)).norm();
      const double xy_dist2 = (point_bi.point.head(2) - point_bj.point.head(2)).norm();
      const double z_diff1 = point_ai.point(2) - point_aj.point(2);
      const double z_diff2 = point_bi.point(2) - point_bj.point(2);

      // consistency score
      const double c_xy = std::abs(xy_dist1 - xy_dist2);
      const double c_z = std::abs(z_diff1 - z_diff2);

      double sigma_xy = params_.sigma_dist + noise_inflate;
      double sigma_z = params_.sigma_dist + noise_inflate;
      double epsilon_xy = params_.epsilon_dist + noise_inflate;
      double epsilon_z = params_.epsilon_dist + noise_inflate;
      
      if (params_.rot_unc_ang_rad > 0.0) {
        const double xy_dist_mean = 0.5*(xy_dist1 + xy_dist2);
        const double z_dist_mean = 0.5*(std::abs(z_diff1) + std::abs(z_diff2));

        // adjust sigma and epsilon based on gravity uncertainty
        sigma_xy += std::abs(xy_dist_mean * rot_unc_ang_cos_ - xy_dist_mean);
        sigma_z += std::abs(z_dist_mean * rot_unc_ang_sin_);
        epsilon_xy += std::abs(xy_dist_mean * rot_unc_ang_cos_ - xy_dist_mean);
        epsilon_z += std::abs(z_dist_mean * rot_unc_ang_sin_);
      }

      if (c_xy > SQRT_TWO_THIRDS*epsilon_xy || c_z > SQRT_ONE_THIRD*epsilon_z) {
        return 0.0;
      } else {
        return std::exp(-0.5*(c_xy*c_xy/(2.0/3.0*sigma_xy*sigma_xy) + 
            c_z*c_z/(sigma_z*sigma_z/3.0)));
      }

    } else {
      // standard distance similarity
      const double c = std::abs(l1 - l2);
      double sigma = params_.sigma_dist + noise_inflate;
      if (c > params_.epsilon_dist) {
        return 0.0;
      } else {
        return std::exp(-0.5*c*c/(sigma*sigma));
      }

    }
    return 0.0;
  }


  double GeneralSegmentDistance::pairwise_point_to_line_sim(const Point& point_ai, const Line& line_aj, 
      const Point& point_bi, const Line& line_bj) 
  {
    // find point on the line j closest to point i
    Point nearest_point_aj = nearest_point_to_line(point_ai, line_aj);
    Point nearest_point_bj = nearest_point_to_line(point_bi, line_bj);

    // compute additional noise inflation from angular uncertainty
    double point_noise_from_angle = 0.0;
    if (params_.point_noise_from_angle) {
      Eigen::VectorXd line_aj_ref = line_aj.point;
      Eigen::VectorXd line_bj_ref = line_bj.point;
      if (line_aj.num_endpoints == 2)
        line_aj_ref = 0.5 * (line_aj.endpoint1 + line_aj.endpoint2);
      if (line_bj.num_endpoints == 2)
        line_bj_ref = 0.5 * (line_bj.endpoint1 + line_bj.endpoint2);

      const double directional_diff_avg = 0.5 * ((line_aj_ref - nearest_point_aj.point).norm() 
                                                + (line_bj_ref - nearest_point_bj.point).norm());
      point_noise_from_angle = directional_diff_avg * std::tan(params_.sigma_angle_rad);
    }

    // TODO: do this for line point to point similarity as well
    Eigen::MatrixXd R_a = Eigen::Matrix2d::Identity();
    R_a << line_aj.direction(0), line_aj.direction(1),
          -line_aj.direction(1), line_aj.direction(0);
    Eigen::MatrixXd R_b = Eigen::Matrix2d::Identity();
    R_b << line_bj.direction(0), line_bj.direction(1),
          -line_bj.direction(1), line_bj.direction(0);

    Eigen::VectorXd p_ai = R_a * point_ai.point.head(2);
    Eigen::VectorXd p_aj = R_a * nearest_point_aj.point.head(2);
    Eigen::VectorXd p_bi = R_b * point_bi.point.head(2);
    Eigen::VectorXd p_bj = R_b * nearest_point_bj.point.head(2);

    double score = 1.0;
    for (int i=0; i<2; i++) {
      score *= pairwise_single_dim_sim(p_ai(i), p_aj(i), p_bi(i), p_bj(i), true, i == 1 ? point_noise_from_angle : 0.0);
    }
    // return score; //comment out
    
    double point_only_score = pairwise_point_to_point_sim(point_ai, nearest_point_aj, point_bi, nearest_point_bj, false, point_noise_from_angle);
    return std::min(score, point_only_score);

  }

  double GeneralSegmentDistance::pairwise_point_to_plane_sim(const Point& point_ai, const Plane& plane_aj, 
      const Point& point_bi, const Plane& plane_bj) 
  {
    // find point on the plane j closest to point i
    Point nearest_point_aj = nearest_point_to_plane(point_ai, plane_aj);
    Point nearest_point_bj = nearest_point_to_plane(point_bi, plane_bj);
    return pairwise_point_to_point_sim(point_ai, nearest_point_aj, point_bi, nearest_point_bj, false);
  }
    

  double GeneralSegmentDistance::pairwise_line_to_line_sim(const Line& line_ai, const Line& line_aj, 
      const Line& line_bi, const Line& line_bj)
  {
    // angle similarity score
    double theta_score = pairwise_angle_sim(
      line_ai.direction, line_aj.direction,
      line_bi.direction, line_bj.direction);
    if (theta_score == 0.0) {
      return 0.0;
    }

    // find maximum distance between the two lines
    // if both infinite, then max_dist is infinite
    // if (maximum_line_distance(line_ai, line_aj) < params_.min_dist ||
    //    maximum_line_distance(line_bi, line_bj) < params_.min_dist) {
    //   return 0.0;
    // }


    // find nearest points between the two lines
    std::tuple<Point, Point> nearest_points_a = nearest_points_on_lines(line_ai, line_aj);
    Point nearest_point_ai = std::get<0>(nearest_points_a);
    Point nearest_point_aj = std::get<1>(nearest_points_a);
    std::tuple<Point, Point> nearest_points_b = nearest_points_on_lines(line_bi, line_bj);
    Point nearest_point_bi = std::get<0>(nearest_points_b);
    Point nearest_point_bj = std::get<1>(nearest_points_b);

    if (violates_line_min_dist(line_ai, line_aj, nearest_point_ai, nearest_point_aj) ||
        violates_line_min_dist(line_bi, line_bj, nearest_point_bi, nearest_point_bj)) {
      return 0.0;
    }

    // compute additional noise inflation from angular uncertainty
    double point_noise_from_angle = 0.0;
    if (params_.point_noise_from_angle) {
      Eigen::VectorXd line_aj_ref = line_aj.point;
      Eigen::VectorXd line_bj_ref = line_bj.point;
      Eigen::VectorXd line_ai_ref = line_ai.point;
      Eigen::VectorXd line_bi_ref = line_bi.point;
      if (line_aj.num_endpoints == 2)
        line_aj_ref = 0.5 * (line_aj.endpoint1 + line_aj.endpoint2);
      if (line_bj.num_endpoints == 2)
        line_bj_ref = 0.5 * (line_bj.endpoint1 + line_bj.endpoint2);
      if (line_ai.num_endpoints == 2)
        line_ai_ref = 0.5 * (line_ai.endpoint1 + line_ai.endpoint2);
      if (line_bi.num_endpoints == 2)
        line_bi_ref = 0.5 * (line_bi.endpoint1 + line_bi.endpoint2);

      const double dir_diff_ai = (line_ai_ref - nearest_point_ai.point).norm();
      const double dir_diff_bi = (line_bi_ref - nearest_point_bi.point).norm();
      const double dir_diff_aj = (line_aj_ref - nearest_point_aj.point).norm();
      const double dir_diff_bj = (line_bj_ref - nearest_point_bj.point).norm();

      const double directional_diff_avg = 0.25 * (dir_diff_ai + dir_diff_bi + dir_diff_aj + dir_diff_bj);
      point_noise_from_angle = directional_diff_avg * std::tan(params_.sigma_angle_rad);
    }

    double distance_score = pairwise_point_to_point_sim(
      nearest_point_ai, nearest_point_aj, nearest_point_bi, nearest_point_bj, false, point_noise_from_angle);
    
    return sqrt(theta_score * distance_score);

  }

  double GeneralSegmentDistance::pairwise_line_to_plane_sim(const Line& line_ai, const Plane& plane_aj, 
      const Line& line_bi, const Plane& plane_bj)
  {
    return pairwise_angle_sim(
      line_ai.direction, plane_aj.normal,
      line_bi.direction, plane_bj.normal);
  }

  double GeneralSegmentDistance::pairwise_plane_to_plane_sim(const Plane& plane_ai, const Plane& plane_aj,
      const Plane& plane_bi, const Plane& plane_bj)
  {
    return pairwise_angle_sim(
      plane_ai.normal, plane_aj.normal,
      plane_bi.normal, plane_bj.normal);
  }

  double GeneralSegmentDistance::pairwise_angle_sim(
      const Eigen::VectorXd& dir_ai, 
      const Eigen::VectorXd& dir_aj, 
      const Eigen::VectorXd& dir_bi, 
      const Eigen::VectorXd& dir_bj)
  {
    const double dot1 = dir_ai.dot(dir_aj);
    const double dot2 = dir_bi.dot(dir_bj);
    const double theta1 = std::acos(params_.bidirectional ? std::abs(dot1) : dot1);
    const double theta2 = std::acos(params_.bidirectional ? std::abs(dot2) : dot2);

    // check min_angle_rad
    if (theta1 < params_.min_angle_rad || theta2 < params_.min_angle_rad) {
      return 0.0;
    }

    // check consistency
    const double diff_theta = std::abs(theta1 - theta2);
    if (diff_theta > params_.epsilon_angle_rad) {
      return 0.0;
    } else {
      return std::exp(-0.5*diff_theta*diff_theta/(params_.sigma_angle_rad*params_.sigma_angle_rad));
    }
  }


  GeneralSegmentDistance::Point GeneralSegmentDistance::datum_to_point(const Datum& datum) 
  {
    Point point;
    point.point = datum.segment(1, params_.dim);
    if (params_.ratio_feature_dim > 0)
      point.ratio_feature = datum.segment(1 + params_.dim, params_.ratio_feature_dim);
    if (params_.cos_feature_dim > 0)
      point.cos_feature = datum.segment(1 + params_.dim + params_.ratio_feature_dim, params_.cos_feature_dim);
    return point;
  }

  GeneralSegmentDistance::Line GeneralSegmentDistance::datum_to_line(const Datum& datum) 
  {
    Line line;
    uint32_t idx = 1;

    line.point = datum.segment(idx, params_.dim);
    idx += params_.dim;
    line.direction = datum.segment(idx, params_.dim);
    idx += params_.dim;

    line.num_endpoints = static_cast<uint32_t>(datum(idx));
    idx += 1;
    if (line.num_endpoints >= 1) {
      line.endpoint1 = datum.segment(idx, params_.dim);
      idx += params_.dim;
    }
    if (line.num_endpoints == 2) {
      line.endpoint2 = datum.segment(idx, params_.dim);
      idx += params_.dim;
    }

    if (params_.ratio_feature_dim > 0)
      line.ratio_feature = datum.segment(idx, params_.ratio_feature_dim);
    idx += params_.ratio_feature_dim;
    
    if (params_.cos_feature_dim > 0)
      line.cos_feature = datum.segment(idx, params_.cos_feature_dim);
    return line;
  }

  GeneralSegmentDistance::Plane GeneralSegmentDistance::datum_to_plane(const Datum& datum) 
  {
    Plane plane;
    uint32_t idx = 1;

    plane.point = datum.segment(idx, params_.dim);
    idx += params_.dim;
    plane.normal = datum.segment(idx, params_.dim);
    idx += params_.dim;

    if (params_.ratio_feature_dim > 0)
      plane.ratio_feature = datum.segment(idx, params_.ratio_feature_dim);
    idx += params_.ratio_feature_dim;

    if (params_.cos_feature_dim > 0)
      plane.cos_feature = datum.segment(idx, params_.cos_feature_dim);
    return plane;
  }


  Datum GeneralSegmentDistance::cos_feature_from_datum(const Datum& datum)
  {
    const GeneralSegmentDistance::SegmentType type_datum = 
      static_cast<GeneralSegmentDistance::SegmentType>(static_cast<int>(datum(0)));
    switch (type_datum)
    {
      case GeneralSegmentDistance::POINT: {
        return datum_to_point(datum).cos_feature;
      } case GeneralSegmentDistance::LINE: {
        return datum_to_line(datum).cos_feature;
      } case GeneralSegmentDistance::PLANE: {
        return datum_to_plane(datum).cos_feature;
        break;
      } default:
        return Datum();
        break;
    }
  }

  Datum GeneralSegmentDistance::ratio_feature_from_datum(const Datum& datum)
  {
    const GeneralSegmentDistance::SegmentType type_datum = 
      static_cast<GeneralSegmentDistance::SegmentType>(static_cast<int>(datum(0)));
    switch (type_datum)
    {
      case GeneralSegmentDistance::POINT: {
        return datum_to_point(datum).ratio_feature;
      } case GeneralSegmentDistance::LINE: {
        return datum_to_line(datum).ratio_feature;
      } case GeneralSegmentDistance::PLANE: {
        return datum_to_plane(datum).ratio_feature;
        break;
      } default:
        return Datum();
        break;
    }
  }

  Datum GeneralSegmentDistance::point_to_datum(const Point& point) 
  {
    uint32_t datum_size = 1 + params_.dim;
    if (params_.ratio_feature_dim > 0)
      datum_size += params_.ratio_feature_dim;
    if (params_.cos_feature_dim > 0)
      datum_size += params_.cos_feature_dim;

    Datum datum = Datum::Zero(datum_size);
    datum(0) = static_cast<double>(GeneralSegmentDistance::SegmentType::POINT);
    datum.segment(1, params_.dim) = point.point;
    if (params_.ratio_feature_dim > 0)
      datum.segment(1 + params_.dim, params_.ratio_feature_dim) = point.ratio_feature;
    if (params_.cos_feature_dim > 0)
      datum.segment(1 + params_.dim + params_.ratio_feature_dim, params_.cos_feature_dim) = point.cos_feature;
    return datum;
  }

  Datum GeneralSegmentDistance::line_to_datum(const Line& line) 
  {
    uint32_t datum_size = 1 + 2*params_.dim + 1; // type, point, direction, num_endpoints
    if (line.num_endpoints >= 1)
      datum_size += params_.dim;
    if (line.num_endpoints == 2)
      datum_size += params_.dim;
    if (params_.ratio_feature_dim > 0)
      datum_size += params_.ratio_feature_dim;
    if (params_.cos_feature_dim > 0)
      datum_size += params_.cos_feature_dim;

    Datum datum = Datum::Zero(datum_size);
    datum(0) = static_cast<double>(GeneralSegmentDistance::SegmentType::LINE);
    uint32_t idx = 1;
    datum.segment(idx, params_.dim) = line.point;
    idx += params_.dim;
    datum.segment(idx, params_.dim) = line.direction;
    idx += params_.dim;
    datum(idx) = static_cast<double>(line.num_endpoints);
    idx += 1;
    if (line.num_endpoints >= 1) {
      datum.segment(idx, params_.dim) = line.endpoint1;
      idx += params_.dim;
    }
    if (line.num_endpoints == 2) {
      datum.segment(idx, params_.dim) = line.endpoint2;
      idx += params_.dim;
    }
    if (params_.ratio_feature_dim > 0)
      datum.segment(idx, params_.ratio_feature_dim) = line.ratio_feature;
    idx += params_.ratio_feature_dim;
    if (params_.cos_feature_dim > 0)
      datum.segment(idx, params_.cos_feature_dim) = line.cos_feature;
    return datum;
  }

  GeneralSegmentDistance::Point GeneralSegmentDistance::nearest_point_to_line(
    const Point& point, const Line& line, const bool on_infinite_line) 
  {
    Point nearest_point;
    nearest_point.point = line.point
      + (line.direction.dot((point.point - line.point))) / line.direction.squaredNorm()
      * line.direction;

    if (on_infinite_line) {
      return nearest_point;
    }

    if (line.num_endpoints == 0) {
      return nearest_point;
    } else if (point_is_on_line(nearest_point, line)) {
      return nearest_point;
    } else {
      // if the nearest point is not on the line segment, then return the nearest endpoint
      if (line.num_endpoints == 1) {
        nearest_point.point = line.endpoint1;
      } else if (line.num_endpoints == 2) {
        double dist_to_ep1 = (point.point - line.endpoint1).squaredNorm();
        double dist_to_ep2 = (point.point - line.endpoint2).squaredNorm();
        nearest_point.point = dist_to_ep1 < dist_to_ep2 ? line.endpoint1 : line.endpoint2;
      } else {
        throw std::runtime_error("Line has invalid number of endpoints");
      }
      return nearest_point;
    }
  }

  GeneralSegmentDistance::Point GeneralSegmentDistance::nearest_point_to_plane(
    const Point& point, const Plane& plane) 
  {
    GeneralSegmentDistance::Point nearest_point;
    double dist_to_plane = (point.point - plane.point).dot(plane.normal);
    nearest_point.point = point.point - dist_to_plane * plane.normal;
    return nearest_point;
  }

  std::tuple<GeneralSegmentDistance::Point, GeneralSegmentDistance::Point> 
    GeneralSegmentDistance::nearest_points_on_lines(
    const Line& line1, const Line& line2)
  {
    Point point1;
    Point point2;

    if (is_parallel(line1.direction, line2.direction)) {
      // lines are parallel, so just find nearest point on line2 to point on line1 (if no endpoints)
      if (std::min(line1.num_endpoints, line2.num_endpoints) == 0) {
        point1.point = line1.point;
        point2 = nearest_point_to_line(point1, line2);
      } else {        
        // both lines have at least one endpoint, so just find the minimum of all the endpoint combos
        std::vector<Point> candidates1;
        std::vector<Point> candidates2;
        // add each endpoint and its nearest point on the other line

        for (uint32_t i = 0; i < line1.num_endpoints; i++) {
          Point l1_ep;
          l1_ep.point = (i == 0) ? line1.endpoint1 : line1.endpoint2;
          candidates1.push_back(l1_ep);
          candidates2.push_back(nearest_point_to_line(l1_ep, line2));
        }
        for (uint32_t j=0; j<line2.num_endpoints; j++) {
          Point l2_ep;
          l2_ep.point = (j == 0) ? line2.endpoint1 : line2.endpoint2;
          candidates2.push_back(l2_ep);
          candidates1.push_back(nearest_point_to_line(l2_ep, line1));
        }

        double min_dist_sq = std::numeric_limits<double>::max();
        uint32_t min_idx = 0;
        for (size_t i = 0; i < candidates1.size(); i++) {
          double dist_sq = (candidates1[i].point - candidates2[i].point).squaredNorm();
          if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            min_idx = i;
          }
        }
        point1 = candidates1.at(min_idx);
        point2 = candidates2.at(min_idx);

      }
    } else {
      // lines are not parallel, so find nearest points using formula
      Eigen::VectorXd t = Eigen::VectorXd::Zero(params_.dim);

      if (params_.dim == 2) {
        // 2D case
        Eigen::MatrixXd A(2,2);
        A.col(0) = line1.direction;
        A.col(1) = -line2.direction;
        Eigen::VectorXd b = line2.point - line1.point;

        t = A.inverse() * b;
      } else if (params_.dim == 3) {
        Eigen::Vector3d line1_dir = line1.direction;
        Eigen::Vector3d line2_dir = line2.direction;
        // 3D case
        // solve the system derived in user2255770's answer from 
        // StackExchange: https://math.stackexchange.com/q/1993990
        Eigen::MatrixXd A(3,3);
        Eigen::VectorXd perp(3);
        perp = line1_dir.cross(line2_dir);
        perp = perp / perp.norm();
        A.col(0) = line1_dir;
        A.col(1) = -line2_dir;
        A.col(2) = perp;
        Eigen::VectorXd b = line2.point - line1.point;
        t = A.inverse() * b;
      } else {
        throw std::runtime_error("Nearest points between lines only implemented for 2D/3D");
      }
      point1.point = line1.point + t(0) * line1.direction;
      point2.point = line2.point + t(1) * line2.direction;
    }

    if (point_is_on_line(point1, line1) && point_is_on_line(point2, line2)) {
      return std::make_tuple(point1, point2);
    }

    // closest points are not on both line segments, so check endpoints
    std::vector<Point> candidates1;
    std::vector<Point> candidates2;
    // add each endpoint and its nearest point on the other line
    if (line1.num_endpoints >= 1) {
      Point l1_ep1;
      l1_ep1.point = line1.endpoint1;
      candidates1.push_back(l1_ep1);
      candidates2.push_back(nearest_point_to_line(l1_ep1, line2));
    }
    if (line1.num_endpoints == 2) {
      Point l1_ep2;
      l1_ep2.point = line1.endpoint2;
      candidates1.push_back(l1_ep2);
      candidates2.push_back(nearest_point_to_line(l1_ep2, line2));
    }
    if (line2.num_endpoints >= 1) {
      Point l2_ep1;
      l2_ep1.point = line2.endpoint1;
      candidates2.push_back(l2_ep1);
      candidates1.push_back(nearest_point_to_line(l2_ep1, line1));
    }
    if (line2.num_endpoints == 2) {
      Point l2_ep2;
      l2_ep2.point = line2.endpoint2;
      candidates2.push_back(l2_ep2);
      candidates1.push_back(nearest_point_to_line(l2_ep2, line1));
    }
    double min_dist_sq = std::numeric_limits<double>::max();
    uint32_t min_idx = 0;
    for (size_t i=0; i<candidates1.size(); i++) {
      double dist_sq = (candidates1[i].point - candidates2[i].point).squaredNorm();
      if (dist_sq < min_dist_sq) {
        min_dist_sq = dist_sq;
        min_idx = i;
      }
    }
    point1 = candidates1.at(min_idx);
    point2 = candidates2.at(min_idx);

    return std::make_tuple(point1, point2);
  }

  bool GeneralSegmentDistance::point_is_on_line(const Point& point, const Line& line) 
  {
    const Point nearest_point_on_infinite_line = nearest_point_to_line(point, line, true);
    bool on_infinite_line = (point.point - nearest_point_on_infinite_line.point).norm()
       < params_.nearby_eps;

    // reject immediately if not on infinite line
    if (!on_infinite_line) {
      return false;
    }

    // if no endpoints, then point is on the line
    if (line.num_endpoints == 0) {
      return true;
    }

    // Check if the point is within the line segment (if endpoints are defined)
    if (line.num_endpoints == 2) {
      Eigen::VectorXd ep1_to_ep2 = line.endpoint2 - line.endpoint1;
      Eigen::VectorXd ep1_to_point = point.point - line.endpoint1;
      double t = ep1_to_point.dot(ep1_to_ep2);
      return t >= 0 && t <= ep1_to_ep2.squaredNorm();
    } else if (line.num_endpoints == 1) {
      Eigen::VectorXd ep1_to_point = point.point - line.endpoint1;
      double t = ep1_to_point.dot(line.direction);
      return t >= 0;
    }

    return true;
  }

  // double GeneralSegmentDistance::maximum_line_distance(const Line& line1, const Line& line2) 
  // {
  //   if (line1.num_endpoints != 2 && line2.num_endpoints != 2) {
  //     return std::numeric_limits<double>::infinity();
  //   }

  //   std::vector<double> distances;
  //   if (line1.num_endpoints > 0) {
  //     distances.push_back((line1.endpoint1 -  nearest_point_to_line(Point{line1.endpoint1}, line2).point).norm());
  //   }
  //   if (line1.num_endpoints == 2) {
  //     distances.push_back((line1.endpoint2 -  nearest_point_to_line(Point{line1.endpoint2}, line2).point).norm());
  //   }
  //   if (line2.num_endpoints > 0) {
  //     distances.push_back((line2.endpoint1 -  nearest_point_to_line(Point{line2.endpoint1}, line1).point).norm());
  //   }
  //   if (line2.num_endpoints == 2) {
  //     distances.push_back((line2.endpoint2 -  nearest_point_to_line(Point{line2.endpoint2}, line1).point).norm());
  //   }
  //   // distances.push_back((line1.endpoint1 -  nearest_point_to_line(Point{line1.endpoint1}, line2).point).norm());
  //   // distances.push_back((line1.endpoint2 -  nearest_point_to_line(Point{line1.endpoint2}, line2).point).norm());
  //   // distances.push_back((line2.endpoint1 -  nearest_point_to_line(Point{line2.endpoint1}, line1).point).norm());
  //   // distances.push_back((line2.endpoint2 -  nearest_point_to_line(Point{line2.endpoint2}, line1).point).norm());

  //   return *std::max_element(distances.begin(), distances.end());
  // }

  bool GeneralSegmentDistance::violates_line_min_dist(const Line& line_i, const Line& line_j, 
      const Point& nearest_point_i, const Point& nearest_point_j) 
  {
    if (params_.min_dist <= 0.0 || params_.min_angle_rad <= 0.0) {
      return false;
    }

    // check whether angle is < min angle
    const double dot = line_i.direction.dot(line_j.direction);
    const double theta = std::acos(params_.bidirectional ? std::abs(dot) : dot);
    if (theta >= params_.min_angle_rad) {
      return false;
    }

    // if both lines are infinite, check the distance between the lines at their reference points
    if (line_i.num_endpoints == 0 && line_j.num_endpoints == 0) {
      const double dist_at_ref_point_i = (line_i.point - nearest_point_to_line(Point{line_i.point}, line_j, true).point).norm();
      const double dist_at_ref_point_j = (line_j.point - nearest_point_to_line(Point{line_j.point}, line_i, true).point).norm();
      return std::min(dist_at_ref_point_i, dist_at_ref_point_j) < params_.min_dist;
    } else {
      // if either line has endpoints, check the distance between the nearest points
      return (nearest_point_i.point - nearest_point_j.point).norm() < params_.min_dist;
    }

  }

} // ns invariants
} // ns clipper
