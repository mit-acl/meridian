import numpy as np
import clipperpy
from dataclasses import dataclass, field

@dataclass
class SegmentMatchParams:

    dim: int = 3;                                   # dimension of points (2 or 3)
    ratio_feature_dim: int = 0;                     # number of ratio features (e.g., volume)
    cos_feature_dim: int = 0;                       # number of features used for cosine similarity
    sigma_dist: float = 0.4;                        # spread / "variance" of exponential kernel
    epsilon_dist: float = 0.6;                      # bound on consistency score, determines if inlier/outlier
    min_dist: float = 0.0;                          # minimum allowable distance between inlier points in the same dataset
    sigma_angle_rad: float = np.deg2rad(10.0);      # spread / "variance" of exponential kernel
    epsilon_angle_rad: float = np.deg2rad(20.0);    # bound on consistency score, determines if inlier/outlier
    min_angle_rad: float = 0.0;                     # minimum allowable angle (in radians) between inlier segments in the same dataset
    distance_weight: float = 1.0;                   # weight of pairwise similarity in single/pairwise fusion
    ratio_weight: float = 1.0;                      # weight of cosine similarity in single similarity fusion
    cosine_weight: float = 1.0;                     # weight of cosine similarity in single similarity fusion
    ratio_epsilon: np.ndarray = \
        field(default_factory=lambda: np.zeros(0)); # bound on feature ratio score, determines if inlier/outlier
    cosine_min: float = 0.5;                        # cosine similarity scaled so that cosine_min maps to 0.0 similarity score
    cosine_max: float = 0.7;                        # cosine similarity scaled so that cosine_max maps to 1.0 similarity score
    gravity_guided: bool = False;                   # whether to use gravity-guided prior
    gravity_unc_ang_rad: float = 0.0;               # uncertainty adjustment for gravity direction in radians

    def to_clipper(self,):
        iparams = clipperpy.invariants.GeneralSegmentDistanceParams()
        iparams.dim = self.dim
        iparams.ratio_feature_dim = self.ratio_feature_dim
        iparams.cos_feature_dim = self.cos_feature_dim
        iparams.sigma_dist = self.sigma_dist
        iparams.epsilon_dist = self.epsilon_dist
        iparams.min_dist = self.min_dist
        iparams.sigma_angle_rad = self.sigma_angle_rad
        iparams.epsilon_angle_rad = self.epsilon_angle_rad
        iparams.min_angle_rad = self.min_angle_rad
        iparams.distance_weight = self.distance_weight
        iparams.ratio_weight = self.ratio_weight
        iparams.cosine_weight = self.cosine_weight
        iparams.ratio_epsilon = self.ratio_epsilon
        iparams.cosine_min = self.cosine_min
        iparams.cosine_max = self.cosine_max
        iparams.gravity_guided = self.gravity_guided
        iparams.gravity_unc_ang_rad = self.gravity_unc_ang_rad
        return iparams