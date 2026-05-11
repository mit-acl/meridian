import numpy as np
import clipperpy
from dataclasses import dataclass, field
from typing import ClassVar
from meridian.params.params_base import ParamsBase


@dataclass
class LangevinMatcherParams(ParamsBase):
    """Parameters for the Langevin dynamics multi-hypothesis solver."""

    params_key: ClassVar[str] = "langevin"

    n_particles: int = 1000
    n_iter: int = 1000
    step_size: float = 1.0
    adagrad: bool = True
    alpha: float = 0.9
    device: str = "cuda"
    min_associations: int = 3
    anneal_noise: bool = True
    no_noise: bool = False
    early_stop: bool = True
    check_interval: int = 10
    obj_tol: float = 1e-5
    patience: int = 3
    jaccard_pruning: bool = False
    jaccard_thresh: float = 0.3


@dataclass
class PrimitiveMatchParams(ParamsBase):
    """
    PrimitiveMatchParams: dataclass for primitive matching parameters.

    Args:
        dim (int = 3): dimension of points (2 or 3).
        ratio_feature_dim (int = 0): number of ratio features (e.g., volume)
        cos_feature_dim (int = 0): number of features used for cosine similarity.
        sigma_dist (float = 0.4): spread / "variance" of exponential kernel for distance.
        epsilon_dist (float = 0.6): bound on consistency score for distance,
            determines if inlier/outlier.
        min_dist (float = 0.0): minimum allowable distance between inlier points in the same
            dataset.
        sigma_angle_rad (float = np.deg2rad(10.0)): spread / "variance" of exponential kernel
            for angle.
        epsilon_angle_rad (float = np.deg2rad(20.0)): bound on consistency score for angle,
            determines if inlier/outlier.
        min_angle_rad (float = 0.0): minimum allowable angle (in radians) between inlier segments
            in the same dataset.
        distance_weight (float = 1.0): weight of pairwise similarity in single/pairwise fusion.
        ratio_weight (float = 1.0): weight of ratio similarity in single similarity fusion.
        cosine_weight (float = 1.0): weight of cosine similarity in single similarity fusion.
        ratio_epsilon (np.ndarray = np.zeros(0)): bound on feature ratio score.
        cosine_min (float = 0.5): cosine similarity scaled so that cosine_min maps to 0.0
            similarity score.
        cosine_max (float = 0.7): cosine similarity scaled so that cosine_max maps to 1.0
            similarity score.
        z_dir_constrained (bool = False): whether to use z-direction (gravity) constrained prior.
        xyz_dir_constrained (bool = False): whether to use rotation prior in all directions (3D only).
        xy_dir_constrained_2d (bool = False): whether to use rotation prior in all directions (2D only).
        rot_unc_ang_rad (float = 0.0): uncertainty adjustment for known rotation direction in radians.
    """

    # class attribute
    params_key: ClassVar[str] = "primitive_match"

    # Parameters
    dim: int = 3
    ratio_feature_dim: int = 0
    cos_feature_dim: int = 1024
    sigma_dist: float = 2.0
    epsilon_dist: float = 2.0
    min_dist: float = 3.0
    sigma_angle_rad: float = np.deg2rad(20.0)
    epsilon_angle_rad: float = np.deg2rad(20.0)
    min_angle_rad: float = np.deg2rad(10.0)
    distance_weight: float = 1.0
    ratio_weight: float = 1.0
    cosine_weight: float = 1.0
    ratio_epsilon: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cosine_min: float = 0.1
    cosine_max: float = 0.5
    z_dir_constrained: bool = False
    xyz_dir_constrained: bool = False
    xy_dir_constrained_2d: bool = False
    rot_unc_ang_rad: float = np.deg2rad(20.0)
    point_noise_from_angle: bool = True
    k_nearest_neighbors: int = 10  # None = all-to-all

    solver: str = "langevin"  # "clipper" or "langevin"
    langevin_params: LangevinMatcherParams = None

    def __post_init__(self):
        if isinstance(self.langevin_params, dict):
            self.langevin_params = LangevinMatcherParams(**self.langevin_params)
        elif self.langevin_params is None and self.solver == "langevin":
            self.langevin_params = LangevinMatcherParams()

    def to_clipper(
        self,
    ):
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
        iparams.z_dir_constrained = self.z_dir_constrained
        iparams.xyz_dir_constrained = self.xyz_dir_constrained
        iparams.xy_dir_constrained_2d = self.xy_dir_constrained_2d
        iparams.rot_unc_ang_rad = self.rot_unc_ang_rad
        iparams.point_noise_from_angle = self.point_noise_from_angle
        return iparams
