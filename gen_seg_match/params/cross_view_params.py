from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gen_seg_match.params.params_base import ParamsBase


@dataclass
class CrossViewPlaceRecognitionParams(ParamsBase):
    params_key: ClassVar[str] = "cross_view_place_recognition"

    method: str = "semantic-gem"  # "semantic-gem" or "semantic-point-line"
    ground_descriptor_dist_m: float = 5.0
    k_nearest_neighbors: int = 5


@dataclass
class CrossViewRPGOParams(ParamsBase):
    params_key: ClassVar[str] = "cross_view_rpgo"

    # CLIPPER consistency
    rot_consistency_sigma_deg: float = 10.0
    rot_consistency_eps_deg: float = 20.0
    trans_consistency_sigma_m: float = 5.0
    trans_consistency_eps_m: float = 10.0
    min_num_associations: int = 3

    # Optimization method
    optimization_method: str = "pgo"  # "frame_align" or "pgo"

    # PGO noise parameters
    odom_trans_sigma_m: float = 0.1
    odom_rot_sigma_deg: float = 0.5
    prior_trans_sigma_m: float = 5.0
    prior_rot_sigma_deg: float = 10.0

    @property
    def rot_consistency_sigma_rad(self) -> float:
        return np.deg2rad(self.rot_consistency_sigma_deg)

    @property
    def rot_consistency_eps_rad(self) -> float:
        return np.deg2rad(self.rot_consistency_eps_deg)

    @property
    def odom_rot_sigma_rad(self) -> float:
        return np.deg2rad(self.odom_rot_sigma_deg)

    @property
    def prior_rot_sigma_rad(self) -> float:
        return np.deg2rad(self.prior_rot_sigma_deg)
