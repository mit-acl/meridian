import numpy as np
import torch
from dataclasses import dataclass, field
from typing import ClassVar, Optional
from meridian.params.params_base import ParamsBase


@dataclass
class RegisterParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "register"

    # Parameters
    only_use_points: bool = False
    use_gravity: bool = True

    point_weight: float = 1.0
    line_direction_weight: float = 1.0
    line_moment_weight: float = 1.0
    plane_normal_weight: float = 1.0
    plane_offset_weight: float = 1.0
    gravity_weight: float = 1.0

    eps: Optional[float] = 0.05
    dup_eps: Optional[float] = 0.05
    run_gd: Optional[bool] = False
    device: Optional[torch.device] = "cpu"

    refine_transforms_kwargs: dict = None

    # Hypothesis clustering params (for multi-hypothesis matching)
    cluster_trans_thresh_m: float = 1.5
    cluster_rot_thresh_deg: float = 5.0
    max_hypotheses: int = 25  # 0 = no limit

    # Compute full-submap alignment fitness per hypothesis, rank by it, and make
    # it available downstream (see CrossViewRPGOParams.lc_score_method).
    compute_fitness: bool = True
    # Score this many count-ranked clusters with fitness before keeping the
    # best max_hypotheses of them. 0 = score all.
    fitness_prescreen_hypotheses: int = 50
    fitness_point_inlier_thresh_m: float = 1.0
    fitness_line_inlier_thresh_m: float = 2.0
    fitness_line_angle_thresh_deg: float = 5.0
    fitness_line_min_overlap: float = 0.5
    # Wilson lower-bound confidence (standard errors; 3.29 = two-sided 99.9 %)
    fitness_wilson_z: float = 3.29
    # Fewer in-patch ground primitives than this scores 0.
    fitness_min_in_patch: int = 10
    # Fewer inliers than this scores 0 and skips refinement.
    fitness_min_inliers: int = 3

    # ICP re-fit of each returned hypothesis to its own fitness inliers. 
    # Requires compute_fitness.
    refine_hypotheses: bool = True
    refine_max_iters: int = 3
    # Reject a refinement moving the patch center further than this. 0 = uncapped.
    refine_max_correction_m: float = 3.0

    def __post_init__(self):
        if self.refine_transforms_kwargs is None:
            self.refine_transforms_kwargs = {}
