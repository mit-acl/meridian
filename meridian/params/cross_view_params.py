import os
import warnings
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import numpy as np
import yaml

from meridian.params.params_base import ParamsBase


@dataclass
class CrossViewMatchingParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "cross_view_matching"

    matching_mode: str = "vpr"  # "all", "gt", "vpr", or "max_intersection"
    max_intersection_patches_per_ground_sm: int = 2

    match_min_len_m: float = 2.0

    ground_dist_from_aerial_patch_center_m: float = 25.0

    # Use CLIPPER instead of Langevin for pass 2 (rerun with known rotation)
    clipper_pass2: bool = False

    translation_only: bool = False
    rot_bias_deg: float = 0.0
    uniform_rot_noise_bounds_deg: Tuple[float, float] = (0.0, 0.0)

    points_only: bool = False
    lines_only: bool = False

    match_trans_err_m: float = 5.0
    match_rot_err_deg: float = 10.0

    # Compute full-submap alignment fitness per hypothesis, rank by it, and make
    # it available downstream (see CrossViewRPGOParams.lc_score_method).
    compute_fitness: bool = True
    # Score this many count-ranked clusters with fitness before keeping the
    # best RegisterParams.max_hypotheses of them. 0 = score all.
    fitness_prescreen_hypotheses: int = 40
    fitness_point_inlier_thresh_m: float = 1.0
    fitness_line_inlier_thresh_m: float = 1.5
    fitness_line_angle_thresh_deg: float = 5.0
    fitness_line_min_overlap: float = 0.25
    fitness_min_in_patch: int = 10
    # Wilson lower-bound confidence (standard errors)
    fitness_wilson_z: float = 2.576

    # ICP re-fit of each returned hypothesis to its own fitness inliers, for
    # sub-meter alignment. Requires compute_fitness; ~1 fitness eval per iter.
    refine_hypotheses: bool = False
    refine_max_iters: int = 3
    # Below this many correspondences the hypothesis is left alone.
    refine_min_inliers: int = 6
    # Reject a refinement moving the patch center further than this. 0 = uncapped.
    refine_max_correction_m: float = 2.0


@dataclass
class CrossViewVisualizationParams(ParamsBase):
    params_key: ClassVar[str] = "cross_view_visualization"

    aerial_viz_pixel_size_m: float = 0.05
    aerial_viz_line_width_m: float = 0.2
    aerial_viz_target_size_kb: int = 200
    match_viz_target_size_kb: int = 200

    # Trajectory plot colors
    estimated_trajectory_color: str = "#fa5ff7"  # light magenta
    gt_trajectory_color: str = "#89fe05"  # lime green

# Values `frame_descriptor` accepts, on either segmenter.
# Pooled from the segmenter's DINO patch features
POOLED_DESCRIPTORS = ("dino-gap", "dino-gmp", "dino-gem")
# Own model, run on the raw image
STANDALONE_DESCRIPTORS = ("anyloc", "meridian-vpr", "salad")
FRAME_DESCRIPTORS = POOLED_DESCRIPTORS + STANDALONE_DESCRIPTORS

COMPARISONS = ("image", "semantic-point-line")

# Old `method:` values; every image one only ever selected "image".
_LEGACY_METHODS = {name: "image" for name in FRAME_DESCRIPTORS}
_LEGACY_METHODS["semantic-point-line"] = "semantic-point-line"


@dataclass
class CrossViewPlaceRecognitionParams(ParamsBase):
    params_key: ClassVar[str] = "cross_view_place_recognition"

    # What to compare. One of COMPARISONS.
    comparison: str = "image"
    ground_descriptor_dist_m: float = 5.0
    k_nearest_neighbors: int = 25

    method: Optional[str] = None  # deprecated spelling of `comparison`

    def __post_init__(self):
        if self.method is not None:
            mapped = _LEGACY_METHODS.get(self.method)
            if mapped is None:
                raise ValueError(f"Unknown legacy method: {self.method!r}")
            warnings.warn(
                f"`method: {self.method}` is deprecated; use "
                f"`comparison: {mapped}`. The model comes from "
                "frame_descriptor, not from here.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.comparison = mapped
            self.method = None
        if self.comparison not in COMPARISONS:
            raise ValueError(
                f"comparison={self.comparison!r} must be one of {COMPARISONS}."
            )


@dataclass
class CrossViewRPGOParams(ParamsBase):
    params_key: ClassVar[str] = "cross_view_rpgo"

    # CLIPPER consistency
    rot_consistency_sigma_deg: float = 5.0
    rot_consistency_eps_deg: float = 5.0
    trans_consistency_sigma_m: float = 2.0
    trans_consistency_eps_m: float = 2.0
    added_trans_noise_m_per_m: float = 0.05
    added_rot_noise_deg_per_m: float = 0.02
    single_lc_per_ground_sm: bool = True
    single_lc_per_ground_aerial_pair: bool = True
    fuse_lc_score: bool = True
    # "frequency-ratio" (particle count) or "fitness-ratio" (alignment fitness),
    # both normalized per aerial-ground pair, or "fitness-norm" (fitness scaled
    # by lc_score_fitness_ref, so it stays comparable across submaps). The
    # fitness methods require CrossViewMatchingParams.compute_fitness.
    lc_score_method: str = "fitness-norm"
    # Fitness mapping to a score of 1.0. Raw fitness tops out near 0.29, which
    # sits far below CLIPPER's unit diagonal and collapses the densest clique.
    lc_score_fitness_ref: float = 0.3
    min_num_associations: int = 3
    min_num_associations_rerun: Optional[int] = None

    # Optimization method
    optimization_method: str = "pgo"  # "frame_align" or "pgo"

    # PGO noise parameters
    odom_trans_sigma_m: float = 0.05
    odom_rot_sigma_deg: float = 0.05
    prior_trans_sigma_m: float = 2.0
    prior_rot_sigma_deg: float = 5.0

    rerun_match_with_known_rot: bool = True

    # GT inlier selection (bypasses CLIPPER)
    gt_inliers: bool = False
    gt_inliers_rot_err_deg: float = 5.0
    gt_inliers_trans_err_m: float = 1.0

    # initialize pose only with n submap registrations - this gaurds the
    # loop closure rejection from growing too large
    outlier_rejection_max_num_lcs: Optional[int] = 100_000

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


@dataclass
class CrossViewIncrementalParams(ParamsBase):
    params_key: ClassVar[str] = "cross_view_incremental"

    consistent_loop_closure_thresh: int = 1
    rot_constrained_consistent_lc_thresh: int = 6
    rot_constrained_consistent_lc_frac: float = 0.5

    # POST-state guard: if the loop-closure outlier-rejection objective
    # (u^T M u / u^T u) drops by more than this versus the last accepted POST
    # objective, reject the new solve and keep the previous lastopt anchors.
    # Pipeline continues to the next ground submap.
    allowable_outlier_lc_obj_drop: float = np.inf

    # Abort the incremental run when the live instantaneous translation error
    # (vs data.gt_pose_data) exceeds this many meters. Saves whatever outputs
    # exist and exits. Only active when gt_pose_data is set.
    early_termination_err_m: float = 20.0

    # If True, once we have accepted a loop closure as an inlier in the
    # post-global-localization mode, then all other corresponding submap
    # potential loop closures are removed from the outlier rejection processing
    commit_accepted_inliers: bool = True

    # If > 0, then the most recent n loop closures will not be committed to.
    # In other words, all hypotheses will remain for the n most recent lcs,
    # until newer loop closure distributions have been acquired.
    delay_most_recent_lc_commit_num: int = 2


def check_frame_descriptors_match(params_source, comparison=None, run=None):
    """Ground and aerial descriptors must come from the same model.

    Args:
        params_source: params YAML path or directory
        comparison: CrossViewPlaceRecognitionParams.comparison, if known
        run: run key within the YAML

    Returns:
        The agreed frame_descriptor, to stamp onto whatever it produces, or
        None if neither block is present.
    """
    from meridian.params.segmenter_params import (
        AerialSegmenterParams,
        SegmenterParams,
    )

    found = {
        key: cls.load(params_source, run=run).frame_descriptor
        for key, cls in (
            ("segmenter", SegmenterParams),
            ("aerial_segmenter", AerialSegmenterParams),
        )
        if _block_present(params_source, key)
    }
    if len(set(found.values())) > 1:
        pairs = ", ".join(f"{k}.frame_descriptor={v!r}" for k, v in found.items())
        raise ValueError(
            f"{pairs}; cross-view similarity between two different models is "
            "meaningless."
        )
    if not found:
        return None

    descriptor = next(iter(found.values()))
    if comparison == "image" and descriptor is None:
        raise ValueError(
            "cross_view_place_recognition.comparison='image' compares frame "
            "descriptors, but frame_descriptor is unset; every similarity "
            "would be nan."
        )
    return descriptor


def _block_present(params_source, key):
    """Whether `key` is configured in `params_source` at all."""
    if os.path.isdir(params_source):
        return os.path.exists(os.path.join(params_source, f"{key}.yaml"))
    with open(params_source, "r") as f:
        return key in (yaml.full_load(f) or {})
