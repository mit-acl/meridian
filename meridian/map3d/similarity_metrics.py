###########################################################
#
# similarity_metrics.py
#
# Classes with methods to calculate similarity metrics
#
# Authors: Qingyuan Li
#
# January 27. 2025
#
###########################################################

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from meridian.primitive.primitive import PointPrimitive, LinePrimitive
from meridian.primitive.primitive_list import PrimitiveList


@dataclass
class AlignmentFitnessResult:
    """Result of an ICP-style full-submap alignment fitness evaluation.

    ``fitness`` is a support-aware inlier ratio in [0, 1] obtained under the
    estimated transform, restricted to the aerial patch footprint so
    non-overlapping ground primitives are not counted as outliers. It refines
    the plain Open3D-style inlier ratio by replacing the raw fraction with its
    Wilson lower confidence bound, so a handful of lucky matches (e.g. 1/1)
    cannot outrank a well-supported hypothesis (e.g. 45/70). The inlier
    thresholds alone separate signal from noise; residual magnitude within the
    band is not graded.

    ``fitness = wilson_lb(n_inliers, n_in_patch)``.

    The raw ``inlier_ratio`` and ``mean_inlier_quality`` are kept for debugging.
    """

    fitness: float
    n_inliers: int
    n_in_patch: int
    inlier_ratio: float = 0.0
    mean_inlier_quality: float = 0.0
    inlier_pairs: List[Tuple[int, int]] = field(default_factory=list)  # (ground_id, aerial_id)
    patch_bounds: Optional[Tuple[float, float, float, float]] = None  # xmin, ymin, xmax, ymax


_LINE_EXTENT_M = 1e5  # half-length used to treat infinite lines/rays as long segments


def _wilson_lower_bound(k: int, n: int, z: float) -> float:
    """Lower bound of the Wilson score interval for a binomial proportion k/n.

    For a fixed ratio this shrinks toward 0 as ``n`` shrinks, so small-support
    hypotheses are penalized relative to well-supported ones with the same
    ratio. Returns 0 for n == 0."""
    if n <= 0:
        return 0.0
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2.0 * n)
    margin = z * np.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)
    return float(max(0.0, (center - margin) / denom))


def _assert_2d(primitives: PrimitiveList, name: str) -> None:
    """Fitness is defined only in the 2D matching frame. Reject 3D primitives
    rather than silently dropping z (which would mix 2D and 3D geometry)."""
    for seg in primitives:
        if seg.dim != 2:
            raise ValueError(
                f"AlignmentFitness operates in the 2D matching frame, but "
                f"{name} contains a dim={seg.dim} primitive; project with "
                f".to_dim(2) first."
            )
        break


def _line_angle(a: LinePrimitive, b: LinePrimitive) -> float:
    """Unsigned angle in [0, pi/2] between two lines, folded so that a line and
    its direction-reversed twin (same geometric line) are treated as parallel."""
    ang = a.angle_between(b)
    return min(ang, np.pi - ang)


def _aerial_patch_bounds(
    aerial_segments: PrimitiveList,
) -> Optional[Tuple[float, float, float, float]]:
    """Axis-aligned bounds (xmin, ymin, xmax, ymax) of the aerial submap's
    coverage, taken from point coordinates and finite line endpoints. Infinite
    lines (no endpoints) are skipped since they don't bound a region."""
    xs: List[float] = []
    ys: List[float] = []
    for seg in aerial_segments.get_points():
        p = seg.get_point()
        xs.append(float(p[0]))
        ys.append(float(p[1]))
    for seg in aerial_segments.get_lines():
        for ep in seg.endpoints:
            if ep is not None:
                xs.append(float(ep[0]))
                ys.append(float(ep[1]))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _line_as_segment(line: LinePrimitive) -> Tuple[np.ndarray, np.ndarray]:
    """Represent a line/ray/segment as a finite (p0, p1) segment so a single
    segment-vs-box clip handles all endpoint configurations."""
    d = line.get_direction().flatten()[:2]
    if line.endpoints[0] is not None and line.endpoints[1] is not None:
        return line.endpoints[0][:2], line.endpoints[1][:2]
    if line.num_endpoints == 1:
        ep = line.endpoints[0] if line.endpoints[0] is not None else line.endpoints[1]
        ep = ep[:2]
        return ep, ep + d * _LINE_EXTENT_M
    pt = line.get_point().flatten()[:2]
    return pt - d * _LINE_EXTENT_M, pt + d * _LINE_EXTENT_M


def _segment_intersects_box(
    p0: np.ndarray, p1: np.ndarray, bounds: Tuple[float, float, float, float]
) -> bool:
    """Liang-Barsky: True if segment p0->p1 at least partly lies in the box."""
    xmin, ymin, xmax, ymax = bounds
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    p = [-dx, dx, -dy, dy]
    q = [p0[0] - xmin, xmax - p0[0], p0[1] - ymin, ymax - p0[1]]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            if qi < 0.0:  # parallel to this edge and outside the slab
                return False
        else:
            r = qi / pi
            if pi < 0.0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    return t0 <= t1


def _primitive_in_patch(
    seg, bounds: Tuple[float, float, float, float]
) -> bool:
    xmin, ymin, xmax, ymax = bounds
    if isinstance(seg, PointPrimitive):
        p = seg.get_point()
        return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax
    p0, p1 = _line_as_segment(seg)
    return _segment_intersects_box(p0, p1, bounds)


class AlignmentFitness:
    """ICP-style full-submap alignment fitness for point + line primitive maps.

    The ground submap is transformed into the aerial frame by the estimated
    transform and scored against the *full* aerial submap (not just the matched
    inliers). Only ground primitives that at least partly lie within the aerial
    patch are counted, so partial overlap is not penalized. Ground points are
    matched to aerial points and ground lines to aerial lines.

    Each in-patch ground primitive is matched to at most one aerial primitive
    (and vice versa) by greedy one-to-one assignment: all within-threshold
    (ground, aerial) candidate pairs are ranked by a soft quality in [0, 1] that
    decays linearly with residual distance (and, for lines, with the folded
    angle), and pairs are accepted highest-quality-first while skipping any
    whose ground or aerial primitive is already taken. Quality is used only to
    resolve this assignment; the reported ``fitness`` is the Wilson lower bound
    of the resulting inlier ratio, rewarding dense overlap while resisting
    small-support inflation. Hypotheses with fewer than ``min_in_patch``
    in-patch primitives score 0 (too little overlap to trust).
    """

    @classmethod
    def compute(
        cls,
        aerial_segments: PrimitiveList,
        ground_segments: PrimitiveList,
        T_aerial_ground: np.ndarray,
        point_inlier_thresh_m: float,
        line_inlier_thresh_m: float,
        line_angle_thresh_rad: float,
        wilson_z: float,
        patch_bounds: Optional[Tuple[float, float, float, float]] = None,
        min_in_patch: int = 3,
    ) -> AlignmentFitnessResult:
        _assert_2d(aerial_segments, "aerial_segments")
        _assert_2d(ground_segments, "ground_segments")

        if patch_bounds is None:
            patch_bounds = _aerial_patch_bounds(aerial_segments)

        if patch_bounds is None:
            return AlignmentFitnessResult(0.0, 0, 0)

        # Transform a copy of the full ground map into the aerial frame.
        ground = ground_segments.copy()
        ground.transform(T_aerial_ground)

        aerial_points = aerial_segments.get_points()
        aerial_lines = aerial_segments.get_lines()
        aerial_pts_xy = (
            np.array([p.get_point()[:2] for p in aerial_points])
            if len(aerial_points) > 0
            else np.zeros((0, 2))
        )

        # Gather every within-threshold (quality, ground_id, aerial_id) pair,
        # then resolve to a one-to-one matching greedily below.
        n_in_patch = 0
        candidates: List[Tuple[float, int, int]] = []
        for g in ground:
            if not _primitive_in_patch(g, patch_bounds):
                continue
            n_in_patch += 1

            if isinstance(g, PointPrimitive):
                if len(aerial_points) == 0:
                    continue
                gp = g.get_point()[:2]
                dists = np.linalg.norm(aerial_pts_xy - gp, axis=1)
                for j in np.nonzero(dists < point_inlier_thresh_m)[0]:
                    q = 1.0 - dists[j] / point_inlier_thresh_m
                    candidates.append((float(q), g.id, aerial_points[j].id))
            else:  # LinePrimitive
                for a in aerial_lines:
                    ang = _line_angle(g, a)
                    if ang >= line_angle_thresh_rad:
                        continue
                    d = g.min_dist_to(a)
                    if d >= line_inlier_thresh_m:
                        continue
                    q_dist = 1.0 - d / line_inlier_thresh_m
                    q_ang = 1.0 - ang / line_angle_thresh_rad
                    q = np.sqrt(q_dist * q_ang)  # geometric mean
                    candidates.append((float(q), g.id, a.id))

        # Greedy one-to-one: take pairs best-first, skipping any primitive that
        # has already been claimed on either side.
        candidates.sort(key=lambda c: c[0], reverse=True)
        used_ground: set = set()
        used_aerial: set = set()
        inlier_pairs: List[Tuple[int, int]] = []
        qualities: List[float] = []
        for q, gid, aid in candidates:
            if gid in used_ground or aid in used_aerial:
                continue
            used_ground.add(gid)
            used_aerial.add(aid)
            inlier_pairs.append((gid, aid))
            qualities.append(q)

        n_inliers = len(inlier_pairs)
        inlier_ratio = n_inliers / n_in_patch if n_in_patch > 0 else 0.0
        mean_quality = float(np.mean(qualities)) if qualities else 0.0
        if n_in_patch < min_in_patch:
            fitness = 0.0
        else:
            fitness = _wilson_lower_bound(n_inliers, n_in_patch, wilson_z)
        return AlignmentFitnessResult(
            fitness=fitness,
            n_inliers=n_inliers,
            n_in_patch=n_in_patch,
            inlier_ratio=inlier_ratio,
            mean_inlier_quality=mean_quality,
            inlier_pairs=inlier_pairs,
            patch_bounds=patch_bounds,
        )


class Wasserstein:
    @classmethod
    def principle_square_root(cls, A):
        """
        Compute the principle square root of a symmetric positive semi-definite matrix A

        - Source: https://en.wikipedia.org/wiki/Square_root_of_a_matrix#Positive_semidefinite_matrices
            - see "Solutions in close form/By diagonalization"
        - using Eigendecomposition: V D^{1/2} V^T * (V D^{1/2} V^T) = V D V^T = A
        """
        eigvals, eigvecs = np.linalg.eigh(A)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    @classmethod
    def wasserstein_metric(
        cls,
        gaussian_1: Tuple[np.ndarray, np.ndarray],
        gaussian_2: Tuple[np.ndarray, np.ndarray],
    ):
        """
        Compute the Wasserstein metric between two Gaussian distributions

        - Source: https://en.wikipedia.org/wiki/Wasserstein_metric#Normal_distributions

        Args:
            gaussian_1 (Tuple[np.ndarray, np.ndarray]): mean and covariance of the first Gaussian distribution
            gaussian_2 (Tuple[np.ndarray, np.ndarray]): mean and covariance of the second Gaussian distribution
        """
        mu1, sigma1 = gaussian_1
        mu2, sigma2 = gaussian_2
        sigma2_sqrt = cls.principle_square_root(sigma2)
        return np.linalg.norm(mu1 - mu2) + np.trace(
            sigma1
            + sigma2
            - 2 * cls.principle_square_root(sigma2_sqrt @ sigma1 @ sigma2_sqrt)
        )


class ChamferDistance:
    @classmethod
    def chamfer_distance(cls, pcd1, pcd2):
        """
        See [1] https://github.com/UM-ARM-Lab/Chamfer-Distance-API and [2] https://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#Point-Cloud-Distance.

        The champer distance from pcd1 to pcd2 is the average of the distances from each point in pcd1 to its nearest point in pcd2.
            - o3d.geometry.PointCloud.compute_point_cloud_distance [2] returns an array of the distances for each point in the calling pointcloud to
              the other pointcloud, so we take the mean to get chamfer distance as a single metric.

        Instead of adding the directional champer distances like [1], we take the minimum, as we want to measure overlap and de-value extent.
        The champer distance from a small pointcloud to a large enclosing pointcloud will be small, but this is not true in the other direction.

        Args:
            pcd1 (o3d.geometry.PointCloud): first point cloud
            pcd2 (o3d.geometry.PointCloud): second point cloud
        """
        if not pcd1.has_points() or not pcd2.has_points():
            return np.inf
        return min(
            np.mean(pcd1.compute_point_cloud_distance(pcd2)),
            np.mean(pcd2.compute_point_cloud_distance(pcd1)),
        )

    @classmethod
    def norm_chamfer_distance(cls, pcd1, pcd2):
        """
        Compute the normalized chamfer distance between two point clouds.

        Normalization is done by dividing the chamfer distance by the
        diagonal of the axis-aligned bounding box that contains both point clouds.

        Args:
            pcd1 (o3d.geometry.PointCloud): first point cloud
            pcd2 (o3d.geometry.PointCloud): second point cloud
        """
        chamfer_dist = cls.chamfer_distance(pcd1, pcd2)
        aabb1 = pcd1.get_axis_aligned_bounding_box()
        aabb2 = pcd2.get_axis_aligned_bounding_box()
        merged_min_bound = np.minimum(aabb1.min_bound, aabb2.min_bound)
        merged_max_bound = np.maximum(aabb1.max_bound, aabb2.max_bound)
        merged_spread = merged_max_bound - merged_min_bound
        diag = np.linalg.norm(merged_spread)
        return 1 - (chamfer_dist / diag) if diag > 0 else 1.0
