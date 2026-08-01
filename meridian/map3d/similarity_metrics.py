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
from dataclasses import dataclass, field, fields
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


@dataclass
class _RawLines:
    """Array-of-structs view of a LineList, straight out of the primitives.

    ``ep0``/``ep1`` hold the present endpoints (``ep0`` first, so a ray always
    stores its endpoint in ``ep0``) and are NaN where absent.
    """

    ids: np.ndarray  # (n,) int
    pts: np.ndarray  # (n, 2) a point on the line
    dirs: np.ndarray  # (n, 2) unit
    ep0: np.ndarray  # (n, 2)
    ep1: np.ndarray  # (n, 2)
    n_eps: np.ndarray  # (n,) 0, 1 or 2


@dataclass
class _Lines:
    """Lines reduced to finite segments clipped to the patch, so every pairwise
    quantity below is one (n_ground x n_aerial) numpy expression over a uniform
    representation -- no endpoint cases, no infinities.
    """

    ids: np.ndarray  # (n,) int
    dirs: np.ndarray  # (n, 2) unit
    seg0: np.ndarray  # (n, 2)
    seg1: np.ndarray  # (n, 2)


def _points_xy(points, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(n, 2) point coordinates under the rigid transform (R, t)."""
    if len(points) == 0:
        return np.zeros((0, 2))
    xy = np.array([p.get_point().flatten()[:2] for p in points])
    return xy @ R.T + t


def _select_lines(lines: _Lines, mask: np.ndarray) -> _Lines:
    return _Lines(*(getattr(lines, f.name)[mask] for f in fields(lines)))


def _extract_lines(lines, R: np.ndarray, t: np.ndarray) -> _RawLines:
    """Pack lines into arrays, applying the rigid transform (R, t) on the way."""
    n = len(lines)
    ids = np.zeros(n, dtype=np.int64)
    pts = np.zeros((n, 2))
    dirs = np.zeros((n, 2))
    ep0 = np.full((n, 2), np.nan)
    ep1 = np.full((n, 2), np.nan)
    n_eps = np.zeros(n, dtype=np.int64)
    for i, line in enumerate(lines):
        ids[i] = line.id
        pts[i] = line.get_point().flatten()[:2]
        dirs[i] = line.get_direction().flatten()[:2]
        present = [ep for ep in line.endpoints if ep is not None]
        n_eps[i] = len(present)
        if len(present) > 0:
            ep0[i] = present[0][:2]
        if len(present) > 1:
            ep1[i] = present[1][:2]

    pts = pts @ R.T + t
    ep0 = ep0 @ R.T + t
    ep1 = ep1 @ R.T + t
    dirs = dirs @ R.T
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.where(norms > 0.0, norms, 1.0)
    return _RawLines(ids, pts, dirs, ep0, ep1, n_eps)


def _clip_lines(raw: _RawLines, bounds: Tuple[float, float, float, float]) -> _Lines:
    """Reduce every line to the finite segment lying inside ``bounds``.

    Bounded segments keep their endpoints. Rays and infinite lines are clipped to
    the box, so alignment is judged on the geometry the two submaps actually share
    -- without this, two near-parallel unbounded lines that meet kilometers
    outside the patch would register as touching. A line that misses the box
    entirely collapses to a single point on it (its endpoint, or its defining
    point), which is outside the box and so cannot be an inlier of anything in it.
    """
    lo = np.array([bounds[0], bounds[1]])
    hi = np.array([bounds[2], bounds[3]])
    origin = np.where((raw.n_eps > 0)[:, None], raw.ep0, raw.pts)
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = (lo - origin) / raw.dirs
        tb = (hi - origin) / raw.dirs
    # A slab the line is parallel to either contains it (no constraint) or rules
    # it out entirely.
    parallel = raw.dirs == 0.0
    inside = (origin >= lo) & (origin <= hi)
    t_lo = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(ta, tb))
    t_hi = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(ta, tb))
    t_lo = t_lo.max(axis=1)
    t_hi = t_hi.min(axis=1)

    # A ray extends from its endpoint along +direction only.
    is_ray = raw.n_eps == 1
    t_lo = np.where(is_ray, np.maximum(t_lo, 0.0), t_lo)
    t_hi = np.where(is_ray, np.maximum(t_hi, 0.0), t_hi)

    misses = t_lo > t_hi
    t_lo = np.where(misses, 0.0, t_lo)
    t_hi = np.where(misses, 0.0, t_hi)

    bounded = (raw.n_eps > 1)[:, None]
    seg0 = np.where(bounded, raw.ep0, origin + t_lo[:, None] * raw.dirs)
    seg1 = np.where(bounded, raw.ep1, origin + t_hi[:, None] * raw.dirs)
    return _Lines(raw.ids, raw.dirs, seg0, seg1)


def _point_to_segment_dist2(
    p: np.ndarray, a: np.ndarray, ab: np.ndarray, ab_len2: np.ndarray
) -> np.ndarray:
    """Squared distance from point ``p`` to segment ``a -> a + ab``. All arguments
    broadcast against each other with xy on the last axis. Squared, so the caller
    can take the minimum over several of these and pay for one sqrt."""
    ap = p - a
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.clip((ap * ab).sum(-1) / ab_len2, 0.0, 1.0)
    s = np.nan_to_num(s)  # zero-length segment: clamp to its start point
    d = ap - s[..., None] * ab
    return d[..., 0] ** 2 + d[..., 1] ** 2


def _segment_dist_matrix(g: _Lines, a: _Lines) -> np.ndarray:
    """(n_ground, n_aerial) minimum distance between every pair of lines.

    Crossing segments are 0; otherwise the minimum is attained at one of the
    four endpoints, which is what the endpoint terms below enumerate."""
    u = g.seg1 - g.seg0  # (G, 2)
    v = a.seg1 - a.seg0  # (A, 2)
    w = a.seg0[None, :, :] - g.seg0[:, None, :]  # (G, A, 2)
    cross_uv = np.outer(u[:, 0], v[:, 1]) - np.outer(u[:, 1], v[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (w[..., 0] * v[None, :, 1] - w[..., 1] * v[None, :, 0]) / cross_uv
        r = (w[..., 0] * u[:, None, 1] - w[..., 1] * u[:, None, 0]) / cross_uv
    crossing = (
        (cross_uv != 0.0) & (s >= 0.0) & (s <= 1.0) & (r >= 0.0) & (r <= 1.0)
    )

    u_len2 = (u * u).sum(-1)[:, None]
    v_len2 = (v * v).sum(-1)[None, :]
    dist2 = np.minimum(
        np.minimum(
            _point_to_segment_dist2(
                g.seg0[:, None, :], a.seg0[None, :, :], v[None, :, :], v_len2
            ),
            _point_to_segment_dist2(
                g.seg1[:, None, :], a.seg0[None, :, :], v[None, :, :], v_len2
            ),
        ),
        np.minimum(
            _point_to_segment_dist2(
                a.seg0[None, :, :], g.seg0[:, None, :], u[:, None, :], u_len2
            ),
            _point_to_segment_dist2(
                a.seg1[None, :, :], g.seg0[:, None, :], u[:, None, :], u_len2
            ),
        ),
    )
    return np.where(crossing, 0.0, np.sqrt(dist2))


def _overlap_matrix(g: _Lines, a: _Lines) -> np.ndarray:
    """(n_ground, n_aerial) extent overlap of each line pair projected onto the
    aerial line's direction, as intersection over minimum length (IoM). IoM rather
    than IoU, so a short ground fragment lying fully alongside a long aerial road
    counts as complete overlap."""
    g_s0 = g.seg0 @ a.dirs.T  # (G, A)
    g_s1 = g.seg1 @ a.dirs.T
    g_lo, g_hi = np.minimum(g_s0, g_s1), np.maximum(g_s0, g_s1)

    a_s0 = (a.seg0 * a.dirs).sum(1)  # (A,); each projected onto its own direction
    a_s1 = (a.seg1 * a.dirs).sum(1)
    a_lo = np.minimum(a_s0, a_s1)[None, :]
    a_hi = np.maximum(a_s0, a_s1)[None, :]

    intersection = np.minimum(g_hi, a_hi) - np.maximum(g_lo, a_lo)
    shortest = np.minimum(g_hi - g_lo, a_hi - a_lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.minimum(1.0, intersection / shortest)
    return np.where((intersection <= 0.0) | (shortest <= 0.0), 0.0, ratio)


def _aerial_patch_bounds(
    points_xy: np.ndarray, lines: _RawLines
) -> Optional[Tuple[float, float, float, float]]:
    """Axis-aligned bounds (xmin, ymin, xmax, ymax) of the aerial submap's
    coverage, taken from point coordinates and finite line endpoints. Infinite
    lines (no endpoints) are skipped since they don't bound a region."""
    xy = np.concatenate([points_xy, lines.ep0, lines.ep1])
    xy = xy[~np.isnan(xy).any(axis=1)]
    if len(xy) == 0:
        return None
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def _points_in_box(
    points_xy: np.ndarray, bounds: Tuple[float, float, float, float]
) -> np.ndarray:
    xmin, ymin, xmax, ymax = bounds
    return (
        (points_xy[:, 0] >= xmin)
        & (points_xy[:, 0] <= xmax)
        & (points_xy[:, 1] >= ymin)
        & (points_xy[:, 1] <= ymax)
    )


def _lines_in_box(
    lines: _Lines, bounds: Tuple[float, float, float, float]
) -> np.ndarray:
    """Liang-Barsky slab clip: True where the line at least partly lies in the
    box."""
    xmin, ymin, xmax, ymax = bounds
    lo_bound = np.array([xmin, ymin])
    hi_bound = np.array([xmax, ymax])
    p0 = lines.seg0
    d = lines.seg1 - p0
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = (lo_bound - p0) / d
        tb = (hi_bound - p0) / d
    # A slab the segment is parallel to either contains it (no constraint) or
    # rules it out entirely.
    inside = (p0 >= lo_bound) & (p0 <= hi_bound)
    parallel = d == 0.0
    lo = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(ta, tb))
    hi = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(ta, tb))
    t0 = np.maximum(0.0, lo.max(axis=1))
    t1 = np.minimum(1.0, hi.min(axis=1))
    return t0 <= t1


class AlignmentFitness:
    """ICP-style full-submap alignment fitness for point + line primitive maps.

    The ground submap is transformed into the aerial frame by the estimated
    transform and scored against the *full* aerial submap (not just the matched
    inliers). Only ground primitives that at least partly lie within the aerial
    patch are counted, so partial overlap is not penalized. Ground points are
    matched to aerial points and ground lines to aerial lines.

    Quality only selects each ground primitive's best match; the reported
    ``fitness`` is the Wilson lower bound of the resulting inlier ratio,
    rewarding dense overlap while resisting small-support inflation. Hypotheses
    with fewer than ``min_in_patch`` in-patch primitives score 0 (too little
    overlap to trust).

    Everything is evaluated as whole-matrix numpy expressions over
    (n_ground x n_aerial), since the per-pair Python geometry calls this
    replaces dominated the matcher's runtime.
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
        line_min_overlap: float,
        wilson_z: float,
        min_in_patch: int,
        patch_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> AlignmentFitnessResult:
        _assert_2d(aerial_segments, "aerial_segments")
        _assert_2d(ground_segments, "ground_segments")

        eye = np.eye(2)
        zero = np.zeros(2)
        # Ground primitives are brought into the aerial frame as arrays rather
        # than by copying and transforming the submap.
        R = T_aerial_ground[:2, :2]
        t = T_aerial_ground[:2, -1]

        aerial_point_list = aerial_segments.get_points()
        ground_point_list = ground_segments.get_points()
        aerial_pts = _points_xy(aerial_point_list, eye, zero)
        ground_pts = _points_xy(ground_point_list, R, t)
        raw_aerial_lines = _extract_lines(aerial_segments.get_lines(), eye, zero)
        raw_ground_lines = _extract_lines(ground_segments.get_lines(), R, t)

        if patch_bounds is None:
            patch_bounds = _aerial_patch_bounds(aerial_pts, raw_aerial_lines)

        if patch_bounds is None:
            return AlignmentFitnessResult(0.0, 0, 0)

        # Unbounded lines are clipped to the patch, widened by the inlier
        # threshold so geometry that could still match something in the patch is
        # kept. Beyond that, a line is too far away to align with anything here.
        clip_bounds = (
            patch_bounds[0] - line_inlier_thresh_m,
            patch_bounds[1] - line_inlier_thresh_m,
            patch_bounds[2] + line_inlier_thresh_m,
            patch_bounds[3] + line_inlier_thresh_m,
        )
        aerial_lines = _clip_lines(raw_aerial_lines, clip_bounds)
        ground_lines = _clip_lines(raw_ground_lines, clip_bounds)

        # Each in-patch ground primitive claims its best aerial primitive.
        pt_in_patch = _points_in_box(ground_pts, patch_bounds)
        line_in_patch = _lines_in_box(ground_lines, patch_bounds)
        n_in_patch = int(pt_in_patch.sum() + line_in_patch.sum())

        inlier_pairs: List[Tuple[int, int]] = []
        qualities: List[float] = []

        if pt_in_patch.any() and len(aerial_pts) > 0:
            dists = np.linalg.norm(
                ground_pts[pt_in_patch][:, None, :] - aerial_pts[None, :, :], axis=2
            )
            best = dists.argmin(axis=1)
            best_dist = dists[np.arange(len(best)), best]
            hit = best_dist < point_inlier_thresh_m
            aerial_ids = np.array([p.id for p in aerial_point_list])
            ground_ids = np.array([p.id for p in ground_point_list])[pt_in_patch]
            inlier_pairs.extend(
                zip(ground_ids[hit].tolist(), aerial_ids[best[hit]].tolist())
            )
            qualities.extend((1.0 - best_dist[hit] / point_inlier_thresh_m).tolist())

        if line_in_patch.any() and len(aerial_lines.ids) > 0:
            g_lines = _select_lines(ground_lines, line_in_patch)
            cos_ang = np.abs(g_lines.dirs @ aerial_lines.dirs.T).clip(0.0, 1.0)
            ang = np.arccos(cos_ang)
            dist = _segment_dist_matrix(g_lines, aerial_lines)
            overlap = _overlap_matrix(g_lines, aerial_lines)

            q_dist = 1.0 - dist / line_inlier_thresh_m
            q_ang = 1.0 - ang / line_angle_thresh_rad
            quality = np.cbrt(q_dist * q_ang * overlap)  # geometric mean
            quality[
                (ang >= line_angle_thresh_rad)
                | (dist >= line_inlier_thresh_m)
                | (overlap < line_min_overlap)
            ] = 0.0

            best = quality.argmax(axis=1)
            best_quality = quality[np.arange(len(best)), best]
            hit = best_quality > 0.0
            inlier_pairs.extend(
                zip(g_lines.ids[hit].tolist(), aerial_lines.ids[best[hit]].tolist())
            )
            qualities.extend(best_quality[hit].tolist())

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
