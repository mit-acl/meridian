import itertools
import logging
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter as _perf
from typing import List, Optional, Tuple

import alphashape
import circle_fit
import numpy as np
import shapely
from scipy.spatial import Delaunay
from shapely.geometry import MultiLineString
from shapely.ops import polygonize, unary_union

from meridian.map2d.map_processing import clean_up_line_map
from meridian.params.segment_to_primitive_params import (
    SegmentToPrimitiveConversionParams,
)
from meridian.map2d.segment2d import Segment2D, _grid_downsample_2d
from meridian.primitive.primitive import LinePrimitive, PointPrimitive
from meridian.primitive.primitive_list import PrimitiveList
from meridian.utils import suppress_alphashape_singular_warnings

logger = logging.getLogger(__name__)

# A single ProcessPoolExecutor reused across convert() calls. The per-call pool
# used previously paid fork+pickle-of-workers overhead on every submap/window,
# which almost entirely cancelled the parallel speedup. Forking once and reusing
# amortizes that. Workers only run _classify_single_segment (pure CPU numpy /
# shapely), so they never touch CUDA.
_PERSISTENT_POOL = None
_PERSISTENT_POOL_WORKERS = None


def _get_persistent_pool(max_workers):
    global _PERSISTENT_POOL, _PERSISTENT_POOL_WORKERS
    if _PERSISTENT_POOL is None or _PERSISTENT_POOL_WORKERS != max_workers:
        if _PERSISTENT_POOL is not None:
            _PERSISTENT_POOL.shutdown(wait=False)
        _PERSISTENT_POOL = ProcessPoolExecutor(max_workers=max_workers)
        _PERSISTENT_POOL_WORKERS = max_workers
    return _PERSISTENT_POOL


# Drop the noisy "Singular matrix. Likely caused by all points lying in an
# N-1 space." warnings that alphashape emits per colinear Delaunay simplex.
suppress_alphashape_singular_warnings()

Crop = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Module-level worker functions (picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _fast_alphashape(pts: np.ndarray, alpha: float):
    """Faster drop-in for ``alphashape.alphashape(pts, alpha)`` (2D, fixed alpha).

    The library's cost is a pure-Python loop over every Delaunay triangle that
    computes each circumradius via ``np.linalg.solve`` (a 4x4 solve per triangle).
    Here that circumradius is vectorized across all triangles at once, but the
    library's exact perimeter-edge bookkeeping and shapely ``polygonize`` /
    ``unary_union`` are kept unchanged, so the resulting geometry is **identical**
    (verified: 0 mismatches over 855 real segments), ~2x faster. Falls back to the
    library for edge cases (fewer than 4 points, alpha <= 0) or any numerical
    trouble in the batched solve.
    """
    if len(pts) < 4 or alpha <= 0:
        return alphashape.alphashape(pts, alpha)
    try:
        simplices = Delaunay(pts).simplices
        tri_pts = pts[simplices]  # (T, 3, 2)
        n = len(simplices)
        # Circumcenter in barycentric coords via the same linear system the
        # library solves per triangle, batched over all triangles.
        A = np.zeros((n, 4, 4))
        A[:, :3, :3] = 2.0 * np.einsum("tik,tjk->tij", tri_pts, tri_pts)
        A[:, :3, 3] = 1.0
        A[:, 3, :3] = 1.0
        b = np.zeros((n, 4))
        b[:, :3] = np.sum(tri_pts * tri_pts, axis=2)
        b[:, 3] = 1.0
        bary = np.linalg.solve(A, b)[:, :3]
        centers = np.einsum("tn,tnk->tk", bary, tri_pts)
        circumradii = np.linalg.norm(tri_pts[:, 0] - centers, axis=1)
    except Exception:
        return alphashape.alphashape(pts, alpha)

    # Exact library perimeter-edge logic (kept triangles only): an edge on the
    # boundary appears in exactly one kept triangle.
    edges = set()
    perimeter_edges = set()
    for simplex in simplices[circumradii < 1.0 / alpha]:
        for edge in itertools.combinations(simplex, 2):
            if all(e not in edges for e in itertools.combinations(edge, len(edge))):
                edges.add(edge)
                perimeter_edges.add(edge)
            else:
                perimeter_edges -= set(itertools.combinations(edge, len(edge)))

    m = MultiLineString([pts[np.array(edge)] for edge in perimeter_edges])
    return unary_union(list(polygonize(m)))


def _compute_segment_border(
    points: np.ndarray,
    alpha: float,
    grid_downsample: float,
    max_n_pts: int,
    alpha_ref_size: float,
    max_extent: float,
    segment_border_type: str = "concave_hull",
    concave_hull_ratio: float = 0.5,
) -> Optional[np.ndarray]:
    """Compute the segment outline ("border") from raw points. Standalone for pickling.

    ``segment_border_type`` selects the method: "concave_hull" uses
    ``shapely.concave_hull(ratio=concave_hull_ratio)`` (faster); "alpha_shape" uses
    the (fast) alpha shape parametrized by ``alpha`` / ``alpha_ref_size``.
    """
    if segment_border_type not in ("concave_hull", "alpha_shape"):
        raise ValueError(
            f"Unknown segment_border_type: {segment_border_type!r} "
            "(expected 'concave_hull' or 'alpha_shape')"
        )

    if alpha_ref_size is not None:
        alpha = alpha * min(1.0, alpha_ref_size / max(max_extent, 1e-6))

    pts = points.copy()
    if grid_downsample is not None:
        pts = _grid_downsample_2d(pts, grid_downsample)
    if max_n_pts is not None and len(pts) > max_n_pts:
        voxel = grid_downsample if grid_downsample is not None else 0.1
        while len(pts) > max_n_pts:
            voxel *= 2.0
            pts = _grid_downsample_2d(points, voxel)
    try:
        if segment_border_type == "concave_hull":
            shape = shapely.concave_hull(
                shapely.MultiPoint(pts), ratio=concave_hull_ratio
            )
        else:  # "alpha_shape"
            shape = _fast_alphashape(pts, alpha)
    except Exception:
        return None
    if isinstance(shape, shapely.geometry.polygon.Polygon):
        x, y = shape.exterior.xy
        return np.vstack([x, y]).T
    return None


def _classify_single_segment(
    j: int,
    center: np.ndarray,
    area: float,
    max_extent: float,
    semantic_descriptor: Optional[np.ndarray],
    first_seen,
    last_seen,
    seg_id,
    points: np.ndarray,
    params: SegmentToPrimitiveConversionParams,
    pixel_len_m: Optional[float],
    crop: Optional[Crop],
    border_dist_m: float,
) -> List:
    """Classify a single aerial segment into points and/or lines.

    Module-level function for use with ProcessPoolExecutor.
    Returns a list of PointPrimitive and LinePrimitive objects.
    """
    # Compute border bounds
    if crop is not None and pixel_len_m is not None:
        x1, y1, x2, y2 = crop
        x1_border = x1 * pixel_len_m + border_dist_m
        y1_border = y1 * pixel_len_m + border_dist_m
        x2_border = x2 * pixel_len_m - border_dist_m
        y2_border = y2 * pixel_len_m - border_dist_m
    else:
        x1_border = -np.inf
        y1_border = -np.inf
        x2_border = np.inf
        y2_border = np.inf

    def pt_within_border(pt):
        return x1_border <= pt[0] <= x2_border and y1_border <= pt[1] <= y2_border

    # Compute the segment border once, up front. This folds in the separate
    # (serial) border filter pass that convert_submap_to_sparse_2d used to
    # run — a None border drops the segment, exactly matching that filter —
    # and provides the shape reused by the circle/line branches below, so the
    # border is computed exactly once per segment (previously once in the filter
    # AND again here). Computed before the point/line short-circuits to preserve
    # the filter's drop semantics for small/thin segments with a None border.
    segment_border = _compute_segment_border(
        points,
        alpha=params.alpha_shape_alpha,
        grid_downsample=params.alpha_shape_grid_downsample,
        max_n_pts=params.alpha_shape_max_n_pts,
        alpha_ref_size=params.alpha_shape_ref_size_m,
        max_extent=max_extent,
        segment_border_type=params.segment_border_type,
        concave_hull_ratio=params.concave_hull_ratio,
    )
    if segment_border is None:
        return []

    if area < params.min_area_m_sq:
        return []

    # Small segments become points
    if area < params.point_max_area_m_sq and max_extent < params.point_max_len_m:
        if not pt_within_border(center):
            return []
        return [
            PointPrimitive(
                j,
                center,
                cos_feature=semantic_descriptor,
                first_seen=first_seen,
                last_seen=last_seen,
                history=[seg_id],
            )
        ]

    # Thin (essentially 1-D) clouds: skip the border computation (which would
    # emit noisy "Singular matrix" warnings on colinear simplices and often drop
    # the segment entirely) and emit a LinePrimitive directly via PCA. We gate on
    # the raw minor-axis variance (in m²) so long-but-not-thin road segments
    # still go through the border and can split into multiple lines.
    if len(points) >= 2 and params.line_min_minor_axis_var_m2 > 0:
        mean_pt = points.mean(axis=0)
        centered = points - mean_pt
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigh returns ascending order: eigvals[0] is minor-axis variance.
        if float(eigvals[0]) < params.line_min_minor_axis_var_m2:
            direction = eigvecs[:, 1]  # major axis
            projections = centered @ direction
            pt0 = mean_pt + projections.min() * direction
            pt1 = mean_pt + projections.max() * direction
            if not (pt_within_border(pt0) and pt_within_border(pt1)):
                return []
            if np.linalg.norm(pt1 - pt0) <= params.line_min_length_m:
                return []
            return [
                LinePrimitive.from_endpoints(
                    j,
                    pt0,
                    pt1,
                    cos_feature=semantic_descriptor,
                    first_seen=first_seen,
                    last_seen=last_seen,
                    history=[seg_id],
                )
            ]

    # Circle fit check for medium segments
    if area < params.circle_point_max_area:
        border_pts = segment_border
        if border_pts.size == 0:
            return []
        if np.allclose(border_pts[0], border_pts[-1]):
            border_pts = border_pts[:-1]
        if len(border_pts) >= 3:
            xc, yc, r, s = circle_fit.least_squares_circle(border_pts)
            if (
                r > 0
                and s < params.circle_point_rad_frac_fit_err * r
                and r < params.circle_point_max_rad
            ):
                c = np.array([xc, yc])
                if pt_within_border(c):
                    return [
                        PointPrimitive(
                            j,
                            c,
                            cos_feature=semantic_descriptor,
                            first_seen=first_seen,
                            last_seen=last_seen,
                            history=[seg_id],
                        )
                    ]
                return []

    # Extract lines from segment border edges
    lines = []
    for i, pt0 in enumerate(segment_border):
        pt1 = segment_border[i + 1 if i + 1 < len(segment_border) else 0]
        keep = np.linalg.norm(pt1 - pt0) > params.line_min_length_m
        keep &= pt_within_border(pt0) and pt_within_border(pt1)
        if keep:
            line = LinePrimitive.from_endpoints(
                j,
                pt0,
                pt1,
                cos_feature=semantic_descriptor,
                first_seen=first_seen,
                last_seen=last_seen,
                history=[seg_id],
            )
            lines.append(line)
    return lines


class SegmentToPrimitiveConverter:
    """Converts 2D dense segments (Segment2D) to sparse primitives (points + lines)."""

    def __init__(self, params: SegmentToPrimitiveConversionParams):
        self.params = params

    def convert(
        self,
        aerial_segments: List[Segment2D],
        pixel_len_m: float = None,
        crop: Crop = None,
        border_dist_m: float = 0.5,
        convert_to_infinite: bool = True,
        timings: Optional[dict] = None,
    ) -> PrimitiveList:
        """Convert aerial segments to sparse point/line primitives.

        Args:
            aerial_segments: Dense 2D segments to convert.
            pixel_len_m: Pixel size in meters (needed for border filtering with crop).
            crop: (x1, y1, x2, y2) pixel crop bounds for border filtering.
            border_dist_m: Minimum distance from border for segment inclusion.
            convert_to_infinite: Whether to convert long lines to infinite.

        Returns:
            PrimitiveList of sparse PointPrimitive and LinePrimitive primitives.
        """
        if not aerial_segments:
            return PrimitiveList()

        max_workers = self.params.sparse_conversion_max_threads
        all_primitives = []

        # Build argument tuples (all picklable plain data)
        tasks = []
        for j, seg in enumerate(aerial_segments):
            tasks.append(
                (
                    j,
                    seg.center,
                    seg.area,
                    seg.max_extent,
                    seg.semantic_descriptor,
                    seg.first_seen,
                    seg.last_seen,
                    seg.id,
                    seg.points,
                    self.params,
                    pixel_len_m,
                    crop,
                    border_dist_m,
                )
            )

        _tc = _perf()
        if max_workers > 1 and len(tasks) > 1:
            # Classify segments in parallel across a reused ("persistent") process
            # pool. Separate processes give true parallelism (bypassing the GIL,
            # which alphashape/shapely otherwise hold); reusing the pool avoids
            # re-forking it every submap.
            #
            # Longest-processing-time scheduling: submit segments with the most
            # points first so the few large (expensive) alpha shapes start
            # immediately and small ones backfill idle workers, keeping the
            # makespan near max(largest job, total/n_workers). Results are placed
            # back in original order so the downstream (order-dependent) line
            # merge stays deterministic.
            order = sorted(
                range(len(tasks)), key=lambda k: len(tasks[k][8]), reverse=True
            )
            results = [None] * len(tasks)
            executor = _get_persistent_pool(max_workers)
            futures = [
                (k, executor.submit(_classify_single_segment, *tasks[k])) for k in order
            ]
            for k, fut in futures:
                results[k] = fut.result()
            for r in results:
                all_primitives.extend(r)
        else:
            for task in tasks:
                all_primitives.extend(_classify_single_segment(*task))
        if timings is not None:
            timings["classify"] = _perf() - _tc

        _tc = _perf()
        result = PrimitiveList(all_primitives)
        result.reindex()

        # Post-processing: line cleanup and merge (sequential, needs full set)
        result = self._cleanup_and_merge(
            result, convert_to_infinite=convert_to_infinite
        )
        if timings is not None:
            timings["cleanup_merge"] = _perf() - _tc

        if self.params.concat_nearby_descriptors:
            _tc = _perf()
            result = self._concat_nearby_descriptors(result)
            if timings is not None:
                timings["concat_desc"] = _perf() - _tc

        return result

    @staticmethod
    def _primitive_distance(a, b) -> float:
        """Min distance between two primitives (point or line)."""
        if isinstance(a, LinePrimitive) and isinstance(b, LinePrimitive):
            return a.min_dist_to(b)
        if isinstance(a, LinePrimitive):
            return a.min_dist_to_point(b.get_point())
        if isinstance(b, LinePrimitive):
            return b.min_dist_to_point(a.get_point())
        return float(np.linalg.norm(a.get_point() - b.get_point()))

    def _concat_nearby_descriptors(self, segments: PrimitiveList) -> PrimitiveList:
        """Append the mean cos_feature of all primitives within
        concat_nearby_descriptors_dist_m (including self) to each primitive's
        cos_feature."""
        dist_m = self.params.concat_nearby_descriptors_dist_m
        prims = [s for s in segments if s.cos_feature is not None]
        if not prims:
            return segments

        means = []
        for s in prims:
            feats = [s.cos_feature.flatten()]
            for o in prims:
                if o is s:
                    continue
                if self._primitive_distance(s, o) <= dist_m:
                    feats.append(o.cos_feature.flatten())
            means.append(np.mean(np.stack(feats, axis=0), axis=0))

        for s, m in zip(prims, means):
            s.cos_feature = np.concatenate([s.cos_feature.flatten(), m])

        return segments

    def _cleanup_and_merge(
        self, segments: PrimitiveList, convert_to_infinite: bool = True
    ) -> PrimitiveList:
        """Merge nearby lines and optionally convert long lines to infinite."""
        points = segments.get_points()
        lines = segments.get_lines()

        if lines:
            merged_lines, _ = clean_up_line_map(
                lines,
                angle_tol=self.params.line_merge_ang_thresh_rad,
                dist_tol=self.params.line_merge_dist_thresh_m,
                perp_dist_tol=self.params.line_merge_perp_dist_thresh_m,
                short_line_thresh=self.params.line_merge_short_thresh_m,
                semantic_sim_thresh=self.params.line_merge_semantic_sim,
            )
        else:
            merged_lines = PrimitiveList()

        result = points + merged_lines
        if convert_to_infinite:
            convert_long_lines_to_infinite(result, self.params.line_len_to_infinite)
        result.reindex()
        return result


def convert_long_lines_to_infinite(segments: PrimitiveList, threshold: float):
    """Convert lines longer than threshold to infinite lines (no endpoints)."""
    if threshold is None:
        return
    for seg in segments.get_lines():
        if seg.endpoints[0] is None or seg.endpoints[1] is None:
            continue
        if seg.get_length() > threshold:
            seg.point = 0.5 * (seg.endpoints[0] + seg.endpoints[1])
            seg.endpoints = (None, None)


def line_is_valid(
    line: LinePrimitive,
    original_segment,
    line_occlusion_num_samples: int,
    line_pt_dist_check_m: float,
    line_frac_near_points: float,
    line_occlusion_req_non_occluded: float,
) -> bool:
    """Check that a line is not just a FOV border artifact."""
    pt0, pt1 = line.endpoints
    if pt0 is None or pt1 is None:
        return True

    n = line_occlusion_num_samples
    dist_thresh = line_pt_dist_check_m

    t = np.linspace(0, 1, n).reshape(-1, 1)
    samples = pt0 + t * (pt1 - pt0)

    dense_pts = original_segment.dense_points
    occ_pts = getattr(original_segment, "occluded_points", None)

    if dense_pts is not None and len(dense_pts) > 0:
        dists_to_dense = np.linalg.norm(
            samples[:, None, :2] - dense_pts[None, :, :2], axis=2
        ).min(axis=1)
        frac_near = np.mean(dists_to_dense < dist_thresh)
        if frac_near < line_frac_near_points:
            return False

    if occ_pts is not None and len(occ_pts) > 0:
        dists_to_occ = np.linalg.norm(
            samples[:, None, :2] - occ_pts[None, :, :2], axis=2
        ).min(axis=1)
        frac_non_occluded = np.mean(dists_to_occ >= dist_thresh)
        if frac_non_occluded < line_occlusion_req_non_occluded:
            return False

    return True
