import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import alphashape
import circle_fit
import numpy as np
import shapely

from gen_seg_match.map2d.map_processing import clean_up_line_map
from gen_seg_match.params.segment_to_primitive_params import (
    SegmentToPrimitiveConversionParams,
)
from gen_seg_match.segment.aerial_segment import AerialSegment, _grid_downsample_2d
from gen_seg_match.segment.segment_types import (
    SegmentLine,
    SegmentList,
    SegmentPoint,
)

logger = logging.getLogger(__name__)

Crop = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Module-level worker functions (picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _compute_alpha_shape(
    points: np.ndarray,
    alpha: float,
    grid_downsample: float,
    max_n_pts: int,
    alpha_ref_size: float,
    max_extent: float,
) -> Optional[np.ndarray]:
    """Compute alpha shape from raw points. Standalone for pickling."""
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
        shape = alphashape.alphashape(pts, alpha=alpha)
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
    Returns a list of SegmentPoint and SegmentLine objects.
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

    if area < params.min_area_m_sq:
        return []

    # Small segments become points
    if area < params.point_max_area_m_sq and max_extent < params.point_max_len_m:
        if not pt_within_border(center):
            return []
        return [
            SegmentPoint(
                j,
                center,
                cos_feature=semantic_descriptor,
                first_seen=first_seen,
                last_seen=last_seen,
                history=[seg_id],
            )
        ]

    # Compute alpha shape
    alpha_shape = _compute_alpha_shape(
        points,
        alpha=params.alpha_shape_alpha,
        grid_downsample=params.alpha_shape_grid_downsample,
        max_n_pts=params.alpha_shape_max_n_pts,
        alpha_ref_size=params.alpha_shape_ref_size_m,
        max_extent=max_extent,
    )
    if alpha_shape is None:
        return []

    # Circle fit check for medium segments
    if area < params.circle_point_max_area:
        alpha_pts = alpha_shape
        if alpha_pts.size == 0:
            return []
        if np.allclose(alpha_pts[0], alpha_pts[-1]):
            alpha_pts = alpha_pts[:-1]
        if len(alpha_pts) >= 3:
            xc, yc, r, s = circle_fit.least_squares_circle(alpha_pts)
            if (
                r > 0
                and s < params.circle_point_rad_frac_fit_err * r
                and r < params.circle_point_max_rad
            ):
                c = np.array([xc, yc])
                if pt_within_border(c):
                    return [
                        SegmentPoint(
                            j,
                            c,
                            cos_feature=semantic_descriptor,
                            first_seen=first_seen,
                            last_seen=last_seen,
                            history=[seg_id],
                        )
                    ]
                return []

    # Extract lines from alpha shape edges
    lines = []
    for i, pt0 in enumerate(alpha_shape):
        pt1 = alpha_shape[i + 1 if i + 1 < len(alpha_shape) else 0]
        keep = np.linalg.norm(pt1 - pt0) > params.line_min_length_m
        keep &= pt_within_border(pt0) and pt_within_border(pt1)
        if keep:
            line = SegmentLine.from_endpoints(
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
    """Converts 2D dense segments (AerialSegment) to sparse primitives (points + lines)."""

    def __init__(self, params: SegmentToPrimitiveConversionParams):
        self.params = params

    def convert(
        self,
        aerial_segments: List[AerialSegment],
        pixel_len_m: float = None,
        crop: Crop = None,
        border_dist_m: float = 0.5,
        convert_to_infinite: bool = True,
    ) -> SegmentList:
        """Convert aerial segments to sparse point/line primitives.

        Args:
            aerial_segments: Dense 2D segments to convert.
            pixel_len_m: Pixel size in meters (needed for border filtering with crop).
            crop: (x1, y1, x2, y2) pixel crop bounds for border filtering.
            border_dist_m: Minimum distance from border for segment inclusion.
            convert_to_infinite: Whether to convert long lines to infinite.

        Returns:
            SegmentList of sparse SegmentPoint and SegmentLine primitives.
        """
        if not aerial_segments:
            return SegmentList()

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

        if max_workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_classify_single_segment, *task): i
                    for i, task in enumerate(tasks)
                }
                for future in as_completed(futures):
                    all_primitives.extend(future.result())
        else:
            for task in tasks:
                all_primitives.extend(_classify_single_segment(*task))

        result = SegmentList(all_primitives)
        result.reindex()

        # Post-processing: line cleanup and merge (sequential, needs full set)
        result = self._cleanup_and_merge(
            result, convert_to_infinite=convert_to_infinite
        )

        return result

    def _cleanup_and_merge(
        self, segments: SegmentList, convert_to_infinite: bool = True
    ) -> SegmentList:
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
            merged_lines = SegmentList()

        result = points + merged_lines
        if convert_to_infinite:
            convert_long_lines_to_infinite(result, self.params.line_len_to_infinite)
        result.reindex()
        return result


def convert_long_lines_to_infinite(segments: SegmentList, threshold: float):
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
    line: SegmentLine,
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
