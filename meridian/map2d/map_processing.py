import numpy as np
from typing import List
from copy import deepcopy

from meridian.primitive.primitive import LinePrimitive, PointPrimitive
from meridian.primitive.primitive_list import PrimitiveList


def _point_to_infinite_line_dist(point, line_point, line_dir):
    """Distance from a point to an infinite line defined by a point and unit direction."""
    v = point - line_point
    proj = np.dot(v, line_dir) * line_dir
    return np.linalg.norm(v - proj)


def merge_lines(line1: LinePrimitive, line2: LinePrimitive) -> LinePrimitive:
    """Merge two lines using least-squares fit of all constituent endpoints.

    Accumulates endpoints through successive merges so that the fitted direction
    is robust to noisy short-line endpoints.
    """
    # Collect all constituent endpoints (carried forward through merges)
    eps1 = getattr(line1, "_merged_endpoints", [line1.endpoints[0], line1.endpoints[1]])
    eps2 = getattr(line2, "_merged_endpoints", [line2.endpoints[0], line2.endpoints[1]])
    all_endpoints = eps1 + eps2

    # Least-squares line fit via PCA of endpoint cloud
    pts = np.array(all_endpoints)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]  # first principal component = line direction

    # Project all endpoints onto the fitted line, pick the two extremes
    projections = centered @ direction
    min_idx = int(np.argmin(projections))
    max_idx = int(np.argmax(projections))

    # Projected endpoints lie on the best-fit line (perpendicular noise removed)
    ep1 = centroid + projections[min_idx] * direction
    ep2 = centroid + projections[max_idx] * direction

    # TODO: we should probably keep track of the history of cosine features as we are
    # merging lines. Also, should probably weight by length.
    if line1.cos_feature is not None and line2.cos_feature is not None:
        merged_cos_feature = (
            line1.cos_feature * line1.get_length()
            + line2.cos_feature * line2.get_length()
        )
        merged_cos_feature /= np.linalg.norm(merged_cos_feature)
    else:
        merged_cos_feature = None
    assert type(line1.first_seen) == type(line2.first_seen)
    assert type(line1.last_seen) == type(line2.last_seen)
    first_seen = line1.first_seen
    last_seen = line1.last_seen
    if line2.first_seen is not None and line2.first_seen < first_seen:
        first_seen = line2.first_seen
    if line2.last_seen is not None and line2.last_seen > last_seen:
        last_seen = line2.last_seen
    merged = LinePrimitive.from_endpoints(
        -1,
        ep1,
        ep2,
        cos_feature=merged_cos_feature,
        first_seen=first_seen,
        last_seen=last_seen,
        history=list(set(line1.history).union(set(line2.history))),
    )
    merged._merged_endpoints = all_endpoints
    return merged


def merge_points(pt1, pt2):
    new_point = (pt1.get_point() + pt2.get_point()) / 2
    return PointPrimitive(-1, new_point)


def clean_up_line_map(
    lines: List[LinePrimitive],
    max_iter: int = 1000,
    angle_tol: float = np.deg2rad(5),
    dist_tol: float = 0.5,
    perp_dist_tol: float = 0.5,
    short_line_thresh: float = None,
    semantic_sim_thresh: float = None,
) -> PrimitiveList:
    def merge_check(line1: LinePrimitive, line2: LinePrimitive):
        len1 = line1.get_length()
        len2 = line2.get_length()
        # Cheap necessary condition (exact prune): line{1,2}.point lies on the
        # respective segment, so every point of a segment is within its length
        # of that point. Hence min segment-to-segment distance >=
        # ||p1 - p2|| - len1 - len2. If that already exceeds dist_tol the pair
        # can never satisfy min_dist < dist_tol, so skip before the expensive
        # closest-point geometry below. Far-apart pairs dominate, so this cuts
        # most of the O(iter*n^2) work without changing which pairs merge.
        if (
            np.linalg.norm(line1.point.ravel() - line2.point.ravel())
            > len1 + len2 + dist_tol
        ):
            return False

        d1 = line1.direction
        d2 = line2.direction
        cross_norm = np.linalg.norm(np.cross(d1, d2))

        # Relax angle tolerance for short lines
        effective_angle_tol = angle_tol
        if short_line_thresh is not None:
            min_len = min(len1, len2)
            if min_len < short_line_thresh:
                effective_angle_tol = angle_tol * (
                    short_line_thresh / max(min_len, 1e-6)
                )
                effective_angle_tol = min(effective_angle_tol, np.pi / 4)

        # Check angle tolerance first (most permissive — fast exit)
        if cross_norm >= effective_angle_tol:
            return False

        # Check semantic similarity (cosine sim of unit-normalized cos_feature)
        if semantic_sim_thresh is not None:
            if line1.cos_feature is not None and line2.cos_feature is not None:
                if np.dot(line1.cos_feature, line2.cos_feature) < semantic_sim_thresh:
                    return False

        is_parallel = cross_norm < 1e-3

        if is_parallel:
            # Parallel case: perpendicular distance is point-to-infinite-line
            perp = _point_to_infinite_line_dist(line2.point, line1.point, d1)
            if perp > perp_dist_tol:
                return False
            # Segment-to-segment distance for dist_tol
            min_dist = line1.min_dist_to(line2)
            return min_dist < dist_tol
        else:
            # Near-parallel case: use closest points for perpendicular distance
            closest_pts = line1.closest_points(line2)
            perp_1 = _point_to_infinite_line_dist(closest_pts[1], line1.point, d1)
            perp_2 = _point_to_infinite_line_dist(closest_pts[0], line2.point, d2)
            if perp_1 > perp_dist_tol or perp_2 > perp_dist_tol:
                return False
            min_dist = np.linalg.norm(closest_pts[0] - closest_pts[1])
            return min_dist < dist_tol

    assert perp_dist_tol <= dist_tol, (
        "perp_dist_tol should be less than or equal to dist_tol"
    )

    return _clean_up_map(lines, merge_check, merge_lines, max_iter)


def clean_up_point_map(
    points: List[PointPrimitive], max_iter: int = 1000, dist_tol: float = 0.5
) -> List[PointPrimitive]:
    def merge_check(pt1, pt2):
        return np.linalg.norm(pt1.get_point() - pt2.get_point()) < dist_tol

    return _clean_up_map(points, merge_check, merge_points, max_iter)


def split_long_lines(lines: PrimitiveList, max_length: float) -> PrimitiveList:
    new_lines = []
    for line in lines:
        line_length = line.get_length()
        if line_length <= max_length:
            new_lines.append(line)
        elif line.num_endpoints < 2:
            new_lines.append(line)
        else:
            num_splits = int(np.ceil(line_length / max_length))
            start_pt = line.endpoints[0]
            end_pt = line.endpoints[1]
            direction = (end_pt - start_pt) / line_length
            segment_length = line_length / num_splits
            for i in range(num_splits):
                seg_start = start_pt + i * segment_length * direction
                seg_end = start_pt + (i + 1) * segment_length * direction
                new_line = deepcopy(line)
                new_line.endpoints = (seg_start, seg_end)
                new_lines.append(new_line)
    return PrimitiveList(new_lines)


def _clean_up_map(
    objects: list,
    merge_check: callable,
    merge_objects: callable,
    max_iter: int = 1000,
) -> PrimitiveList:
    # only shallow copy the list, not the objects themselves, since we are modifying in place
    objects = list(objects)
    prev_objects = list(objects)

    for outer_iter in range(max_iter):
        outer_changed = False
        for i in range(len(prev_objects) - 1, -1, -1):
            for j in range(i + 1, len(objects)):
                obj_i = prev_objects[i]
                obj_j = objects[j]
                if merge_check(obj_i, obj_j):
                    objects[i] = merge_objects(obj_i, obj_j)
                    del objects[j]
                    outer_changed = True
                    break
        if not outer_changed:
            break
        prev_objects = list(objects)

    return PrimitiveList(objects), outer_iter + 1
