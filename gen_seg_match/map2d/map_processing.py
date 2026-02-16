import numpy as np
from typing import List
from copy import deepcopy

from gen_seg_match.segment.segment_types import SegmentLine, SegmentPoint, SegmentList


def _point_to_infinite_line_dist(point, line_point, line_dir):
    """Distance from a point to an infinite line defined by a point and unit direction."""
    v = point - line_point
    proj = np.dot(v, line_dir) * line_dir
    return np.linalg.norm(v - proj)


def merge_lines(line1: SegmentLine, line2: SegmentLine) -> SegmentLine:
    """Merge two lines into one by taking the endpoints that are farthest apart."""
    len_line1 = np.linalg.norm(line1.endpoints[1] - line1.endpoints[0])
    len_line2 = np.linalg.norm(line2.endpoints[1] - line2.endpoints[0])
    direction1 = line1.get_direction() * len_line1 + line2.get_direction() * len_line2
    direction2 = line1.get_direction() * len_line1 - line2.get_direction() * len_line2
    direction = (
        direction1
        if np.linalg.norm(direction1) > np.linalg.norm(direction2)
        else direction2
    )
    direction /= np.linalg.norm(direction)

    # project endpoints of both lines onto the merged line
    endpoints = [
        line1.endpoints[0],
        line1.endpoints[1],
        line2.endpoints[0],
        line2.endpoints[1],
    ]

    max_dist = -1
    pt1_idx = -1
    pt2_idx = -1
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            dist = np.linalg.norm(endpoints[i] - endpoints[j])
            if dist > max_dist:
                max_dist = dist
                pt1_idx = i
                pt2_idx = j

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
    return SegmentLine.from_endpoints(
        -1,
        endpoints[pt1_idx],
        endpoints[pt2_idx],
        cos_feature=merged_cos_feature,
        first_seen=first_seen,
        last_seen=last_seen,
        history=list(set(line1.history).union(set(line2.history))),
    )


def merge_points(pt1, pt2):
    new_point = (pt1.get_point() + pt2.get_point()) / 2
    return SegmentPoint(-1, new_point)


def clean_up_line_map(
    lines: List[SegmentLine],
    max_iter: int = 1000,
    angle_tol: float = np.deg2rad(5),
    dist_tol: float = 0.5,
    perp_dist_tol: float = 0.5,
    short_line_thresh: float = None,
) -> SegmentList:
    def merge_check(line1: SegmentLine, line2: SegmentLine):
        d1 = line1.direction
        d2 = line2.direction
        cross_norm = np.linalg.norm(np.cross(d1, d2))

        # Relax angle tolerance for short lines
        effective_angle_tol = angle_tol
        if short_line_thresh is not None:
            min_len = min(line1.get_length(), line2.get_length())
            if min_len < short_line_thresh:
                effective_angle_tol = angle_tol * (
                    short_line_thresh / max(min_len, 1e-6)
                )
                effective_angle_tol = min(effective_angle_tol, np.pi / 4)

        # Check angle tolerance first (most permissive — fast exit)
        if cross_norm >= effective_angle_tol:
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
    points: List[SegmentPoint], max_iter: int = 1000, dist_tol: float = 0.5
) -> List[SegmentPoint]:
    def merge_check(pt1, pt2):
        return np.linalg.norm(pt1.get_point() - pt2.get_point()) < dist_tol

    return _clean_up_map(points, merge_check, merge_points, max_iter)


def split_long_lines(lines: SegmentList, max_length: float) -> SegmentList:
    new_lines = []
    for line in lines:
        line_length = line.get_length()
        if line_length <= max_length:
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
    return SegmentList(new_lines)


def _clean_up_map(
    objects: list,
    merge_check: callable,
    merge_objects: callable,
    max_iter: int = 1000,
) -> SegmentList:
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

    return SegmentList(objects), outer_iter + 1
