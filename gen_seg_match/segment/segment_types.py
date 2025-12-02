import numpy as np
from dataclasses import dataclass
import clipperpy
from robotdatapy import transform as transform
from typing import Tuple, List

from gen_seg_match.viz.utils import color_from_seed


class GeneralSegment:
    id: int
    point: np.ndarray
    ratio_feature: np.ndarray = None  # optional ratio feature vector
    cos_feature: np.ndarray = None  # optional cosine feature vector
    first_seen: float = None  # optional timestamp of first observation
    last_seen: float = None  # optional timestamp of last observation
    dense_points: np.ndarray = None  # optional dense point cloud

    @property
    def dim(self) -> int:
        return self.point.shape[0]

    def to_array(self, include_ratio=True, include_cos=True) -> np.ndarray:
        raise NotImplementedError("to_array not implemented")

    def get_point(self) -> np.ndarray:
        return self.point.flatten()

    def transform(self, T: np.ndarray):
        raise NotImplementedError("transform not implemented")

    def copy(self):
        raise NotImplementedError("copy not implemented")

    def color_from_id(self, order="rgb", num_type=int) -> tuple:
        """Returns a color tuple based on the segment ID."""
        return color_from_seed(self.id, order, num_type)

    def reference_time(self, use_avg_time=True):
        if not use_avg_time:
            return self.first_seen
        else:
            return (self.first_seen + self.last_seen) / 2.0

    def clear_dense_points(self):
        self.dense_points = None

    def _copy_optional_array(self, arr: np.ndarray) -> np.ndarray:
        return arr.copy() if arr is not None else None

    def _to_array_features(self, include_ratio=True, include_cos=True) -> np.ndarray:
        if self.ratio_feature is not None and include_ratio:
            ratio_feature = self.ratio_feature.flatten()
        else:
            ratio_feature = []
        if self.cos_feature is not None and include_cos:
            cos_feature = self.cos_feature.flatten()
        else:
            cos_feature = []
        return np.concatenate([ratio_feature, cos_feature])


@dataclass
class SegmentPoint(GeneralSegment):
    id: int
    point: np.ndarray
    ratio_feature: np.ndarray = None  # optional ratio feature vector
    cos_feature: np.ndarray = None  # optional cosine feature vector
    first_seen: float = None  # optional timestamp of first observation
    last_seen: float = None  # optional timestamp of last observation
    dense_points: np.ndarray = None  # optional dense point cloud

    def __post_init__(self):
        if self.cos_feature is not None:
            self.cos_feature /= np.linalg.norm(self.cos_feature)

    def to_array(self, include_ratio=True, include_cos=True) -> np.ndarray:
        features = self._to_array_features(include_ratio, include_cos)
        return np.concatenate(
            [
                [clipperpy.invariants.GeneralSegmentDistance.POINT.value],
                self.get_point(),
                features,
            ]
        )

    def transform(self, T):
        self.point = transform.transform(T, self.point)
        if self.dense_points is not None:
            self.dense_points = transform.transform(T, self.dense_points)
        return self

    def copy(self):
        return SegmentPoint(
            self.id,
            self.point.copy(),
            self._copy_optional_array(self.ratio_feature),
            self._copy_optional_array(self.cos_feature),
        )


@dataclass
class SegmentLine(GeneralSegment):
    id: int
    point: np.ndarray
    direction: np.ndarray = None
    endpoints: Tuple[np.ndarray, np.ndarray] = (None, None)
    ratio_feature: np.ndarray = None  # optional ratio feature vector
    cos_feature: np.ndarray = None  # optional cosine feature vector
    first_seen: float = None  # optional timestamp of first observation
    last_seen: float = None  # optional timestamp of last observation
    dense_points: np.ndarray = None  # optional dense point cloud

    # endpoints can be given such that if only one endpoint is given, it is assumed
    # that the ray extends from that endpoint infinitely along the positive direction vector

    def __post_init__(self):
        if self.direction is None:
            raise ValueError("Missing required field 'direction' for SegmentLine")
        if self.cos_feature is not None:
            self.cos_feature /= np.linalg.norm(self.cos_feature)
        self.direction = self.direction / np.linalg.norm(self.direction)

    @classmethod
    def from_endpoints(cls, id: int, pt1: np.ndarray, pt2: np.ndarray, **kwargs):
        direction = pt2 - pt1
        direction = direction / np.linalg.norm(direction)
        return cls(
            id=id, point=pt1, direction=direction, endpoints=(pt1, pt2), **kwargs
        )

    @property
    def num_endpoints(self) -> int:
        num_endpoints = 2
        for ep in self.endpoints:
            if ep is None:
                num_endpoints -= 1
        return num_endpoints

    def to_array(self, include_ratio=True, include_cos=True) -> np.ndarray:
        endpoints1 = []
        endpoints2 = []
        num_endpoints = 0
        if self.endpoints[0] is not None:
            endpoints1 = self.endpoints[0]
            num_endpoints += 1
        if self.endpoints[1] is not None:
            if num_endpoints == 0:
                endpoints1 = self.endpoints[1]
            else:
                endpoints2 = self.endpoints[1]
            num_endpoints += 1
        features = self._to_array_features(include_ratio, include_cos)

        return np.concatenate(
            [
                [clipperpy.invariants.GeneralSegmentDistance.LINE.value],
                self.get_point(),
                self.get_direction(),
                [num_endpoints],
                endpoints1,
                endpoints2,
                features,
            ]
        )

    def get_direction(self) -> np.ndarray:
        return self.direction.flatten() / np.linalg.norm(self.direction)

    def transform(self, T: np.ndarray):
        self.point = transform.transform(T, self.point)
        self.direction = (T[0:3, 0:3] @ self.direction.reshape((3, 1))).flatten()
        if self.endpoints[0] is not None:
            self.endpoints = (
                transform.transform(T, self.endpoints[0]),
                self.endpoints[1],
            )
        if self.endpoints[1] is not None:
            self.endpoints = (
                self.endpoints[0],
                transform.transform(T, self.endpoints[1]),
            )
        if self.dense_points is not None:
            self.dense_points = transform.transform(T, self.dense_points)
        return self

    def copy(self):
        return SegmentLine(
            self.id,
            self.point.copy(),
            self.direction.copy(),
            (
                self.endpoints[0].copy() if self.endpoints[0] is not None else None,
                self.endpoints[1].copy() if self.endpoints[1] is not None else None,
            ),
            self._copy_optional_array(self.ratio_feature),
            self._copy_optional_array(self.cos_feature),
        )

    def get_length(self):
        if self.endpoints[0] is not None and self.endpoints[1] is not None:
            return np.linalg.norm(self.endpoints[1] - self.endpoints[0])
        else:
            return np.inf

    def is_parallel_to(self, other: "SegmentLine", tol: float = 1e-3) -> bool:
        assert self.dim == other.dim, "Lines must be in the same dimension"
        cross_product = np.cross(self.get_direction(), other.get_direction())
        return np.linalg.norm(cross_product) < tol

    def has_point_on_line(self, point: np.ndarray, tol: float = 1e-3) -> bool:
        assert self.dim == point.shape[0], "Point must be in the same dimension as line"
        # point_vec = point - self.get_point()
        # cross_product = np.cross(self.get_direction(), point_vec)
        # on_infinite_line = np.linalg.norm(cross_product) < tol
        closest_point_on_infinite = self.closest_point_to_point(
            point, use_infinite_line=True
        )
        on_infinite_line = np.linalg.norm(closest_point_on_infinite - point) < tol

        if not on_infinite_line:
            return False
        if self.endpoints[0] is None and self.endpoints[1] is None:
            return True
        elif self.endpoints[0] is not None and self.endpoints[1] is not None:
            ep1_to_ep0 = self.endpoints[1] - self.endpoints[0]
            point_to_ep0 = point - self.endpoints[0]
            dot_product = np.dot(ep1_to_ep0, point_to_ep0)
            # dot product should be between 0 and |ep1_to_ep0|^2
            return 0 <= dot_product <= np.linalg.norm(ep1_to_ep0) ** 2
        else:
            ep0 = (
                self.endpoints[0]
                if self.endpoints[0] is not None
                else self.endpoints[1]
            )
            point_to_ep0 = point - ep0
            dot_product = np.dot(self.get_direction(), point_to_ep0)
            return dot_product >= 0

    def angle_between(self, other: "SegmentLine") -> float:
        """Returns the angle in radians between two lines."""
        assert self.dim == other.dim, "Lines must be in the same dimension"
        dot_product = np.dot(self.get_direction(), other.get_direction())
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return angle

    def closest_points(self, other: "SegmentLine") -> np.ndarray:
        if self.is_parallel_to(other):
            raise ParallelLinesException(self, other)

        # find unit direction vector for line C, which is perpendicular to lines A and B
        if self.dim == 2:
            LHS = np.array([self.get_direction(), -other.get_direction()]).T
            RHS = other.get_point() - self.get_point()
        elif self.dim == 3:
            perp = np.cross(self.get_direction(), other.get_direction())
            perp /= np.linalg.norm(perp)
            RHS = other.get_point() - self.get_point()
            LHS = np.array([self.get_direction(), -other.get_direction(), perp]).T
        else:
            raise NotImplementedError("Closest points only implemented for 2D and 3D")

        # solve the system derived in user2255770's answer from StackExchange: https://math.stackexchange.com/q/1993990
        t = np.linalg.inv(LHS) @ RHS
        closest_pt_self = self.get_point() + t[0] * self.get_direction()
        closest_pt_other = other.get_point() + t[1] * other.get_direction()
        if self.has_point_on_line(closest_pt_self) and other.has_point_on_line(
            closest_pt_other
        ):
            return np.array([closest_pt_self, closest_pt_other])

        # closest points are not on both line segments, so check endpoints
        closest_pt_self = None
        closest_pt_other = None
        for ep in self.endpoints:
            if ep is None:
                continue
            pt_on_other = other.closest_point_to_point(ep)
            if closest_pt_self is None or np.linalg.norm(
                ep - pt_on_other
            ) < np.linalg.norm(closest_pt_self - closest_pt_other):
                closest_pt_self = ep
                closest_pt_other = pt_on_other

        for ep in other.endpoints:
            if ep is None:
                continue
            pt_on_self = self.closest_point_to_point(ep)
            if closest_pt_self is None or np.linalg.norm(
                ep - pt_on_self
            ) < np.linalg.norm(closest_pt_self - closest_pt_other):
                closest_pt_self = pt_on_self
                closest_pt_other = ep

        return np.array([closest_pt_self, closest_pt_other])

    def closest_point_to_point(
        self, point: np.ndarray, use_infinite_line=False
    ) -> np.ndarray:
        assert self.dim == point.shape[0], "Point must be in the same dimension as line"
        nearest_point_on_infinite = (
            self.get_point()
            + np.dot(self.get_direction(), point - self.get_point())
            * self.get_direction()
        )

        if use_infinite_line:
            return nearest_point_on_infinite
        elif self.endpoints[0] is None and self.endpoints[1] is None:
            return nearest_point_on_infinite
        elif self.has_point_on_line(nearest_point_on_infinite):
            return nearest_point_on_infinite

        # has at least one endpoint and the point on the infinite version of this line
        # is not on the line segment with endpoints
        # means that the closest point must be one of the endpoints
        closest_pt = None
        for ep in self.endpoints:
            if ep is None:
                continue
            if closest_pt is None:
                closest_pt = ep
            elif np.linalg.norm(point - ep) < np.linalg.norm(point - closest_pt):
                closest_pt = ep
        return closest_pt

    def min_dist_to_point(self, point: np.ndarray) -> float:
        """Returns the minimum distance between the line segment and a point."""
        closest_pt = self.closest_point_to_point(point)
        return np.linalg.norm(closest_pt - point)

    def min_dist_to(self, other: "SegmentLine") -> float:
        """Returns the minimum distance between two line segments."""
        if self.is_parallel_to(other):
            if min(self.num_endpoints, other.num_endpoints) == 0:
                # both lines are infinite, so just compute distance from point to line
                return self.min_dist_to_point(other.get_point())
            else:
                # at least one line has endpoints, so check all endpoint-to-line distances
                min_dist = np.inf
                for ep in self.endpoints:
                    if ep is not None:
                        dist = other.min_dist_to_point(ep)
                        if dist < min_dist:
                            min_dist = dist
                for ep in other.endpoints:
                    if ep is not None:
                        dist = self.min_dist_to_point(ep)
                        if dist < min_dist:
                            min_dist = dist
                return min_dist
        else:
            closest_pts = self.closest_points(other)
            return np.linalg.norm(closest_pts[0] - closest_pts[1])


@dataclass
class SegmentPlane(GeneralSegment):
    id: int
    point: np.ndarray
    normal: np.ndarray = None  # required normal vector
    cos_feature: np.ndarray = None  # optional cosine feature vector
    ratio_feature: np.ndarray = None  # optional ratio feature vector
    first_seen: float = None  # optional timestamp of first observation
    last_seen: float = None  # optional timestamp of last observation
    dense_points: np.ndarray = None  # optional dense point cloud

    def __post_init__(self):
        if self.normal is None:
            raise ValueError("Missing required field 'normal' for SegmentLine")
        if self.cos_feature is not None:
            self.cos_feature /= np.linalg.norm(self.cos_feature)

    def to_array(self, include_ratio=True, include_cos=True) -> np.ndarray:
        features = self._to_array_features(include_ratio, include_cos)

        return np.concatenate(
            [
                [clipperpy.invariants.GeneralSegmentDistance.PLANE.value],
                self.get_point(),
                self.get_normal(),
                features,
            ]
        )

    def get_normal(self) -> np.ndarray:
        return self.normal.flatten()


class ParallelLinesException(Exception):
    def __init__(self, line1: SegmentLine, line2: SegmentLine):
        self.line1 = line1
        self.line2 = line2
        message = f"Parallel lines detected: {line1} and {line2}"
        super().__init__(message)


class SegmentList(list):
    """A list of GeneralSegment objects with some helper functions."""

    @property
    def first_seen(self) -> float:
        return min(seg.first_seen for seg in self)

    @property
    def last_seen(self) -> float:
        return max(seg.last_seen for seg in self)

    def get_points(self) -> "SegmentList":
        return SegmentList([seg for seg in self if type(seg) is SegmentPoint])

    def get_lines(self) -> "SegmentList":
        return SegmentList([seg for seg in self if type(seg) is SegmentLine])

    def get_planes(self) -> "SegmentList":
        return SegmentList([seg for seg in self if type(seg) is SegmentPlane])

    def type_ordered(self) -> "SegmentList":
        points = self.get_points()
        lines = self.get_lines()
        planes = self.get_planes()
        return SegmentList(points + lines + planes)

    def get_type_ordered_idx(self, idx) -> GeneralSegment:
        return self.type_ordered()[idx]

    def get_segment_from_id(self, id) -> GeneralSegment:
        matching_segs = [seg for seg in self if seg.id == id]
        assert len(matching_segs) <= 1, f"Multiple segments with id {id} found"
        return matching_segs[0] if len(matching_segs) == 1 else None

    def sublist_from_ids(self, ids: List[int]) -> "SegmentList":
        return SegmentList([self.get_segment_from_id(id_i) for id_i in ids])

    def has_id(self, id) -> bool:
        return any(seg.id == id for seg in self)

    def transform(self, T: np.ndarray) -> "SegmentList":
        for seg in self:
            seg.transform(T)
        return self

    def get_mean_point(self) -> np.ndarray:
        all_points = np.array([seg.get_point() for seg in self])
        return np.mean(all_points, axis=0)
