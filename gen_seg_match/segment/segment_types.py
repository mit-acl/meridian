import numpy as np
from dataclasses import dataclass
import clipperpy

class GeneralSegment:
    
    @property
    def dim(self) -> int:
        return self.point.shape[0]
    
    def to_array(self) -> np.ndarray:
        raise NotImplementedError("to_array not implemented")
    
    def get_point(self) -> np.ndarray:
        return self.point.flatten()

@dataclass
class SegmentPoint(GeneralSegment):

    id: int
    point: np.ndarray

    def to_array(self) -> np.ndarray:
        return np.concatenate([[clipperpy.invariants.GeneralSegmentDistance.POINT.value], 
                               self.get_point()])

@dataclass
class SegmentLine(GeneralSegment):

    id: int
    point: np.ndarray
    direction: np.ndarray
    endpoints: np.ndarray = (None, None)
    
    # endpoints can be given such that if only one endpoint is given, it is assumed
    # that the ray extends from that endpoint infinitely along the positive direction vector

    @classmethod
    def from_endpoints(cls, id: int, pt1: np.ndarray, pt2: np.ndarray):
        direction = pt2 - pt1
        direction = direction / np.linalg.norm(direction)
        return cls(id=id, point=pt1, direction=direction, endpoints=(pt1, pt2))

    def to_array(self) -> np.ndarray:
        return np.concatenate([[clipperpy.invariants.GeneralSegmentDistance.LINE.value], 
                               self.get_point(), self.get_direction()])

    def get_direction(self) -> np.ndarray:
        return self.direction.flatten() / np.linalg.norm(self.direction)
    
    def is_parallel_to(self, other: 'SegmentLine', tol: float = 1e-3) -> bool:
        assert self.dim == other.dim, "Lines must be in the same dimension"
        cross_product = np.cross(self.get_direction(), other.get_direction())
        return np.linalg.norm(cross_product) < tol
    
    def has_point_on_line(self, point: np.ndarray, tol: float = 1e-3) -> bool:
        assert self.dim == point.shape[0], "Point must be in the same dimension as line"
        point_vec = point - self.get_point()
        cross_product = np.cross(self.get_direction(), point_vec)
        on_infinite_line = np.linalg.norm(cross_product) < tol
        
        if not on_infinite_line:
            return False
        if self.endpoints[0] is None and self.endpoints[1] is None:
            return True
        elif self.endpoints[0] is not None and self.endpoints[1] is not None:
            ep1_to_ep0 = self.endpoints[1] - self.endpoints[0]
            point_to_ep0 = point - self.endpoints[0]
            dot_product = np.dot(ep1_to_ep0, point_to_ep0)
            # dot product should be between 0 and |ep1_to_ep0|^2
            return 0 <= dot_product <= np.linalg.norm(ep1_to_ep0)**2
        else:
            ep0 = self.endpoints[0] if self.endpoints[0] is not None else self.endpoints[1]
            point_to_ep0 = point - ep0
            dot_product = np.dot(self.get_direction(), point_to_ep0)
            return dot_product >= 0

    def angle_between(self, other: 'SegmentLine') -> float:
        """Returns the angle in radians between two lines."""
        assert self.dim == other.dim, "Lines must be in the same dimension"
        dot_product = np.dot(self.get_direction(), other.get_direction())
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return angle
    
    def closest_points(self, other: 'SegmentLine') -> np.ndarray:
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
        if self.has_point_on_line(closest_pt_self) and other.has_point_on_line(closest_pt_other):
            return np.array([closest_pt_self, closest_pt_other])
        
        # closest points are not on both line segments, so check endpoints
        closest_pt_self = None
        closest_pt_other = None
        for ep in self.endpoints:
            if ep is None:
                continue
            pt_on_other = other.closest_point_to_point(ep)
            if closest_pt_self is None or np.linalg.norm(ep - pt_on_other) < np.linalg.norm(closest_pt_self - closest_pt_other):
                closest_pt_self = ep
                closest_pt_other = pt_on_other
                
        for ep in other.endpoints:
            if ep is None:
                continue
            pt_on_self = self.closest_point_to_point(ep)
            if closest_pt_self is None or np.linalg.norm(ep - pt_on_self) < np.linalg.norm(closest_pt_self - closest_pt_other):
                closest_pt_self = pt_on_self
                closest_pt_other = ep
                
        return np.array([closest_pt_self, closest_pt_other])
    
    def closest_point_to_point(self, point: np.ndarray) -> np.ndarray:
        assert self.dim == point.shape[0], "Point must be in the same dimension as line"
        nearest_point_on_infinite = (self.get_point() + 
            np.dot(self.get_direction(), point - self.get_point()) / np.linalg.norm(self.get_direction())**2 
            * self.get_direction())
        
        if self.endpoints[0] is None and self.endpoints[1] is None:
            return nearest_point_on_infinite
        elif self.has_point_on_line(point):
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
    
    def min_dist_to(self, other: 'SegmentLine') -> float:
        """Returns the minimum distance between two line segments."""
        if self.is_parallel_to(other):
            # can use any point on one line and find closest point on the other
            return self.min_dist_to_point(other.get_point())
        else:
            closest_pts = self.closest_points(other)
            return np.linalg.norm(closest_pts[0] - closest_pts[1])


@dataclass
class SegmentPlane(GeneralSegment):

    id: int
    point: np.ndarray
    normal: np.ndarray

    def to_array(self) -> np.ndarray:
        return np.concatenate([[clipperpy.invariants.GeneralSegmentDistance.PLANE.value], 
                               self.get_point(), self.get_normal()])

    def get_normal(self) -> np.ndarray:
        return self.normal.flatten()
    
    
class ParallelLinesException(Exception):

    def __init__(self, line1: SegmentLine, line2: SegmentLine):
        self.line1 = line1
        self.line2 = line2
        message = f"Parallel lines detected: {line1} and {line2}"
        super().__init__(message)
