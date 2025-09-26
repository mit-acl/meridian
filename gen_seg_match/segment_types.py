import numpy as np
from dataclasses import dataclass
import clipperpy


class GeneralSegment:
    
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

    def get_direction(self) -> np.ndarray:
        return self.direction.flatten()

    def to_array(self) -> np.ndarray:
        return np.concatenate([[clipperpy.invariants.GeneralSegmentDistance.LINE.value], 
                               self.get_point(), self.get_direction()])

@dataclass
class SegmentPlane(GeneralSegment):

    id: int
    point: np.ndarray
    normal: np.ndarray

    def get_normal(self) -> np.ndarray:
        return self.normal.flatten()
    
    def to_array(self) -> np.ndarray:
        return np.concatenate([[clipperpy.invariants.GeneralSegmentDistance.PLANE.value], 
                               self.get_point(), self.get_normal()])