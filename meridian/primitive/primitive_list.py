import numpy as np
import pickle
from typing import List

from meridian.primitive.primitive import Primitive, PointPrimitive, LinePrimitive


class PrimitiveList(List[Primitive]):
    """A list of Primitive objects with some helper functions."""

    def __add__(self, other: "PrimitiveList") -> "PrimitiveList":
        return PrimitiveList(super().__add__(other))

    @classmethod
    def load(cls, filepath: str) -> "PrimitiveList":
        """Loads a primitive list from a pickle file."""
        with open(filepath, "rb") as f:
            primitive_list = pickle.load(f)
        return primitive_list

    def save(self, filepath: str):
        """Saves the primitive list to a pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @property
    def first_seen(self) -> float:
        return min(seg.first_seen for seg in self)

    @property
    def last_seen(self) -> float:
        return max(seg.last_seen for seg in self)

    @property
    def ids(self) -> List[int]:
        return [seg.id for seg in self]

    def get_points(self) -> "PointList":
        return PointList([seg for seg in self if type(seg) is PointPrimitive])

    def get_lines(self) -> "LineList":
        return LineList([seg for seg in self if type(seg) is LinePrimitive])

    def type_ordered(self) -> "PrimitiveList":
        cache = self.__dict__.get("_type_ordered_cache")
        if cache is None or cache[0] != len(self):
            points = self.get_points()
            lines = self.get_lines()
            ordered = PrimitiveList(points + lines)
            ids = np.fromiter(
                (s.id for s in ordered), dtype=np.int64, count=len(ordered)
            )
            self.__dict__["_type_ordered_cache"] = (len(self), ordered, ids)
            return ordered
        return cache[1]

    def type_ordered_ids(self) -> np.ndarray:
        self.type_ordered()
        return self.__dict__["_type_ordered_cache"][2]

    def get_type_ordered_idx(self, idx) -> Primitive:
        return self.type_ordered()[idx]

    def _id_index(self) -> dict:
        cache = self.__dict__.get("_id_index_cache")
        if cache is None or cache[0] != len(self):
            index = {seg.id: seg for seg in self}
            self.__dict__["_id_index_cache"] = (len(self), index)
            return index
        return cache[1]

    def get_segment_from_id(self, id) -> Primitive:
        return self._id_index().get(id)

    def sublist_from_ids(self, ids: List[int]) -> "PrimitiveList":
        index = self._id_index()
        return PrimitiveList([index[id_i] for id_i in ids if id_i in index])

    def has_id(self, id) -> bool:
        return id in self._id_index()

    def transform(self, T: np.ndarray) -> "PrimitiveList":
        for seg in self:
            seg.transform(T)
        return self

    def copy(self) -> "PrimitiveList":
        return PrimitiveList([seg.copy() for seg in self])

    def get_mean_point(self) -> np.ndarray:
        all_points = np.array([seg.get_point() for seg in self])
        return np.mean(all_points, axis=0)

    def reindex(self):
        for new_id, seg in enumerate(self):
            seg.id = new_id
        return self

    def to_dim(self, dim: int) -> "PrimitiveList":
        return PrimitiveList([seg.to_dim(dim) for seg in self])

    @property
    def dim(self) -> int:
        if len(self) == 0:
            return 0
        dim = self[0].dim
        for seg in self:
            if seg.dim != dim:
                return None
        return dim


class PointList(PrimitiveList[PointPrimitive]):
    def __post_init__(self):
        for seg in self:
            assert type(seg) == PointPrimitive, (
                "All primitives must be of type PointPrimitive"
            )

    @property
    def points(self) -> np.ndarray:
        return np.array([seg.point for seg in self]).reshape(len(self), self.dim)


class LineList(PrimitiveList[LinePrimitive]):
    def __post_init__(self):
        for seg in self:
            assert type(seg) == LinePrimitive, (
                "All primitives must be of type LinePrimitive"
            )

    @property
    def directions(self) -> np.ndarray:
        return np.array([seg.direction for seg in self]).reshape(len(self), self.dim)

    @property
    def moments(self) -> np.ndarray:
        return np.array([np.cross(seg.point, seg.direction) for seg in self]).reshape(
            len(self), self.dim
        )
