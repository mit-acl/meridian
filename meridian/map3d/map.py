import numpy as np
import os
import pickle
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List

from meridian.map3d.map_segment import MapSegment


@dataclass(frozen=True)
class SegmentMap:
    segments: List[MapSegment]
    trajectory: List[np.ndarray]  # 4x4 poses (FLU frame)
    times: List[float]
    descriptors: List[np.ndarray] = None
    descriptor_type: str = None

    def __post_init__(self):
        assert len(self.trajectory) == len(self.times), (
            "Trajectory and times must have the same length"
        )
        if self.descriptors is not None:
            assert len(self.descriptors) == len(self.times), (
                "Descriptors and times must have the same length"
            )
        for pose in self.trajectory:
            assert pose.shape == (4, 4), "Trajectory poses must be 4x4 matrices"

    def get_segment_by_id(self, seg_id) -> MapSegment:
        for seg in self.segments:
            if seg.id == seg_id:
                return seg
        return None

    def make_picklable(self):
        for seg in self.segments:
            seg.reset_memoized()

    def save(self, path):
        self.make_picklable()
        with open(os.path.expanduser(path), "wb") as f:
            pickle.dump(self, f, -1)

    @classmethod
    def from_pickle(cls, pickle_file: str):
        with open(os.path.expanduser(pickle_file), "rb") as f:
            segment_map = pickle.load(f)
            assert type(segment_map) == cls
            return segment_map

    @classmethod
    def concatenate(cls, maps: list):
        reference = maps[0]
        if len(maps) == 1:
            return reference
        elif len(maps) == 2:
            if len(reference.times) == 0:
                return maps[1]
            if len(maps[1].times) == 0:
                return reference

            other = deepcopy(maps[1])
            max_seg_id = max([seg.id for seg in reference.segments])
            for segment in other.segments:
                segment.id += max_seg_id
            return cls(
                segments=reference.segments + other.segments,
                trajectory=reference.trajectory + other.trajectory,
                times=reference.times + other.times,
                descriptors=reference.descriptors + other.descriptors
                if reference.descriptors is not None and other.descriptors is not None
                else None,
                descriptor_type=reference.descriptor_type or other.descriptor_type,
            )
        else:
            while len(maps) > 1:
                concatenated = cls.concatenate(maps[:2])
                maps = [concatenated] + maps[2:]
            return concatenated
