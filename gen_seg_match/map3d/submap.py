import numpy as np
from dataclasses import dataclass
from typing import List
from robotdatapy.transform import transform
from roman.utils import transform_rm_roll_pitch

from gen_seg_match.segment.segment_types import GeneralSegment

@dataclass
class Submap:

    id: int
    time: float
    segments: List[GeneralSegment]
    segment_ids: List[int]
    pose_flu: np.ndarray
    pose_flu_gt: np.ndarray = None
    segment_frame: str = 'submap_gravity_aligned'
    descriptor: np.ndarray = None

    @property
    def pose_gravity_aligned(self):
        return transform_rm_roll_pitch(self.pose_flu)
    
    @property
    def pose_gravity_aligned_gt(self):
        return transform_rm_roll_pitch(self.pose_flu_gt)
    
    @property
    def position(self):
        return self.pose_flu[:3,3]
    
    @property
    def position_gt(self):
        return self.pose_flu_gt[:3,3]
    
    @property
    def has_gt(self):
        return self.pose_flu_gt is not None
    
    @property
    def segments_as_global_points(self):
        # self.pose_gravity_aligned returns T_odom_center
        # which is transformation from center frame to odom frame
        # so this transforms segments back to the global (odom) frame
        T_odom_center = self.pose_gravity_aligned_gt if self.pose_flu_gt is not None else self.pose_gravity_aligned
        return transform(T_odom_center, np.vstack([seg.center.T for seg in self.segments])) # (1, 3) -> (N, 3)

    def __len__(self):
        return len(self.segments)
