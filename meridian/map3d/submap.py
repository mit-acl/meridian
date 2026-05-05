import numpy as np
from dataclasses import dataclass
from robotdatapy.transform import transform
import pickle
from enum import Enum

from meridian.primitive.primitive_list import PrimitiveList


class FrameType(Enum):
    BASE_LINK = "base_link"
    CAMERA = "camera"
    GRAVITY_ALIGNED_FLU = "gravity_aligned_flu"
    UTM = "utm"
    ODOMETRY = "odometry"
    IMG_PATCH_TOP_LEFT_CORNER = "img_patch_top_left_corner"


@dataclass
class Submap:
    id: int
    time: float
    segments: PrimitiveList
    pose: np.ndarray
    segment_frame: FrameType
    gravity_dir: np.ndarray = None
    descriptor: np.ndarray = None
    metadata: dict = None

    @property
    def position(self):
        return self.pose[:3, 3]

    @property
    def segments_as_global_points(self):
        # self.pose returns T_odom_submap
        # which is transformation from submap center frame to odom frame
        # so this transforms segments back to the global (odom) frame
        T_odom_center = self.pose
        return transform(
            T_odom_center, np.vstack([seg.center.T for seg in self.segments])
        )  # (1, 3) -> (N, 3)

    @property
    def segment_ids(self):
        return [seg.id for seg in self.segments]

    def __len__(self):
        return len(self.segments)

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    def load(filepath: str) -> "Submap":
        with open(filepath, "rb") as f:
            submap = pickle.load(f)
        return submap
