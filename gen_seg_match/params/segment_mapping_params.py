import numpy as np
from dataclasses import dataclass
from typing import ClassVar, List, Tuple, Union

from gen_seg_match.params.params_base import ParamsBase


@dataclass
class SegmentMappingParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "segment_mapping"

    ##################

    # Association
    geometric_association_method: str = "iom"
    semantic_association_method: str = "cosine_similarity"
    geometric_score_range: Tuple[float] = (0.2, 1.0)
    semantic_score_range: Tuple[float] = (0.7, 1.0)
    min_2d_iou: Union[float, None] = None

    # Lifecycle
    min_sightings: int = 2
    max_t_no_sightings: float = 0.4
    mask_downsample_factor: int = 8

    # Segment filtering
    min_max_extent: float = 0.25
    clustering_epsilon: float = 0.25

    # Graveyard
    segment_graveyard_time: float = 15.0
    segment_graveyard_dist: float = 10.0

    # Voxelization
    iou_voxel_size: float = 0.25
    segment_voxel_size: float = 0.05
    segment_outlier_removal_std: float = 0.0  # disabled by default

    # Pipeline
    dt: float = 1 / 6  # time step for iterating through data (mapping frequency)
    T_camera_flu: Union[List, None] = None  # transform from camera to FLU frame

    def __post_init__(self):
        if (
            self.semantic_association_method is not None
            and self.semantic_association_method.lower() == "none"
        ):
            self.semantic_association_method = None

        if self.T_camera_flu is not None:
            self.T_camera_flu = np.array(self.T_camera_flu).reshape((4, 4))
        else:
            self.T_camera_flu = np.eye(4)

    def get_map_segment_params(self):
        from gen_seg_match.segment.map_segment import MapSegmentParams

        return MapSegmentParams(
            voxel_size=self.segment_voxel_size,
            outlier_removal_std=self.segment_outlier_removal_std,
        )
