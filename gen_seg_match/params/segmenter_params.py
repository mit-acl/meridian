from dataclasses import dataclass
from typing import ClassVar, Tuple

from gen_seg_match.params.segmenter_params_base import SegmenterParamsBase
from gen_seg_match.utils import expandvars_recursive


@dataclass
class SegmenterParams(SegmenterParamsBase):
    params_key: ClassVar[str] = "segmenter"

    # Override base default
    # None now

    # Ground-specific fields
    yolo_weights_path: str = "$ROMAN_WEIGHTS/yolov7.pt"
    erosion_size: int = 3
    voxel_size: float = 0.05
    ignore_labels: list = tuple([])
    use_keep_labels: bool = False
    keep_labels: list = tuple([])
    keep_labels_option: dict = None
    keep_mask_minimal_intersection: float = 0.3
    rotate_img: str = None
    yolo_imgsz: Tuple[int, int] = None
    use_point_cloud: bool = False
    depth_scale: float = 1e3
    max_depth: float = 7.5
    mask_downsample_factor: int = 8
    pcd_stride: int = 4
    min_mask_pixels: int = 0
    min_mask_image_fraction: float = 0.0
    occlusion_edge_img_frac: float = 0.02
    occlusion_edge_max_img_frac: float = 0.15
    occlusion_max_depth: float = 7.0
    outlier_removal_std: float = 1.0
    outlier_removal_dbscan_eps: float = 0.5
    outlier_removal_dbscan_min_points: int = 10
    min_occluded_unoccluded_dist_m: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        self.yolo_weights_path = expandvars_recursive(self.yolo_weights_path)
        if self.yolo_imgsz is None:
            self.yolo_imgsz = self.imgsz
