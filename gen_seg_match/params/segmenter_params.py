import numpy as np
from dataclasses import dataclass
from typing import ClassVar, List, Tuple

from gen_seg_match.params.params_base import ParamsBase
from gen_seg_match.utils import expandvars_recursive


@dataclass
class SegmenterParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "segmenter"

    ##################

    model_type: str = "fastsam"
    weights_path: str = "$ROMAN_WEIGHTS/FastSAM-x.pt"
    yolo_weights_path: str = "$ROMAN_WEIGHTS/yolov7.pt"
    imgsz: Tuple[int, int] = (256, 256)
    conf: float = 0.5
    iou: float = 0.9
    device: str = "cuda"
    semantics: str = "dino"
    semantics_dim: int = 768
    erosion_size: int = 3
    voxel_size: float = 0.05
    ignore_labels: list = tuple(["person"])
    use_keep_labels: bool = False
    keep_labels: list = tuple([])
    keep_labels_option: dict = None
    keep_mask_minimal_intersection: float = 0.3
    rotate_img: str = None
    semantics: str = "dino"
    frame_descriptor: str = "dino-gem"
    yolo_imgsz: Tuple[int, int] = None
    use_point_cloud: bool = False
    depth_scale: float = 1e3
    max_depth: float = 7.5
    mask_downsample_factor: int = 8
    pcd_stride: int = 4
    triangle_ignore_masks: List[
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    ] = None

    def get_model_type(self):
        return self.model_type.lower()

    def __post_init__(self):
        if self.model_type is not None:
            self.model_type = self.model_type.lower()
        if self.frame_descriptor.lower() == "none":
            self.frame_descriptor = None
        self.weights_path = expandvars_recursive(self.weights_path)
        self.yolo_weights_path = expandvars_recursive(self.yolo_weights_path)
        if self.yolo_imgsz is None:
            self.yolo_imgsz = self.imgsz
