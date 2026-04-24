from dataclasses import dataclass
from typing import ClassVar, List, Tuple

import numpy as np

from gen_seg_match.params.params_base import ParamsBase
from gen_seg_match.utils import expandvars_recursive


@dataclass
class SegmenterParamsBase(ParamsBase):
    params_key: ClassVar[str] = "segmenter"

    # Segmentation model
    model_type: str = "fastsam"
    weights_path: str = "$ROMAN_WEIGHTS/FastSAM-x.pt"
    imgsz: Tuple[int, int] = (1024, 1024)
    conf: float = 0.2
    iou: float = 0.9
    device: str = "cuda"

    # Semantics
    semantics: str = "dino"
    semantics_dim: int = 1024
    semantics_size: str = "large"
    dino_half: bool = False
    dinov3_path: str = "~/code/dinov3"
    dinov3_weights: str = None

    # Shared
    triangle_ignore_masks: List[
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    ] = None
    frame_descriptor: str = "anyloc"

    # AnyLoc params (used when frame_descriptor == "anyloc")
    anyloc_path: str = "${ANYLOC_PATH}"
    anyloc_vocab_dir: str = "${ANYLOC_VOCAB_DIR}"
    anyloc_domain: str = "urban"
    anyloc_num_clusters: int = 32
    anyloc_dino_model: str = "dinov2_vitg14"
    anyloc_layer: int = 31
    anyloc_facet: str = "value"

    # SALAD params (used when frame_descriptor == "salad")
    salad_path: str = "${SALAD_PATH}"

    def get_model_type(self):
        return self.model_type.lower()

    def __post_init__(self):
        if self.model_type is not None:
            self.model_type = self.model_type.lower()
        if (
            self.frame_descriptor is not None
            and self.frame_descriptor.lower() == "none"
        ):
            self.frame_descriptor = None
        self.weights_path = expandvars_recursive(self.weights_path)
        self.dinov3_path = expandvars_recursive(self.dinov3_path)
        if self.dinov3_weights is not None:
            self.dinov3_weights = expandvars_recursive(self.dinov3_weights)
        self.anyloc_path = expandvars_recursive(self.anyloc_path)
        self.anyloc_vocab_dir = expandvars_recursive(self.anyloc_vocab_dir)
        self.salad_path = expandvars_recursive(self.salad_path)


@dataclass
class SegmenterParams(SegmenterParamsBase):
    params_key: ClassVar[str] = "segmenter"

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


@dataclass
class AerialSegmenterParams(SegmenterParamsBase):
    params_key: ClassVar[str] = "aerial_segmenter"

    # Override base defaults for aerial
    model_type: str = "segment_anything"
    weights_path: str = "$ROMAN_WEIGHTS/sam_vit_l_0b3195.pth"
    conf: float = 0.1
    iou: float = 0.1
    imgsz: tuple = (1024, 1024)

    # Aerial-specific
    pixel_len_m: float = 0.01
    min_area: float = 0.01
    max_area: float = np.inf
    downsample_factor: int = 5
