import os
from dataclasses import dataclass
from typing import ClassVar, List, Tuple

from gen_seg_match.params.params_base import ParamsBase
from gen_seg_match.utils import expandvars_recursive


@dataclass
class SegmenterParamsBase(ParamsBase):
    params_key: ClassVar[str] = "segmenter"

    # Segmentation model
    model_type: str = "fastsam"
    weights_path: str = "$ROMAN_WEIGHTS/FastSAM-x.pt"
    imgsz: Tuple[int, int] = (256, 256)
    conf: float = 0.5
    iou: float = 0.9
    device: str = "cuda"

    # Semantics
    semantics: str = "dino"
    semantics_dim: int = 768
    semantics_size: str = "base"
    dino_half: bool = False
    dinov3_path: str = "~/code/dinov3"
    dinov3_weights: str = None

    # Shared
    triangle_ignore_masks: List[
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    ] = None
    frame_descriptor: str = None

    # AnyLoc params (used when frame_descriptor == "anyloc")
    anyloc_path: str = "~/code/AnyLoc"
    anyloc_vocab_dir: str = "~/code/AnyLoc/demo/cache"
    anyloc_domain: str = "urban"
    anyloc_num_clusters: int = 32
    anyloc_dino_model: str = "dinov2_vitg14"
    anyloc_layer: int = 31
    anyloc_facet: str = "value"

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
        self.dinov3_path = os.path.expanduser(self.dinov3_path)
        if self.dinov3_weights is not None:
            self.dinov3_weights = os.path.expanduser(self.dinov3_weights)
        self.anyloc_path = os.path.expanduser(self.anyloc_path)
        self.anyloc_vocab_dir = os.path.expanduser(self.anyloc_vocab_dir)
