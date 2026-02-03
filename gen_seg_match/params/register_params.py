import numpy as np
import torch
from dataclasses import dataclass, field
from typing import ClassVar, Optional
from gen_seg_match.params.params_base import ParamsBase


@dataclass
class RegisterParams(ParamsBase):
    # class attribute
    params_key: ClassVar[str] = "register"

    # Parameters
    only_use_points: bool = False
    use_gravity: bool = True

    point_weight: float = 1.0
    line_direction_weight: float = 1.0
    line_moment_weight: float = 1.0
    plane_normal_weight: float = 1.0
    plane_offset_weight: float = 1.0
    gravity_weight: float = 1.0

    lin_eps: Optional[float] = 1e-4
    dup_eps: Optional[float] = 0.05
    run_gd: Optional[bool] = False
    device: Optional[torch.device] = "cpu"

    refine_transforms_kwargs: dict = None

    def __post_init__(self):
        if self.refine_transforms_kwargs is None:
            self.refine_transforms_kwargs = {}
