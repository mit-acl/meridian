"""Minimal AnyLoc-free VPR pipeline: torch.hub DINOv2 + vpr cosine VLAD.

Reproduces AnyLoc's front-end (truncated backbone, layer-DESC_LAYER "value"
facet) and aggregation (cosine VLAD over AnyLoc's cached centers). 
The extractor copies AnyLoc's usage of the
torch.hub `facebookresearch/dinov2` `dinov2_vitg14` model: a forward hook on
`blocks[DESC_LAYER].attn.qkv` captures the attention "value" facet. We only run
blocks 0..DESC_LAYER (later blocks + final norm are skipped) and use fp16.
"""
from __future__ import annotations

import torch
import torch.hub  # stop torch.hub's GitHub check from hanging when offline
import torch.nn.functional as F

from meridian.segmenter.vlad import VLAD as VprVLAD

torch.hub._validate_not_a_forked_repo = lambda *a, **k: True


def load_centers(path: str) -> torch.Tensor:
    """Load cached cosine-kmeans cluster centers (K, D) straight off disk."""
    centers = torch.load(path, map_location="cpu")
    if not (torch.is_tensor(centers) and centers.dim() == 2):
        raise ValueError(f"expected a (K, D) tensor at {path}, got {type(centers)}")
    return centers.float()


class AnyLocPipeline(torch.nn.Module):
    """torch.hub DINOv2 (truncated, value facet) -> vpr cosine VLAD, no AnyLoc import.

    `fp16=True` runs the backbone in half precision -- halves its resident
    footprint and matches AnyLoc's optimized fp16 front-end.
    """

    MODEL = "dinov2_vitg14"

    def __init__(self, centers: torch.Tensor, desc_layer: int = 31,
                 fp16: bool = True, device: torch.device | str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.desc_layer = desc_layer
        self.fp16 = fp16
        self.dtype = torch.float16 if fp16 else torch.float32

        self.backbone = torch.hub.load("facebookresearch/dinov2", self.MODEL)
        self.backbone = self.backbone.to(self.device, self.dtype).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if desc_layer >= len(self.backbone.blocks):
            raise ValueError(f"desc_layer {desc_layer} >= {len(self.backbone.blocks)} blocks")

        # Capture the "value" facet exactly like AnyLoc: hook blocks[L].attn.qkv.
        self._qkv = None
        self.backbone.blocks[desc_layer].attn.qkv.register_forward_hook(
            lambda _m, _i, out: setattr(self, "_qkv", out))

        self.vlad = VprVLAD(centers, metric="cosine").to(self.device).eval()

    @classmethod
    def from_cached_centers(cls, centers_path: str, **kwargs) -> "AnyLocPipeline":
        return cls(load_centers(centers_path), **kwargs)

    @torch.no_grad()
    def extract(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) ImageNet-normalized -> (B, N, D) value-facet tokens.

        Runs only blocks 0..desc_layer (the hook fires inside block desc_layer;
        later blocks + final norm are skipped), then -- like AnyLoc -- drops the
        CLS token, slices the "value" third of qkv, and L2-normalizes.
        """
        img = img.to(self.device, self.dtype)
        x = self.backbone.prepare_tokens_with_masks(img)
        for blk in self.backbone.blocks[:self.desc_layer + 1]:
            x = blk(x)
        res = self._qkv                        # (B, N+1, 3*D) captured at block L
        self._qkv = None
        res = res[:, 1:, ...]                  # drop CLS (vitg14 has no registers)
        d_len = res.shape[2] // 3
        res = res[:, :, 2 * d_len:]            # "value" facet
        return F.normalize(res.float(), dim=-1)

    @torch.no_grad()
    def aggregate(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, N, D) -> (B, K*D) L2-normalized global descriptor."""
        return self.vlad(feats.to(self.device))

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.aggregate(self.extract(img))
