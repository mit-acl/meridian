"""Image-level VPR descriptors. Both share a frozen DINOv2 ViT-G/14, layer-31
value facet; weights ship in `third_party/vpr`. `describe` is the entry point: raw RGB
in, L2-normalized descriptor out.

    AnyLocPipeline       cosine VLAD over AnyLoc's cached centers; view-agnostic
    MeridianVprPipeline  trained NetVLAD head from `vpr`; two towers, so ground
                         and aerial crops go through different weights
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.hub  # stop torch.hub's GitHub check from hanging when offline
import torch.nn.functional as F

from vpr.data.transforms import default_transform
from vpr.models.vlad import VLAD as VprVLAD

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

        self.transform = default_transform()

    @classmethod
    def from_cached_centers(cls, centers_path: str, **kwargs) -> "AnyLocPipeline":
        return cls(load_centers(centers_path), **kwargs)

    @torch.no_grad()
    def describe(self, img_rgb) -> np.ndarray:
        """RGB image -> 1-D global descriptor (num_clusters * D,), L2-normalized.

        `img_rgb` is an (H, W, 3) uint8 RGB array (or PIL image). Applies AnyLoc's
        ImageNet normalization and center-crops to a multiple of the ViT patch
        size before running the full extract -> aggregate pipeline.
        """
        if isinstance(img_rgb, np.ndarray):
            img_rgb = np.ascontiguousarray(img_rgb)
        img_pt = self.transform(img_rgb)[None, ...]
        gd = self.forward(img_pt).float()
        return gd.squeeze(0).cpu().numpy()

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


# -----------------------------------------------------------------------------
# meridian-vpr: the trained two-tower head from third_party/vpr.
# -----------------------------------------------------------------------------

# Ground and aerial segmenters are separate objects, so without this each loads
# its own ViT-G, which is wasted memory.
_BACKBONE_CACHE: dict = {}


def resolve_vpr_checkpoint(path: str) -> Path:
    """A checkpoint file, or a run directory whose highest `checkpoints/epoch_*.pt` wins."""
    p = Path(path).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        ckpts = sorted(
            p.glob("checkpoints/epoch_*.pt"),
            key=lambda f: int(f.stem.split("_")[-1]),
        )
        if ckpts:
            return ckpts[-1]
        raise FileNotFoundError(f"no checkpoints/epoch_*.pt under {p}")
    raise FileNotFoundError(f"meridian-vpr checkpoint not found: {p}")


class MeridianVprPipeline(torch.nn.Module):
    """Frozen DINOv2 -> trained two-tower NetVLAD head (`vpr.models.cvmnet`).

    The checkpoint stores the backbone and head config it was trained with, so
    nothing here is configured.
    """

    def __init__(self, ckpt_path: str, fp16: bool = True,
                 device: torch.device | str = "cuda"):
        super().__init__()
        from vpr.data.transforms import default_transform
        from vpr.models.backbone import DinoBackbone
        from vpr.models.cvmnet import CVMNetHead

        self.device = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32

        ckpt = torch.load(resolve_vpr_checkpoint(ckpt_path), map_location="cpu",
                          weights_only=False)
        if "config" not in ckpt:
            raise ValueError(
                f"{ckpt_path} has no embedded config -- it is not a vpr "
                f"cross-view checkpoint")
        bcfg, hcfg = ckpt["config"]["backbone"], ckpt["config"]["head"]

        key = (bcfg["model_name"], bcfg["source"], bcfg["layer"], bcfg["facet"],
               str(self.device), self.dtype)
        backbone = _BACKBONE_CACHE.get(key)
        if backbone is None:
            backbone = DinoBackbone(
                bcfg["model_name"], layer=bcfg["layer"], facet=bcfg["facet"],
                source=bcfg["source"],
            ).to(self.device, self.dtype).eval()
            _BACKBONE_CACHE[key] = backbone
        self.backbone = backbone

        self.head = CVMNetHead(
            feature_dim=backbone.feature_dim,
            num_clusters=hcfg["num_clusters"],
            fc_hidden_dims=hcfg["fc_hidden_dims"],
            descriptor_dim=hcfg["descriptor_dim"],
            netvlad_gating=hcfg["netvlad_gating"],
            netvlad_normalize_input=hcfg["netvlad_normalize_input"],
            share_branch_weights=hcfg["share_branch_weights"],
            netvlad_assign_groups=hcfg.get("netvlad_assign_groups", 1),
            netvlad_group_weights=hcfg.get("netvlad_group_weights"),
        ).to(self.device).eval()
        self.head.load_state_dict(ckpt["model"])

        # Tokens are cast to fp32 for the head, whose weights load as fp32.
        patch = bcfg["patch_size"]
        size = bcfg["size"]
        if not isinstance(size, dict):
            size = {"sat": size, "grd": size}
        self.transforms = {
            view: default_transform(
                tuple(s) if isinstance(s, list) else s, patch)
            for view, s in (("satellite", size["sat"]), ("ground", size["grd"]))
        }

    @torch.no_grad()
    def describe(self, img_rgb, view: str) -> np.ndarray:
        """RGB + view ("ground" or "satellite") -> L2-normalized descriptor.

        The view picks the trained branch and the input size it was fit on.
        """
        from PIL import Image as PILImage

        if view not in self.transforms:
            raise ValueError(f"view must be 'ground' or 'satellite', got {view!r}")
        if isinstance(img_rgb, np.ndarray):
            img_rgb = PILImage.fromarray(np.ascontiguousarray(img_rgb))
        img_pt = self.transforms[view](img_rgb)[None].to(self.device)
        tokens = self.backbone(img_pt).float()
        encode = (self.head.encode_ground if view == "ground"
                  else self.head.encode_satellite)
        return encode(tokens).squeeze(0).cpu().numpy()
