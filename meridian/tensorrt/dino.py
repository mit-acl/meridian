"""TensorRT for the two DINO backbones the segmenter runs.

    DinoSemanticsTRT  per-pixel semantics. HuggingFace weights, whichever family
                      and size `semantics`/`semantics_size` name -- DINOv2
                      ViT-L/14 by default, DINOv3 too. Full encoder, last hidden
                      state, one dynamic-shape engine for every input size.
    DinoVprTRT        VPR, shared by AnyLoc and meridian-vpr. torch.hub DINOv2
                      ViT-G/14, pinned by the VPR checkpoint, truncated at layer
                      31, value facet. One fixed-shape engine per input size.
"""

import logging

import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel
from vpr.models.backbone import DinoBackbone

from meridian.tensorrt.engine import TRTEngine, ensure_engine, export_onnx

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def model_geometry(model_name):
    """(patch_size, num_prefix_tokens, short_side) for a HuggingFace DINO model."""
    cfg = AutoConfig.from_pretrained(model_name)
    proc = AutoImageProcessor.from_pretrained(model_name, do_center_crop=False)
    size = proc.size.get("shortest_edge") or min(proc.size["height"], proc.size["width"])
    # 1 [CLS] plus any register tokens (DINOv3, DINOv2-with-registers).
    num_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
    return int(cfg.patch_size), num_prefix, int(size)


def preprocess(img_bgr, imgsz):
    """Resize and ImageNet-normalize a BGR image into a (1, 3, H, W) float32 CUDA tensor.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        imgsz: int to scale the short side to, or an (h, w) tuple for an exact
            resize.
    """
    if isinstance(imgsz, int):
        h, w = img_bgr.shape[:2]
        scale = imgsz / min(h, w)
        new_h, new_w = round(h * scale), round(w * scale)
    else:
        new_h, new_w = imgsz
    img = torch.from_numpy(np.ascontiguousarray(img_bgr)).cuda()
    resized = img.flip(-1).permute(2, 0, 1)[None].float()
    # PIL resizes horizontally then vertically, rounding to uint8 after each pass
    for size in [(resized.shape[2], new_w), (new_h, new_w)]:
        resized = torch.nn.functional.interpolate(
            resized, size=size, mode="bicubic", align_corners=False, antialias=True
        ).round_().clamp_(0, 255)
    mean = torch.as_tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)
    return (resized / 255.0 - mean) / std


###############################################################################
# Semantics: HuggingFace DINOv2/DINOv3, full encoder, dynamic shape.
###############################################################################


def _export_hf(model_name):
    """Return an export_onnx callable for the HuggingFace model `model_name`."""

    def run(onnx_path):
        class Wrapper(torch.nn.Module):
            """A single positional tensor in, last_hidden_state out."""

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, pixel_values):
                return self.model(pixel_values=pixel_values).last_hidden_state

        # SDPA has no opset-17 lowering, so export from the eager attention.
        model = AutoModel.from_pretrained(
            model_name, attn_implementation="eager"
        ).eval().cuda()
        export_onnx(
            Wrapper(model),
            torch.zeros(1, 3, 256, 256, device="cuda"),
            onnx_path,
            input_name="pixel_values",
            output_names=["last_hidden_state"],
            dynamic_axes={
                "pixel_values": {0: "batch", 2: "height", 3: "width"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
        )

    return run


class DinoSemanticsTRT(TRTEngine):
    """The semantics backbone on TensorRT, at dynamic resolution.

    Args:
        model_name: HuggingFace model id, e.g. 'facebook/dinov2-large'.
        directory: Where the cached .onnx and .trt live.
        imgsz: Short side the input is scaled to. Defaults to whatever the
            model's own image processor uses.
        max_scale: Largest supported side, as a multiple of `imgsz`; also the
            widest aspect ratio, since the short side is pinned.
        fp16: Allow FP16 kernels.
        timing: Print a per-call stage breakdown.
    """

    def __init__(self, model_name, directory, imgsz=None, max_scale=3,
                 fp16=False, timing=False):
        self.patch_size, self.num_prefix, default_imgsz = model_geometry(model_name)
        self.imgsz = int(imgsz or default_imgsz)
        self.max_scale = max_scale
        engine_path = ensure_engine(
            model_name.split("/")[-1], directory, "pixel_values",
            (
                (1, self.imgsz, self.imgsz),
                (1, self.imgsz, 2 * self.imgsz),
                (1, max_scale * self.imgsz, max_scale * self.imgsz),
            ),
            _export_hf(model_name), fp16=fp16,
        )
        super().__init__(engine_path, timing=timing)

    def warmup(self, iters=3, input_shape=None):
        super().warmup(input_shape or (1, 3, self.imgsz, 2 * self.imgsz), iters)

    def reshape_patches(self, last_hidden_state, input_hw):
        """Fold the patch tokens into a (1, h, w, D) grid, dropping the prefix."""
        h = input_hw[0] // self.patch_size
        w = input_hw[1] // self.patch_size
        return last_hidden_state[:, self.num_prefix:, :].reshape(1, h, w, -1)

    def embed(self, img_bgr, reshape=False):
        """Patch features for a BGR image.

        Returns:
            (1, num_tokens, D) including the prefix tokens, or (1, h, w, D)
            with them dropped when `reshape` is set.
        """
        t = [self.now()]
        inp = preprocess(img_bgr, self.imgsz)
        limit = self.max_scale * self.imgsz
        if max(inp.shape[2:]) > limit:
            raise ValueError(
                f"{img_bgr.shape[1]}x{img_bgr.shape[0]} scales to "
                f"{inp.shape[2]}x{inp.shape[3]}, past this engine's {limit}px "
                f"limit; rebuild it with max_scale >= "
                f"{-(-max(inp.shape[2:]) // self.imgsz)}"
            )
        t.append(self.now())
        (out,) = self._infer(inp)
        t.append(self.now())
        if reshape:
            out = self.reshape_patches(out, inp.shape[2:])
        t.append(self.now())
        self._report("DinoSemantics_TRT", t)
        return out


###############################################################################
# VPR: torch.hub DINOv2 ViT-G/14, truncated, value facet, fixed shape.
###############################################################################


def _export_hub(model_name, layer, facet, source, input_hw):
    """Return an export_onnx callable for the `vpr` DinoBackbone it describes."""

    def run(onnx_path):
        backbone = DinoBackbone(
            model_name, layer=layer, facet=facet, source=source
        ).eval().cuda()
        # The value facet is hooked, but the tracer follows it into the output.
        export_onnx(
            backbone,
            torch.zeros(1, 3, *input_hw, device="cuda"),
            onnx_path,
            input_name="pixel_values",
            output_names=["tokens"],
            dynamic_axes=None,
        )

    return run


class DinoVprTRT:
    """The frozen ViT-G/14 behind both VPR descriptors, on TensorRT.

    Args:
        model_name: torch.hub backbone id, e.g. 'dinov2_vitg14'.
        directory: Where the cached .onnx and .trt live.
        layer: Transformer block to read out of, 0-based. `DinoBackbone` stops
            there, so later blocks never enter the exported graph.
        facet: 'value' for the AnyLoc facet, or 'token'.
        source: 'hub' for torch.hub DINOv2 weights, 'hf' for HuggingFace.
        shapes: (h, w) sizes to build up front. Anything else is built lazily.
        fp16: Allow FP16 kernels.
        timing: Print a per-call stage breakdown.
    """

    def __init__(self, model_name, directory, layer=31, facet="value",
                 source="hub", shapes=(), fp16=False, timing=False):
        self.model_name = model_name
        self.directory = directory
        self.layer = layer
        self.facet = facet
        self.source = source
        self.fp16 = fp16
        self.timing = timing
        self.engines = {}
        self.feature_dim = None
        # Compile first
        for hw in shapes:
            self._engine_path(tuple(hw))
        for hw in shapes:
            self._engine(tuple(hw))

    def _engine_path(self, input_hw):
        """Compile the engine for this (h, w) if it is not already on disk."""
        h, w = input_hw
        name = f"{self.model_name}_l{self.layer}_{self.facet}"
        shape = (1, h, w)
        logger.info(f"{name}: preparing a {h}x{w} engine")
        return ensure_engine(
            name, self.directory, "pixel_values", (shape, shape, shape),
            _export_hub(self.model_name, self.layer, self.facet, self.source, input_hw),
            fp16=self.fp16, onnx_key=f"{name}_{h}x{w}",
        )

    def _engine(self, input_hw):
        """The engine for this (h, w), building and loading it on first use."""
        engine = self.engines.get(input_hw)
        if engine is not None:
            return engine
        h, w = input_hw
        engine = TRTEngine(self._engine_path(input_hw),
                           input_shape=(1, 3, h, w), timing=self.timing)
        engine.warmup((1, 3, h, w))
        self.engines[input_hw] = engine
        self.feature_dim = int(engine.output_tensors[engine.output_names[0]].shape[-1])
        return engine

    def warmup(self, iters=3):
        for engine in self.engines.values():
            engine.warmup(engine._input_shape, iters)

    def forward(self, pixel_values):
        """(1, 3, H, W) normalized pixels -> (1, N, D) patch tokens on the GPU.

        Mirrors `vpr.models.backbone.DinoBackbone.forward`.
        """
        inp = pixel_values
        if torch.is_tensor(inp) and inp.is_cuda:
            inp = inp.detach().float().contiguous()
        else:
            if torch.is_tensor(inp):
                inp = inp.detach().float().numpy()
            inp = np.ascontiguousarray(inp, dtype=np.float32)
        engine = self._engine(tuple(inp.shape[2:]))
        t = [engine.now(), engine.now()]
        (out,) = engine._infer(inp)
        t += [engine.now(), engine.now()]
        engine._report("DinoVpr_TRT", t)
        return out

    __call__ = forward
