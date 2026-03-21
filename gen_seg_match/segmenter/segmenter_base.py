import os

import cv2 as cv
import numpy as np
from numpy.typing import ArrayLike
import torch
from fastsam import FastSAMPrompt, FastSAM
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import AutoImageProcessor, AutoModel

from gen_seg_match.params.segmenter_params_base import SegmenterParamsBase


class SegmenterBase:
    """Base class for ground and aerial segmenters.

    Handles segmentation model loading (FastSAM/SAM), semantics model loading
    (DINO/DINOv3/DINOv3-HF), feature extraction, and frame descriptors.
    """

    def __init__(self, params: SegmenterParamsBase):
        self.params = params
        self.semantic_patches_shape = None

        # Lazy-init flags — models loaded on first use
        self._segmentation_model_loaded = False
        self._semantics_model_loaded = False
        self._anyloc_loaded = False

        self.model = None
        self.semantics_model = None
        self.semantics_preprocess = None

        self.frame_descriptor_type = params.frame_descriptor
        self._anyloc_extractor = None
        self._anyloc_vlad = None
        self._anyloc_transform = None
        if params.frame_descriptor is not None:
            assert (
                params.semantics in ("dino", "dinov3", "dinov3-hf")
                or params.frame_descriptor == "anyloc"
            ), (
                "Frame descriptor only supported with DINO, DINOv3, DINOv3-HF semantics, or 'anyloc'."
            )

    def _ensure_segmentation_model(self):
        if not self._segmentation_model_loaded:
            self._init_segmentation_model()
            self._segmentation_model_loaded = True

    def _ensure_semantics_model(self):
        if not self._semantics_model_loaded:
            self._init_semantics_model()
            self._semantics_model_loaded = True

    def _ensure_anyloc(self):
        if not self._anyloc_loaded:
            self._init_anyloc()
            self._anyloc_loaded = True

    def _init_segmentation_model(self):
        if self.params.get_model_type() == "fastsam":
            self.model = FastSAM(self.params.weights_path)
        elif self.params.get_model_type() == "segment_anything":
            sam = sam_model_registry["vit_l"](checkpoint=self.params.weights_path)
            sam.to(self.params.device)
            sam.eval()
            self.model = SamAutomaticMaskGenerator(sam)
        else:
            raise ValueError(
                f"Unsupported segmenter model type: {self.params.model_type}"
            )

    def _init_semantics_model(self):
        if self.params.semantics is None or self.params.semantics.lower() == "none":
            self.semantics_model = None
            self.semantics_preprocess = None
        elif self.params.semantics.lower() == "dino":
            dino_model_name = f"facebook/dinov2-{self.params.semantics_size}"
            self.semantics_preprocess = AutoImageProcessor.from_pretrained(
                dino_model_name, do_center_crop=False
            )
            self.semantics_model = AutoModel.from_pretrained(dino_model_name)
            self.semantics_model.eval()
            self.semantics_model.to(self.params.device)
            self._num_register_tokens = 0
        elif self.params.semantics.lower() == "dinov3-hf":
            size_to_hf_name = {
                "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            }
            hf_name = size_to_hf_name.get(self.params.semantics_size)
            if hf_name is None:
                raise ValueError(
                    f"Invalid semantics_size for dinov3-hf: {self.params.semantics_size}. "
                    f"Choose from {list(size_to_hf_name.keys())}."
                )
            self.semantics_preprocess = AutoImageProcessor.from_pretrained(
                hf_name, do_center_crop=False
            )
            self.semantics_model = AutoModel.from_pretrained(hf_name)
            self.semantics_model.eval()
            self.semantics_model.to(self.params.device)
            self._num_register_tokens = self.semantics_model.config.num_register_tokens
        elif self.params.semantics.lower() == "dinov3":
            import torchvision.transforms as T

            size_to_hub_fn = {
                "small": "dinov3_vits16",
                "base": "dinov3_vitb16",
                "large": "dinov3_vitl16",
            }
            hub_fn = size_to_hub_fn.get(self.params.semantics_size)
            if hub_fn is None:
                raise ValueError(
                    f"Invalid semantics_size for dinov3: {self.params.semantics_size}. "
                    f"Choose from {list(size_to_hub_fn.keys())}."
                )
            hub_kwargs = {}
            if self.params.dinov3_weights is not None:
                hub_kwargs["weights"] = self.params.dinov3_weights
            self.semantics_model = torch.hub.load(
                self.params.dinov3_path, hub_fn, source="local", **hub_kwargs
            )
            self.semantics_model.eval()
            self.semantics_model.to(self.params.device)
            self.dinov3_transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            raise ValueError(
                f"Invalid semantics option: {self.params.semantics}. "
                f"Choose from 'dino', 'dinov3', 'dinov3-hf', or 'none'."
            )

    def _run_segmentation(self, image_rgb):
        """Run segmentation model on an RGB image.

        Args:
            image_rgb: RGB image as numpy array.

        Returns:
            masks: (N, H, W) numpy array of binary masks, or empty list if none found.
        """
        self._ensure_segmentation_model()
        if self.params.get_model_type() == "fastsam":
            everything_results = self.model(
                image_rgb,
                retina_masks=True,
                device=self.params.device,
                imgsz=self.params.imgsz,
                conf=self.params.conf,
                iou=self.params.iou,
            )
            prompt_process = FastSAMPrompt(
                image_rgb, everything_results, device=self.params.device
            )
            masks = prompt_process.everything_prompt()
        elif self.params.get_model_type() == "segment_anything":
            masks_output = self.model.generate(image_rgb)
            mask_list = []
            for obj in masks_output:
                mask_list.append(obj["segmentation"].astype(np.uint8))
            masks = torch.from_numpy(np.stack(mask_list)).to(self.params.device)
        else:
            raise ValueError(
                f"Unsupported segmenter model type: {self.params.model_type}"
            )

        if len(masks) > 0:
            masks = masks.cpu().numpy()
        else:
            return []

        return masks

    def _extract_dino_features(self, img_bgr):
        """Extract DINO features from a BGR image.

        Args:
            img_bgr: BGR image as numpy array.

        Returns:
            Tuple of (per_pixel_features, output_patches):
                per_pixel_features: (H, W, C) tensor of per-pixel features.
                output_patches: (1, h, w, C) tensor of patch features.
        """
        self._ensure_semantics_model()
        if self.params.semantics in ("dino", "dinov3-hf"):
            img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
            preprocessed = self.semantics_preprocess(
                images=img_rgb, return_tensors="pt"
            ).to(self.params.device)
            dino_output = self.semantics_model(**preprocessed)
            output_patches = self.get_output_patches(
                model_output=dino_output.last_hidden_state,
                img_shape=img_bgr.shape,
                feature_dim=self.params.semantics_dim,
            )
            per_pixel = self.get_per_pixel_features(
                model_output_patches=output_patches, img_shape=img_bgr.shape
            )
            return per_pixel, output_patches
        elif self.params.semantics == "dinov3":
            img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
            img_tensor = (
                self.dinov3_transform(img_rgb).unsqueeze(0).to(self.params.device)
            )
            with torch.no_grad():
                features = self.semantics_model.get_intermediate_layers(
                    img_tensor, n=1, reshape=True, return_class_token=False, norm=True
                )[0]  # (B, C, H_patches, W_patches)
            output_patches = features.permute(0, 2, 3, 1)  # (1, H, W, C)
            per_pixel = torch.nn.functional.interpolate(
                features,
                size=(img_bgr.shape[0], img_bgr.shape[1]),
                mode="bilinear",
            )[0].permute(1, 2, 0)  # (H, W, C)
            return per_pixel, output_patches
        else:
            return None, None

    def _compute_gem_descriptor(self, patch_features):
        """Compute GeM (Generalized Mean) pooling descriptor.

        Args:
            patch_features: Tensor of shape (..., C) — patch-level features.

        Returns:
            Normalized 1-D numpy array of shape (C,).
        """
        with torch.no_grad():
            features_flat = patch_features.reshape(-1, patch_features.shape[-1])
            cubed = torch.mean(features_flat**3, dim=0)
            descriptor = torch.sign(cubed) * (
                torch.abs(cubed).clamp(min=1e-12) ** (1.0 / 3)
            )
            descriptor = descriptor / torch.norm(descriptor)
        return descriptor.cpu().numpy()

    def _compute_mean_dino_descriptor(self, dino_features, mask):
        """Compute mean DINO descriptor over a binary mask.

        Args:
            dino_features: (H, W, C) tensor of per-pixel features.
            mask: (H, W) binary mask.

        Returns:
            Normalized 1-D numpy array of shape (C,).
        """
        dino_mask = dino_features[mask.astype(bool)]  # num-pixels x C
        dino_mask = dino_mask.cpu().detach().numpy()
        mean_dino = np.mean(dino_mask, axis=0)
        mean_dino = mean_dino / np.linalg.norm(mean_dino)
        return mean_dino

    def get_output_patches(
        self, model_output: ArrayLike, img_shape: ArrayLike, feature_dim: int
    ) -> ArrayLike:
        """Reshape flat DINO/DINOv3-HF patch tokens into a spatial grid.

        Args:
            model_output: Last hidden state of model, shape (B, N_tokens, C).
            img_shape: Original image shape (H, W, ...).
            feature_dim: Expected feature dimension C.

        Returns:
            Reshaped output of shape (1, h_patches, w_patches, C).
        """
        model_output_flat_patches = model_output[:, 1 + self._num_register_tokens :, :]
        if self.semantic_patches_shape is None:
            ratio = img_shape[1] / img_shape[0]  # width / height
            num_patches = model_output_flat_patches.shape[1]
            h = np.round(np.sqrt(num_patches / ratio)).astype(int)
            w = np.round(np.sqrt(num_patches * ratio)).astype(int)
            self.semantic_patches_shape = (1, h, w, feature_dim)

        model_output_patches = model_output_flat_patches.reshape(
            self.semantic_patches_shape
        )
        return model_output_patches

    def get_per_pixel_features(
        self, model_output_patches: ArrayLike, img_shape: ArrayLike
    ) -> ArrayLike:
        """Interpolate patch features to per-pixel resolution.

        Args:
            model_output_patches: (1, h, w, C) patch features.
            img_shape: Original image shape (H, W, ...).

        Returns:
            Per-pixel features of shape (H, W, C).
        """
        per_pixel_features = torch.nn.functional.interpolate(
            model_output_patches.permute(0, 3, 1, 2),
            size=(img_shape[0], img_shape[1]),
            mode="bilinear",
        )
        per_pixel_features = per_pixel_features[0].permute(1, 2, 0)
        return per_pixel_features

    def get_frame_descriptor(
        self, dino_features: torch.Tensor, img_bgr=None
    ) -> np.ndarray:
        """Compute a frame-level descriptor from patch features.

        Supports 'dino-gap', 'dino-gmp', 'dino-gem', and 'anyloc'.

        Args:
            dino_features: Patch features tensor.
            img_bgr: BGR image (only needed for 'anyloc').

        Returns:
            Normalized 1-D numpy descriptor, or None if no frame_descriptor configured.
        """
        if self.frame_descriptor_type is None:
            return None

        if self.frame_descriptor_type == "anyloc":
            return self._compute_anyloc_descriptor(img_bgr)

        with torch.no_grad():
            dino_features_flat = dino_features.view(-1, dino_features.shape[-1])
            if self.frame_descriptor_type == "dino-gap":
                frame_descriptor = torch.sum(dino_features_flat, dim=0)
            elif self.frame_descriptor_type == "dino-gmp":
                frame_descriptor = torch.max(dino_features_flat, dim=0).values
            elif self.frame_descriptor_type == "dino-gem":
                cubed_descriptor = torch.mean(dino_features_flat**3, dim=0)
                frame_descriptor = torch.sign(cubed_descriptor) * (
                    torch.abs(cubed_descriptor).clamp(min=1e-12) ** (1.0 / 3)
                )
            else:
                raise ValueError(
                    "frame descriptor must be one of 'dino-gap', 'dino-gmp', 'dino-gem', or 'anyloc'."
                )

            frame_descriptor /= torch.norm(frame_descriptor)

        return frame_descriptor.cpu().detach().numpy()

    def _init_anyloc(self):
        """Initialize AnyLoc DINOv2 extractor and VLAD vocabulary."""
        import sys
        import torchvision.transforms as tvf

        sys.path.insert(0, os.path.join(self.params.anyloc_path, "demo"))
        from utilities import DinoV2ExtractFeatures, VLAD

        self._anyloc_extractor = DinoV2ExtractFeatures(
            self.params.anyloc_dino_model,
            self.params.anyloc_layer,
            self.params.anyloc_facet,
            device=self.params.device,
        )
        self._anyloc_transform = tvf.Compose(
            [
                tvf.ToTensor(),
                tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        ext_specifier = (
            f"{self.params.anyloc_dino_model}/"
            f"l{self.params.anyloc_layer}_{self.params.anyloc_facet}"
            f"_c{self.params.anyloc_num_clusters}"
        )
        c_centers_file = os.path.join(
            self.params.anyloc_vocab_dir,
            "vocabulary",
            ext_specifier,
            self.params.anyloc_domain,
            "c_centers.pt",
        )
        assert os.path.isfile(c_centers_file), (
            f"AnyLoc vocabulary not found: {c_centers_file}"
        )

        self._anyloc_vlad = VLAD(
            self.params.anyloc_num_clusters,
            desc_dim=None,
            cache_dir=os.path.dirname(c_centers_file),
        )
        self._anyloc_vlad.fit(None)

    def _compute_anyloc_descriptor(self, img_bgr):
        """Compute AnyLoc (DINOv2 + VLAD) descriptor from a BGR image.

        Args:
            img_bgr: BGR image as numpy array.

        Returns:
            Normalized 1-D numpy array of shape (num_clusters * desc_dim,).
        """
        self._ensure_anyloc()
        import torchvision.transforms as tvf
        from PIL import Image as PILImage

        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        img_pt = self._anyloc_transform(pil_img).to(self.params.device)

        c, h, w = img_pt.shape
        h_new = (h // 14) * 14
        w_new = (w // 14) * 14
        img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]

        with torch.no_grad():
            ret = self._anyloc_extractor(img_pt)
            gd = self._anyloc_vlad.generate(ret.cpu().squeeze())

        return gd.numpy()
