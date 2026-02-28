import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2 as cv
import numpy as np
from numpy.typing import ArrayLike
import copy
from fastsam import FastSAMPrompt
from fastsam import FastSAM
from dataclasses import dataclass
from typing import Tuple, List

# import shapely
from transformers import AutoImageProcessor, AutoModel
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from roman.utils import expandvars_recursive

from gen_seg_match.segment.aerial_segment import AerialSegment
from gen_seg_match.params import AerialSegmenterParams


class AerialSegmenter:
    def __init__(self, params: AerialSegmenterParams):
        self.params = copy.deepcopy(params)
        if self.params.get_model_type() == "fastsam":
            self.model = FastSAM(expandvars_recursive(params.weights_path))
        elif self.params.get_model_type() == "segment_anything":
            sam = sam_model_registry["vit_l"](
                checkpoint=expandvars_recursive(params.weights_path)
            )
            sam.to(self.params.device)
            sam.eval()
            self.model = SamAutomaticMaskGenerator(sam)
        else:
            raise ValueError(
                f"Invalid model type: {params.model_type}. Choose from 'fastsam' or 'segment_anything'."
            )

        if params.semantics is None or params.semantics.lower() == "none":
            self.semantics_model = None
            self.semantics_preprocess = None
        elif params.semantics.lower() == "dino":
            dino_model_name = f"facebook/dinov2-{params.semantics_size}"
            self.semantics_preprocess = AutoImageProcessor.from_pretrained(
                dino_model_name, do_center_crop=False
            )
            self.semantics_model = AutoModel.from_pretrained(dino_model_name)
            self.semantics_model.eval()
            self.semantics_model.to(self.params.device)
            self._num_register_tokens = 0
        elif params.semantics.lower() == "dinov3-hf":
            size_to_hf_name = {
                "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            }
            hf_name = size_to_hf_name.get(params.semantics_size)
            if hf_name is None:
                raise ValueError(
                    f"Invalid semantics_size for dinov3-hf: {params.semantics_size}. "
                    f"Choose from {list(size_to_hf_name.keys())}."
                )
            self.semantics_preprocess = AutoImageProcessor.from_pretrained(
                hf_name, do_center_crop=False
            )
            self.semantics_model = AutoModel.from_pretrained(hf_name)
            self.semantics_model.eval()
            self.semantics_model.to(self.params.device)
            self._num_register_tokens = self.semantics_model.config.num_register_tokens
        elif params.semantics.lower() == "dinov3":
            import torchvision.transforms as T

            size_to_hub_fn = {
                "small": "dinov3_vits16",
                "base": "dinov3_vitb16",
                "large": "dinov3_vitl16",
            }
            hub_fn = size_to_hub_fn.get(params.semantics_size)
            if hub_fn is None:
                raise ValueError(
                    f"Invalid semantics_size for dinov3: {params.semantics_size}. "
                    f"Choose from {list(size_to_hub_fn.keys())}."
                )
            hub_kwargs = {}
            if params.dinov3_weights is not None:
                hub_kwargs["weights"] = params.dinov3_weights
            self.semantics_model = torch.hub.load(
                params.dinov3_path, hub_fn, source="local", **hub_kwargs
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
                f"Invalid semantics option: {params.semantics}. Choose from 'dino', 'dinov3', 'dinov3-hf', or 'none'."
            )
        self.semantic_patches_shape = None

    def get_crop_descriptor(self, img_bgr, crop=None) -> np.ndarray:
        """Compute a DINO-GeM global descriptor for an image crop.

        Args:
            img_bgr: BGR image (full aerial image).
            crop: Optional (x1, y1, x2, y2) pixel crop.

        Returns:
            Normalized 1-D numpy array of shape (semantics_dim,), or None if
            semantics is disabled.
        """
        if self.semantics_model is None:
            return None

        if crop is not None:
            img_bgr = img_bgr[crop[1] : crop[3], crop[0] : crop[2]]
        image_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        if self.params.downsample_factor > 1:
            image_rgb = cv.resize(
                image_rgb,
                (
                    image_rgb.shape[1] // self.params.downsample_factor,
                    image_rgb.shape[0] // self.params.downsample_factor,
                ),
                interpolation=cv.INTER_LINEAR,
            )

        with torch.no_grad():
            if self.params.semantics in ("dino", "dinov3-hf"):
                preprocessed = self.semantics_preprocess(
                    images=image_rgb, return_tensors="pt"
                ).to(self.params.device)
                dino_output = self.semantics_model(**preprocessed)
                features = dino_output.last_hidden_state[
                    :, 1 + self._num_register_tokens :, :
                ]
            elif self.params.semantics == "dinov3":
                img_tensor = (
                    self.dinov3_transform(image_rgb).unsqueeze(0).to(self.params.device)
                )
                features = self.semantics_model.get_intermediate_layers(
                    img_tensor, n=1, reshape=False, return_class_token=False, norm=True
                )[0]  # (1, N_patches, C)
            else:
                return None

            features_flat = features.reshape(-1, features.shape[-1])

            # GeM pooling
            cubed = torch.mean(features_flat**3, dim=0)
            descriptor = torch.sign(cubed) * (
                torch.abs(cubed).clamp(min=1e-12) ** (1.0 / 3)
            )
            descriptor = descriptor / torch.norm(descriptor)

        return descriptor.cpu().numpy()

    def run(self, img_bgr, crop=None) -> List[AerialSegment]:
        """
        Run FastSAM on the given image and return a list of observations.
        """
        if crop is not None:
            img_bgr = img_bgr[crop[1] : crop[3], crop[0] : crop[2]]
            img_origin = np.array([crop[0], crop[1]])
        else:
            img_origin = np.array([0.0, 0.0])
        image_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        if self.params.downsample_factor > 1:
            image_rgb = cv.resize(
                image_rgb,
                (
                    image_rgb.shape[1] // self.params.downsample_factor,
                    image_rgb.shape[0] // self.params.downsample_factor,
                ),
                interpolation=cv.INTER_LINEAR,
            )

        # Run FastSAM
        if self.params.get_model_type() == "fastsam":
            everything_results = self.model(
                image_rgb,
                conf=self.params.conf,
                iou=self.params.iou,
                imgsz=self.params.imgsz,
                retina_masks=True,
                device=self.params.device,
            )
            prompt_process = FastSAMPrompt(
                image_rgb, everything_results, device=self.params.device
            )
            masks = prompt_process.everything_prompt()
        elif self.params.get_model_type() == "segment_anything":
            masks_output = self.model.generate(image_rgb)

            # Convert SAM result masks into (N,H,W) boolean numpy array like FastSAM
            mask_list = []
            for obj in masks_output:
                mask_list.append(obj["segmentation"])

            masks = torch.from_numpy(np.stack(mask_list)).to(self.params.device)

        if len(masks) > 0:
            masks = masks.cpu().numpy()
        else:
            return []

        if self.params.semantics in ("dino", "dinov3-hf"):
            # Process the image for DINO / DINOv3-HF
            preprocessed = self.semantics_preprocess(
                images=image_rgb, return_tensors="pt"
            ).to(self.params.device)
            dino_output = self.semantics_model(**preprocessed)
            dino_features = self.get_per_pixel_features(
                model_output=dino_output.last_hidden_state,
                img_shape=image_rgb.shape,
                feature_dim=self.params.semantics_dim,
            )
        elif self.params.semantics == "dinov3":
            img_tensor = (
                self.dinov3_transform(image_rgb).unsqueeze(0).to(self.params.device)
            )
            with torch.no_grad():
                features = self.semantics_model.get_intermediate_layers(
                    img_tensor, n=1, reshape=True, return_class_token=False, norm=True
                )[0]  # (B, C, H_patches, W_patches)
            dino_features = torch.nn.functional.interpolate(
                features,
                size=(image_rgb.shape[0], image_rgb.shape[1]),
                mode="bilinear",
            )[0].permute(1, 2, 0)  # (H, W, C)

        aerial_segments = []
        for i, mask in enumerate(masks):
            # if self.params.downsample_factor > 1:
            #     mask = cv.resize(mask, (mask.shape[1] * self.params.downsample_factor,
            # mask.shape[0] * self.params.downsample_factor), interpolation=cv.INTER_NEAREST)

            area = (
                np.sum(mask)
                * self.params.pixel_len_m**2
                * self.params.downsample_factor**2
            )
            if area < self.params.min_area or area > self.params.max_area:
                continue
            points = (
                np.array(np.nonzero(mask)).astype(np.float64).T[:, ::-1]
                * self.params.downsample_factor
                * self.params.pixel_len_m
                + img_origin * self.params.pixel_len_m
            )
            # convex_hull = shapely.convex_hull(shapely.MultiPoint(points))
            semantic_descriptor = None
            if self.params.semantics in ("dino", "dinov3", "dinov3-hf"):
                assert (
                    mask.shape[0] == dino_features.shape[0]
                    and mask.shape[1] == dino_features.shape[1]
                ), "Mask and DINO features must have the same shape."
                dino_mask = dino_features[mask.astype(bool)]  # num-pixels x dino_shape
                dino_mask = dino_mask.cpu().detach().numpy()
                mean_dino = np.mean(dino_mask, axis=0)  # dino_shape
                mean_dino = mean_dino / np.linalg.norm(mean_dino)  # normalize
                semantic_descriptor = mean_dino
            aerial_segments.append(
                AerialSegment(
                    id=i,
                    center=np.mean(points, axis=0),
                    area=area,
                    points=points,
                    semantic_descriptor=semantic_descriptor,
                )
            )
        return aerial_segments

    def get_per_pixel_features(
        self, model_output: ArrayLike, img_shape: ArrayLike, feature_dim: int
    ) -> ArrayLike:
        """
        Extract (Dino) per-pixel features

        Args:
            model_output (ArrayLike): Last hidden state of (Dino) model
            img_shape (ArrayLike): Original image shape
            feature_dim (int): Expected (Dino) feature dimension

        Returns:
            ArrayLike: Reshaped (Dino) output
        """
        model_output_flat_patches = model_output[:, 1 + self._num_register_tokens :, :]
        if self.semantic_patches_shape is None:
            ratio = img_shape[1] / img_shape[0]  # width / height
            num_patches = model_output_flat_patches.shape[1]
            h = np.round(np.sqrt(num_patches / ratio)).astype(
                int
            )  # number of patches along y-axis
            w = np.round(np.sqrt(num_patches * ratio)).astype(
                int
            )  # number of patches along x-axis

            self.semantic_patches_shape = (1, h, w, feature_dim)

        model_output_patches = model_output_flat_patches.reshape(
            self.semantic_patches_shape
        )

        # interpolate the feature map to match the size of the original image
        per_pixel_features = torch.nn.functional.interpolate(
            model_output_patches.permute(
                0, 3, 1, 2
            ),  # permute to be batch, channels, height, width
            size=(img_shape[0], img_shape[1]),
            mode="bilinear",
        )  # 1 x dino_shape x h x w

        # reshape
        per_pixel_features = per_pixel_features[0].permute(
            1, 2, 0
        )  # h x w x feature_dim

        return per_pixel_features  # h x w x feature_dim
