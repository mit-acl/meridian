import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2 as cv
import numpy as np
import torch
from typing import List

from gen_seg_match.segment.aerial_segment import AerialSegment
from gen_seg_match.params import AerialSegmenterParams
from gen_seg_match.segmenter.segmenter_base import SegmenterBase


class AerialSegmenter(SegmenterBase):
    def __init__(self, params: AerialSegmenterParams):
        import copy

        super().__init__(copy.deepcopy(params))

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

        return self._compute_gem_descriptor(features)

    def segment(self, img_bgr, crop=None) -> List[AerialSegment]:
        """
        Run segmentation on the given image and return a list of AerialSegments.
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

        # Run segmentation
        masks = self._run_segmentation(image_rgb)

        if len(masks) == 0:
            return []

        # Extract DINO features
        dino_features = None
        if self.params.semantics in ("dino", "dinov3", "dinov3-hf"):
            # Convert downsampled RGB back to BGR for _extract_dino_features
            img_bgr_ds = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
            dino_features, _ = self._extract_dino_features(img_bgr_ds)

        aerial_segments = []
        for i, mask in enumerate(masks):
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
            semantic_descriptor = None
            if dino_features is not None:
                assert (
                    mask.shape[0] == dino_features.shape[0]
                    and mask.shape[1] == dino_features.shape[1]
                ), "Mask and DINO features must have the same shape."
                semantic_descriptor = self._compute_mean_dino_descriptor(
                    dino_features, mask
                )
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

    def run(self, img_bgr, crop=None) -> List[AerialSegment]:
        """Backward-compatible alias for segment()."""
        return self.segment(img_bgr, crop=crop)
