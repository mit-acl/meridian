import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2 as cv
import numpy as np
import torch
from typing import List

from meridian.map2d.segment2d import Segment2D
from meridian.params import AerialSegmenterParams
from meridian.segmenter.segmenter_base import SegmenterBase


class AerialSegmenter(SegmenterBase):
    def __init__(self, params: AerialSegmenterParams):
        import copy

        super().__init__(copy.deepcopy(params))

    def get_crop_descriptor(self, img_bgr, crop=None) -> np.ndarray:
        """Compute a global descriptor for an image crop.

        Args:
            img_bgr: BGR image (full aerial image).
            crop: Optional (x1, y1, x2, y2) pixel crop.

        Returns:
            Normalized 1-D numpy descriptor array, or None if semantics is
            disabled and frame_descriptor is not "anyloc".
        """
        if crop is not None:
            img_bgr = img_bgr[crop[1] : crop[3], crop[0] : crop[2]]

        if self.params.downsample_factor > 1:
            img_bgr = cv.resize(
                img_bgr,
                (
                    img_bgr.shape[1] // self.params.downsample_factor,
                    img_bgr.shape[0] // self.params.downsample_factor,
                ),
                interpolation=cv.INTER_LINEAR,
            )

        if self.frame_descriptor_type == "anyloc":
            return self._compute_anyloc_descriptor(img_bgr)
        elif self.frame_descriptor_type == "salad":
            return self._compute_salad_descriptor(img_bgr)

        self._ensure_semantics_model()
        if self.semantics_model is None:
            return None

        image_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

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

    def segment(self, img_bgr, crop=None) -> List[Segment2D]:
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
        dino_output_patches = None
        if self.params.semantics in ("dino", "dinov3", "dinov3-hf"):
            # Convert downsampled RGB back to BGR for _extract_dino_features
            img_bgr_ds = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
            dino_features, dino_output_patches = self._extract_dino_features(img_bgr_ds)

        # First pass: filter masks by area and compute geometry
        valid_entries = []
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
            valid_entries.append((i, mask, points, area))

        # Batch-compute DINO descriptors on GPU
        descriptors = [None] * len(valid_entries)
        if dino_features is not None and valid_entries:
            valid_masks = [entry[1] for entry in valid_entries]
            assert (
                valid_masks[0].shape[0] == dino_features.shape[0]
                and valid_masks[0].shape[1] == dino_features.shape[1]
            ), "Mask and DINO features must have the same shape."
            dino_frame_embedding = (
                self._compute_dino_frame_embedding(dino_output_patches)
                if self.params.subtract_frame_descriptor
                and dino_output_patches is not None
                else None
            )
            descriptors = self._compute_batch_mean_dino_descriptors(
                dino_features,
                valid_masks,
                dino_frame_embedding=dino_frame_embedding,
            )

        # Build segments
        aerial_segments = []
        for (i, mask, points, area), desc in zip(valid_entries, descriptors):
            aerial_segments.append(
                Segment2D(
                    id=i,
                    center=np.mean(points, axis=0),
                    area=area,
                    points=points,
                    semantic_descriptor=desc,
                )
            )
        return aerial_segments

    def run(self, img_bgr, crop=None) -> List[Segment2D]:
        """Backward-compatible alias for segment()."""
        return self.segment(img_bgr, crop=crop)
