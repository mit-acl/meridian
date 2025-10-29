import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2 as cv
import numpy as np
from numpy.typing import ArrayLike
import copy
from fastsam import FastSAMPrompt
from fastsam import FastSAM
from dataclasses import dataclass
from typing import Tuple, List
import shapely
from transformers import AutoImageProcessor, AutoModel
import torch

from roman.map.observation import Observation
from roman.utils import expandvars_recursive

from gen_seg_match.segment.aerial_segment import AerialSegment


@dataclass
class AerialFastSAMParams:

    weights_path: str = "$ROMAN_WEIGHTS/FastSAM-x.pt"
    conf: float = 0.25
    iou: float = 0.9
    imgsz: Tuple[int, int] = (256, 256)
    device: str = 'cuda'
    pixel_len_m: float = 0.01
    min_area: float = .01
    max_area: float = 25.0
    semantics: str = 'dino'
    triangle_ignore_masks: List[Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int]]] = None



class AerialFastSAMWrapper():

    def __init__(self, params: AerialFastSAMParams):

        self.params = copy.deepcopy(params)
        self.model = FastSAM(expandvars_recursive(params.weights_path))

        if params.semantics is None or params.semantics.lower() == 'none':
            self.semantics_model = None
            self.semantics_preprocess = None
        elif params.semantics.lower() == 'dino':
            self.semantics_preprocess = AutoImageProcessor.from_pretrained('facebook/dinov2-base', do_center_crop=False)
            self.semantics_model = AutoModel.from_pretrained('facebook/dinov2-base')
            self.semantics_model.eval()
            self.semantics_model.to(self.params.device)
        else:
            raise ValueError(f"Invalid semantics option: {params.semantics}. Choose from 'clip', 'dino', or 'none'.")
        self.semantic_patches_shape = None
        

    def run(self, img_bgr, crop=None, downsample_factor: int = 1) -> List[Observation]:
        """
        Run FastSAM on the given image and return a list of observations.
        """
        if crop is not None:
            img_bgr = img_bgr[crop[1]:crop[3], crop[0]:crop[2]]
            img_origin = np.array([crop[0], crop[1]])
        else:
            img_origin = np.array([0., 0.])
        image_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
            
        if downsample_factor > 1:
            image_rgb = cv.resize(image_rgb, (image_rgb.shape[1] // downsample_factor, image_rgb.shape[0] // downsample_factor), interpolation=cv.INTER_LINEAR)

        # Run FastSAM
        everything_results = self.model(image_rgb, conf=self.params.conf, iou=self.params.iou, imgsz=self.params.imgsz, 
                             retina_masks=True, device=self.params.device)
        prompt_process = FastSAMPrompt(image_rgb, everything_results, device=self.params.device)
        masks = prompt_process.everything_prompt()

        if (len(masks) > 0):
            masks = masks.cpu().numpy()
        else:
            return []
        
        if self.params.semantics == 'dino':
            # Process the image for DINO
            dino_shape = 768
            img_bgr = cv.cvtColor(image_rgb, cv.COLOR_BGR2RGB)
            preprocessed = self.semantics_preprocess(images=img_bgr, return_tensors="pt").to(self.params.device)
            dino_output = self.semantics_model(**preprocessed)
            dino_features = self.get_per_pixel_features(
                model_output=dino_output.last_hidden_state, 
                img_shape=image_rgb.shape, 
                feature_dim=dino_shape
            )

        aerial_segments = []
        for i, mask in enumerate(masks):
            # if downsample_factor > 1:
            #     mask = cv.resize(mask, (mask.shape[1] * downsample_factor, mask.shape[0] * downsample_factor), interpolation=cv.INTER_NEAREST)
            
            area = np.sum(mask) * self.params.pixel_len_m**2 * downsample_factor**2
            # print(area)
            if area < self.params.min_area or area > self.params.max_area:
                continue
            points = np.array(np.nonzero(mask)).astype(np.float64).T[:, ::-1] * downsample_factor * self.params.pixel_len_m \
                                                                 + img_origin * self.params.pixel_len_m
            convex_hull = shapely.convex_hull(shapely.MultiPoint(points))
            semantic_descriptor = None
            if self.params.semantics == 'dino':
                assert mask.shape[0] == dino_features.shape[0] and mask.shape[1] == dino_features.shape[1], \
                    "Mask and DINO features must have the same shape."
                dino_mask = dino_features[mask.astype(bool)] # num-pixels x dino_shape
                dino_mask = dino_mask.cpu().detach().numpy()
                mean_dino = np.mean(dino_mask, axis=0) # dino_shape
                mean_dino = mean_dino / np.linalg.norm(mean_dino) # normalize
                semantic_descriptor = mean_dino
            aerial_segments.append(AerialSegment(
                id=i,
                center=np.mean(points, axis=0),
                area=area,
                points=points,
                semantic_descriptor=semantic_descriptor,
                # img_pixel_location=(np.array(np.nonzero(mask)).astype(np.float64).mean(axis=1)[::-1] + img_origin).astype(np.int64),
                # convex_hull=np.array(convex_hull.exterior.coords)
            ))
        return aerial_segments

    def get_per_pixel_features(self, model_output: ArrayLike, img_shape: ArrayLike, 
            feature_dim: int) -> ArrayLike:
        """
        Extract (Dino) per-pixel features

        Args:
            model_output (ArrayLike): Last hidden state of (Dino) model
            img_shape (ArrayLike): Original image shape
            feature_dim (int): Expected (Dino) feature dimension

        Returns:
            ArrayLike: Reshaped (Dino) output
        """
        model_output_flat_patches = model_output[:,1:, :]
        if self.semantic_patches_shape is None:
            ratio = img_shape[1] / img_shape[0] # width / height
            num_patches = model_output_flat_patches.shape[1]
            h = np.round(np.sqrt(num_patches / ratio)).astype(int) # number of patches along y-axis
            w = np.round(np.sqrt(num_patches * ratio)).astype(int) # number of patches along x-axis

            self.semantic_patches_shape = (1, h, w, feature_dim)
            
        model_output_patches = model_output_flat_patches.reshape(self.semantic_patches_shape)

        # interpolate the feature map to match the size of the original image
        per_pixel_features = torch.nn.functional.interpolate(
            model_output_patches.permute(0, 3, 1, 2), # permute to be batch, channels, height, width
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
        ) # 1 x dino_shape x h x w

        # reshape
        per_pixel_features = per_pixel_features[0].permute(1, 2, 0) # h x w x feature_dim

        return per_pixel_features # h x w x feature_dim
