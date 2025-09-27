import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2 as cv
import numpy as np
import copy
from fastsam import FastSAMPrompt
from fastsam import FastSAM
from dataclasses import dataclass
from typing import Tuple, List
import shapely

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
            aerial_segments.append(AerialSegment(
                id=i,
                center=np.mean(points, axis=0),
                area=area,
                points=points,
                # img_pixel_location=(np.array(np.nonzero(mask)).astype(np.float64).mean(axis=1)[::-1] + img_origin).astype(np.int64),
                # convex_hull=np.array(convex_hull.exterior.coords)
            ))
        return aerial_segments
