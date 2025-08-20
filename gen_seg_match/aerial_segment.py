import numpy as np
from dataclasses import dataclass
from typing import Tuple
import shapely

@dataclass
class AerialSegment:

    id: int
    center: np.ndarray
    area: float
    points: np.ndarray # Points in the segment in meters
    img_pixel_location: np.ndarray = None
    # convex_hull: np.ndarray = None # Convex hull of the segment in meters

    def __post_init__(self):
        self._convex_hull = None

    @property
    def convex_hull(self) -> np.ndarray:
        if self._convex_hull is not None:
            return self._convex_hull
        convex_hull_shapely = shapely.convex_hull(shapely.MultiPoint(self.points))
        self._convex_hull = np.array(convex_hull_shapely.exterior.coords)
        return self._convex_hull

    def convex_hull_pixels(self, img_pixel_scale: float, img_origin_m: Tuple[float, float] = (0.0, 0.0), ) -> np.ndarray:
        if self.convex_hull is None:
            return None
        return ((self.convex_hull - np.array(img_origin_m)) / img_pixel_scale).astype(np.int32)
    
    @property
    def viz_color(self):
        np.random.seed(self.id)
        color = np.random.randint(0, 255, 3).tolist()
        return color
        