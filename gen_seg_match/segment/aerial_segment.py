import numpy as np
from dataclasses import dataclass
from typing import Tuple
import shapely
import open3d as o3d

@dataclass
class AerialSegment:

    id: int
    center: np.ndarray
    area: float
    points: np.ndarray # Points in the segment in meters
    img_pixel_location: np.ndarray = None
    semantic_descriptor: np.ndarray = None
    # convex_hull: np.ndarray = None # Convex hull of the segment in meters

    def __post_init__(self):
        self._convex_hull = None
        self._gaussian = None
        self._eigvals = None
        self._pcd = None

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
        
    @property
    def pcd(self):
        if self._pcd is None:
            self._pcd = o3d.geometry.PointCloud()
            self._pcd.points = o3d.utility.Vector3dVector(np.hstack([self.points, np.zeros((self.points.shape[0], 1))]))  # Add a zero z-coordinate
        return self._pcd

    @property 
    def gaussian(self):
        if self._gaussian is None:
            self._gaussian = self.pcd.compute_mean_and_covariance()
        return self._gaussian
        
    @property
    def normalized_eigenvalues(self):
        """Compute the normalized eigenvalues of the covariance matrix
        as a np array [e1, e2, e3]
        e1 >= e2 >= e3 so that the sum is one
        """
        if self._eigvals is None:
            _, C = self.gaussian
            _, eigvals, _ = np.linalg.svd(C)  # svd return in descending order
            self._eigvals = eigvals / eigvals.sum()
        return self._eigvals

    @property
    def linearity(self):
        """ Large if similar to a 1D line (Weinmann et al. ISPRS 2014)

        Args:
            e (np.ndarray): normalized eigenvalues of this point cloud
        """
        e = self.normalized_eigenvalues
        return (e[0]-e[1]) / e[0]

    @property
    def planarity(self):
        """ Large if similar to a 2D plane (Weinmann et al. ISPRS 2014)
        Args:
            e (np.ndarray): normalized eigenvalues of this point cloud
        """
        e = self.normalized_eigenvalues
        return (e[1]-0.0) / e[0]

    @property
    def scattering(self):
        """Large if this object is 3D, i.e., neither a line nor a plane (Weinmann et al. ISPRS 2014)

        Args:
            e (np.ndarray): normalized eigenvalues of this point cloud
        """
        return 0.0