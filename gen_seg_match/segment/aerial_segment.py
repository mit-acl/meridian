import numpy as np
from dataclasses import dataclass
from typing import Tuple
import shapely
from shapely.geometry import MultiPoint
import open3d as o3d
import alphashape
from typing import Dict

from gen_seg_match.viz.utils import color_from_seed


@dataclass
class AerialSegment:
    id: int
    center: np.ndarray
    area: float
    points: np.ndarray  # Points in the segment in meters
    img_pixel_location: np.ndarray = None
    semantic_descriptor: np.ndarray = None
    first_seen: float = None
    last_seen: float = None
    alpha_shapes: Dict[(float, float)] = None  # (alpha, grid_downsample) -> alpha shape

    def __post_init__(self):
        self._convex_hull = None
        self._gaussian = None
        self._eigvals = None
        self._pcd = None
        self.alpha_shapes = {}

    @property
    def convex_hull(self) -> np.ndarray:
        if self._convex_hull is not None:
            return self._convex_hull
        convex_hull_shapely = shapely.convex_hull(shapely.MultiPoint(self.points))
        self._convex_hull = np.array(convex_hull_shapely.exterior.coords)
        return self._convex_hull

    @property
    def max_extent(self) -> float:
        if self.convex_hull is None:
            return 0.0
        dists = np.linalg.norm(
            self.convex_hull[:, np.newaxis, :] - self.convex_hull[np.newaxis, :, :],
            axis=-1,
        )
        return np.max(dists)

    def convex_hull_pixels(
        self,
        img_pixel_scale: float,
        img_origin_m: Tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        if self.convex_hull is None:
            return None
        return ((self.convex_hull - np.array(img_origin_m)) / img_pixel_scale).astype(
            np.int32
        )

    def get_alpha_shape(self, alpha=0.5, grid_downsample=None):
        if (alpha, grid_downsample) in self.alpha_shapes:
            return self.alpha_shapes[(alpha, grid_downsample)]

        points = self.points.copy()
        # print(len(points))
        if grid_downsample is not None:
            # TODO: just do this in numpy
            points_o3d = o3d.geometry.PointCloud()
            points_o3d.points = o3d.utility.Vector3dVector(
                np.hstack([points, np.zeros((len(points), 1))])
            )
            points_o3d = points_o3d.voxel_down_sample(voxel_size=grid_downsample)
            points = np.asarray(points_o3d.points)[:, :2]
        try:
            alpha_shape = alphashape.alphashape(points, alpha=alpha)
        except Exception as e:
            print(
                f"Error computing alpha shape for segment {self.id} with alpha={alpha}: {e}"
            )
            self.alpha_shapes[(alpha, grid_downsample)] = None
            return None
        if type(alpha_shape) is shapely.geometry.polygon.Polygon:
            x, y = alpha_shape.exterior.xy
            self.alpha_shapes[(alpha, grid_downsample)] = np.vstack([x, y]).T
        elif type(alpha_shape) is shapely.geometry.MultiPolygon:
            # x, y = [alpha_shape.geometry.exterior.xy for poly_item in alpha_shape]
            # x, y = shapely.concave_hull(alpha_shape).exterior.xy
            self.alpha_shapes[(alpha, grid_downsample)] = None
        else:
            self.alpha_shapes[(alpha, grid_downsample)] = None

        return self.alpha_shapes[(alpha, grid_downsample)]

    def get_alpha_shape_pixels(
        self,
        img_pixel_scale: float,
        img_origin_m: Tuple[float, float] = (0.0, 0.0),
        alpha=0.5,
        grid_downsample=None,
    ):
        alpha_shape = self.get_alpha_shape(alpha, grid_downsample)
        if alpha_shape is None:
            return None
        alpha_shape_pixels = (
            (alpha_shape - np.array(img_origin_m)) / img_pixel_scale
        ).astype(np.int32)
        return alpha_shape_pixels

    def calculate_area_from_convex_hull(self) -> float:
        self.area = MultiPoint(self.convex_hull).convex_hull.area
        return self.area

    def color_from_id(self, order="rgb", num_type=int) -> tuple:
        """Returns a color tuple based on the segment ID."""
        return color_from_seed(self.id, order, num_type)

    @property
    def viz_color(self):
        np.random.seed(self.id)
        color = np.random.randint(0, 255, 3).tolist()
        return color

    @property
    def pcd(self):
        if self._pcd is None:
            self._pcd = o3d.geometry.PointCloud()
            self._pcd.points = o3d.utility.Vector3dVector(
                np.hstack([self.points, np.zeros((self.points.shape[0], 1))])
            )  # Add a zero z-coordinate
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
        """Large if similar to a 1D line (Weinmann et al. ISPRS 2014)

        Args:
            e (np.ndarray): normalized eigenvalues of this point cloud
        """
        e = self.normalized_eigenvalues
        return (e[0] - e[1]) / e[0]

    @property
    def planarity(self):
        """Large if similar to a 2D plane (Weinmann et al. ISPRS 2014)
        Args:
            e (np.ndarray): normalized eigenvalues of this point cloud
        """
        e = self.normalized_eigenvalues
        return (e[1] - 0.0) / e[0]

    @property
    def scattering(self):
        """Large if this object is 3D, i.e., neither a line nor a plane (Weinmann et al. ISPRS 2014)

        Args:
            e (np.ndarray): normalized eigenvalues of this point cloud
        """
        return 0.0
