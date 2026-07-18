import numpy as np
from dataclasses import dataclass
from typing import Tuple
import shapely
from shapely.geometry import MultiPoint
import open3d as o3d
from typing import Dict, Optional

from meridian.utils import suppress_alphashape_singular_warnings
from meridian.viz.utils import color_from_seed

# Drop the noisy "Singular matrix. Likely caused by all points lying in an
# N-1 space." warnings that alphashape emits per colinear Delaunay simplex.
suppress_alphashape_singular_warnings()


def _grid_downsample_2d(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample 2D points by keeping one point per grid cell.

    Uses floor-division to assign each point to a grid cell, then keeps
    only unique cells and returns the mean point in each cell.
    """
    grid_coords = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(grid_coords, axis=0, return_index=True)
    return points[unique_idx]


@dataclass
class Segment2D:
    id: int
    center: np.ndarray
    area: float
    points: np.ndarray  # Points in the segment in meters
    img_pixel_location: np.ndarray = None
    semantic_descriptor: np.ndarray = None
    first_seen: float = None
    last_seen: float = None
    segment_borders: Dict = None  # cache_key -> border coords

    def __post_init__(self):
        self._convex_hull = None
        self._gaussian = None
        self._eigvals = None
        self._pcd = None
        self._obb_extents = None
        self.segment_borders = {}

    @property
    def obb_extents(self) -> Tuple[float, float]:
        """Oriented bounding box extents (minor_axis, major_axis) in meters."""
        if self._obb_extents is None:
            cov = np.cov(self.points.T)
            _, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalue order
            projected = (self.points - self.points.mean(axis=0)) @ eigvecs
            extents = projected.max(axis=0) - projected.min(axis=0)
            self._obb_extents = (extents[0], extents[1])  # (minor, major)
        return self._obb_extents

    @property
    def convex_hull(self) -> np.ndarray:
        if self._convex_hull is not None:
            return self._convex_hull
        convex_hull_shapely = shapely.convex_hull(shapely.MultiPoint(self.points))
        if convex_hull_shapely.is_empty:
            self._convex_hull = None
        elif hasattr(convex_hull_shapely, "exterior"):
            # Polygon (normal case: 3+ non-collinear points)
            self._convex_hull = np.array(convex_hull_shapely.exterior.coords)
        else:
            # LineString (collinear points) or Point (single point)
            self._convex_hull = np.array(convex_hull_shapely.coords)
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

    def get_segment_border(
        self,
        alpha=0.5,
        grid_downsample=None,
        max_n_pts: Optional[int] = None,
        alpha_ref_size: float = None,
        segment_border_type: str = "concave_hull",
        concave_hull_ratio: float = 0.5,
    ):
        cache_key = (
            alpha,
            grid_downsample,
            max_n_pts,
            segment_border_type,
            concave_hull_ratio,
        )
        if cache_key in self.segment_borders:
            return self.segment_borders[cache_key]

        # Local import breaks the segment_to_primitive -> segment2d import cycle;
        # viz and conversion now share the exact same border computation.
        from meridian.map2d.segment_to_primitive import _compute_segment_border

        border = _compute_segment_border(
            self.points,
            alpha=alpha,
            grid_downsample=grid_downsample,
            max_n_pts=max_n_pts,
            alpha_ref_size=alpha_ref_size,
            max_extent=self.max_extent,
            segment_border_type=segment_border_type,
            concave_hull_ratio=concave_hull_ratio,
        )
        self.segment_borders[cache_key] = border
        return border

    def get_segment_border_pixels(
        self,
        img_pixel_scale: float,
        img_origin_m: Tuple[float, float] = (0.0, 0.0),
        alpha=0.5,
        grid_downsample=None,
        max_n_pts: Optional[int] = None,
        alpha_ref_size: float = None,
        segment_border_type: str = "concave_hull",
        concave_hull_ratio: float = 0.5,
    ):
        border = self.get_segment_border(
            alpha,
            grid_downsample,
            max_n_pts,
            alpha_ref_size=alpha_ref_size,
            segment_border_type=segment_border_type,
            concave_hull_ratio=concave_hull_ratio,
        )
        if border is None:
            return None
        return ((border - np.array(img_origin_m)) / img_pixel_scale).astype(np.int32)

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
