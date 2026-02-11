from dataclasses import dataclass, field
from typing import Dict
import numpy as np
import open3d as o3d

from robotdatapy.transform import transform
from gen_seg_match.viz.utils import color_from_seed
from gen_seg_match.map3d.voxel_grid import VoxelGrid


@dataclass
class Observation:
    """
    Segment observation data class
    """

    id: int
    time: float
    pose: np.ndarray
    mask: np.ndarray = None
    mask_downsampled: np.ndarray = None
    point_cloud: np.ndarray = None  # n-by-3 matrix. Each row is a 3D point.
    occluded_points: np.ndarray = None  # n-by-3 matrix. Each row is a 3D point.
    points_beyond_max_depth: np.ndarray = None  # n-by-3 matrix. Each row is a 3D point.
    semantic_descriptor: np.ndarray = None
    voxel_grid: Dict[float, VoxelGrid] = field(default_factory=dict)
    _transformed_points: np.ndarray = None
    _pcd: o3d.geometry.PointCloud = None

    def copy(self, include_mask: bool = True, include_ptcld=False):
        ptcld_copy = None
        if self.point_cloud is not None and include_ptcld:
            ptcld_copy = self.point_cloud.copy()
        if include_mask:
            return Observation(
                self.id,
                self.time,
                self.pose.copy(),
                self.mask,
                self.mask_downsampled,
                ptcld_copy,
            )
        else:
            return Observation(
                self.id, self.time, self.pose.copy(), None, None, ptcld_copy
            )

    def get_voxel_grid(self, voxel_size: float):
        """
        Get the voxel bounding box for the point cloud
        """
        if voxel_size not in self.voxel_grid:
            self.voxel_grid[voxel_size] = VoxelGrid.from_points(
                self.transformed_points, voxel_size
            )
        return self.voxel_grid[voxel_size]

    @property
    def transformed_points(self):
        if self._transformed_points is None:
            self._transformed_points = transform(self.pose, self.point_cloud, axis=0)
        return self._transformed_points

    @property
    def pcd(self):
        if self._pcd is None:
            self._pcd = o3d.geometry.PointCloud()
            self._pcd.points = o3d.utility.Vector3dVector(self.transformed_points)
        return self._pcd

    def color_from_id(self, order="rgb", num_type=int) -> tuple:
        """Returns a color tuple based on the segment ID."""
        return color_from_seed(self.id, order, num_type)
