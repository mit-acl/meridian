import numpy as np
from dataclasses import dataclass
from typing import List
from copy import deepcopy
import open3d as o3d
from robotdatapy import transform as transform

from meridian.primitive.primitive import Primitive


def get_roman_ratio_feature(roman_segment) -> np.ndarray:
    """Pack ROMAN segment volume/linearity/planarity/scattering into a 4-vector."""
    try:
        volume = roman_segment.volume
    except Exception:
        volume = 0.0
    return np.array(
        [
            volume,
            roman_segment.linearity,
            roman_segment.planarity,
            roman_segment.scattering,
        ]
    )


@dataclass
class DenseSegment(Primitive):
    id: int
    dense_points: np.ndarray
    ratio_feature: np.ndarray = None  # optional ratio feature vector
    cos_feature: np.ndarray = None  # optional cosine feature vector
    first_seen: float = None  # optional timestamp of first observation
    last_seen: float = None  # optional timestamp of last observation
    history: List[int] = None  # optional list of past segment ids
    occluded_points: np.ndarray = None  # optional occlusion points

    _gaussian: np.ndarray = None
    _pcd: o3d.geometry.PointCloud = None
    _eigvals: np.ndarray = None

    @property
    def points(self) -> np.ndarray:
        return self.dense_points

    @property
    def pcd(self):
        if self._pcd is None:
            self._pcd = o3d.geometry.PointCloud()
            self._pcd.points = o3d.utility.Vector3dVector(self.points)
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
    def num_points(self) -> int:
        return self.dense_points.shape[1]

    @property
    def semantic_descriptor(self) -> np.ndarray:
        return self.cos_feature

    @property
    def linearity(self):
        """Large if similar to a 1D line (Weinmann et al. ISPRS 2014)."""
        e = self.normalized_eigenvalues
        return (e[0] - e[1]) / e[0]

    @property
    def planarity(self):
        """Large if similar to a 2D plane (Weinmann et al. ISPRS 2014)."""
        e = self.normalized_eigenvalues
        return (e[1] - e[2]) / e[0]

    @property
    def scattering(self):
        """Large if this object is 3D, i.e., neither a line nor a plane (Weinmann et al. ISPRS 2014)."""
        e = self.normalized_eigenvalues
        return e[2] / e[0]

    def __post_init__(self):
        if self.cos_feature is not None:
            self.cos_feature /= np.linalg.norm(self.cos_feature)
        super().__post_init__()

    def transform(self, T):
        self.dense_points = transform.transform(T, self.dense_points)
        if self.occluded_points is not None:
            self.occluded_points = transform.transform(T, self.occluded_points)
        return self

    def copy(self):
        return DenseSegment(
            self.id,
            self.dense_points.copy(),
            self._copy_optional_array(self.ratio_feature),
            self._copy_optional_array(self.cos_feature),
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            occluded_points=self._copy_optional_array(self.occluded_points),
            history=deepcopy(self.history),
        )
