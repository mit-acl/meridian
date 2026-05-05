import numpy as np
from typing import List, Tuple, Union
import shapely

from roman.object.segment import Segment as RomanSegment
from meridian.segment.segment_types import (
    GeneralSegment,
    SegmentPoint,
    SegmentLine,
    SegmentPlane,
    SegmentList,
    DenseSegment,
)
from meridian.segment.map_segment import MapSegment

DenseSegment = Union[DenseSegment, RomanSegment, MapSegment]
from meridian.params import DenseToSparseParams


class DenseToSparseConverter:
    def __init__(self, params: DenseToSparseParams):
        self.params = params

    def convert(
        self, segments: List[DenseSegment], use_occlusion: bool = True
    ) -> SegmentList:
        general_segments = SegmentList([])

        for dense_segment in segments:
            if dense_segment.num_points == 0:
                continue

            mean, eigvals, U, principal_components = self.pca(dense_segment)
            max_extents = np.max(principal_components, axis=0) - np.min(
                principal_components, axis=0
            )
            if self.params.force_points_only:
                general_segments.append(self.to_point_segment(dense_segment, mean))
                continue

            # since extents may not perfectly align with eigvals, reorder extents
            max_extents = np.sort(
                max_extents,
            )[::-1]

            if self.is_point(max_extents):
                general_segments.append(self.to_point_segment(dense_segment, mean))
            elif self.is_line(eigvals=eigvals, max_extents=max_extents):
                general_segments.extend(
                    self.to_line_segment(
                        dense_segment,
                        mean,
                        U,
                        principal_components,
                        use_occlusion=use_occlusion,
                    )
                )
            elif self.is_plane(eigvals=eigvals, max_extents=max_extents):
                general_segments.extend(
                    self.to_plane_segment(
                        dense_segment,
                        mean,
                        U,
                        use_occlusion=use_occlusion,
                    )
                )
            else:
                general_segments.append(self.to_point_segment(dense_segment, mean))

        general_segments.reindex()
        return general_segments

    def pca(
        self, dense_segment: DenseSegment
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mean, C = dense_segment.gaussian
        U, eigvals, _ = np.linalg.svd(C)
        mean_center_points = dense_segment.points - mean
        principal_components = (
            mean_center_points @ U
        )  # U[:, i] is the i-th principal component
        # principal_components shape: (num_points, 3)

        return mean, eigvals, U, principal_components

    def is_point(self, max_extents: np.ndarray) -> bool:
        return (
            max_extents[2]
            < max_extents[1]
            < max_extents[0]
            < self.params.non_point_min_extent
        )

    def is_line(self, eigvals: np.ndarray, max_extents: np.ndarray) -> bool:
        # alternative conversion
        # if (
        #     eigvals[1] / eigvals[0] > self.params.line_max_e1_e0
        #     or eigvals[2] / eigvals[0] > self.params.line_max_e2_e0
        # ):
        #     return False

        # dist_from_line = np.linalg.norm(principal_components[:, 1:], axis=1)
        # rms_dist_from_line = np.sqrt(np.mean(dist_from_line**2))

        # return rms_dist_from_line <= self.params.line_rms_threshold
        return (
            max_extents[2] < max_extents[1] < self.params.max_minor_axis_extent
            and eigvals[2] / eigvals[0]
            < eigvals[1] / eigvals[0]
            < self.params.max_eigval_ratio
        )

    def is_plane(self, eigvals: np.ndarray, max_extents: np.ndarray) -> bool:
        # alternative conversion
        # if (
        #     eigvals[2] / eigvals[1] > self.params.plane_max_e2_e1
        #     or eigvals[2] / eigvals[0] > self.params.plane_max_e2_e0
        # ):
        #     return False

        # dist_from_plane = principal_components[:, 2]
        # rms_dist_from_plane = np.sqrt(np.mean(dist_from_plane**2))

        # return rms_dist_from_plane <= self.params.plane_rms_threshold
        return (
            max_extents[2] < self.params.max_minor_axis_extent
            and eigvals[2] / eigvals[0]
            < eigvals[2] / eigvals[1]
            < self.params.max_eigval_ratio
        )

    def to_point_segment(
        self, roman_segment: RomanSegment, pt: np.ndarray
    ) -> SegmentPoint:
        dense_points = roman_segment.points if self.params.copy_dense_points else None
        return SegmentPoint(
            id=roman_segment.id,
            point=pt,  # TODO: is _center_ref == 'bottom-middle' an issue?
            ratio_feature=self.get_roman_ratio_feature(roman_segment),
            cos_feature=roman_segment.semantic_descriptor,
            first_seen=roman_segment.first_seen,
            last_seen=roman_segment.last_seen,
            dense_points=dense_points,
            history=getattr(roman_segment, "history", []),
        )

    def to_line_segment(
        self,
        dense_segment: DenseSegment,
        mean: np.ndarray,
        U: np.ndarray,
        principal_components: np.ndarray,
        use_occlusion: bool = False,
    ) -> GeneralSegment:
        direction = U[:, 0]
        line_projections = principal_components[:, 0]
        line_min = mean + np.min(line_projections) * direction
        line_max = mean + np.max(line_projections) * direction
        dense_points = dense_segment.points if self.params.copy_dense_points else None

        end_pts = (line_min, line_max)
        offset = mean.flatten()

        if use_occlusion:
            end_pts, offset, direction = self.get_occluded_line_borders(
                dense_segment, mean, U
            )

        line = SegmentLine(
            id=dense_segment.id,
            point=offset,  # avoid using dense_segment.center in case _center_ref == 'bottom-middle'
            direction=direction,
            endpoints=end_pts if not self.params.line_always_infinite else (None, None),
            ratio_feature=self.get_roman_ratio_feature(dense_segment),
            cos_feature=dense_segment.semantic_descriptor,
            first_seen=dense_segment.first_seen,
            last_seen=dense_segment.last_seen,
            dense_points=dense_points,
            history=getattr(dense_segment, "history", []),
        )
        line_segments = []
        if self.params.line_inclusion:
            line_segments.append(line)
        if self.params.line_separate_endpoints:
            for pt in end_pts:
                if pt is None:
                    continue
                line_segments.append(self.to_point_segment(dense_segment, pt=pt))
        if self.params.line_separate_center_point:
            line_segments.append(self.to_point_segment(dense_segment, pt=mean))
        return line_segments

    def to_plane_segment(
        self,
        segment: DenseSegment,
        mean: np.ndarray,
        U: np.ndarray,
        use_occlusion: bool = True,
    ) -> SegmentList:
        normal = U[:, 2]
        dense_points = segment.points if self.params.copy_dense_points else None

        sparse_segments = SegmentList()

        sparse_segments.append(
            SegmentPlane(
                id=segment.id,
                point=mean,  # avoid using roman_segment.center in case _center_ref == 'bottom-middle'
                normal=normal,
                ratio_feature=self.get_roman_ratio_feature(segment),
                cos_feature=segment.semantic_descriptor,
                first_seen=segment.first_seen,
                last_seen=segment.last_seen,
                dense_points=dense_points,
                history=getattr(segment, "history", []),
            )
        )
        if not use_occlusion:
            return sparse_segments

        occluded_points = segment.occluded_points
        # project points and occluded points to 2D plane
        plane_basis_vectors = U[:, :2]  # first two principal components
        points_2d = self.get_points_on_2d_plane(
            segment.points, normal, mean, plane_basis_vectors
        )
        occluded_points_2d = self.get_points_on_2d_plane(
            occluded_points, normal, mean, plane_basis_vectors
        )
        line_borders = self.get_line_borders_from_2d_points(
            points_2d, occluded_points_2d
        )
        for pt1, pt2 in line_borders:
            pt1_3d = (
                mean
                + pt1[0] * plane_basis_vectors[:, 0]
                + pt1[1] * plane_basis_vectors[:, 1]
            )
            pt2_3d = (
                mean
                + pt2[0] * plane_basis_vectors[:, 0]
                + pt2[1] * plane_basis_vectors[:, 1]
            )

            if occluded_points.size > 0:
                t = (np.ones((3, 10)) * np.linspace(0, 1, 10)).T
                pts_3d = t * pt1_3d + (1 - t) * pt2_3d
                occluded_points_dists_mins = []
                for pt in pts_3d:
                    occluded_points_dists = np.linalg.norm(pt - occluded_points, axis=1)
                    occluded_points_dists_mins.append(np.min(occluded_points_dists))
                if np.max(occluded_points_dists_mins) < 1.0:
                    continue

            if (
                pt1_3d[2] > self.params.plane_border_max_depth
                and pt2_3d[2] > self.params.plane_border_max_depth
            ):
                continue
            # if pt1_3d[2] < 1.0 or pt2_3d[2] < 1.0:
            #     continue
            sparse_segments.append(
                SegmentLine.from_endpoints(
                    id=segment.id,
                    pt1=pt1_3d,
                    pt2=pt2_3d,
                    ratio_feature=self.get_roman_ratio_feature(segment),
                    cos_feature=segment.semantic_descriptor,
                    first_seen=segment.first_seen,
                    last_seen=segment.last_seen,
                    dense_points=None,
                    history=getattr(segment, "history", []),
                )
            )

        return sparse_segments

    @classmethod
    def get_roman_ratio_feature(cls, roman_segment: RomanSegment) -> np.ndarray:
        # volume computation may fail for flat segments
        try:
            volume = roman_segment.volume
        except:
            volume = 0.0
        return np.array(
            [
                volume,
                roman_segment.linearity,
                roman_segment.planarity,
                roman_segment.scattering,
            ]
        )

    def get_points_on_2d_plane(
        self,
        points: np.ndarray,
        plane_normal: np.ndarray,
        plane_offset: np.ndarray,
        plane_basis_vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Projects 3D points onto a 2D plane defined by its normal, offset, and basis vectors.

        Args:
            points (np.ndarray): Nx3 array of 3D points to be projected.
            plane_normal (np.ndarray): 3 dimensional normal vector of the plane.
            plane_offset (np.ndarray): 3 dimensional point on the plane.
            plane_basis_vectors (np.ndarray): 3x2 array where each column is a basis vector of the plane.

        Returns:
            np.ndarray: Nx2 array of 2D projected points.
        """

        # Project points onto plane
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        e1 = plane_basis_vectors[:, 0] / np.linalg.norm(plane_basis_vectors[:, 0])
        e2 = plane_basis_vectors[:, 1] / np.linalg.norm(plane_basis_vectors[:, 1])
        centered_points = points - plane_offset
        points_on_2d = np.stack(
            [np.dot(e1, centered_points.T), np.dot(e2, centered_points.T)], axis=-1
        )
        return points_on_2d

    def get_line_borders_from_2d_points(
        self, points_2d: np.ndarray, occluded_points_2d: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Computes the line borders of a 2D point cloud. If line borders are all close to occluded points,
            or if lines are short, they are not included.

        Args:
            points_2d (np.ndarray): Nx2 array of 2D points.
            occluded_points_2d (np.ndarray): Nx2 array of 2D occluded points.

        Returns:
            SegmentList: sparse line segments representing the line borders.
        """
        lines = []
        if len(points_2d) == 0:
            return lines

        convex_hull_shapely = shapely.convex_hull(shapely.MultiPoint(points_2d))
        convex_hull = np.array(convex_hull_shapely.exterior.coords)

        # if edge angles are small, combine them
        smoothed_convex_hull = [convex_hull[0]]
        for i in range(1, len(convex_hull) - 1):
            pt_prev = convex_hull[i - 1]
            pt_curr = convex_hull[i]
            pt_next = convex_hull[i + 1]
            v1 = pt_curr - pt_prev
            v2 = pt_next - pt_curr
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            # if angle < np.deg2rad(self.params.plane_border_smoothing_angle_deg):
            if angle < np.deg2rad(10.0):  # TODO: magic number
                # skip current point
                continue
            else:
                smoothed_convex_hull.append(pt_curr)
        convex_hull = smoothed_convex_hull

        for i, pt0 in enumerate(np.array(convex_hull)):
            pt1 = np.array(convex_hull)[i + 1 if i + 1 < len(convex_hull) else 0]
            # first check if the line is long enough
            if np.linalg.norm(pt1 - pt0) > self.params.non_point_min_extent:
                # if not occluded poitns, add line directly
                if occluded_points_2d.size == 0:
                    lines.append((pt0, pt1))
                    continue

                # check if the line is far enough from occluded points
                occluded = False
                # for frac in np.linspace(0, 1, num=10):
                #     test_pt = pt0 + frac * (pt1 - pt0)
                #     dists_to_occluded = np.linalg.norm(occluded_points_2d - test_pt, axis=1)
                #     # TODO: magic number - num = 10 and 1.0
                #     if np.min(dists_to_occluded) < 1.0:
                #         occluded = True
                #         break
                if not occluded:
                    lines.append((pt0, pt1))

        return lines

    def get_occluded_line_borders(
        self,
        segment: DenseSegment,
        mean: np.ndarray,
        U: np.ndarray,
    ):
        occluded_points = segment.occluded_points
        assert segment.points.shape[1] == 3
        mean_center_points = segment.points - mean
        axis_aligned_points = (U.T @ mean_center_points.T).T

        mean = mean.reshape((3, 1))
        num_endpoints = 0
        pt0_axis_aligned = np.array([np.min(axis_aligned_points[:, 0]), 0.0, 0.0])
        pt1_axis_aligned = np.array([np.max(axis_aligned_points[:, 0]), 0.0, 0.0])
        if len(occluded_points) == 0:
            end_pts = [
                U @ pt0_axis_aligned.reshape((3, 1)) + mean,
                U @ pt1_axis_aligned.reshape((3, 1)) + mean,
            ]
            num_endpoints = 2
        else:
            axis_aligned_occluded_points = (
                U.T @ (occluded_points - mean.flatten()).T
            ).T
            dist_to_occluded_pt0 = np.linalg.norm(
                axis_aligned_occluded_points - pt0_axis_aligned, axis=1
            )
            dist_to_occluded_pt1 = np.linalg.norm(
                axis_aligned_occluded_points - pt1_axis_aligned, axis=1
            )

            end_pts = []

            if np.min(dist_to_occluded_pt0) > self.params.occlusion_dist:
                end_pts.append(U @ pt0_axis_aligned.reshape((3, 1)) + mean)
                num_endpoints += 1
            else:
                end_pts.append(None)

            if np.min(dist_to_occluded_pt1) > self.params.occlusion_dist:
                end_pts.append(U @ pt1_axis_aligned.reshape((3, 1)) + mean)
                num_endpoints += 1
            else:
                end_pts.append(None)

        direction = U[:, 0]
        # if there is only one endpoint, need to make sure the vector points the right directions
        if num_endpoints == 1:
            offset = end_pts[0] if end_pts[0] is not None else end_pts[1]
            offset = offset.flatten()
            if np.dot(direction, offset - mean.flatten()) > 0:
                direction = -direction
        else:
            offset = mean.flatten()

        return end_pts, offset, direction
