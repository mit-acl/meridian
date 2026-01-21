import numpy as np
from typing import List, Tuple
import shapely

from roman.object.segment import Segment as RomanSegment
from gen_seg_match.segment.segment_types import (
    GeneralSegment,
    SegmentPoint,
    SegmentLine,
    SegmentPlane,
    SegmentList,
)
from gen_seg_match.params.roman_conversion_params import RomanConversionParams
from gen_seg_match.map3d.segments_from_img import (
    get_line_general_segments as get_line_with_occlusion,
)


class GeneralSegmentConverter:
    def __init__(self, params: RomanConversionParams):
        self.params = params

    def pca(
        self, roman_segment: RomanSegment
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mean, C = roman_segment.gaussian
        U, eigvals, _ = np.linalg.svd(C)
        mean_center_points = roman_segment.points - mean
        principal_components = (
            mean_center_points @ U
        )  # U[:, i] is the i-th principal component
        # principal_components shape: (num_points, 3)

        return mean, eigvals, U, principal_components

    def is_line(self, eigvals: np.ndarray, principal_components: np.ndarray) -> bool:
        if (
            eigvals[1] / eigvals[0] > self.params.line_max_e1_e0
            or eigvals[2] / eigvals[0] > self.params.line_max_e2_e0
        ):
            return False

        dist_from_line = np.linalg.norm(principal_components[:, 1:], axis=1)
        rms_dist_from_line = np.sqrt(np.mean(dist_from_line**2))

        return rms_dist_from_line <= self.params.line_rms_threshold

    def is_plane(self, eigvals: np.ndarray, principal_components: np.ndarray) -> bool:
        if (
            eigvals[2] / eigvals[1] > self.params.plane_max_e2_e1
            or eigvals[2] / eigvals[0] > self.params.plane_max_e2_e0
        ):
            return False

        dist_from_plane = principal_components[:, 2]
        rms_dist_from_plane = np.sqrt(np.mean(dist_from_plane**2))

        return rms_dist_from_plane <= self.params.plane_rms_threshold

    def to_plane_segment(
        self, roman_segment: RomanSegment, mean: np.ndarray, U: np.ndarray
    ) -> GeneralSegment:
        normal = U[:, 2]
        dense_points = roman_segment.points if self.params.copy_dense_points else None

        return SegmentPlane(
            id=roman_segment.id,
            point=mean,  # avoid using roman_segment.center in case _center_ref == 'bottom-middle'
            normal=normal,
            ratio_feature=self.get_roman_ratio_feature(roman_segment),
            cos_feature=roman_segment.semantic_descriptor,
            first_seen=roman_segment.first_seen,
            last_seen=roman_segment.last_seen,
            dense_points=dense_points,
            history=getattr(roman_segment, "history", []),
        )

    def to_line_segment(
        self,
        roman_segment: RomanSegment,
        mean: np.ndarray,
        U: np.ndarray,
        principal_components: np.ndarray,
    ) -> GeneralSegment:
        direction = U[:, 0]
        line_projections = principal_components[:, 0]
        line_min = mean + np.min(line_projections) * direction
        line_max = mean + np.max(line_projections) * direction
        dense_points = roman_segment.points if self.params.copy_dense_points else None

        line = SegmentLine(
            id=roman_segment.id,
            point=mean,  # avoid using roman_segment.center in case _center_ref == 'bottom-middle'
            direction=direction,
            endpoints=(line_min, line_max),
            ratio_feature=self.get_roman_ratio_feature(roman_segment),
            cos_feature=roman_segment.semantic_descriptor,
            first_seen=roman_segment.first_seen,
            last_seen=roman_segment.last_seen,
            dense_points=dense_points,
            history=getattr(roman_segment, "history", []),
        )
        line_segments = []
        if self.params.line_inclusion:
            line_segments.append(line)
        if self.params.line_separate_endpoints:
            for pt in [line_min, line_max]:
                line_segments.append(self.to_point_segment(roman_segment, pt=pt))
        if self.params.line_separate_center_point:
            line_segments.append(self.to_point_segment(roman_segment, pt=mean))
        return line_segments

    def to_point_segment(
        self, roman_segment: RomanSegment, pt: np.ndarray
    ) -> SegmentPoint:
        dense_points = roman_segment.points if self.params.copy_dense_points else None
        segment_history = []
        if hasattr(roman_segment, "history"):
            segment_history = roman_segment.history
        return SegmentPoint(
            id=roman_segment.id,
            point=pt,  # TODO: is _center_ref == 'bottom-middle' an issue?
            ratio_feature=self.get_roman_ratio_feature(roman_segment),
            cos_feature=roman_segment.semantic_descriptor,
            first_seen=roman_segment.first_seen,
            last_seen=roman_segment.last_seen,
            dense_points=dense_points,
            history=segment_history,
        )

    def get_roman_ratio_feature(self, roman_segment: RomanSegment) -> np.ndarray:
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

    def roman_to_general_segments(
        self, roman_segments: List[RomanSegment]
    ) -> List[GeneralSegment]:
        # Temporary patch to get rid of duplicate segment ids
        # TODO: fix this upstream in ROMAN
        max_segment_id = np.max([seg.id for seg in roman_segments]) + 1
        segment_ids = set()
        for seg in roman_segments:
            if seg.id in segment_ids:
                seg.id = max_segment_id
                max_segment_id += 1
            segment_ids.add(seg.id)

        general_segments = SegmentList([])

        for roman_segment in roman_segments:
            mean, eigvals, U, principal_components = self.pca(roman_segment)

            if self.params.force_points_only:
                general_segments.append(self.to_point_segment(roman_segment, pt=mean))
                continue

            # if self.is_plane(eigvals, principal_components):
            #     general_segments.append(self.to_point_segment(roman_segment))
            #     # TODO: enable general segments
            #     # general_segments.append(self.to_plane_segment(roman_segment, mean, U))
            if self.is_line(eigvals, principal_components):
                general_segments.extend(
                    self.to_line_segment(roman_segment, mean, U, principal_components)
                )
            else:
                general_segments.append(self.to_point_segment(roman_segment, pt=mean))

        # Related line and point segments may have the same id -
        # ensure all segment ids are unique
        general_segments.reindex()
        return general_segments

    # alternative to convert segments
    # def roman_to_general_segments(
    #     self, roman_segments: List[RomanSegment]
    # ) -> List[GeneralSegment]:
    #     # Temporary patch to get rid of duplicate segment ids
    #     # TODO: fix this upstream in ROMAN
    #     max_segment_id = np.max([seg.id for seg in roman_segments]) + 1
    #     segment_ids = set()
    #     for seg in roman_segments:
    #         if seg.id in segment_ids:
    #             seg.id = max_segment_id
    #             max_segment_id += 1
    #         segment_ids.add(seg.id)

    #     general_segments = SegmentList()

    #     for roman_segment in roman_segments:
    #         mean, eigvals, U, principal_components = self.pca(roman_segment)
    #         max_extents = np.max(principal_components, axis=0) - np.min(
    #             principal_components, axis=0
    #         )

    #         if self.params.force_points_only:
    #             general_segments.append(self.to_point_segment(roman_segment, mean))
    #             continue
    #         # assert max_extents[0] >= max_extents[1] >= max_extents[2], (
    #         #     "PCA principal components are not sorted correctly."
    #         # )
    #         # since extents may not perfectly align with eigvals, reorder extents
    #         max_extents = np.sort(
    #             max_extents,
    #         )[::-1]

    #         if (
    #             max_extents[2]
    #             < max_extents[1]
    #             < max_extents[0]
    #             < self.params.non_point_min_extent
    #         ):
    #             general_segments.append(self.to_point_segment(roman_segment, mean))
    #         elif (
    #             max_extents[2] < max_extents[1] < self.params.max_minor_axis_extent
    #             and eigvals[2] / eigvals[0]
    #             < eigvals[2] / eigvals[1]
    #             < self.params.max_eigval_ratio
    #         ):
    #             general_segments.extend(
    #                 self.to_line_segment(roman_segment, mean, U, principal_components)
    #             )
    #         # elif (
    #         #     max_extents[2] < self.params.max_minor_axis_extent
    #         #     and eigvals[2] / eigvals[0]
    #         #     < eigvals[2] / eigvals[1]
    #         #     < self.params.max_eigval_ratio
    #         # ):
    #         #     general_segments.append(self.to_plane_segment(roman_segment, mean, U))
    #         else:
    #             general_segments.append(self.to_point_segment(roman_segment, mean))

    #     # Related line and point segments may have the same id -
    #     # ensure all segment ids are unique
    #     general_segments.reindex()
    #     return general_segments

    def segment_with_occlusion_to_general_segments(
        self,
        segments: List[RomanSegment],
    ) -> SegmentList:
        general_segments = SegmentList([])

        for roman_segment in segments:
            if roman_segment.num_points == 0:
                continue

            mean, eigvals, U, principal_components = self.pca(roman_segment)
            max_extents = np.max(principal_components, axis=0) - np.min(
                principal_components, axis=0
            )
            if self.params.force_points_only:
                general_segments.append(self.to_point_segment(roman_segment, mean))
                continue
            # assert max_extents[0] >= max_extents[1] >= max_extents[2], (
            #     "PCA principal components are not sorted correctly."
            # )
            # since extents may not perfectly align with eigvals, reorder extents
            max_extents = np.sort(
                max_extents,
            )[::-1]

            if (
                max_extents[2]
                < max_extents[1]
                < max_extents[0]
                < self.params.non_point_min_extent
            ):
                general_segments.append(self.to_point_segment(roman_segment, mean))
            elif (
                max_extents[2] < max_extents[1] < self.params.max_minor_axis_extent
                and eigvals[2] / eigvals[0]
                < eigvals[1] / eigvals[0]
                < self.params.max_eigval_ratio
            ):
                general_segments.extend(
                    get_line_with_occlusion(
                        roman_segment, roman_segment.occluded_points
                    )
                )
            elif (
                max_extents[2] < self.params.max_minor_axis_extent
                and eigvals[2] / eigvals[0]
                < eigvals[2] / eigvals[1]
                < self.params.max_eigval_ratio
            ):
                general_segments.extend(
                    self.get_plane_with_occlusion(
                        roman_segment,
                        roman_segment.occluded_points,
                        mean,
                        U,
                    )
                )
            else:
                general_segments.append(self.to_point_segment(roman_segment, mean))

        general_segments.reindex()
        return general_segments
    
    def get_plane_with_occlusion(
        self, segment: RomanSegment, occluded_points: np.ndarray, mean: np.ndarray, U: np.ndarray
    ) -> SegmentList:
        normal = U[:, 2]
        dense_points = segment.points if self.params.copy_dense_points else None

        sparse_segments = SegmentList()

        sparse_segments.append(SegmentPlane(
            id=segment.id,
            point=mean,  # avoid using roman_segment.center in case _center_ref == 'bottom-middle'
            normal=normal,
            ratio_feature=self.get_roman_ratio_feature(segment),
            cos_feature=segment.semantic_descriptor,
            first_seen=segment.first_seen,
            last_seen=segment.last_seen,
            dense_points=dense_points,
            history=getattr(segment, "history", []),
        ))
        
        # project points and occluded points to 2D plane
        plane_basis_vectors = U[:, :2]  # first two principal components
        points_2d = self.get_points_on_2d_plane(segment.points, normal, mean, plane_basis_vectors)
        occluded_points_2d = self.get_points_on_2d_plane(occluded_points, normal, mean, plane_basis_vectors)
        line_borders = self.get_line_borders_from_2d_points(points_2d, occluded_points_2d)
        for pt1, pt2 in line_borders:
            pt1_3d = mean + pt1[0] * plane_basis_vectors[:, 0] + pt1[1] * plane_basis_vectors[:, 1]
            pt2_3d = mean + pt2[0] * plane_basis_vectors[:, 0] + pt2[1] * plane_basis_vectors[:, 1]

            if occluded_points.size > 0:
                t = (np.ones((3, 10)) * np.linspace(0, 1, 10)).T
                pts_3d = t * pt1_3d + (1 - t) * pt2_3d
                occluded_points_dists_mins = []
                for pt in pts_3d:
                    occluded_points_dists = np.linalg.norm(pt - occluded_points, axis=1)
                    occluded_points_dists_mins.append(np.min(occluded_points_dists))
                if np.max(occluded_points_dists_mins) < 1.0:
                    continue
            
            # TODO: hard coding here, don't want to include lines
            # are are just at the border of max depth
            if pt1_3d[2] > 7.0 and pt2_3d[2] > 7.0:
                continue
            # if pt1_3d[2] < 1.0 or pt2_3d[2] < 1.0:
            #     continue
            sparse_segments.append(SegmentLine.from_endpoints(
                id=segment.id,
                pt1=pt1_3d,
                pt2=pt2_3d,
                ratio_feature=self.get_roman_ratio_feature(segment),
                cos_feature=segment.semantic_descriptor,
                first_seen=segment.first_seen,
                last_seen=segment.last_seen,
                dense_points=None,
                history=getattr(segment, "history", []),
            ))
            
        return sparse_segments
        
    def get_points_on_2d_plane(self, points: np.ndarray, plane_normal: np.ndarray, plane_offset: np.ndarray, plane_basis_vectors: np.ndarray) -> np.ndarray:
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
        R = np.column_stack((e1, e2, plane_normal))
        points_on_plane = (np.eye(3) - np.outer(plane_normal, plane_normal)) @ (points - plane_offset).T
        points_on_2d = R.T @ points_on_plane
        return points_on_2d[:2, :].T
    
    def get_line_borders_from_2d_points(self, points_2d: np.ndarray, occluded_points_2d: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
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
            if angle < np.deg2rad(10.0): # TODO: magic number
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
    