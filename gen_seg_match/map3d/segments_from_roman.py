import numpy as np
from typing import List, Tuple

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

        general_segments = []

        for roman_segment in roman_segments:
            if self.params.force_points_only:
                general_segments.append(self.to_point_segment(roman_segment, pt=mean))
                continue

            mean, eigvals, U, principal_components = self.pca(roman_segment)
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
                < eigvals[2] / eigvals[1]
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
                general_segments.append(self.to_plane_segment(roman_segment, mean, U))
            else:
                general_segments.append(self.to_point_segment(roman_segment, mean))

        general_segments.reindex()
        return general_segments
