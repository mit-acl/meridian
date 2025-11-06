import numpy as np
from typing import List, Tuple

from roman.object.segment import Segment as RomanSegment
from gen_seg_match.segment.segment_types import (
    GeneralSegment,
    SegmentPoint,
    SegmentLine,
    SegmentPlane,
)
from gen_seg_match.params.roman_conversion_params import RomanConversionParams


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

        return SegmentPlane(
            id=roman_segment.id,
            point=mean,  # avoid using roman_segment.center in case _center_ref == 'bottom-middle'
            normal=normal,
            ratio_feature=None,
            cos_feature=roman_segment.semantic_descriptor,
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

        return SegmentLine(
            id=roman_segment.id,
            point=mean,  # avoid using roman_segment.center in case _center_ref == 'bottom-middle'
            direction=direction,
            endpoints=(line_min, line_max),
            ratio_feature=None,
            cos_feature=roman_segment.semantic_descriptor,
        )

    def to_point_segment(self, roman_segment: RomanSegment) -> SegmentPoint:
        return SegmentPoint(
            id=roman_segment.id,
            point=roman_segment.center,  # TODO: is _center_ref == 'bottom-middle' an issue?
            ratio_feature=None,
            cos_feature=roman_segment.semantic_descriptor,
        )

    def roman_to_general_segments(
        self, roman_segments: List[RomanSegment]
    ) -> List[GeneralSegment]:
        general_segments = []

        for roman_segment in roman_segments:
            mean, eigvals, U, principal_components = self.pca(roman_segment)

            if self.is_plane(eigvals, principal_components):
                general_segments.append(self.to_point_segment(roman_segment))
                # TODO: enable general segments
                # general_segments.append(self.to_plane_segment(roman_segment, mean, U))
            elif self.is_line(eigvals, principal_components):
                general_segments.append(
                    self.to_line_segment(roman_segment, mean, U, principal_components)
                )
            else:
                general_segments.append(self.to_point_segment(roman_segment))

        return general_segments
