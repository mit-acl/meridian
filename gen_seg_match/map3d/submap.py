import numpy as np
from dataclasses import dataclass
from typing import List
from copy import deepcopy
from robotdatapy.transform import transform
from roman.utils import transform_rm_roll_pitch

from gen_seg_match.segment.segment_types import GeneralSegment
from gen_seg_match.params.submap_params import SubmapParams
from gen_seg_match.params.roman_conversion_params import RomanConversionParams
from gen_seg_match.map3d.segments_from_roman import GeneralSegmentConverter
from gen_seg_match.utils import sort_time_intervals
from roman.map.map import ROMANMap


@dataclass
class Submap:
    id: int
    time: float
    segments: List[GeneralSegment]
    segment_ids: List[int]
    pose_flu: np.ndarray
    segment_frame: str = "submap_gravity_aligned"
    descriptor: np.ndarray = None

    @property
    def pose_gravity_aligned(self):
        return transform_rm_roll_pitch(self.pose_flu)

    @property
    def position(self):
        return self.pose_flu[:3, 3]

    @property
    def segments_as_global_points(self):
        # self.pose_gravity_aligned returns T_odom_center
        # which is transformation from submap center frame to odom frame
        # so this transforms segments back to the global (odom) frame
        T_odom_center = self.pose_gravity_aligned
        return transform(
            T_odom_center, np.vstack([seg.center.T for seg in self.segments])
        )  # (1, 3) -> (N, 3)

    def __len__(self):
        return len(self.segments)


def submaps_from_roman_map(
    roman_map: ROMANMap,
    submap_params: SubmapParams,
    roman_conversion_params: RomanConversionParams,
) -> List[Submap]:
    """
    Create submaps from a ROMAN Map

    Args:
        roman_map (ROMANMap): Full ROMAN map.
        submap_params (SubmapParams): Parameters for submap creation.

    Returns:
        List[Submap]: List of created submaps.
    """
    # Temporary patch to get rid of duplicate segment ids
    # TODO: fix this upstream in ROMAN
    max_segment_id = np.max([seg.id for seg in roman_map.segments]) + 1
    segment_ids = set()
    for seg in roman_map.segments:
        if seg.id in segment_ids:
            seg.id = max_segment_id
            max_segment_id += 1
        segment_ids.add(seg.id)

    submaps = []

    # force fill submaps to a set size, with set overlap -----------
    if submap_params.creation_method == "force_fill":
        segments_sorted_by_time = sorted(
            roman_map.segments,
            key=lambda seg: seg.reference_time(
                use_avg_time=submap_params.segment_avg_time
            ),
        )

        for i in range(
            0,
            len(segments_sorted_by_time),
            submap_params.max_size - submap_params.overlap,
        ):
            sm_segments = segments_sorted_by_time[i : i + submap_params.max_size]
            if len(sm_segments) == 0:
                continue

            submap_time = np.average(
                [
                    seg.reference_time(use_avg_time=submap_params.segment_avg_time)
                    for seg in sm_segments
                ]
            )
            submap_roman_map_index = np.argmin(np.abs(roman_map.times - submap_time))

            submaps.append(
                Submap(
                    id=len(submaps),
                    time=roman_map.times[submap_roman_map_index],
                    segments=[deepcopy(seg) for seg in sm_segments],
                    segment_ids=[],
                    pose_flu=roman_map.trajectory[submap_roman_map_index],
                )
            )

    # create submaps adaptively, then use metrics to fill -----------
    elif submap_params.creation_method == "adaptive":
        # create submaps
        for i, (pose, t) in enumerate(zip(roman_map.trajectory, roman_map.times)):
            if (
                i == 0
                or np.linalg.norm(pose[:-1, -1] - submaps[-1].pose_flu[:-1, -1])
                > submap_params.center_dist
                or (t - submaps[-1].time > submap_params.center_time)
            ):
                submaps.append(
                    Submap(
                        id=len(submaps),
                        time=t,
                        segments=[],
                        segment_ids=[],
                        pose_flu=pose,
                    )
                )

        # add segments to submaps
        for i, submap in enumerate(submaps):
            # TODO: do we need this? seems too constrictive for long-duration segments
            # set up timing constraints
            tm1 = submaps[i - 1].time if i > 0 else -np.inf
            tp1 = submaps[i + 1].time if i < len(submaps) - 1 else np.inf

            def meets_time_constraints(seg):
                return not (
                    seg.first_seen > tp1 + submap_params.center_time
                    or seg.last_seen < tm1 - submap_params.center_time
                )

            for seg in roman_map.segments:
                if (
                    submap_params.radius is None
                    or (
                        np.linalg.norm(seg.center.flatten() - submap.pose_flu[:-1, -1])
                        < submap_params.radius
                    )
                ) and meets_time_constraints(seg):
                    submap.segments.append(deepcopy(seg))

            if submap_params.max_size is not None:
                if submap_params.pruning_method == "time":  # time-based pruning

                    def pruning_key(seg):
                        return abs(
                            seg.reference_time(
                                use_avg_time=submap_params.segment_avg_time
                            )
                            - submaps[i].time
                        )
                else:  # distance-based pruning

                    def pruning_key(seg):
                        return np.linalg.norm(
                            seg.center.flatten() - submap.pose_flu[:-1, -1]
                        )

                segments_sorted_by_key = sorted(submap.segments, key=pruning_key)
                submap.segments = segments_sorted_by_key[: submap_params.max_size]

    # create submaps at set times -----------
    elif submap_params.creation_method == "set_times":
        segment_time_intervals = np.array(
            [(seg.first_seen, seg.last_seen) for seg in roman_map.segments]
        )

        for t in submap_params.submap_times:
            submap_roman_map_index = np.argmin(np.abs(np.array(roman_map.times) - t))

            segments = [
                deepcopy(roman_map.segments[i])
                for i in sort_time_intervals(segment_time_intervals, t)[
                    : submap_params.max_size
                ]
            ]
            segments = [
                seg
                for seg in segments
                if np.linalg.norm(
                    seg.center.flatten()
                    - roman_map.trajectory[submap_roman_map_index][:3, 3]
                )
                < submap_params.radius
            ]

            submaps.append(
                Submap(
                    id=len(submaps),
                    time=roman_map.times[submap_roman_map_index],
                    segments=segments,
                    segment_ids=[],
                    pose_flu=roman_map.trajectory[submap_roman_map_index],
                )
            )

    else:
        raise ValueError(
            f"Unknown submap creation method: {submap_params.creation_method}"
        )

    # submap postprocessing

    gen_seg_converter = GeneralSegmentConverter(roman_conversion_params)

    submaps = [submap for submap in submaps if len(submap.segments) > 0]

    for submap in submaps:
        # sm.pose is the pose of center w.r.t. odom, which is T_odom_center, inverse is T_center_odom
        # transforms the segments into the center frame (centered w.r.t submap centroid) since they are in the odom frame
        T_center_odom = np.linalg.inv(submap.pose_gravity_aligned)
        for seg in submap.segments:
            seg.transform(T_center_odom)

        if submap_params.descriptor == "mean_semantic":
            submap.descriptor = np.mean(
                [seg.semantic_descriptor for seg in submap.segments], axis=0
            ).flatten()

        # convert ROMAN to general segments
        submap.segments = gen_seg_converter.roman_to_general_segments(submap.segments)
        submap.segment_ids = [seg.id for seg in submap.segments]

    return submaps


def segment_map_3d_to_2d(map_3d):
    """Convert a 3D map to a 2D map by averaging the z-axis."""
    map_2d = deepcopy(map_3d)
    to_rm = []
    for seg in map_2d.segments:
        seg.points[:, 2] = 0.0
        seg._cleanup_points()
        if len(seg.points) < 2:
            to_rm.append(seg)
            continue
        try:
            seg.final_cleanup()
        except Exception:
            to_rm.append(seg)
    return map_2d
