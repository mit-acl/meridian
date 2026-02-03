import numpy as np
from dataclasses import dataclass
from typing import List
from copy import deepcopy
from robotdatapy.transform import transform
from roman.utils import transform_rm_roll_pitch
import pickle

from gen_seg_match.segment.segment_types import SegmentList
from gen_seg_match.params.submap_params import SubmapParams
from gen_seg_match.params.roman_conversion_params import RomanConversionParams
from gen_seg_match.map3d.segments_from_roman import GeneralSegmentConverter
from gen_seg_match.utils import sort_time_intervals
from roman.map.map import ROMANMap


@dataclass
class Submap:
    id: int
    time: float
    segments: SegmentList
    pose_flu: np.ndarray
    segment_frame: str = "submap_gravity_aligned"
    descriptor: np.ndarray = None
    metadata: dict = None

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

    @property
    def segment_ids(self):
        return [seg.id for seg in self.segments]

    def __len__(self):
        return len(self.segments)

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    def load(filepath: str) -> "Submap":
        with open(filepath, "rb") as f:
            submap = pickle.load(f)
        return submap


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
    gen_seg_converter = GeneralSegmentConverter(roman_conversion_params)

    submaps = []
    general_segments = gen_seg_converter.roman_to_general_segments(roman_map.segments)

    # force fill submaps to a set size, with set overlap -----------
    if submap_params.creation_method == "force_fill":
        segments_sorted_by_time = sorted(
            general_segments,
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

            for seg in general_segments:
                if (
                    submap_params.radius is None
                    or (
                        np.linalg.norm(seg.point.flatten() - submap.pose_flu[:-1, -1])
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
            [(seg.first_seen, seg.last_seen) for seg in general_segments]
        )

        for t in submap_params.submap_times:
            submap_roman_map_index = np.argmin(np.abs(np.array(roman_map.times) - t))

            # either keep the nearest segments in time or in distance
            # TODO: this could probably borrow some code from the adaptive method
            if submap_params.pruning_method == "time":
                segments = [
                    deepcopy(general_segments[i])
                    for i in sort_time_intervals(segment_time_intervals, t)[
                        : submap_params.max_size
                    ]
                ]
                segments = [
                    seg
                    for seg in segments
                    if np.linalg.norm(
                        seg.point.flatten()
                        - roman_map.trajectory[submap_roman_map_index][:3, 3]
                    )
                    < submap_params.radius
                ]
            elif submap_params.pruning_method == "distance":
                segments = [
                    deepcopy(seg)
                    for seg in general_segments
                    if np.linalg.norm(
                        seg.point.flatten()
                        - roman_map.trajectory[submap_roman_map_index][:3, 3]
                    )
                    < submap_params.radius
                ]
                segments_sorted_by_distance = sorted(
                    segments,
                    key=lambda seg: np.linalg.norm(
                        seg.point.flatten()
                        - roman_map.trajectory[submap_roman_map_index][:3, 3]
                    ),
                )
                segments = segments_sorted_by_distance[: submap_params.max_size]

            submaps.append(
                Submap(
                    id=len(submaps),
                    time=roman_map.times[submap_roman_map_index],
                    segments=segments,
                    pose_flu=roman_map.trajectory[submap_roman_map_index],
                )
            )

    else:
        raise ValueError(
            f"Unknown submap creation method: {submap_params.creation_method}"
        )

    # submap postprocessing

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

    for sm in submaps:
        sm.segments = SegmentList(sm.segments)
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


if __name__ == "__main__":
    import argparse
    import pickle
    import matplotlib.pyplot as plt
    from gen_seg_match.viz.viz_segments import viz_segments

    parser = argparse.ArgumentParser(description="Create/save/visualize submaps")

    parser.add_argument("-r", "--roman-map", type=str, help="Input ROMAN map file")
    parser.add_argument(
        "-p", "--submap-params", type=str, help="YAML file with submap parameters"
    )
    parser.add_argument("-s", "--input-submaps", type=str, help="Input submap file")
    parser.add_argument(
        "-v", "--visualize", action="store_true", help="Visualize submaps"
    )
    parser.add_argument(
        "-i", "--submap-idx", type=int, help="Index of submap to visualize"
    )
    parser.add_argument("-o", "--output-submaps", type=str, help="Output submap file")

    args = parser.parse_args()

    can_load_roman_map = args.roman_map is not None and args.submap_params is not None
    can_load_input_submaps = args.input_submaps is not None
    if not (can_load_roman_map or can_load_input_submaps):
        parser.error(
            "Either a submaps file or a ROMAN map with submap parameters must be provided."
        )
    if can_load_roman_map and can_load_input_submaps:
        parser.error(
            "Provide either a submaps file or a ROMAN map with submap parameters, not both."
        )

    if can_load_roman_map:
        submap_params_file = args.submap_params
        roman_map_conversion_file = args.submap_params  # assuming same file for now
        roman_map = ROMANMap.from_pickle(args.roman_map)
        submap_params = SubmapParams.from_yaml(submap_params_file)
        roman_conversion_params = RomanConversionParams.from_yaml(
            roman_map_conversion_file
        )
        submaps = submaps_from_roman_map(
            roman_map, submap_params, roman_conversion_params
        )
    else:
        with open(args.input_submaps, "rb") as f:
            submaps = pickle.load(f)

    if args.visualize:
        if args.submap_idx is not None:
            idx = args.submap_idx
        else:
            submap_centers = np.array([sm.pose_flu[:3, 3] for sm in submaps])
            plt.figure()
            plt.scatter(submap_centers[:, 0], submap_centers[:, 1])
            max_axis_lim_len = (
                np.max(submap_centers.max(axis=0) - submap_centers.min(axis=0)) * 1.1
            )
            for i in range(len(submaps)):
                plt.text(
                    submap_centers[i, 0] + max_axis_lim_len / 100,
                    submap_centers[i, 1] + max_axis_lim_len / 100,
                    str(i),
                )
            plt.show()

            idx_str = input("Please input the desired submap index: \n")
            idx = int(idx_str)

        submap = submaps[args.submap_idx]
        viz_segments(submap.segments)

    if args.output_submaps is not None:
        with open(args.output_submaps, "wb") as f:
            pickle.dump(submaps, f)
