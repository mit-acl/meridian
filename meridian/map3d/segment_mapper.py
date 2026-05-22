###########################################################
#
# segment_mapper.py
#
# Segment mapper class for open-set segment mapping
#
# Authors: Mason Peterson, Yulun Tian, Lucas Jia, Qingyuan Li
#
# Dec. 21, 2024
#
###########################################################

import numpy as np
from typing import Dict, List, Set, Union

from functools import cached_property

from robotdatapy.data.img_data import CameraParams

from meridian.map3d.similarity_metrics import ChamferDistance
from meridian.map3d.map_segment import MapSegment
from meridian.map3d.observation import Observation
from meridian.map3d.global_nearest_neighbor import global_nearest_neighbor
from meridian.params.segment_mapping_params import SegmentMappingParams
from meridian.map3d.submap import FrameType, Submap
from meridian.primitive.primitive_list import PrimitiveList
from meridian.primitive.dense_segment import (
    DenseSegment,
    get_roman_ratio_feature,
)

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SegmentMapper:
    def __init__(
        self,
        params: SegmentMappingParams,
        camera_params: CameraParams,
        ground_submap_mapping=None,
        place_recognition=None,
    ):
        self.params = params
        self.camera_params = camera_params

        self.segment_nursery = []
        self.segments = []
        self.inactive_segments = []
        self.segment_graveyard = []
        self.id_counter = 0
        self.last_pose = None
        self.poses_cam_history = []
        self.times_history = []
        self.frame_descriptors_history = []

        # Incremental 2D ground submap state
        self._ground_submap_mapping = ground_submap_mapping
        self.place_recognition = place_recognition
        self._seg_ids_not_in_sm: List[int] = []
        self._seg_last_updated: Dict[int, float] = {}
        self._known_seg_ids: Set[int] = set()
        self.submaps_2d = []
        self._submap_intermediates = []
        self._submap_counter: int = 0
        self._last_submap_time: float = -np.inf

    def update(
        self,
        t: float,
        pose: np.array,
        observations: List[Observation],
        frame_descriptor: np.ndarray,
    ):
        self.poses_cam_history.append(pose)
        self.times_history.append(t)
        self.frame_descriptors_history.append(frame_descriptor)  # may be None

        if len(observations) == 0:  # nothing to update
            return

        # store last pose
        self.last_pose = pose.copy()

        # associate observations with segments
        associated_pairs = global_nearest_neighbor(
            self.segments + self.segment_nursery,
            observations,
            self.similarity_function,
            self.similarity_range,
        )

        # separate segments associated with nursery and normal segments
        pairs_existing = [
            [seg_idx, obs_idx]
            for seg_idx, obs_idx in associated_pairs
            if seg_idx < len(self.segments)
        ]
        pairs_nursery = [
            [seg_idx - len(self.segments), obs_idx]
            for seg_idx, obs_idx in associated_pairs
            if seg_idx >= len(self.segments)
        ]

        # update segments with associated observations
        for seg_idx, obs_idx in pairs_existing:
            self.segments[seg_idx].update(observations[obs_idx], integrate_points=True)
        for seg_idx, obs_idx in pairs_nursery:
            self.segment_nursery[seg_idx].update(
                observations[obs_idx], integrate_points=True
            )

        # delete masks for segments that were not seen in this frame
        for seg in self.segments:
            if not np.allclose(t, seg.last_seen, rtol=0.0):
                seg.last_observation.mask = None

        # handle moving existing segments to inactive
        to_rm = [
            seg
            for seg in self.segments
            if t - seg.last_seen > self.params.max_t_no_sightings or seg.num_points == 0
        ]
        for seg in to_rm:
            if seg.num_points == 0:
                self.segments.remove(seg)
                continue
            try:
                seg.final_cleanup()
            except Exception:  # too few points to form clusters
                self.segments.remove(seg)
                continue
            # final_cleanup can also succeed but null the points (statistical
            # outlier removal flags everything, or DBSCAN keeps no cluster).
            # Drop those instead of moving to inactive — anything kept must
            # have a usable point cloud since merge/IoU/voxel-grid lookups
            # below will crash on an empty segment.
            if seg.points is None or len(seg.points) == 0:
                self.segments.remove(seg)
                continue
            self.inactive_segments.append(seg)
            self.segments.remove(seg)

        # handle moving inactive segments to graveyard
        to_rm = [
            seg
            for seg in self.inactive_segments
            if t - seg.last_seen > self.params.segment_graveyard_time
            or np.linalg.norm(seg.last_observation.pose[:3, 3] - pose[:3, 3])
            > self.params.segment_graveyard_dist
        ]
        for seg in to_rm:
            self.segment_graveyard.append(seg)
            self.inactive_segments.remove(seg)

        to_rm = [
            seg
            for seg in self.segment_nursery
            if t - seg.last_seen > self.params.max_t_no_sightings or seg.num_points == 0
        ]
        for seg in to_rm:
            self.segment_nursery.remove(seg)

        # handle moving segments from nursery to normal segments
        to_upgrade = [
            seg
            for seg in self.segment_nursery
            if seg.num_sightings >= self.params.min_sightings
        ]
        for seg in to_upgrade:
            self.segment_nursery.remove(seg)
            self.segments.append(seg)

        # add new segments
        associated_obs = [obs_idx for _, obs_idx in associated_pairs]
        new_observations = [
            obs for idx, obs in enumerate(observations) if idx not in associated_obs
        ]
        for obs in new_observations:
            new_seg = MapSegment(
                obs,
                self.camera_params,
                self.id_counter,
                self.params.get_map_segment_params(),
            )
            if (
                new_seg.num_points == 0
            ):  # guard from observations coming in with no points
                continue
            self.segment_nursery.append(new_seg)
            self.id_counter += 1

        self.merge()

        return

    @cached_property
    def similarity_function(self):
        """
        Get the similarity function based on the association method
        """

        geometric_methods = {
            "iou": self.iou_similarity,
            "iom": self.iom_similarity,
            "chamfer": self.chamfer_distance_similarity,
        }
        semantic_methods = {
            "cosine_similarity": self.cosine_similarity,
        }

        if self.params.semantic_association_method is None:
            return geometric_methods[self.params.geometric_association_method]
        else:
            return lambda segment, segment_or_observation: np.array(
                [
                    geometric_methods[self.params.geometric_association_method](
                        segment, segment_or_observation
                    ),
                    semantic_methods[self.params.semantic_association_method](
                        segment, segment_or_observation
                    ),
                ]
            )

    @cached_property
    def min_similarity(self):
        """
        Get the minimum similarity threshold for self.similarity_function required to associate two items
        """
        return self.similarity_range[0, :]

    @cached_property
    def similarity_range(self):
        """
        Get an (2, N) array of minimum, threshold, and maximum similarity scores for the similarity function
        """
        return (
            np.array(self.params.geometric_score_range).reshape(2, 1)
            if self.params.semantic_association_method is None
            else np.array(
                [self.params.geometric_score_range, self.params.semantic_score_range]
            ).T
        )

    def iou_similarity(
        self,
        segment: MapSegment,
        segment_or_observation: Union[MapSegment, Observation],
    ):
        return self.voxel_grid_similarity(segment, segment_or_observation)

    def iom_similarity(
        self,
        segment: MapSegment,
        segment_or_observation: Union[MapSegment, Observation],
    ):
        return self.voxel_grid_similarity(
            segment, segment_or_observation, iom_as_iou=True
        )

    def voxel_grid_similarity(
        self,
        segment: MapSegment,
        segment_or_observation: Union[MapSegment, Observation],
        iom_as_iou: bool = False,
    ):
        """
        Compute the similarity between the voxel grids of a segment and an observation/other segment. Always [0, 1].
        """
        voxel_size = self.params.iou_voxel_size
        segment_voxel_grid = segment.get_voxel_grid(voxel_size)
        segment_or_observation_voxel_grid = segment_or_observation.get_voxel_grid(
            voxel_size
        )
        return segment_voxel_grid.iou(
            segment_or_observation_voxel_grid, iom_as_iou=iom_as_iou
        )

    def chamfer_distance_similarity(
        self,
        segment: MapSegment,
        segment_or_observation: Union[MapSegment, Observation],
    ):
        """
        Compute the similarity between a segment and observation/other segment using their chamfer distance.
        """
        # larger distance is less similar
        return -ChamferDistance.chamfer_distance(
            segment.pcd, segment_or_observation.pcd
        )

    def cosine_similarity(
        self,
        segment: MapSegment,
        segment_or_observation: Union[MapSegment, Observation],
    ):
        """
        Compute the cosine similarity between the semantic descriptors of a segment and an observation/other segment.
        """
        if (
            segment.semantic_descriptor is None
            or segment_or_observation.semantic_descriptor is None
        ):
            return 1.0
        return np.dot(
            segment.semantic_descriptor, segment_or_observation.semantic_descriptor
        ) / (
            np.linalg.norm(segment.semantic_descriptor)
            * np.linalg.norm(segment_or_observation.semantic_descriptor)
        )

    def remove_bad_segments(
        self,
        segments: List[MapSegment],
        min_volume: float = 0.0,
        min_max_extent: float = 0.0,
    ):
        """
        Remove segments that have small volumes or have no points

        Args:
            segments (List[MapSegment]): List of segments
            min_volume (float, optional): Minimum allowable segment volume. Defaults to 0.0.
            min_max_extent (float, optional): Minimum allowable max extent. Defaults to 0.0.

        Returns:
            segments (List[MapSegment]): Filtered list of segments
        """
        to_delete = []
        for seg in segments:
            try:
                extent = np.sort(seg.extent)  # in ascending order
                if seg.num_points == 0:
                    to_delete.append(seg)
                elif seg.volume < min_volume:
                    to_delete.append(seg)
                elif extent[-1] < min_max_extent:
                    to_delete.append(seg)
            except:
                to_delete.append(seg)
        for seg in to_delete:
            segments.remove(seg)
        return segments

    def merge(self):
        """
        Merge segments with high overlap
        """

        max_iter = 100
        n = 0
        edited = True

        self.inactive_segments = self.remove_bad_segments(
            self.inactive_segments,
            min_max_extent=self.params.min_max_extent,
        )
        self.segments = self.remove_bad_segments(self.segments)

        # repeatedly try to merge until no further merges are possible
        while n < max_iter and edited:
            edited = False
            n += 1

            for i, seg1 in enumerate(self.segments):
                for j, seg2 in enumerate(self.segments + self.inactive_segments):
                    if i >= j:
                        continue

                    # if segments are very far away, don't worry about doing extra checking
                    c1 = seg1.centroid
                    c2 = seg2.centroid
                    if c1 is None or c2 is None:
                        continue
                    if np.linalg.norm(c1 - c2) > 0.5 * (
                        np.max(seg1.extent) + np.max(seg2.extent)
                    ):
                        continue

                    merge_flag = False

                    # 2D IOU check
                    if self.params.min_2d_iou is not None:
                        mask1 = seg1.reconstruct_mask(self.last_pose)
                        mask2 = seg2.reconstruct_mask(self.last_pose)
                        intersection2d = np.logical_and(mask1, mask2).sum()
                        union2d = np.logical_or(mask1, mask2).sum()
                        iou2d = intersection2d / union2d if union2d > 0 else 0.0

                        merge_flag |= iou2d >= self.params.min_2d_iou

                    # Similarity check
                    merge_flag |= np.all(
                        self.similarity_function(seg1, seg2) >= self.min_similarity
                    )

                    if merge_flag:
                        seg1.update_from_segment(seg2)
                        seg1.id = min(seg1.id, seg2.id)
                        if seg1.num_points == 0:
                            self.segments.pop(i)
                        elif j < len(self.segments):
                            self.segments.pop(j)
                        else:
                            self.inactive_segments.pop(j - len(self.segments))
                        edited = True
                        break
                if edited:
                    break
        return

    def make_pickle_compatible(self):
        """
        Make the SegmentMapper object pickle compatible
        """
        for seg in (
            self.segments
            + self.segment_nursery
            + self.inactive_segments
            + self.segment_graveyard
        ):
            seg.reset_memoized()
        return

    def get_segment_map(self) -> List[MapSegment]:
        """
        Get the segment map
        """
        # Segments still active or in nursery never hit the active->inactive
        # transition that triggers final_cleanup. Run it now so they ship out
        # outlier-removed and DBSCAN-pruned like the rest.
        for seg in list(self.segments) + list(self.segment_nursery):
            if seg.points is not None and len(seg.points) > 0:
                try:
                    seg.final_cleanup()
                except Exception as e:
                    logger.debug(
                        f"end-of-run final_cleanup failed for seg {seg.id}: {e}"
                    )
        segment_map = self.remove_bad_segments(
            self.segment_graveyard + self.inactive_segments + self.segments
        )
        for seg in segment_map:
            seg.reset_memoized()
        return segment_map

    # ------------------------------------------------------------------
    # Incremental 2D ground submap creation
    # ------------------------------------------------------------------

    def process_submaps_2d(self, t: float, pose: np.ndarray):
        """Check for new/updated segments and create 2D ground submaps when triggered.

        Creates a new submap when the number of segments not yet included in any
        submap exceeds ``params.sm2d_num_new_segments``.
        """
        if self._ground_submap_mapping is None:
            return

        all_segs = self.segments + self.inactive_segments + self.segment_graveyard
        current_ids = {seg.id for seg in all_segs}

        # Track new segments and update last-seen times
        for seg in all_segs:
            self._seg_last_updated[seg.id] = seg.last_seen
            if seg.id not in self._known_seg_ids:
                self._known_seg_ids.add(seg.id)
                self._seg_ids_not_in_sm.append(seg.id)

        # Remove IDs that no longer exist
        self._seg_ids_not_in_sm = [
            sid for sid in self._seg_ids_not_in_sm if sid in current_ids
        ]
        self._seg_last_updated = {
            sid: t_last
            for sid, t_last in self._seg_last_updated.items()
            if sid in current_ids
        }

        # Check trigger
        if len(self._seg_ids_not_in_sm) < self.params.sm2d_num_new_segments:
            return

        # Select segments for submap
        new_ids = set(self._seg_ids_not_in_sm)
        remaining_slots = self.params.sm2d_num_segments - len(new_ids)

        # Fill overlap slots with most recently updated non-new segments
        candidates = sorted(
            [
                (sid, t_last)
                for sid, t_last in self._seg_last_updated.items()
                if sid not in new_ids
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        fill_ids = {sid for sid, _ in candidates[: max(0, remaining_slots)]}
        selected_ids = new_ids | fill_ids

        seg_by_id = {seg.id: seg for seg in all_segs}
        selected_segs = [seg_by_id[sid] for sid in selected_ids if sid in seg_by_id]

        if not selected_segs:
            return

        # Apply final_cleanup (statistical outlier removal + DBSCAN largest-cluster
        # pruning) to active segments before they go into the submap. Inactive /
        # graveyard segments already had it applied at transition time.
        active_ids = {seg.id for seg in self.segments}
        for seg in selected_segs:
            if seg.id in active_ids and seg.points is not None and len(seg.points) > 0:
                try:
                    seg.final_cleanup()
                except Exception as e:
                    logger.debug(f"final_cleanup failed for active seg {seg.id}: {e}")
        # Cleanup may zero out points for some segments; drop those from the
        # submap AND from self.segments so the next frame's global_nearest_neighbor
        # doesn't try to compute IoU on a zero-point segment (which raises in
        # MapSegment.get_voxel_grid).
        emptied_ids = {
            seg.id
            for seg in selected_segs
            if seg.id in active_ids and (seg.points is None or len(seg.points) == 0)
        }
        if emptied_ids:
            self.segments = [s for s in self.segments if s.id not in emptied_ids]
        selected_segs = [
            seg
            for seg in selected_segs
            if seg.points is not None and len(seg.points) > 0
        ]
        if not selected_segs:
            return

        # Compute submap time from segment reference times
        ref_times = [
            (seg.first_seen + seg.last_seen) / 2
            for seg in selected_segs
            if seg.first_seen is not None and seg.last_seen is not None
        ]
        if ref_times:
            submap_time = float(np.mean(ref_times))
        else:
            submap_time = t
        # Enforce monotonicity
        if submap_time < self._last_submap_time:
            submap_time = self._last_submap_time

        # Time gate: skip submaps whose center is within `sm2d_consec_min_time_s`
        # of the previous submap's center. `_last_submap_time` is -inf for the
        # first submap, so this no-ops on first creation.
        if np.isfinite(self._last_submap_time):
            dt_since_last = submap_time - self._last_submap_time
            if dt_since_last < self.params.sm2d_consec_min_time_s:
                logger.info(
                    f"Skipping 2D submap at t={submap_time:.2f}: time since last "
                    f"{dt_since_last:.2f}s < {self.params.sm2d_consec_min_time_s}s"
                )
                self._seg_ids_not_in_sm = []
                return

        # Path-length gate: skip submaps whose center is too close (along the
        # trajectory) to the previous submap's center. `_last_submap_time` is
        # -inf for the first submap, so this no-ops on first creation.
        if np.isfinite(self._last_submap_time):
            times_arr_full = np.array(self.times_history)
            mask = (times_arr_full >= self._last_submap_time) & (
                times_arr_full <= submap_time
            )
            sel = np.where(mask)[0]
            if len(sel) >= 2:
                positions = np.array([self.poses_cam_history[i][:3, 3] for i in sel])
                path_len = float(
                    np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                )
            else:
                path_len = 0.0
            if path_len < self.params.sm2d_consec_min_path_len:
                logger.info(
                    f"Skipping 2D submap at t={submap_time:.2f}: path length "
                    f"{path_len:.2f}m < {self.params.sm2d_consec_min_path_len}m"
                )
                self._seg_ids_not_in_sm = []
                return

        # Find nearest pose to submap time
        times_arr = np.array(self.times_history)
        nearest_idx = int(np.argmin(np.abs(times_arr - submap_time)))
        submap_pose = self.poses_cam_history[nearest_idx]
        center = submap_pose[:3, 3]

        # Apply same radius cap as the post-processed ground submap path
        # (ground_submap_primitive_mapping.py:131-167) so long road segments
        # don't trail far past the current submap center.
        gs_params = (
            self._ground_submap_mapping.submap_params
            if self._ground_submap_mapping is not None
            else None
        )
        rad_m = gs_params.ground_submap_rad_m if gs_params is not None else None

        # Convert MapSegments to DenseSegments
        dense_segments = []
        for seg in selected_segs:
            if seg.points is None or len(seg.points) == 0:
                continue
            pts = seg.points
            occ = seg.occluded_points

            if rad_m is not None:
                dists = np.linalg.norm(pts - center, axis=1)
                mask = dists <= rad_m
                if not np.any(mask):
                    continue

                if occ is not None and len(occ) > 0:
                    occ = occ[np.linalg.norm(occ - center, axis=1) <= rad_m]
                    if len(occ) > 0:
                        grid = gs_params.occluded_grid_voxel_size_m
                        dp_xy = np.floor(pts[mask][:, :2] / grid).astype(int)
                        regular_keys = set(map(tuple, dp_xy))
                        occ_xy = np.floor(occ[:, :2] / grid).astype(int)
                        occ_keep = np.array(
                            [tuple(k_) not in regular_keys for k_ in occ_xy]
                        )
                        occ = occ[occ_keep] if np.any(occ_keep) else None
                    else:
                        occ = None

                border_mask = mask & (
                    dists > rad_m - gs_params.occluded_radius_thresh_m
                )
                border_occluded = pts[border_mask].copy()

                all_occluded = [border_occluded]
                if occ is not None and len(occ) > 0:
                    all_occluded.append(occ)
                total = sum(len(a) for a in all_occluded)
                occ_final = np.concatenate(all_occluded, axis=0) if total > 0 else None
                pts = pts[mask].copy()
            else:
                occ_final = occ.copy() if occ is not None and len(occ) > 0 else None
                pts = pts.copy()

            ds = DenseSegment(
                id=seg.id,
                dense_points=pts,
                ratio_feature=get_roman_ratio_feature(seg),
                cos_feature=seg.semantic_descriptor,
                first_seen=seg.first_seen,
                last_seen=seg.last_seen,
                occluded_points=occ_final,
                history=getattr(seg, "history", []),
            )
            ds.point = np.mean(ds.dense_points, axis=0)
            ds.voxel_size = self.params.segment_voxel_size
            dense_segments.append(ds)

        if not dense_segments:
            return

        # Attach place recognition descriptor (semantic-gem / anyloc)
        submap_descriptor = None
        if self.place_recognition is not None and self.place_recognition.method in (
            "semantic-gem",
            "anyloc",
            "salad",
        ):
            submap_descriptor = self._compute_gem_descriptor_from_history(
                dense_segments, center=center, max_dist_m=rad_m
            )

        # Build 3D submap in CAMERA frame
        submap_3d = Submap(
            id=self._submap_counter,
            time=submap_time,
            segments=PrimitiveList(dense_segments),
            pose=submap_pose,
            segment_frame=FrameType.CAMERA,
            descriptor=submap_descriptor,
        )

        # Transform segments to submap-local (camera) frame
        T_submap_odom = np.linalg.inv(submap_pose)
        for seg in submap_3d.segments:
            seg.transform(T_submap_odom)

        # convert_submap_to_sparse_2d transforms segments back to odom before
        # flattening, so the flattened_submap consumed by the ground viz is in
        # odom frame. Anchor coordinate-frame axes at the camera's odom pose
        # (matches the metadata stamp at ground_submap_primitive_mapping.py:291).
        submap_3d.metadata = {"camera_pose": submap_pose}

        # Convert to sparse 2D
        submap_2d, intermediate = (
            self._ground_submap_mapping.convert_submap_to_sparse_2d(
                submap_3d, return_intermediates=True
            )
        )

        # Recompute descriptor for semantic-point-line
        if (
            self.place_recognition is not None
            and self.place_recognition.method == "semantic-point-line"
        ):
            submap_2d = Submap(
                id=submap_2d.id,
                time=submap_2d.time,
                segments=submap_2d.segments,
                pose=submap_2d.pose,
                segment_frame=submap_2d.segment_frame,
                descriptor=self.place_recognition.ground_descriptor(
                    None, submap_segments=submap_2d.segments
                ),
                metadata=submap_2d.metadata,
            )

        self.submaps_2d.append(submap_2d)
        self._submap_intermediates.append(intermediate)
        self._last_submap_time = submap_time
        self._seg_ids_not_in_sm = []
        self._submap_counter += 1

        logger.info(
            f"Created 2D submap {self._submap_counter - 1} with "
            f"{len(submap_2d.segments)} primitives at t={submap_time:.2f}"
        )

    def _compute_gem_descriptor_from_history(
        self, submap_segments, center=None, max_dist_m=None
    ):
        """Compute semantic-gem descriptor from frame descriptor history.

        Replicates the logic of CrossViewPlaceRecognition._ground_descriptor_gem()
        but reads directly from the mapper's live history arrays. When
        ``center`` and ``max_dist_m`` are provided, frames captured from camera
        poses farther than ``max_dist_m`` from ``center`` are excluded so the
        descriptor reflects only the area inside the submap radius.
        """
        seg_first = [s.first_seen for s in submap_segments if s.first_seen is not None]
        seg_last = [s.last_seen for s in submap_segments if s.last_seen is not None]
        if not seg_first or not seg_last:
            return None

        start_time = min(
            s.last_seen for s in submap_segments if s.last_seen is not None
        )
        end_time = max(
            s.first_seen for s in submap_segments if s.first_seen is not None
        )
        if start_time > end_time:
            start_time = min(seg_first)
            end_time = max(seg_last)

        # Filter history to time range, skipping None descriptors
        stacked = []
        last_pos = None
        dist_thresh = (
            self.place_recognition.params.ground_descriptor_dist_m
            if self.place_recognition is not None
            else 5.0
        )
        for i, (t_i, desc_i) in enumerate(
            zip(self.times_history, self.frame_descriptors_history)
        ):
            if t_i < start_time or t_i > end_time:
                continue
            if desc_i is None:
                continue
            pos_i = self.poses_cam_history[i][:3, 3]
            if center is not None and max_dist_m is not None:
                if np.linalg.norm(pos_i - center) > max_dist_m:
                    continue
            if last_pos is None or np.linalg.norm(pos_i - last_pos) >= dist_thresh:
                stacked.append(desc_i)
                last_pos = pos_i

        if stacked:
            return np.vstack(stacked)
        return None
