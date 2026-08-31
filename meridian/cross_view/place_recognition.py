import logging

import numpy as np

from meridian.params.cross_view_params import (
    IMAGE_METHODS,
    CrossViewPlaceRecognitionParams,
)

logger = logging.getLogger(__name__)


class CrossViewPlaceRecognition:
    """Descriptor computation and similarity for cross-view place recognition.

    Supports five methods:
    - "dino-gem": DINO-GeM image-level descriptors
    - "anyloc": AnyLoc (DINOv2 + VLAD) image-level descriptors
    - "meridian-vpr": trained two-tower cross-view NetVLAD (third_party/vpr)
    - "salad": SALAD (DINOv2 + optimal transport) image-level descriptors
    - "semantic-point-line": segment-level cosine features (mean point + mean line)
    """

    def __init__(self, params: CrossViewPlaceRecognitionParams):
        self.params = params
        self.method = params.method

        # Frame cache for the image methods; filled by precompute_ground_map_data.
        self._map_times = None
        self._map_descriptors = None
        self._map_positions = None

    def precompute_ground_map_data(self, ground_map):
        """Cache frame times/descriptors/positions for `ground_descriptor`.

        Call once before the submap loop in ground_map_to_submaps.

        Args:
            ground_map: SegmentMap with .descriptors, .times, .trajectory
        """
        self._map_times = self._map_descriptors = self._map_positions = None
        if ground_map.descriptors is None:
            return
        valid_mask = np.array([d is not None for d in ground_map.descriptors])
        if not valid_mask.any():
            return
        self._map_times = np.array(ground_map.times)[valid_mask]
        self._map_descriptors = np.vstack(
            [d for d in ground_map.descriptors if d is not None]
        )
        self._map_positions = np.array(
            [p[:3, 3] for p in ground_map.trajectory]
        )[valid_mask]

    def aerial_descriptor(
        self, aerial_submap, aerial_segmenter=None, img_bgr=None, crop=None
    ):
        """Compute a descriptor for an aerial submap.

        Args:
            aerial_submap: Submap with .segments (semantic-point-line)
            aerial_segmenter: AerialSegmenter (required for the image methods)
            img_bgr: BGR image (required for the image methods)
            crop: (x1, y1, x2, y2) pixel crop (required for the image methods)

        Returns:
            np.ndarray descriptor, or None
        """
        if self.method in IMAGE_METHODS:
            return aerial_segmenter.get_crop_descriptor(img_bgr, crop=crop)
        elif self.method == "semantic-point-line":
            return self._segment_cos_descriptor(aerial_submap.segments)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def ground_descriptor(
        self,
        ground_submap,
        submap_segments=None,
        center=None,
        max_dist_m=None,
    ):
        """Compute a descriptor for a ground submap.

        Args:
            ground_submap: Submap, or None if submap_segments is provided
            submap_segments: segments for this submap. Image methods take
                DenseSegments (time-window extraction); semantic-point-line a
                PrimitiveList.
            center: optional 3D submap center (odom frame)
            max_dist_m: with ``center``, drops frames captured farther than
                this from it

        Returns:
            np.ndarray descriptor, or None
        """
        if self.method in IMAGE_METHODS:
            return self._stacked_frame_descriptors(
                submap_segments, center=center, max_dist_m=max_dist_m
            )
        elif self.method == "semantic-point-line":
            segments = (
                submap_segments
                if submap_segments is not None
                else ground_submap.segments
            )
            return self._segment_cos_descriptor(segments)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _stacked_frame_descriptors(self, submap_segments, center=None,
                                   max_dist_m=None):
        """Stack cached frame descriptors for a ground submap. Works for every
        image method -- frames hold whatever backend the segmenter ran.

        Args:
            submap_segments: segments with .first_seen/.last_seen
            center: optional 3D submap center (odom frame)
            max_dist_m: with ``center``, drops frames captured farther than
                this from it

        Returns:
            np.ndarray of shape (N, D), frames spaced
            >= ground_descriptor_dist_m apart, or None
        """
        if self._map_descriptors is None or not submap_segments:
            return None

        seg_first = [s.first_seen for s in submap_segments if s.first_seen is not None]
        seg_last = [s.last_seen for s in submap_segments if s.last_seen is not None]
        if not seg_first or not seg_last:
            return None

        # Window where every segment is in view; full span if that is empty.
        start_time, end_time = min(seg_last), max(seg_first)
        if start_time > end_time:
            start_time, end_time = min(seg_first), max(seg_last)

        time_mask = (self._map_times >= start_time) & (self._map_times <= end_time)
        if not np.any(time_mask):
            return None

        frame_descs = self._map_descriptors[time_mask]
        frame_pos = self._map_positions[time_mask]

        if center is not None and max_dist_m is not None:
            radius_mask = (
                np.linalg.norm(frame_pos - np.asarray(center), axis=1) <= max_dist_m
            )
            if not np.any(radius_mask):
                return None
            frame_descs = frame_descs[radius_mask]
            frame_pos = frame_pos[radius_mask]

        stacked = []
        last_pos = None
        for fd, fp in zip(frame_descs, frame_pos):
            if (
                last_pos is None
                or np.linalg.norm(fp - last_pos)
                >= self.params.ground_descriptor_dist_m
            ):
                stacked.append(fd)
                last_pos = fp

        return np.vstack(stacked) if stacked else None

    def _segment_cos_descriptor(self, segments):
        """Mean-point + mean-line cosine features, L2-normalized.

        Args:
            segments: PrimitiveList

        Returns:
            np.ndarray of shape (2 * cos_feature_dim,), or None
        """
        point_features = [
            seg.cos_feature
            for seg in segments.get_points()
            if seg.cos_feature is not None
        ]
        line_features = [
            seg.cos_feature
            for seg in segments.get_lines()
            if seg.cos_feature is not None
        ]
        if not point_features and not line_features:
            return None

        dim = (point_features or line_features)[0].shape[0]
        v1 = np.mean(point_features, axis=0) if point_features else np.zeros(dim)
        v2 = np.mean(line_features, axis=0) if line_features else np.zeros(dim)
        descriptor = np.concatenate([v1, v2])
        norm = np.linalg.norm(descriptor)
        if norm < 1e-12:
            return None
        return descriptor / norm

    def similarity(self, ground_desc, aerial_desc):
        """Similarity between a ground and an aerial descriptor.

        Image methods: max cosine over the ground frame stack (N, D) vs the
        aerial (D,). semantic-point-line: plain cosine.

        Args:
            ground_desc: ground descriptor array
            aerial_desc: aerial descriptor array

        Returns:
            float score, nan if either descriptor is None
        """
        if ground_desc is None or aerial_desc is None:
            return np.nan

        if self.method in IMAGE_METHODS:
            g = ground_desc
            if g.ndim == 1:
                g = g.reshape(1, -1)
            g_norms = np.linalg.norm(g, axis=1, keepdims=True)
            g_norms = np.where(g_norms < 1e-12, 1.0, g_norms)
            g_norm = g / g_norms

            a_norm_val = np.linalg.norm(aerial_desc)
            if a_norm_val < 1e-12:
                return np.nan
            a_norm = aerial_desc / a_norm_val

            dots = g_norm @ a_norm
            return float(np.max(dots))
        elif self.method == "semantic-point-line":
            g_norm_val = np.linalg.norm(ground_desc)
            a_norm_val = np.linalg.norm(aerial_desc)
            if g_norm_val < 1e-12 or a_norm_val < 1e-12:
                return np.nan
            return float(np.dot(ground_desc / g_norm_val, aerial_desc / a_norm_val))
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def compute_similarity_matrix(self, ground_submaps, aerial_submaps):
        """Pairwise `similarity` over precomputed submap.descriptor values.

        Args:
            ground_submaps: dict of str -> Submap
            aerial_submaps: dict of str -> Submap

        Returns:
            sim_matrix: (num_ground, num_aerial) ndarray
            ground_keys: sorted ground submap keys
            aerial_keys: sorted aerial submap keys
        """
        ground_keys = sorted(ground_submaps.keys(), key=lambda k: int(k))
        aerial_keys = sorted(aerial_submaps.keys())

        sim_matrix = np.full((len(ground_keys), len(aerial_keys)), np.nan)

        for gi, gk in enumerate(ground_keys):
            g_desc = ground_submaps[gk].descriptor
            for ai, ak in enumerate(aerial_keys):
                a_desc = aerial_submaps[ak].descriptor
                sim_matrix[gi, ai] = self.similarity(g_desc, a_desc)

        return sim_matrix, ground_keys, aerial_keys
