import logging

import numpy as np

from meridian.params.cross_view_params import CrossViewPlaceRecognitionParams

logger = logging.getLogger(__name__)


class CrossViewPlaceRecognition:
    """Frame-descriptor place recognition between ground and aerial submaps.

    A ground submap's descriptor is a stack of the segmenter's frame descriptors
    (N, D); an aerial submap's is the aerial segmenter's crop descriptor (D,).
    Similarity is the max cosine over the ground stack.

    Submaps are tagged with the model (frame_descriptor type) behind their
    descriptor; the type always comes from the data source (segmenter, ground
    map, or a submap's tag), never from this class.
    """

    def __init__(self, params: CrossViewPlaceRecognitionParams):
        self.params = params

        # Frame cache for ground_descriptor; filled by precompute_ground_map_data.
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
        self._map_positions = np.array([p[:3, 3] for p in ground_map.trajectory])[
            valid_mask
        ]

    def ground_descriptor(self, submap_segments, center=None, max_dist_m=None):
        """Stacked frame descriptors for a ground submap, from the cache filled by
        precompute_ground_map_data. Works for every model -- frames hold whatever
        backend the segmenter ran.

        Args:
            submap_segments: DenseSegments with .first_seen/.last_seen
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
                or np.linalg.norm(fp - last_pos) >= self.params.ground_descriptor_dist_m
            ):
                stacked.append(fd)
                last_pos = fp

        return np.vstack(stacked) if stacked else None

    def similarity(self, ground_desc, aerial_desc):
        """Similarity between a ground and an aerial descriptor: max cosine over
        the ground frame stack (N, D) vs the aerial (D,).

        Args:
            ground_desc: ground descriptor array
            aerial_desc: aerial descriptor array

        Returns:
            float score, nan if either descriptor is None
        """
        if ground_desc is None or aerial_desc is None:
            return np.nan

        g = np.asarray(ground_desc, dtype=np.float32)
        aerial_desc = np.asarray(aerial_desc, dtype=np.float32)
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

    def compute_similarity_matrix(self, ground_submaps, aerial_submaps):
        """Pairwise `similarity` over precomputed submap.descriptor values.

        Args:
            ground_submaps: dict of str -> Submap
            aerial_submaps: dict of str -> Submap

        Returns:
            sim_matrix: (num_ground, num_aerial) ndarray
            ground_keys: sorted ground submap keys
            aerial_keys: sorted aerial submap keys

        Raises:
            ValueError: either side's descriptors came from another model
        """
        self.check_descriptors_match(
            ground=self.get_submaps_tag(ground_submaps),
            aerial=self.get_submaps_tag(aerial_submaps),
        )

        ground_keys = sorted(ground_submaps.keys(), key=lambda k: int(k))
        aerial_keys = sorted(aerial_submaps.keys())

        sim_matrix = np.full((len(ground_keys), len(aerial_keys)), np.nan)

        for gi, gk in enumerate(ground_keys):
            g_desc = ground_submaps[gk].descriptor
            for ai, ak in enumerate(aerial_keys):
                a_desc = aerial_submaps[ak].descriptor
                sim_matrix[gi, ai] = self.similarity(g_desc, a_desc)

        return sim_matrix, ground_keys, aerial_keys

    @staticmethod
    def tag(submap, frame_descriptor):
        """Record which model made `submap.descriptor`, for the disk round trip."""
        if submap.descriptor is None or frame_descriptor is None:
            return
        if submap.metadata is None:
            submap.metadata = {}
        submap.metadata["descriptor_type"] = frame_descriptor

    @staticmethod
    def get_tag(submap):
        """The model `submap.descriptor` was tagged with, or None if untagged."""
        if submap.descriptor is None or not submap.metadata:
            return None
        return submap.metadata.get("descriptor_type")

    @classmethod
    def get_submaps_tag(cls, submaps):
        """First tagged model among `submaps` (dict of key -> Submap), or None."""
        for submap in submaps.values():
            tag = cls.get_tag(submap)
            if tag is not None:
                return tag
        return None

    @staticmethod
    def check_descriptors_match(ground, aerial):
        """Raise if the ground and aerial descriptor models are both known and differ.

        Pipelines call this whenever they have both sources (segmenter params,
        a ground map's descriptor_type, or prebuilt submaps' tags). None means
        unknown (e.g. untagged submaps) and skips the check.
        """
        if ground is None or aerial is None or ground == aerial:
            return
        raise ValueError(
            f"Ground descriptors come from {ground!r} but aerial from {aerial!r}; "
            "cross-view similarity between two different models is meaningless. "
            "Rebuild one side, or set segmenter / aerial_segmenter frame_descriptor "
            "to match."
        )
