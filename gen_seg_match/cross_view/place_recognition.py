import logging

import numpy as np

from gen_seg_match.params.cross_view_params import CrossViewPlaceRecognitionParams

logger = logging.getLogger(__name__)


class CrossViewPlaceRecognition:
    """Descriptor computation and similarity for cross-view place recognition.

    Supports four methods:
    - "semantic-gem": DINO-GeM image-level descriptors
    - "anyloc": AnyLoc (DINOv2 + VLAD) image-level descriptors
    - "salad": SALAD (DINOv2 + optimal transport) image-level descriptors
    - "semantic-point-line": segment-level cosine features (mean point + mean line)
    """

    def __init__(self, params: CrossViewPlaceRecognitionParams):
        self.params = params
        self.method = params.method

        # Lazy cache for ground map data (semantic-gem)
        self._map_times = None
        self._map_descriptors = None
        self._map_positions = None

    def precompute_ground_map_data(self, ground_map):
        """Precompute and cache ground map arrays for semantic-gem ground descriptors.

        Call once before the submap loop in ground_map_to_submaps.

        Args:
            ground_map: SegmentMap with .descriptors, .times, .trajectory
        """
        if ground_map.descriptors is not None:
            self._map_times = np.array(ground_map.times)
            self._map_descriptors = np.vstack(ground_map.descriptors)
            self._map_positions = np.array([p[:3, 3] for p in ground_map.trajectory])
        else:
            self._map_times = None
            self._map_descriptors = None
            self._map_positions = None

    def aerial_descriptor(
        self, aerial_submap, aerial_segmenter=None, img_bgr=None, crop=None
    ):
        """Compute a descriptor for an aerial submap.

        Args:
            aerial_submap: Submap with .segments
            aerial_segmenter: AerialSegmenter (required for "semantic-gem")
            img_bgr: BGR image (required for "semantic-gem")
            crop: (x1, y1, x2, y2) pixel crop (required for "semantic-gem")

        Returns:
            np.ndarray descriptor, or None
        """
        if self.method in ("semantic-gem", "anyloc", "salad"):
            return aerial_segmenter.get_crop_descriptor(img_bgr, crop=crop)
        elif self.method == "semantic-point-line":
            return self._segment_cos_descriptor(aerial_submap.segments)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def ground_descriptor(self, ground_submap, submap_segments=None):
        """Compute a descriptor for a ground submap.

        Args:
            ground_submap: Submap, or None if submap_segments is provided
            submap_segments: segment list for this submap. For "semantic-gem",
                a list of DenseSegment (time range extraction). For
                "semantic-point-line", a SegmentList of SegmentPoint/SegmentLine.

        Returns:
            np.ndarray descriptor, or None
        """
        if self.method in ("semantic-gem", "anyloc", "salad"):
            return self._ground_descriptor_gem(submap_segments)
        elif self.method == "semantic-point-line":
            segments = (
                submap_segments
                if submap_segments is not None
                else ground_submap.segments
            )
            return self._segment_cos_descriptor(segments)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _ground_descriptor_gem(self, submap_segments):
        """Extract stacked frame descriptors for a ground submap (semantic-gem).

        Uses cached _map_times/_map_descriptors/_map_positions from
        precompute_ground_map_data().

        Args:
            submap_segments: list of segments with .first_seen/.last_seen

        Returns:
            np.ndarray of shape (N, D) or None
        """
        if self._map_descriptors is None:
            return None

        try:
            seg_first = [
                s.first_seen for s in submap_segments if s.first_seen is not None
            ]
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

            time_mask = (self._map_times >= start_time) & (self._map_times <= end_time)
            if not np.any(time_mask):
                return None

            frame_descs = self._map_descriptors[time_mask]
            frame_pos = self._map_positions[time_mask]

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

            if stacked:
                return np.vstack(stacked)
        except Exception:
            pass

        return None

    def _segment_cos_descriptor(self, segments):
        """Compute mean-point + mean-line cosine feature descriptor.

        Args:
            segments: SegmentList

        Returns:
            np.ndarray of shape (2*cos_feature_dim,) or None
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

        # Determine dimension from whichever features we have
        if point_features:
            dim = point_features[0].shape[0]
        else:
            dim = line_features[0].shape[0]

        if point_features:
            v1 = np.mean(point_features, axis=0)
        else:
            v1 = np.zeros(dim)

        if line_features:
            v2 = np.mean(line_features, axis=0)
        else:
            v2 = np.zeros(dim)

        descriptor = np.concatenate([v1, v2])
        norm = np.linalg.norm(descriptor)
        if norm < 1e-12:
            return None
        return descriptor / norm

    def similarity(self, ground_desc, aerial_desc):
        """Compute similarity between a ground and aerial descriptor.

        Args:
            ground_desc: ground descriptor array
            aerial_desc: aerial descriptor array

        Returns:
            float similarity score
        """
        if ground_desc is None or aerial_desc is None:
            return np.nan

        if self.method in ("semantic-gem", "anyloc", "salad"):
            # ground is (N, D), aerial is (D,)
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
            # both are (D,)
            g_norm_val = np.linalg.norm(ground_desc)
            a_norm_val = np.linalg.norm(aerial_desc)
            if g_norm_val < 1e-12 or a_norm_val < 1e-12:
                return np.nan
            return float(np.dot(ground_desc / g_norm_val, aerial_desc / a_norm_val))
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def compute_similarity_matrix(self, ground_submaps, aerial_submaps):
        """Compute similarity matrix between ground and aerial submaps.

        Uses precomputed descriptors stored on submap.descriptor.

        Args:
            ground_submaps: dict of str -> Submap
            aerial_submaps: dict of str -> Submap

        Returns:
            sim_matrix: (num_ground, num_aerial) ndarray
            ground_keys: list of ground submap keys
            aerial_keys: list of aerial submap keys
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
