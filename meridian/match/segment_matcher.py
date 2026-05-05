import logging
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import clipperpy
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors

from meridian.segment.segment_types import (
    SegmentPoint,
    GeneralSegment,
    SegmentList,
    SegmentLine,
    SegmentPlane,
)
from meridian.params.segment_match_params import SegmentMatchParams
from meridian.match.match_result import MatchResult

import time

logger = logging.getLogger(__name__)


class InsufficientAssociationsException(Exception):
    def __init__(self, map1_len, map2_len, n_associations=None):
        self.map1_len = map1_len
        self.map2_len = map2_len
        self.n_associations = n_associations
        message = f"Insufficient associations. Map 1 length: {map1_len}. Map 2 length: {map2_len}. Associations: {n_associations}"
        super().__init__(message)


class SegmentMatcher:
    def __init__(self, params: SegmentMatchParams):
        self.params = params
        self._langevin_matcher = None

    @property
    def langevin_matcher(self):
        if self._langevin_matcher is None and self.params.langevin_params is not None:
            from meridian.match.langevin_matcher import LangevinMatcher

            self._langevin_matcher = LangevinMatcher(self.params.langevin_params)
        return self._langevin_matcher

    def set_solver(self, solver: str):
        """Switch solver at runtime (e.g., 'clipper' for pass 2)."""
        self.params.solver = solver

    def match(
        self,
        map1: List[GeneralSegment],
        map2: List[GeneralSegment],
        global_x_dir1: np.ndarray = None,
        global_y_dir1: np.ndarray = None,
        global_z_dir1: np.ndarray = None,
        global_x_dir2: np.ndarray = None,
        global_y_dir2: np.ndarray = None,
        global_z_dir2: np.ndarray = None,
        bidirectional: bool = True,
        putative_match_matrix: np.ndarray = None,
    ) -> MatchResult:
        # Langevin multi-hypothesis path
        if self.params.solver == "langevin" and self.langevin_matcher is not None:
            return self._match_langevin(
                map1,
                map2,
                global_x_dir1=global_x_dir1,
                global_y_dir1=global_y_dir1,
                global_x_dir2=global_x_dir2,
                global_y_dir2=global_y_dir2,
            )

        # CLIPPER single-hypothesis path
        return self._match_clipper(
            map1,
            map2,
            global_x_dir1=global_x_dir1,
            global_y_dir1=global_y_dir1,
            global_z_dir1=global_z_dir1,
            global_x_dir2=global_x_dir2,
            global_y_dir2=global_y_dir2,
            global_z_dir2=global_z_dir2,
            bidirectional=bidirectional,
            putative_match_matrix=putative_match_matrix,
        )

    def _match_clipper(
        self,
        map1: List[GeneralSegment],
        map2: List[GeneralSegment],
        global_x_dir1: np.ndarray = None,
        global_y_dir1: np.ndarray = None,
        global_z_dir1: np.ndarray = None,
        global_x_dir2: np.ndarray = None,
        global_y_dir2: np.ndarray = None,
        global_z_dir2: np.ndarray = None,
        bidirectional: bool = True,
        putative_match_matrix: np.ndarray = None,
    ) -> MatchResult:
        """CLIPPER single-hypothesis matching."""
        map1 = SegmentList(deepcopy(map1))
        map2 = SegmentList(deepcopy(map2))
        map1 = map1.get_points() + map1.get_lines() + map1.get_planes()
        map2 = map2.get_points() + map2.get_lines() + map2.get_planes()

        empty_result = MatchResult([np.array([])], [0.0], [1])

        # return empty associations if map is empty
        if len(map1) == 0 or len(map2) == 0:
            return empty_result

        # transform into direction-aligned frame
        if self.params.z_dir_constrained:
            for map_i, z_dir_i in zip([map1, map2], [global_z_dir1, global_z_dir2]):
                assert z_dir_i is not None, (
                    "Must supply z direction if using z_dir_constrained"
                )
                T_world_aligned = self._construct_z_aligned_frame(z_dir_i)
                map_i.transform(np.linalg.inv(T_world_aligned))
        elif self.params.xyz_dir_constrained:
            for map_i, (x_dir_i, y_dir_i, z_dir_i) in zip(
                [map1, map2],
                [
                    (global_x_dir1, global_y_dir1, global_z_dir1),
                    (global_x_dir2, global_y_dir2, global_z_dir2),
                ],
            ):
                assert (
                    x_dir_i is not None and y_dir_i is not None and z_dir_i is not None
                ), "Must supply x, y, and z directions if using xyz_dir_constrained"
                T_world_aligned = self._construct_xyz_aligned_frame(
                    x_dir_i, y_dir_i, z_dir_i
                )
                map_i.transform(np.linalg.inv(T_world_aligned))
        elif self.params.xy_dir_constrained_2d:
            for map_i, (x_dir_i, y_dir_i) in zip(
                [map1, map2],
                [
                    (global_x_dir1, global_y_dir1),
                    (global_x_dir2, global_y_dir2),
                ],
            ):
                assert x_dir_i is not None and y_dir_i is not None, (
                    "Must supply x and y directions if using xy_dir_constrained_2d"
                )
                T_world_aligned = self._construct_xy_aligned_frame_2d(x_dir_i, y_dir_i)
                map_i.transform(np.linalg.inv(T_world_aligned))

        clipper = self._setup_solver(bidirectional=bidirectional)

        if putative_match_matrix is None:
            clipper, A_init = self._setup_problem(clipper, map1, map2)
            if len(A_init) == 0:
                return empty_result
        else:
            map1_lists = [self._get_seg_array(obj) for obj in map1]
            map2_lists = [self._get_seg_array(obj) for obj in map2]
            map1_cl, map2_cl = self._create_padded_map_arrays(map1_lists, map2_lists)
            clipper.score_pairwise_and_single_consistency(
                map1_cl.T, map2_cl.T, putative_match_matrix.astype(np.int32)
            )

        clipper.solve()
        Ain = clipper.get_selected_associations()

        if putative_match_matrix is None:
            Ain_by_ids = self._assoc_idx_to_ids(Ain, map1, map2)
        else:
            Ain_by_ids = np.array(
                [[map1[pair[0]].id, map2[pair[1]].id] for pair in Ain]
            )

        for Ain_pair in Ain_by_ids:
            assert type(map1.get_segment_from_id(Ain_pair[0])) == type(
                map2.get_segment_from_id(Ain_pair[1])
            ), (
                "Corresponded segments must be of the same type. "
                + f"Got match between {type(map1.get_segment_from_id(Ain_pair[0]))} and {type(map2.get_segment_from_id(Ain_pair[1]))}."
            )

        # Compute CLIPPER objective score
        u_sol = clipper.get_solution().u
        M = clipper.get_affinity_matrix()
        solution_nodes = clipper.get_solution().nodes
        if len(solution_nodes) == 0:
            score = 0.0
        else:
            score = float(u_sol.T @ M @ u_sol / (u_sol.T @ u_sol))

        return MatchResult([Ain_by_ids], [score], [1])

    def _match_langevin(
        self,
        map1: List[GeneralSegment],
        map2: List[GeneralSegment],
        global_x_dir1: np.ndarray = None,
        global_y_dir1: np.ndarray = None,
        global_x_dir2: np.ndarray = None,
        global_y_dir2: np.ndarray = None,
    ) -> MatchResult:
        """Langevin multi-hypothesis matching."""
        dir_kwargs = {}
        if global_x_dir1 is not None:
            dir_kwargs["global_x_dir1"] = global_x_dir1
        if global_y_dir1 is not None:
            dir_kwargs["global_y_dir1"] = global_y_dir1
        if global_x_dir2 is not None:
            dir_kwargs["global_x_dir2"] = global_x_dir2
        if global_y_dir2 is not None:
            dir_kwargs["global_y_dir2"] = global_y_dir2

        start_time = time.time()
        M, C, A, map1_ordered, map2_ordered = self.get_MCA_with_maps(
            map1, map2, **dir_kwargs
        )
        end_time = time.time()
        logger.debug(
            f"get_MCA_with_maps took {end_time - start_time:.3f} seconds for maps of size {len(map1)} and {len(map2)}"
        )

        empty_result = MatchResult([np.array([])], [0.0], [1])
        if M.shape[0] == 0:
            return empty_result

        raw_results = self.langevin_matcher.match(M, C, A)
        if not raw_results:
            return empty_result

        # Convert index pairs to segment IDs
        results = []
        for assoc_idx, score, count in raw_results:
            ids = self.assoc_idx_to_ids(assoc_idx, map1_ordered, map2_ordered)
            results.append((ids, score, count))

        # Optional Jaccard pruning
        if self.params.langevin_params.jaccard_pruning:
            n_before = len(results)
            results = self._cluster_by_jaccard(
                results, self.params.langevin_params.jaccard_thresh
            )
            logger.debug(f"Jaccard pruning: {n_before} -> {len(results)} hypotheses")

        return MatchResult(
            association_arrays=[r[0] for r in results],
            scores=[r[1] for r in results],
            counts=[r[2] for r in results],
        )

    @staticmethod
    def _cluster_by_jaccard(
        hypotheses: List[Tuple[np.ndarray, float, int]],
        jaccard_thresh: float = 0.5,
    ) -> List[Tuple[np.ndarray, float, int]]:
        """Pre-cluster hypotheses by Jaccard similarity of association sets.

        Greedy clustering: iterate hypotheses (already sorted by objective
        descending), assign each to the first cluster within the Jaccard
        threshold, or start a new cluster. The representative is the
        highest-objective member; particle counts are summed.
        """
        if not hypotheses:
            return []

        assoc_sets = []
        for matches, score, count in hypotheses:
            assoc_sets.append(frozenset(map(tuple, matches.tolist())))

        clusters = []
        for i, (matches, score, count) in enumerate(hypotheses):
            s = assoc_sets[i]
            assigned = False
            for cluster in clusters:
                rep_set = cluster[1]
                intersection = len(s & rep_set)
                union = len(s | rep_set)
                if union > 0 and intersection / union >= jaccard_thresh:
                    cluster[2] += count
                    assigned = True
                    break
            if not assigned:
                clusters.append([i, s, count])

        return [
            (hypotheses[idx][0], hypotheses[idx][1], total_count)
            for idx, _, total_count in clusters
        ]

    def get_MCA(self, map1: List[GeneralSegment], map2: List[GeneralSegment]):
        M, C, A_init, _, _ = self.get_MCA_with_maps(map1, map2)
        return M, C, A_init

    def get_MCA_with_maps(
        self,
        map1: List[GeneralSegment],
        map2: List[GeneralSegment],
        global_x_dir1: np.ndarray = None,
        global_y_dir1: np.ndarray = None,
        global_x_dir2: np.ndarray = None,
        global_y_dir2: np.ndarray = None,
    ):
        """Return affinity matrix, constraint matrix, putative associations, and ordered maps.

        The returned map1/map2 SegmentLists are in type-ordered form (points + lines + planes)
        matching the index space of M, C, and A_init. These are needed for
        assoc_idx_to_ids() to convert association indices back to segment IDs.
        """
        map1 = SegmentList(deepcopy(map1))
        map2 = SegmentList(deepcopy(map2))
        map1 = map1.get_points() + map1.get_lines() + map1.get_planes()
        map2 = map2.get_points() + map2.get_lines() + map2.get_planes()

        # Return empty matrices if either map is empty
        if len(map1) == 0 or len(map2) == 0:
            empty = np.zeros((0, 0))
            return empty, empty, np.zeros((0, 2), dtype=np.int32), map1, map2

        # Apply direction-aligned frame transform (same as match())
        if self.params.xy_dir_constrained_2d:
            for map_i, (x_dir_i, y_dir_i) in zip(
                [map1, map2],
                [
                    (global_x_dir1, global_y_dir1),
                    (global_x_dir2, global_y_dir2),
                ],
            ):
                assert x_dir_i is not None and y_dir_i is not None, (
                    "Must supply x and y directions if using xy_dir_constrained_2d"
                )
                T_world_aligned = self._construct_xy_aligned_frame_2d(x_dir_i, y_dir_i)
                map_i.transform(np.linalg.inv(T_world_aligned))

        clipper = self._setup_solver()
        clipper, A_init = self._setup_problem(clipper, map1, map2)
        M = clipper.get_affinity_matrix()
        C = clipper.get_constraint_matrix()
        return M, C, A_init, map1, map2

    def match_multiple(
        self, map1: List[GeneralSegment], map2: List[GeneralSegment], num_solutions=2
    ):
        map1 = SegmentList(map1)
        map2 = SegmentList(map2)
        M, C, A = self.get_MCA(map1, map2)
        M_orig = M.copy()
        clipper = clipperpy.CLIPPER(
            clipperpy.invariants.PairwiseInvariant(), clipperpy.Params()
        )
        solutions = []

        for k in range(num_solutions):
            clipper.set_matrix_data(M=M, C=C)
            clipper.solve()

            solution_nodes = clipper.get_solution().nodes
            Ain = np.zeros((len(solution_nodes), 2)).astype(np.int64)
            for i in range(len(solution_nodes)):
                Ain[i, :] = A[solution_nodes[i], :]

            u_sol = clipper.get_solution().u.copy()
            for i in range(u_sol.shape[0]):
                u_sol[i] = u_sol[i] if i in solution_nodes else 0.0
            if len(solution_nodes) == 0:
                score = 0
            else:
                score = u_sol.T @ M_orig @ u_sol / (u_sol.T @ u_sol)
            Ain_by_ids = self._assoc_idx_to_ids(Ain, map1, map2)
            solutions.append((Ain_by_ids, score))

            if k + 1 < num_solutions:
                row_indices, col_indices = np.meshgrid(
                    solution_nodes, solution_nodes, indexing="ij"
                )
                if len(row_indices) != 0 and len(col_indices) != 0:
                    M[row_indices, col_indices] = 0.0

        return solutions

    def register(
        self,
        map1: List[GeneralSegment],
        map2: List[GeneralSegment],
        global_x_dir1: np.ndarray = None,
        global_y_dir1: np.ndarray = None,
        global_z_dir1: np.ndarray = None,
        global_x_dir2: np.ndarray = None,
        global_y_dir2: np.ndarray = None,
        global_z_dir2: np.ndarray = None,
        correspondences: np.array = None,
    ):
        """
        Computes the transformation that aligns map2 to map1 (T^map1_map2). Currently only uses SegmentPoint correspondences.

        Args:
            map1 (List[GeneralSegment]): Segment list in frame 1
            map2 (List[GeneralSegment]): Segment list in frame 2
            correspondences (np.array, shape=(n,2), optional): If correspondences have already
                been found, set to None. Otherwise, performs register before aligning. Defaults to None.

        Returns:
            np.array: Transformation matrix that aligns map2 to map1 (T^map1_map2).
        """
        if len(map1) == 0 or len(map2) == 0:
            raise InsufficientAssociationsException(len(map1), len(map2))

        if correspondences is None:
            correspondences = self.match(
                map1,
                map2,
                global_x_dir1=global_x_dir1,
                global_y_dir1=global_y_dir1,
                global_z_dir1=global_z_dir1,
                global_x_dir2=global_x_dir2,
                global_y_dir2=global_y_dir2,
                global_z_dir2=global_z_dir2,
            ).association_array
        if len(correspondences) == 0:
            raise InsufficientAssociationsException(len(map1), len(map2))

        map1 = SegmentList(map1)
        map2 = SegmentList(map2)

        filtered_correspondences = [
            corr
            for corr in correspondences
            if type(map1.get_segment_from_id(corr[0])) is SegmentPoint
            and type(map2.get_segment_from_id(corr[1])) is SegmentPoint
        ]

        all_pairs = [
            (map1.get_segment_from_id(corr[0]), map2.get_segment_from_id(corr[1]))
            for corr in correspondences
        ]
        correspondences = np.array([[corr[0].id, corr[1].id] for corr in all_pairs])

        if len(filtered_correspondences) < self.params.dim:
            raise InsufficientAssociationsException(
                len(map1), len(map2), len(filtered_correspondences)
            )

        pts1 = np.array(
            [
                map1.get_segment_from_id(corr[0]).get_point()[: self.params.dim]
                for corr in filtered_correspondences
            ]
        )
        pts2 = np.array(
            [
                map2.get_segment_from_id(corr[1]).get_point()[: self.params.dim]
                for corr in filtered_correspondences
            ]
        )

        mean1 = np.mean(pts1, axis=0)
        mean2 = np.mean(pts2, axis=0)

        pts1_mean_reduced = pts1 - mean1
        pts2_mean_reduced = pts2 - mean2
        assert pts1_mean_reduced.shape == pts2_mean_reduced.shape

        H = pts1_mean_reduced.T @ (pts2_mean_reduced)
        U, s, Vh = np.linalg.svd(H)
        R = U @ Vh

        if np.allclose(np.linalg.det(R), -1.0):
            Vh_prime = Vh.copy()
            Vh_prime[-1, :] *= -1.0
            R = U @ Vh_prime

        t = mean1.reshape((-1, 1)) - R @ mean2.reshape((-1, 1))
        T = np.concatenate(
            [
                np.concatenate([R, t], axis=1),
                np.hstack([np.zeros((1, R.shape[0])), [[1]]]),
            ],
            axis=0,
        )
        return T

    def view_registration(
        self,
        map1: List[GeneralSegment],
        map2: List[GeneralSegment],
        correspondences: np.array,
        T: np.array,
        ax=None,
        **kwargs,
    ):
        """
        Visualize the registration between map1 and map2 (currently only supports SegmentPoint)

        Args:
            map1 (List[GeneralSegment]): Segment list in frame 1
            map2 (List[GeneralSegment]): Segment list in frame 2
            correspondences (np.array, shape=(n,2)): Correspondences between map1 and map2
            T (np.array): Transformation matrix that aligns map2 to map1
        """
        if ax is None:
            _, ax = plt.subplots()

        map1 = SegmentList([seg for seg in map1 if type(seg) is SegmentPoint])
        map2 = SegmentList([seg.copy() for seg in map2 if type(seg) is SegmentPoint])

        map2.transform(T)

        for seg in map1:
            if type(seg) is SegmentPoint:
                ax.plot(
                    seg.get_point()[0],
                    seg.get_point()[1],
                    "o",
                    color="maroon",
                    **kwargs,
                )

        for seg in map2:
            if type(seg) is SegmentPoint:
                ax.plot(
                    seg.get_point()[0], seg.get_point()[1], "o", color="blue", **kwargs
                )

        for corr in correspondences:
            if not map1.has_id(corr[0]) or not map2.has_id(corr[1]):
                continue
            ax.plot(
                [
                    map1.get_segment_from_id(corr[0]).get_point()[0],
                    map2.get_segment_from_id(corr[1]).get_point()[0],
                ],
                [
                    map1.get_segment_from_id(corr[0]).get_point()[1],
                    map2.get_segment_from_id(corr[1]).get_point()[1],
                ],
                color="lawngreen",
                linestyle="dotted",
            )

        ax.set_aspect("equal")
        return ax

    def _setup_solver(self, bidirectional=True):
        iparams = self.params.to_clipper()
        iparams.bidirectional = bidirectional
        invariant = clipperpy.invariants.GeneralSegmentDistance(iparams)
        params = clipperpy.Params()
        clipper = clipperpy.CLIPPERPairwiseAndSingle(invariant, params)
        return clipper

    def _setup_problem(
        self,
        clipper,
        map1: SegmentList,
        map2: SegmentList,
    ):
        points1 = map1.get_points()
        lines1 = map1.get_lines()
        planes1 = map1.get_planes()
        points2 = map2.get_points()
        lines2 = map2.get_lines()
        planes2 = map2.get_planes()

        # set up putative associations between points, lines, and planes separately
        k = self.params.k_nearest_neighbors
        use_knn = k is not None and self.params.cos_feature_dim > 0

        if use_knn and len(points1) > 0 and len(points2) > 0:
            A_init_points = self._knn_filter_associations(points1, points2, k)
        else:
            A_init_points = clipperpy.utils.create_all_to_all(
                len(points1), len(points2)
            )
        if use_knn and len(lines1) > 0 and len(lines2) > 0:
            A_init_lines = self._knn_filter_associations(lines1, lines2, k)
        else:
            A_init_lines = clipperpy.utils.create_all_to_all(len(lines1), len(lines2))
        if use_knn and len(planes1) > 0 and len(planes2) > 0:
            A_init_planes = self._knn_filter_associations(planes1, planes2, k)
        else:
            A_init_planes = clipperpy.utils.create_all_to_all(
                len(planes1), len(planes2)
            )
        A_init_lines[:, 0] += len(points1)
        A_init_lines[:, 1] += len(points2)
        A_init_planes[:, 0] += len(points1) + len(lines1)
        A_init_planes[:, 1] += len(points2) + len(lines2)
        A_init = np.vstack([A_init_points, A_init_lines, A_init_planes])

        map1_arrays = (
            [self._get_seg_array(obj) for obj in points1]
            + [self._get_seg_array(obj) for obj in lines1]
            + [self._get_seg_array(obj) for obj in planes1]
        )
        map2_arrays = (
            [self._get_seg_array(obj) for obj in points2]
            + [self._get_seg_array(obj) for obj in lines2]
            + [self._get_seg_array(obj) for obj in planes2]
        )
        map1_cl, map2_cl = self._create_padded_map_arrays(map1_arrays, map2_arrays)

        clipper.score_pairwise_and_single_consistency(map1_cl.T, map2_cl.T, A_init)
        return clipper, A_init

    def _knn_filter_associations(self, segs1, segs2, k):
        """Return putative associations filtered by k-nearest cos_feature neighbors.

        Bidirectional: for each seg in segs1, find k nearest in segs2 by cosine
        similarity, and vice versa. Returns union of both directions.
        """
        feats1 = np.array([seg.cos_feature.flatten() for seg in segs1])
        feats2 = np.array([seg.cos_feature.flatten() for seg in segs2])

        pairs = set()

        # segs1 -> segs2
        k1 = min(k, len(segs2))
        nn1 = NearestNeighbors(n_neighbors=k1, metric="cosine", algorithm="brute")
        nn1.fit(feats2)
        indices1 = nn1.kneighbors(feats1, return_distance=False)
        for i, neighbors in enumerate(indices1):
            for j in neighbors:
                pairs.add((i, j))

        # segs2 -> segs1
        k2 = min(k, len(segs1))
        nn2 = NearestNeighbors(n_neighbors=k2, metric="cosine", algorithm="brute")
        nn2.fit(feats1)
        indices2 = nn2.kneighbors(feats2, return_distance=False)
        for j, neighbors in enumerate(indices2):
            for i in neighbors:
                pairs.add((i, j))

        if len(pairs) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        return np.array(sorted(pairs), dtype=np.int32)

    def _create_padded_map_arrays(self, map1_lists, map2_lists):
        max_d = max([arr.shape[0] for arr in map1_lists + map2_lists])

        map1_lists = [
            np.pad(arr, (0, max_d - arr.shape[0]), "constant", constant_values=0.0)
            for arr in map1_lists
        ]
        map2_lists = [
            np.pad(arr, (0, max_d - arr.shape[0]), "constant", constant_values=0.0)
            for arr in map2_lists
        ]
        return np.array(map1_lists), np.array(map2_lists)

    def assoc_idx_to_ids(
        self,
        association_matrix: np.ndarray,
        map1: SegmentList,
        map2: SegmentList,
    ) -> np.ndarray:
        """Convert association index pairs to segment ID pairs.

        Args:
            association_matrix: (n, 2) array of type-ordered indices.
            map1: Source SegmentList (as returned by get_MCA_with_maps).
            map2: Target SegmentList (as returned by get_MCA_with_maps).

        Returns:
            (n, 2) array of segment IDs.
        """
        if association_matrix.size == 0:
            return np.zeros_like(association_matrix)
        ids1 = map1.type_ordered_ids()
        ids2 = map2.type_ordered_ids()
        return np.column_stack(
            (ids1[association_matrix[:, 0]], ids2[association_matrix[:, 1]])
        )

    # Keep backward-compatible alias
    _assoc_idx_to_ids = assoc_idx_to_ids

    def _get_seg_array(self, seg: GeneralSegment) -> np.ndarray:
        return seg.to_array(
            include_ratio=self.params.ratio_feature_dim > 0,
            include_cos=self.params.cos_feature_dim > 0,
        )

    def _construct_z_aligned_frame(self, z_dir: np.ndarray):
        e2 = z_dir.reshape((3, 1))

        # find a vector, v0, that is non-parallel to e2
        smallest_component_ax = np.argmin(np.abs(z_dir))
        v0 = np.zeros((3, 1))
        v0[smallest_component_ax] = 1.0

        # using v0 and e2, find a vector e0 that is orthogonal to e2
        # P2 is the projection matrix that projects a vector onto the plane that is orthogonal to e2
        P2 = np.eye(3) - e2 @ e2.T
        e0 = P2 @ v0
        e0 /= np.linalg.norm(e0)

        # finally, take the cross product of e0 and e2 to get an (already unit vector) e1,
        # that is orthogonal to both of the original vectors
        e1 = np.cross(e2.reshape(-1), e0.reshape(-1)).reshape((3, 1))
        # cross product of z vector to x vector yields right hand coordinate system

        transform = np.eye(4)
        transform[:3, :3] = np.hstack([e0, e1, e2])
        return transform

    def _construct_xyz_aligned_frame(
        self, x_dir: np.ndarray, y_dir: np.ndarray, z_dir: np.ndarray
    ):
        transform = np.eye(4)
        transform[:3, 0] = x_dir.flatten()
        transform[:3, 1] = y_dir.flatten()
        transform[:3, 2] = z_dir.flatten()
        return transform

    def _construct_xy_aligned_frame_2d(self, x_dir: np.ndarray, y_dir: np.ndarray):
        transform = np.eye(4)
        transform[0, 0] = x_dir[0]
        transform[1, 0] = x_dir[1]
        transform[0, 1] = y_dir[0]
        transform[1, 1] = y_dir[1]
        return transform
