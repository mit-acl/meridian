import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import clipperpy
from copy import deepcopy

from gen_seg_match.segment.segment_types import (
    SegmentPoint,
    GeneralSegment,
    SegmentList,
    SegmentLine,
    SegmentPlane,
)
from gen_seg_match.params.segment_match_params import SegmentMatchParams


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

    def match(
        self,
        map1: List[GeneralSegment],
        map2: List[GeneralSegment],
        gravity_dir1: np.ndarray = None,
        gravity_dir2: np.ndarray = None,
        bidirectional: bool = True,
        putative_match_matrix: np.ndarray = None,
    ):
        map1 = SegmentList(deepcopy(map1))
        map2 = SegmentList(deepcopy(map2))

        # return empty associations if map is empty
        if len(map1) == 0 or len(map2) == 0:
            return np.array([[]])

        # transform into gravity aligned frame
        if self.params.gravity_guided:
            for map_i, gravity_dir_i in zip([map1, map2], [gravity_dir1, gravity_dir2]):
                assert gravity_dir_i is not None, (
                    "Must supply gravity direction if using gravity guided"
                )
                T_world_gravity = self._construct_gravity_aligned_frame(gravity_dir_i)
                map_i.transform(np.linalg.inv(T_world_gravity))

        clipper = self._setup_solver(bidirectional=bidirectional)

        if putative_match_matrix is None:
            clipper, A_init = self._setup_problem(
                clipper,
                map1.get_points(),
                map1.get_lines(),
                map2.get_points(),
                map2.get_lines(),
            )
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
        return Ain_by_ids

    def get_MCA(self, map1: List[GeneralSegment], map2: List[GeneralSegment]):
        map1 = SegmentList(map1)
        map2 = SegmentList(map2)
        clipper = self._setup_solver()
        clipper, A_init = self._setup_problem(
            clipper,
            map1.get_points(),
            map1.get_lines(),
            map2.get_points(),
            map2.get_lines(),
        )
        M = clipper.get_affinity_matrix()
        C = clipper.get_constraint_matrix()
        return M, C, A_init

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
        gravity_dir1: np.ndarray = None,
        gravity_dir2: np.ndarray = None,
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
            correspondences = self.match(map1, map2, gravity_dir1, gravity_dir2)
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
        correspondences = np.array(
            [[corr[0].id, corr[1].id] for corr in all_pairs]
        )

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
        points1: List[GeneralSegment],
        lines1: List[GeneralSegment],
        points2: List[GeneralSegment],
        lines2: List[GeneralSegment],
    ):
        # set up all to all matching between points and lines separately
        A_init_points = clipperpy.utils.create_all_to_all(len(points1), len(points2))
        A_init_lines = clipperpy.utils.create_all_to_all(len(lines1), len(lines2))
        A_init_lines[:, 0] += len(points1)
        A_init_lines[:, 1] += len(points2)
        A_init = np.vstack([A_init_points, A_init_lines])

        map1_arrays = [self._get_seg_array(obj) for obj in points1] + [
            self._get_seg_array(obj) for obj in lines1
        ]
        map2_arrays = [self._get_seg_array(obj) for obj in points2] + [
            self._get_seg_array(obj) for obj in lines2
        ]
        map1_cl, map2_cl = self._create_padded_map_arrays(map1_arrays, map2_arrays)

        clipper.score_pairwise_and_single_consistency(map1_cl.T, map2_cl.T, A_init)
        return clipper, A_init

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

    def _assoc_idx_to_ids(
        self,
        association_matrix: np.ndarray,
        map1: List[GeneralSegment],
        map2: List[GeneralSegment],
    ) -> np.ndarray:
        Ain_by_ids = np.zeros_like(association_matrix)
        for i in range(association_matrix.shape[0]):
            Ain_by_ids[i, 0] = map1.get_type_ordered_idx(association_matrix[i, 0]).id
            Ain_by_ids[i, 1] = map2.get_type_ordered_idx(association_matrix[i, 1]).id
        return Ain_by_ids

    def _get_seg_array(self, seg: GeneralSegment) -> np.ndarray:
        return seg.to_array(
            include_ratio=self.params.ratio_feature_dim > 0,
            include_cos=self.params.cos_feature_dim > 0,
        )

    def _construct_gravity_aligned_frame(self, gravity_dir: np.ndarray):
        e2 = gravity_dir.reshape((3, 1))

        # find a vector, v0, that is non-parallel to e2
        smallest_component_ax = np.argmin(np.abs(gravity_dir))
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