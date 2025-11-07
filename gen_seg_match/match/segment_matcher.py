import numpy as np
from typing import List
import matplotlib.pyplot as plt
import clipperpy
from dataclasses import dataclass

from gen_seg_match.segment.segment_types import SegmentPoint, SegmentLine, \
    SegmentPlane, GeneralSegment, SegmentList
from gen_seg_match.params.segment_match_params import SegmentMatchParams

class InsufficientAssociationsException(Exception):
    
    def __init__(self, map1_len, map2_len, n_associations=None):
        self.map1_len = map1_len
        self.map2_len = map2_len
        self.n_associations = n_associations
        message = f"Insufficient associations. Map 1 length: {map1_len}. Map 2 length: {map2_len}. Associations: {n_associations}"
        super().__init__(message)

class SegmentMatcher():

    def __init__(self, params: SegmentMatchParams):
        self.params = params

    def match(self, map1: List[GeneralSegment], map2: List[GeneralSegment]):
        map1 = SegmentList(map1)
        map2 = SegmentList(map2)
        if len(map1) == 0 or len(map2) == 0:
            return np.array([[]])
            
        clipper = self._setup_solver()
        clipper, A_init = self._setup_problem(clipper, map1.get_points(), map1.get_lines(),
                                                map2.get_points(), map2.get_lines())
        clipper.solve()
        Ain = clipper.get_selected_associations()
        Ain_by_ids = self._assoc_idx_to_ids(Ain, map1, map2)
        return Ain_by_ids
    
    def get_MCA(self, map1: List[GeneralSegment], map2: List[GeneralSegment]):
        map1 = SegmentList(map1)
        map2 = SegmentList(map2)
        clipper = self._setup_solver()
        clipper, A_init = self._setup_problem(clipper, map1.get_points(), map1.get_lines(),
                                                map2.get_points(), map2.get_lines())
        M = clipper.get_affinity_matrix()
        C = clipper.get_constraint_matrix()
        return M, C, A_init
    
    def match_multiple(self, map1: List[GeneralSegment], 
                       map2: List[GeneralSegment], num_solutions=2):
        map1 = SegmentList(map1)
        map2 = SegmentList(map2)
        M, C, A = self.get_MCA(map1, map2)
        M_orig = M.copy()
        clipper = clipperpy.CLIPPER(clipperpy.invariants.PairwiseInvariant(), clipperpy.Params())
        solutions = []

        for k in range(num_solutions):
            clipper.set_matrix_data(M=M, C=C)
            clipper.solve()

            solution_nodes = clipper.get_solution().nodes
            Ain = np.zeros((len(solution_nodes), 2)).astype(np.int64)
            for i in range(len(solution_nodes)):
                Ain[i,:] = A[solution_nodes[i],:]
            
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
                row_indices, col_indices = np.meshgrid(solution_nodes, solution_nodes, indexing='ij')
                if len(row_indices) != 0 and len(col_indices) != 0:
                    M[row_indices,col_indices] = 0.0

        return solutions
    
    def _setup_solver(self):
        invariant = clipperpy.invariants.GeneralSegmentDistance(self.params.to_clipper())
        params = clipperpy.Params()
        clipper = clipperpy.CLIPPERPairwiseAndSingle(invariant, params)
        return clipper
    
    def _setup_problem(self, clipper, 
            points1: List[GeneralSegment], lines1: List[GeneralSegment],
            points2: List[GeneralSegment], lines2: List[GeneralSegment]):

        # set up all to all matching between points and lines separately
        A_init_points = clipperpy.utils.create_all_to_all(len(points1), len(points2))
        A_init_lines = clipperpy.utils.create_all_to_all(len(lines1), len(lines2))
        A_init_lines[:,0] += len(points1)
        A_init_lines[:,1] += len(points2)
        A_init = np.vstack([A_init_points, A_init_lines])

        map1_arrays = [obj.to_array() for obj in points1] + [obj.to_array() for obj in lines1]
        map2_arrays = [obj.to_array() for obj in points2] + [obj.to_array() for obj in lines2]
        max_d = max([arr.shape[0] for arr in map1_arrays + map2_arrays])

        map1_arrays = [np.pad(arr, (0, max_d - arr.shape[0]), 'constant', constant_values=0.0) for arr in map1_arrays]
        map2_arrays = [np.pad(arr, (0, max_d - arr.shape[0]), 'constant', constant_values=0.0) for arr in map2_arrays]
        map1_cl = np.array(map1_arrays)
        map2_cl = np.array(map2_arrays)

        clipper.score_pairwise_and_single_consistency(map1_cl.T, map2_cl.T, A_init)
        return clipper, A_init
    
    def _assoc_idx_to_ids(self, association_matrix: np.ndarray, 
                           map1: List[GeneralSegment], 
                           map2: List[GeneralSegment]) -> np.ndarray:
        Ain_by_ids = np.zeros_like(association_matrix)
        for i in range(association_matrix.shape[0]):
            Ain_by_ids[i,0] = map1.get_type_ordered_idx(association_matrix[i,0]).id
            Ain_by_ids[i,1] = map2.get_type_ordered_idx(association_matrix[i,1]).id
        return Ain_by_ids

    def register(self, map1: List[GeneralSegment], map2: List[GeneralSegment], correspondences: np.array = None):
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
            correspondences = self.match(map1, map2)

        map1 = SegmentList(map1)
        map2 = SegmentList(map2)

        filtered_correspondences = [corr for corr in correspondences if type(map1.get_segment_from_id(corr[0])) == SegmentPoint 
                                                                    and type(map2.get_segment_from_id(corr[1])) == SegmentPoint]
        
        if len(filtered_correspondences) < self.params.dim:
            raise InsufficientAssociationsException(len(map1), len(map2), len(filtered_correspondences))

        pts1 = np.array([map1.get_segment_from_id(corr[0]).get_point()[:self.params.dim] for corr in filtered_correspondences])
        pts2 = np.array([map2.get_segment_from_id(corr[1]).get_point()[:self.params.dim] for corr in filtered_correspondences])

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
            Vh_prime[-1,:] *= -1.0
            R = U @ Vh_prime

        t = mean1.reshape((-1,1)) - R @ mean2.reshape((-1,1))
        T = np.concatenate([np.concatenate([R, t], axis=1), np.hstack([np.zeros((1, R.shape[0])), [[1]]])], axis=0)
        return T
    
    def view_registration(self, map1: List[GeneralSegment], map2: List[GeneralSegment], correspondences: np.array, T: np.array, ax=None, **kwargs):
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

        map1 = SegmentList([seg for seg in map1 if type(seg) == SegmentPoint])
        map2 = SegmentList([seg.copy() for seg in map2 if type(seg) == SegmentPoint])

        map2.transform(T)

        for seg in map1:
            if type(seg) == SegmentPoint:
                ax.plot(seg.get_point()[0], seg.get_point()[1], 'o', color='maroon', **kwargs)

        for seg in map2:
            if type(seg) == SegmentPoint:
                ax.plot(seg.get_point()[0], seg.get_point()[1], 'o', color='blue', **kwargs)

        for corr in correspondences:
            ax.plot([map1.get_segment_from_id(corr[0]).get_point()[0], map2.get_segment_from_id(corr[1]).get_point()[0]], 
                     [map1.get_segment_from_id(corr[0]).get_point()[1], map2.get_segment_from_id(corr[1]).get_point()[1]], 
                     color='lawngreen', linestyle='dotted')
        
        ax.set_aspect('equal')
        return ax