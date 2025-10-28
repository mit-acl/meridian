import numpy as np
from typing import List
import matplotlib.pyplot as plt
import clipperpy
from dataclasses import dataclass

from gen_seg_match.segment.segment_types import SegmentPoint, SegmentLine, \
    SegmentPlane, GeneralSegment
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

    # def match(self, map1: List[GeneralSegment], map2: List[GeneralSegment]):
    def match(self, points1: List[GeneralSegment], lines1: List[GeneralSegment],
              points2: List[GeneralSegment], lines2: List[GeneralSegment]):
        if len(points1) + len(lines1) == 0 or len(points2) + len(lines2) == 0:
            return np.array([[]])
        # if len(map1) == 0 or len(map2) == 0:
        #     return np.array([[]])
        clipper = self._setup_solver()
        clipper, A_init = self._setup_problem(clipper, points1, lines1, points2, lines2)
        clipper.solve()
        Ain = clipper.get_selected_associations()
        return Ain
    
    def _setup_solver(self):
        invariant = clipperpy.invariants.GeneralSegmentDistance(self.params.to_clipper())
        params = clipperpy.Params()
        clipper = clipperpy.CLIPPERPairwiseAndSingle(invariant, params)
        return clipper
    
    # def _setup_problem(self, clipper, map1: list, map2: list):
    def _setup_problem(self, clipper, 
            points1: List[GeneralSegment], lines1: List[GeneralSegment],
            points2: List[GeneralSegment], lines2: List[GeneralSegment]):
        # points1 = [obj for obj in map1 if type(obj) == SegmentPoint]
        # lines1 = [obj for obj in map1 if type(obj) == SegmentLine]
        # points2 = [obj for obj in map2 if type(obj) == SegmentPoint]
        # lines2 = [obj for obj in map2 if type(obj) == SegmentLine]

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
    
    def get_MCA(self, points1: List[GeneralSegment], lines1: List[GeneralSegment],
              points2: List[GeneralSegment], lines2: List[GeneralSegment]):
        clipper = self._setup_solver()
        clipper, A_init = self._setup_problem(clipper, points1, lines1, points2, lines2)
        M = clipper.get_affinity_matrix()
        C = clipper.get_constraint_matrix()
        return M, C, A_init
    
    def match_multiple(self, points1: List[GeneralSegment], lines1: List[GeneralSegment],
              points2: List[GeneralSegment], lines2: List[GeneralSegment], num_solutions=2):
        M, C, A = self.get_MCA(points1, lines1, points2, lines2)
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
            solutions.append((Ain.copy(), score))

            if k + 1 < num_solutions:
                row_indices, col_indices = np.meshgrid(solution_nodes, solution_nodes, indexing='ij')
                if len(row_indices) != 0 and len(col_indices) != 0:
                    M[row_indices,col_indices] = 0.0

        return solutions

    # def T_align(self, map1: List[Object], map2: List[Object], correspondences: np.array = None):
    #     """
    #     Computes the transformation that aligns map2 to map1.

    #     Args:
    #         map1 (List[Object]): Object list in frame 1
    #         map2 (List[Object]): Object list in frame 2
    #         correspondences (np.array, shape=(n,2), optional): If correspondences have already 
    #             been found, set to None. Otherwise, performs register before aligning. Aligns using 
    #             Arun's method. Defaults to None.

    #     Returns:
    #         np.array: Transformation matrix that aligns map2 to map1
    #     """
    #     if len(map1) == 0 or len(map2) == 0:
    #         raise InsufficientAssociationsException(len(map1), len(map2))

    #     if correspondences is None:
    #         correspondences = self.register(map1, map2)
    #     if len(correspondences) < self.dim:
    #         raise InsufficientAssociationsException(len(map1), len(map2), len(correspondences))

    #     pts1 = np.array([map1[corr[0]].center.reshape(-1)[:self.dim] for corr in correspondences])
    #     pts2 = np.array([map2[corr[1]].center.reshape(-1)[:self.dim] for corr in correspondences])

    #     weights = np.ones((pts1.shape[0],1))
    #     weights = weights.reshape((-1,1))
    #     mean1 = (np.sum(pts1 * weights, axis=0) / np.sum(weights)).reshape(-1)
    #     mean2 = (np.sum(pts2 * weights, axis=0) / np.sum(weights)).reshape(-1)
    #     pts1_mean_reduced = pts1 - mean1
    #     pts2_mean_reduced = pts2 - mean2
    #     assert pts1_mean_reduced.shape == pts2_mean_reduced.shape
    #     H = pts1_mean_reduced.T @ (pts2_mean_reduced * weights)
    #     U, s, Vh = np.linalg.svd(H)
    #     R = U @ Vh
    #     if np.allclose(np.linalg.det(R), -1.0):
    #         Vh_prime = Vh.copy()
    #         Vh_prime[-1,:] *= -1.0
    #         R = U @ Vh_prime
    #     t = mean1.reshape((-1,1)) - R @ mean2.reshape((-1,1))
    #     T = np.concatenate([np.concatenate([R, t], axis=1), np.hstack([np.zeros((1, R.shape[0])), [[1]]])], axis=0)
    #     return T
    
    # def view_registration(self, map1: List[Object], map2: List[Object], correspondences: np.array, T: np.array, ax=None, **kwargs):
    #     """
    #     Visualize the registration between map1 and map2

    #     Args:
    #         map1 (List[Object]): Object list in frame 1
    #         map2 (List[Object]): Object list in frame 2
    #         correspondences (np.array, shape=(n,2)): Correspondences between map1 and map2
    #         T (np.array): Transformation matrix that aligns map2 to map1
    #     """
    #     if ax is None:
    #         _, ax = plt.subplots()

    #     map2_cp = [obj.copy() for obj in map2]
    #     for obj in map2_cp:
    #         obj.transform(T)

    #     for obj in map1:
    #         obj.plot2d(ax, color='maroon', **kwargs)

    #     for obj in map2_cp:
    #         obj.plot2d(ax, color='blue', **kwargs)

    #     for corr in correspondences:
    #         ax.plot([map1[corr[0]].centroid[0], map2_cp[corr[1]].centroid[0]], 
    #                  [map1[corr[0]].centroid[1], map2_cp[corr[1]].centroid[1]], 
    #                  color='lawngreen', linestyle='dotted')
        
    #     ax.set_aspect('equal')
    #     return ax