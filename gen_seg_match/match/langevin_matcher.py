import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from gen_seg_match.match.langevin_dynamics import LangevinDynamics
from gen_seg_match.match.segment_matcher import SegmentMatcher
from gen_seg_match.segment.segment_types import GeneralSegment

logger = logging.getLogger(__name__)


@dataclass
class LangevinMatcherParams:
    n_particles: int = 5000
    n_iter: int = 1000
    step_size: float = 1.0
    adagrad: bool = True
    alpha: float = 0.9
    device: str = "cuda"
    min_associations: int = 3
    anneal_noise: bool = False
    no_noise: bool = False
    early_stop: bool = True
    check_interval: int = 10
    obj_tol: float = 1e-3
    patience: int = 3


class LangevinMatcher:
    """Multi-hypothesis matcher using Langevin dynamics on CLIPPER affinity matrices."""

    def __init__(self, segment_matcher: SegmentMatcher, params: LangevinMatcherParams):
        self.segment_matcher = segment_matcher
        self.params = params

    def match(
        self,
        map1: List[GeneralSegment],
        map2: List[GeneralSegment],
    ) -> List[Tuple[np.ndarray, float, int]]:
        """Run Langevin dynamics to find multiple association hypotheses.

        Args:
            map1: Source segments.
            map2: Target segments.

        Returns:
            List of (association_ids, objective_score, particle_count) tuples
            sorted descending by objective. Each association_ids is an (n, 2)
            array of segment IDs. particle_count is the number of particles
            that converged to this association set. Returns empty list if no
            valid hypotheses are found.
        """
        # Get affinity matrix, constraint matrix, putative associations, and ordered maps
        M, C, A, map1_ordered, map2_ordered = self.segment_matcher.get_MCA_with_maps(
            map1, map2
        )

        if M.shape[0] == 0:
            return []

        # Run Langevin dynamics
        C_ld = (M == 0).astype(int)
        dim = M.shape[0]
        ld_solver = LangevinDynamics(M, C_ld, dim, dim, device=self.params.device)

        u = torch.rand(self.params.n_particles, dim).to(self.params.device)
        u = ld_solver.updateParticles(
            u,
            stepsize=self.params.step_size,
            n_iter=self.params.n_iter,
            adagrad=self.params.adagrad,
            alpha=self.params.alpha,
            no_noise=self.params.no_noise,
            anneal_noise=self.params.anneal_noise,
            early_stop=self.params.early_stop,
            check_interval=self.params.check_interval,
            obj_tol=self.params.obj_tol,
            patience=self.params.patience,
        )

        logger.debug(
            f"Langevin dynamics: {ld_solver._actual_iters}/{self.params.n_iter} iterations"
        )

        # Extract association sets from converged particles
        sorted_values = ld_solver.extract_associations(u, A)

        # Build lookup for objective computation
        A_np = np.asarray(A)
        A_lookup = {(int(r[0]), int(r[1])): i for i, r in enumerate(A_np)}

        # Convert to segment IDs, filter, and compute objectives
        results = []
        for assoc_set, count in sorted_values:
            if len(assoc_set) < self.params.min_associations:
                continue

            # Check one-to-one constraints
            sources = [p[0] for p in assoc_set]
            targets = [p[1] for p in assoc_set]
            if len(set(sources)) < len(sources) or len(set(targets)) < len(targets):
                continue

            # Compute objective score: x^T M x / x^T x
            indices = [A_lookup[pair] for pair in assoc_set if pair in A_lookup]
            if len(indices) < self.params.min_associations:
                continue
            x = np.zeros(M.shape[0])
            x[indices] = 1.0
            obj = (x @ M @ x) / np.dot(x, x)

            # Convert to segment IDs
            assoc_matrix = np.array(list(assoc_set), dtype=np.int64)
            ids = self.segment_matcher.assoc_idx_to_ids(
                assoc_matrix, map1_ordered, map2_ordered
            )
            results.append((ids, obj, count))

        # Sort by objective descending (best first)
        results.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"Langevin matcher: {len(results)} valid hypotheses from "
            f"{len(sorted_values)} unique association sets"
        )

        return results
