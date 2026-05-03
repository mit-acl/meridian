import logging
from typing import List, Tuple

import numpy as np
import torch

from gen_seg_match.match.langevin_dynamics import LangevinDynamics
from gen_seg_match.params.segment_match_params import LangevinMatcherParams


import time

logger = logging.getLogger(__name__)


class LangevinMatcher:
    """Multi-hypothesis matcher using Langevin dynamics on CLIPPER affinity matrices."""

    def __init__(self, params: LangevinMatcherParams):
        self.params = params

    def match(
        self,
        M: np.ndarray,
        C: np.ndarray,
        A: np.ndarray,
    ) -> List[Tuple[np.ndarray, float, int]]:
        """Run Langevin dynamics to find multiple association hypotheses.

        Args:
            M: Affinity matrix from CLIPPER.
            C: Constraint matrix from CLIPPER.
            A: Putative associations array (N, 2) of index pairs.

        Returns:
            List of (association_indices, objective_score, particle_count) tuples
            sorted descending by objective. Each association_indices is an (n, 2)
            array of index pairs (in the same index space as A). particle_count
            is the number of particles that converged to this association set.
            Returns empty list if no valid hypotheses are found.
        """
        if M.shape[0] == 0:
            return []

        # Run Langevin dynamics
        t_setup0 = time.time()
        C_ld = (M == 0).astype(int)
        dim = M.shape[0]
        ld_solver = LangevinDynamics(M, C_ld, dim, dim, device=self.params.device)

        u = torch.rand(self.params.n_particles, dim).to(self.params.device)
        t_setup = time.time() - t_setup0

        start_time = time.time()
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
        end_time = time.time()
        logger.debug(
            f"Langevin dynamics took {end_time - start_time:.3f} seconds "
            f"({ld_solver._actual_iters}/{self.params.n_iter} iters)"
        )

        start_time = time.time()
        # Extract association sets from converged particles
        sorted_values = ld_solver.extract_associations(u, A)
        end_time = time.time()
        logger.debug(f"Extracting association sets took {end_time - start_time:.3f} seconds")

        # Build lookup for objective computation
        t_lookup0 = time.time()
        A_np = np.asarray(A)
        A_lookup = {(int(r[0]), int(r[1])): i for i, r in enumerate(A_np)}
        t_lookup = time.time() - t_lookup0

        start_time = time.time()
        # Filter and compute objectives
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

            # Return raw index pairs (caller converts to segment IDs)
            assoc_matrix = np.array(list(assoc_set), dtype=np.int64)
            results.append((assoc_matrix, obj, count))

        end_time = time.time()
        logger.debug(f"Filtering and computing objectives for {len(sorted_values)} association sets took {end_time - start_time:.3f} seconds")

        # Sort by objective descending (best first)
        t_sort0 = time.time()
        results.sort(key=lambda x: x[1], reverse=True)
        t_sort = time.time() - t_sort0

        logger.debug(
            f"LM_INNER setup={t_setup*1000:.1f}ms lookup={t_lookup*1000:.1f}ms "
            f"sort={t_sort*1000:.1f}ms n_results={len(results)} |A|={len(A_np)}"
        )

        logger.debug(
            f"Langevin matcher: {len(results)} valid hypotheses from "
            f"{len(sorted_values)} unique association sets"
        )

        return results
