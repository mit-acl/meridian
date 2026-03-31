import numpy as np
from collections import defaultdict

import torch


class LangevinDynamics:
    def __init__(self, M, C, d, dim, device="cpu"):
        self.device = device
        self.M = torch.from_numpy(M).float().to(device)
        self.C = torch.from_numpy(C).float().to(device)
        self.d = d
        assert d >= dim, "d must be greater than or equal to dim"
        self.Md = self.M - d * self.C
        self.dim = dim

    def grad_particles(self, X):
        """
        Compute gradient for all particles.

        Args:
            X: shape (n, d) where each row is a particle x_i.

        Returns:
            grad: shape (n, d), where row i is grad(x_i).
        """
        return 2.0 * (X @ self.Md)

    def updateParticles(self, u, stepsize, n_iter, adagrad=False, alpha=0.9, debug=False, **kwargs):
        u = self.update(
            theta=u, n_iter=n_iter, stepsize=stepsize,
            adagrad=adagrad, alpha=alpha, debug=debug, **kwargs,
        )
        return u

    @torch.no_grad()
    def update(
        self, theta, n_iter=1000, stepsize=1e-3, alpha=0.9, adagrad=False,
        debug=False, no_noise=False, anneal_noise=False, early_stop=True,
        check_interval=50, obj_tol=5e-3, patience=3,
    ):
        """
        Run Langevin dynamics with optional early stopping.

        Args:
            early_stop: If True, stop when the smoothed objective stabilizes.
            check_interval: Number of iterations between convergence checks.
            obj_tol: Relative change threshold (smoothed) for convergence.
            patience: Number of consecutive checks below threshold before stopping.
        """
        fudge_factor = 1e-6
        historical_grad = 0
        self._actual_iters = 0

        # EMA of objective for smoothing stochastic noise
        obj_ema = None
        ema_beta = 0.5  # smoothing factor (higher = more smoothing)
        converge_count = 0

        for iter in range(n_iter):
            logpgrad = self.grad_particles(theta)

            if no_noise:
                direct = 1 / 2 * logpgrad
            else:
                # Annealed noise: linearly decay noise from sqrt(stepsize) to 0
                if anneal_noise:
                    noise_scale = np.sqrt(stepsize) * (1.0 - iter / max(n_iter - 1, 1))
                else:
                    noise_scale = np.sqrt(stepsize)
                direct = 1 / 2 * logpgrad + torch.randn_like(theta) * noise_scale

            if adagrad:
                if iter == 0:
                    historical_grad = historical_grad + direct ** 2
                else:
                    historical_grad = alpha * historical_grad + (1 - alpha) * (direct ** 2)
                adj_grad = torch.divide(direct, fudge_factor + torch.sqrt(historical_grad))
                theta = theta + stepsize * adj_grad
            else:
                theta = theta + stepsize * direct

            # Projection onto non-negative unit sphere
            theta = torch.clamp(theta, min=0.0)
            norm = torch.norm(theta, dim=1, keepdim=True).clamp(min=1e-6)
            theta = theta / norm

            self._actual_iters = iter + 1

            # Early stopping check
            if early_stop and (iter + 1) % check_interval == 0:
                obj = ((theta @ self.Md) * theta).sum(dim=1).mean().item()
                if obj_ema is None:
                    obj_ema = obj
                else:
                    prev_ema = obj_ema
                    obj_ema = ema_beta * obj_ema + (1 - ema_beta) * obj
                    rel_change = abs(obj_ema - prev_ema) / (abs(prev_ema) + 1e-12)
                    if debug:
                        print(f"  iter {iter+1}: obj={obj:.4f}, ema={obj_ema:.4f}, rel_change={rel_change:.2e}")
                    if rel_change < obj_tol:
                        converge_count += 1
                        if converge_count >= patience:
                            if debug:
                                print(f"  Converged at iter {iter+1}")
                            break
                    else:
                        converge_count = 0
                if debug and obj_ema == obj:
                    print(f"  iter {iter+1}: obj={obj:.4f} (initial)")

        return theta

    def extract_associations(self, u, A_put):
        """Extract and deduplicate association sets from converged particles.

        Args:
            u: Converged particle tensor of shape (n_particles, d).
            A_put: Putative associations, shape (n_assoc, 2). Each row is
                (source_idx, target_idx).

        Returns:
            List of (association_set, count) tuples sorted by count ascending.
            Each association_set is a tuple of (source_idx, target_idx) tuples.
        """
        _, sorted_idx = torch.sort(u, dim=1, descending=True)

        if isinstance(A_put, torch.Tensor):
            A_cpu = A_put.detach().cpu().numpy()
        else:
            A_cpu = np.asarray(A_put)
        sorted_idx_cpu = sorted_idx.cpu().numpy()

        # Estimate clique size per particle: k = round(x^T Md x)
        Md_u = u @ self.Md                                # (n_particles, n)
        omega_hat = torch.round((Md_u * u).sum(dim=1))    # (n_particles,)
        omega_hat = omega_hat.clamp(min=0, max=u.size(1)).to(torch.long)

        associations = []
        for i in range(u.size(0)):
            k = int(omega_hat[i].item())
            if k == 0:
                associations.append(tuple())
                continue
            sel = sorted_idx_cpu[i, :k]
            associations.append(tuple(map(tuple, A_cpu[sel])))

        # Merge duplicates (normalize by sorting each set)
        merged_dict = defaultdict(int)
        for assoc in associations:
            normalized_key = tuple(sorted(assoc))
            merged_dict[normalized_key] += 1

        return sorted(merged_dict.items(), key=lambda item: item[1])
