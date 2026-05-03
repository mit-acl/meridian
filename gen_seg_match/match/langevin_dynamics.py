import numpy as np
from collections import defaultdict

import torch

torch.set_float32_matmul_precision("high")


class LangevinDynamics:
    def __init__(self, M, C, d, dim, device="cpu"):
        self.device = device
        self.M = torch.from_numpy(M).float().to(device)
        self.C = torch.from_numpy(C).float().to(device)
        self.d = d
        assert d >= dim, "d must be greater than or equal to dim"
        self.Md = self.M - d * self.C
        self.Md_lp = self.Md.to(torch.bfloat16)
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

    def updateParticles(
        self, u, stepsize, n_iter, adagrad=False, alpha=0.9, debug=False, **kwargs
    ):
        u = self.update(
            theta=u,
            n_iter=n_iter,
            stepsize=stepsize,
            adagrad=adagrad,
            alpha=alpha,
            debug=debug,
            **kwargs,
        )
        return u

    @torch.no_grad()
    def update(
        self,
        theta,
        n_iter=1000,
        stepsize=1e-3,
        alpha=0.9,
        adagrad=False,
        debug=False,
        no_noise=False,
        anneal_noise=False,
        early_stop=True,
        check_interval=50,
        obj_tol=5e-3,
        patience=3,
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
        historical_grad = None
        self._actual_iters = 0

        obj_ema = None
        ema_beta = 0.5
        converge_count = 0

        # Drop the redundant "2.0 *" in grad and the "0.5 *" everywhere by
        # absorbing them: define grad = X @ Md_lp directly. The convergence
        # objective then becomes (grad * theta).sum, with no factor of 0.5.
        sqrt_stepsize = float(np.sqrt(stepsize)) * 0.5
        # half_sqrt_stepsize is the noise scale that produces the same theta
        # update as the original "0.5 * logpgrad + randn * sqrt(stepsize)".
        # Equivalent to multiplying the noise by 0.5 and the grad by 0.5.
        # Re-derive: original direct = 0.5 * (2 * theta@Md) + randn*sqrt(s)
        #   = theta@Md + randn * sqrt(s)
        # so we'll treat 'logpgrad' as theta@Md_lp (no *2), and use noise scale
        # 'sqrt_stepsize_full' = sqrt(stepsize).
        sqrt_stepsize_full = float(np.sqrt(stepsize))
        inv_iter_decay = 1.0 / max(n_iter - 1, 1)

        theta = theta.to(torch.bfloat16)
        noise_buf = torch.empty_like(theta)

        for iter in range(n_iter):
            logpgrad = theta @ self.Md_lp  # was 2.0 * (...)

            if early_stop and iter > 0 and iter % check_interval == 0:
                obj = (logpgrad * theta).sum(dim=1).mean().item()
                if obj_ema is None:
                    obj_ema = obj
                else:
                    prev_ema = obj_ema
                    obj_ema = ema_beta * obj_ema + (1 - ema_beta) * obj
                    rel_change = abs(obj_ema - prev_ema) / (abs(prev_ema) + 1e-12)
                    if debug:
                        print(
                            f"  iter {iter}: obj={obj:.4f}, ema={obj_ema:.4f}, rel_change={rel_change:.2e}"
                        )
                    if rel_change < obj_tol:
                        converge_count += 1
                        if converge_count >= patience:
                            if debug:
                                print(f"  Converged at iter {iter}")
                            break
                    else:
                        converge_count = 0
                if debug and obj_ema == obj:
                    print(f"  iter {iter}: obj={obj:.4f} (initial)")

            if no_noise:
                direct = logpgrad
            else:
                if anneal_noise:
                    noise_scale = sqrt_stepsize_full * (1.0 - iter * inv_iter_decay)
                else:
                    noise_scale = sqrt_stepsize_full
                # Refill noise buffer in-place (no allocation).
                torch.randn(theta.shape, out=noise_buf)
                direct = logpgrad.add_(noise_buf, alpha=noise_scale)

            if adagrad:
                if historical_grad is None:
                    historical_grad = direct * direct
                else:
                    # historical_grad = alpha * historical_grad + (1-alpha) * direct**2
                    historical_grad.mul_(alpha).addcmul_(direct, direct, value=1.0 - alpha)
                # adj_grad = direct / (fudge + sqrt(historical_grad))
                adj_grad = direct.div_(historical_grad.sqrt().add_(fudge_factor))
                theta.add_(adj_grad, alpha=stepsize)
            else:
                theta.add_(direct, alpha=stepsize)

            theta.clamp_(min=0.0)
            norm = torch.linalg.vector_norm(theta, dim=1, keepdim=True).clamp_(min=1e-6)
            theta.div_(norm)

            self._actual_iters = iter + 1

        return theta.float()

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
        # All GPU work in one fused block, then a single sync to host.
        sorted_idx = torch.argsort(u, dim=1, descending=True)
        Md_u = u @ self.Md
        omega_hat = (
            torch.round((Md_u * u).sum(dim=1))
            .clamp(min=0, max=u.size(1))
            .to(torch.long)
        )
        sorted_idx_cpu = sorted_idx.cpu().numpy()
        omega_hat_cpu = omega_hat.cpu().numpy()

        if isinstance(A_put, torch.Tensor):
            A_cpu = A_put.detach().cpu().numpy()
        else:
            A_cpu = np.asarray(A_put)

        merged_dict = defaultdict(int)
        for i in range(sorted_idx_cpu.shape[0]):
            k = int(omega_hat_cpu[i])
            if k == 0:
                merged_dict[()] += 1
                continue
            sel = sorted_idx_cpu[i, :k]
            assoc = A_cpu[sel]
            order = np.lexsort((assoc[:, 1], assoc[:, 0]))
            normalized_key = tuple(map(tuple, assoc[order]))
            merged_dict[normalized_key] += 1

        return sorted(merged_dict.items(), key=lambda item: item[1])
