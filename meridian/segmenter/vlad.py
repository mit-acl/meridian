"""Classical VLAD with hard assignment.

Parameter-free aggregation: assign each local descriptor to its nearest cluster
center, accumulate residuals per cluster, intra-normalize, flatten, and
L2-normalize. Used by the AnyLoc-style VPR pipeline.

Reference: Jégou, Douze, Schmid, Pérez, *Aggregating local descriptors into a
compact image representation*, CVPR 2010.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VLAD(nn.Module):
    """Hard-assignment VLAD. No learnable parameters.

    Assign each of the N descriptors to its nearest center, sum residuals per
    cluster, intra-normalize each cluster slice, flatten, then L2-normalize.

    `metric` — how descriptors are assigned to clusters:
      * "euclidean": nearest center by squared-L2 distance (classical VLAD).
      * "cosine":    nearest center by cosine similarity. Centers are
                     unit-normalized for *assignment only*; residuals use the
                     raw (un-normalized) centers. Reproduces AnyLoc, whose
                     cosine k-means vocabulary is not unit-norm and which
                     decouples assignment from the residual.

    `normalize_input` — L2-normalize each descriptor before assignment and
    residuals. Pairs with k-means on L2-normalized descriptors. The original
    NetVLAD does not do this; leaving it off preserves descriptor magnitude
    (feature confidence), turning it on makes assignment a pure cosine.
    """

    def __init__(self, centers: torch.Tensor, normalize_input: bool = False,
                 metric: str = "euclidean"):
        """centers: (K, D) cluster centers. `metric`/`normalize_input`: see class."""
        super().__init__()
        if centers.dim() != 2:
            raise ValueError(f"centers must be (K, D), got {tuple(centers.shape)}")
        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"metric must be 'euclidean' or 'cosine', got {metric!r}")
        self.register_buffer("centers", centers.clone())
        self.register_buffer("centers_unit", F.normalize(centers.clone(), dim=1))
        self.normalize_input = normalize_input
        self.metric = metric

    @property
    def num_clusters(self) -> int:
        return self.centers.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.centers.shape[1]

    @property
    def output_dim(self) -> int:
        return self.num_clusters * self.feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, K * D), L2-normalized."""
        if self.normalize_input:
            x = F.normalize(x, dim=-1)
        B, N, D = x.shape
        K = self.num_clusters
        c = self.centers.to(dtype=x.dtype)  # (K, D), used for residuals

        if self.metric == "cosine":
            cu = self.centers_unit.to(dtype=x.dtype)  # (K, D)
            sim = F.normalize(x, dim=-1) @ cu.t()  # (B, N, K)
            assign_idx = sim.argmax(dim=-1)  # (B, N)
        else:
            c_sq = (c * c).sum(dim=1)  # (K,)
            d = -2.0 * (x @ c.t()) + c_sq  # (B, N, K)
            assign_idx = d.argmin(dim=-1)  # (B, N)
        assignment = F.one_hot(assign_idx, num_classes=K).to(x.dtype)  # (B, N, K)

        a_sum = assignment.sum(dim=1, keepdim=True)  # (B, 1, K)
        a = a_sum * c.t().unsqueeze(0)  # (B, D, K)
        vlad = torch.einsum("bnk,bnd->bdk", assignment, x) - a

        vlad = F.normalize(vlad, dim=1)
        vlad = vlad.reshape(B, K * D)
        vlad = F.normalize(vlad, dim=1)
        return vlad
