import numpy as np
import torch

# import pypose as pp
from typing import Tuple, Optional, List

from meridian.register.r3d.geometry import transform_plucker_torch
from meridian.primitive.primitive_list import PrimitiveList

# def refine_transform(R_init: np.ndarray, t_init: np.ndarray,
#                     source: PointLineCloud, target: PointLineCloud,
#                     device: Optional[torch.device]='cpu', **kwargs) -> Tuple[np.ndarray, np.ndarray, List[float]]:
#     """
#     Refine the initial transformation between point/line clouds using gradient descent on SE(3).
#     Wrapper for refine_transform_torch with numpy inputs/outputs.

#     Parameters:
#     R_init : np.ndarray
#         Initial guess for 3x3 rotation matrix.
#     t_init : np.ndarray
#         Initial guess for 3x1 translation vector.
#     source : PointLineCloud
#         Source point-line cloud with points (Nx3) and lines (Mx6 (d, m) Plücker coordinates).
#     target : PointLineCloud
#         Target point-line cloud with points (Nx3) and lines (Mx6 (d, m) Plücker coordinates).
#     device : Optional[torch.device]
#         Device to run the optimization on.
#     **kwargs : dict
#         Additional keyword arguments to pass to refine_transform_torch.

#     Returns:
#     R_refined : np.ndarray
#         Refined 3x3 rotation matrix.
#     t_refined : np.ndarray
#         Refined 3x1 translation vector.
#     losses : List[float]
#         List of loss values during optimization.
#     """

#     assert source.N == target.N and source.M == target.M, "Source and target must have the same number of points and lines."

#     R_init_torch = torch.tensor(R_init, dtype=float, device=device)
#     t_init_torch = torch.tensor(t_init, dtype=float, device=device)
#     source_points_torch = torch.tensor(source.points, dtype=float, device=device)
#     target_points_torch = torch.tensor(target.points, dtype=float, device=device)
#     source_lines_torch = torch.tensor(source.lines, dtype=float, device=device)
#     target_lines_torch = torch.tensor(target.lines, dtype=float, device=device)
#     R_refined_torch, t_refined_torch, losses = refine_transform_torch(
#         R_init_torch, t_init_torch,
#         source_points_torch, target_points_torch,
#         source_lines_torch, target_lines_torch,
#         device=device, **kwargs
#     )

#     R_refined = R_refined_torch.cpu().numpy()
#     t_refined = t_refined_torch.cpu().numpy()
#     return R_refined, t_refined, losses

# def refine_transform_torch(R_init: torch.Tensor, t_init: torch.Tensor,
#                           source_points: torch.Tensor, target_points: torch.Tensor,
#                           source_lines: torch.Tensor, target_lines: torch.Tensor,
#                           lr: float=1e-3, iters: int=100, min_loss_delta: float=1e-7,
#                           device: Optional[torch.device]='cpu', **kwargs) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
#     """
#     Refine the initial transformation between point/line clouds using gradient descent on SE(3).

#     Parameters:
#     R_init : torch.Tensor
#         Initial guess for 3x3 rotation matrix.
#     t_init : torch.Tensor
#         Initial guess for 3x1 translation vector.
#     source_points : torch.Tensor
#         An Nx3 tensor of 3D source points.
#     target_points : torch.Tensor
#         An Nx3 tensor of 3D target points.
#     source_lines : torch.Tensor
#         An Mx6 tensor of 3D source lines ((d, m) Plücker coordinates).
#     target_lines : torch.Tensor
#         An Mx6 tensor of 3D target lines ((d, m) Plücker coordinates).
#     device : Optional[torch.device]
#         Device to run the optimization on.
#     lr : float
#         Learning rate for the optimizer.
#     iters : int
#         Maximum number of optimization iterations.
#     min_loss_delta : float
#         Minimum change in loss to continue optimization.
#     **kwargs : dict
#         Additional arguments for PointLineLoss

#     Returns:
#     R_refined : torch.Tensor
#         Refined 3x3 rotation matrix.
#     t_refined : torch.Tensor
#         Refined 3x1 translation vector.
#     losses : List[float]
#         List of loss values during optimization.
#     """

#     delta = pp.Parameter(pp.so3(torch.zeros(1,3, dtype=R_init.dtype, device=device)))
#     t_corr = torch.nn.Parameter(torch.zeros(3, dtype=R_init.dtype, device=device))

#     optimizer = torch.optim.Adam([delta, t_corr],  lr=lr)
#     loss_fn = PointLineLoss(**kwargs).to(device)

#     losses = []

#     for _ in range(iters):
#         optimizer.zero_grad()

#         R_delta = delta.Exp().matrix().squeeze(0)

#         R_current = R_delta @ R_init
#         t_current = t_init + t_corr.squeeze(0)

#         loss = loss_fn(R_current, t_current,
#                        source_points, target_points,
#                        source_lines, target_lines)

#         loss.backward()
#         optimizer.step()

#         losses.append(loss.item())
#         if len(losses) > 1 and abs(losses[-2] - losses[-1]) < min_loss_delta:
#             break

#     R_final = (delta.Exp().matrix().squeeze(0) @ R_init).detach()
#     t_final = (t_init + t_corr.squeeze(0)).detach()

#     with torch.no_grad():
#         losses.append(loss_fn(R_final, t_final,
#                             source_points, target_points,
#                             source_lines, target_lines).item())

#     return R_final, t_final, losses


# class PointLineLoss(torch.nn.Module):
#     def __init__(self, W_P=1.0, W_L_D=1.0, W_L_M=1.0, bidirectional: bool = False):
#         super().__init__()
#         self.W_P = W_P
#         self.W_L_D = W_L_D
#         self.W_L_M = W_L_M
#         self.bidirectional = bidirectional

#     def forward(
#         self,
#         R: torch.Tensor,
#         t: torch.Tensor,
#         source_points: torch.Tensor,
#         target_points: torch.Tensor,
#         source_lines: torch.Tensor,
#         target_lines: torch.Tensor,
#     ):
#         """
#         Compute the combined point-line loss for given transformation.

#         Parameters:
#         R : torch.Tensor
#             A 3x3 rotation matrix.
#         t : torch.Tensor
#             A 3x1 translation vector.
#         source_points : torch.Tensor
#             An Nx3 array of source 3D points.
#         target_points : torch.Tensor
#             An Nx3 array of target 3D points.
#         source_lines : torch.Tensor
#             An Mx6 array of (d, m) source Plücker line coordinates.
#         target_lines : torch.Tensor
#             An Mx6 array of (d, m) target Plücker line coordinates.
#         """

#         N, M = len(source_points), len(source_lines)
#         assert N == len(target_points) and M == len(target_lines), (
#             "Source and target must have the same number of points and lines."
#         )

#         loss = 0.0

#         # -----------------------
#         # Point correspondences
#         # -----------------------
#         if N > 0:
#             # Point L2 loss
#             transformed_points = (R @ source_points.T).T + t
#             point_diff = transformed_points - target_points
#             loss += 0.5 * self.W_P * torch.sum(point_diff**2)

#         # -----------------------
#         # Line correspondences (Plücker)
#         # -----------------------
#         if M > 0:
#             transformed_lines = transform_plucker_torch(R, t, source_lines)
#             target_directions, target_moments = target_lines[:, :3], target_lines[:, 3:]
#             transformed_directions, transformed_moments = (
#                 transformed_lines[:, :3],
#                 transformed_lines[:, 3:],
#             )

#             # Directional alignment loss
#             dir_cosine_sim = torch.sum(
#                 target_directions * transformed_directions, dim=1
#             )
#             if self.bidirectional:
#                 dir_cosine_sim = torch.abs(dir_cosine_sim)
#             loss += self.W_L_D * torch.sum(1 - dir_cosine_sim)

#             # Moment directional L2 loss
#             sq_moment_diff = torch.sum(
#                 (transformed_moments - target_moments) ** 2, dim=1
#             )
#             if self.bidirectional:
#                 sq_moment_diff = torch.minimum(
#                     sq_moment_diff,
#                     torch.sum((transformed_moments + target_moments) ** 2, dim=1),
#                 )
#             loss += 0.5 * self.W_L_M * torch.sum(sq_moment_diff)

#         return loss


# def PointLineLoss_numpy(
#     PLS: PointLineLoss,
#     R: np.ndarray,
#     t: np.ndarray,
#     source: PrimitiveList,
#     target: PrimitiveList,
#     device: torch.device = "cpu",
# ) -> float:
#     """
#     Run PointLineLoss with numpy inputs.

#     Parameters:
#     PLS : PointLineLoss
#         An instance of the PointLineLoss class.
#     R : np.ndarray
#         Candidate 3x3 rotation matrix.
#     t : np.ndarray
#         Candidate 3x1 translation vector.
#     source : PrimitiveList
#         Source point-line cloud with points (Nx3) and lines (Mx6 (d, m) Plücker coordinates).
#     target : PrimitiveList
#         Target point-line cloud with points (Nx3) and lines (Mx6 (d, m) Plücker coordinates).
#     device : torch.device
#         Device to run the computation on.

#     Returns:
#     loss : float
#         Computed loss value.
#     """

#     R_torch = torch.tensor(R, dtype=float, device=device)
#     t_torch = torch.tensor(t, dtype=float, device=device)
#     source_points_torch = torch.tensor(
#         source.get_points().points, dtype=float, device=device
#     )
#     target_points_torch = torch.tensor(
#         target.get_points().points, dtype=float, device=device
#     )
#     source_lines_torch = torch.tensor(
#         np.hstack((source.get_lines().directions, source.get_lines().moments)),
#         dtype=float,
#         device=device,
#     )
#     target_lines_torch = torch.tensor(
#         np.hstack((target.get_lines().directions, target.get_lines().moments)),
#         dtype=float,
#         device=device,
#     )

#     with torch.no_grad():
#         loss = PLS(
#             R_torch,
#             t_torch,
#             source_points_torch,
#             target_points_torch,
#             source_lines_torch,
#             target_lines_torch,
#         )

#     return loss.item()
