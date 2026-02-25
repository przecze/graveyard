# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""
Gradient-descent losses and step logic for deaths/yr (no UI).

Used by death_model for post-fit grad optim (period integral + curvature).
"""

from __future__ import annotations

import torch


def period_cumulative_deaths_loss(
    D: torch.Tensor,
    deaths_tgt: torch.Tensor,
    anchor_indices: list[int],
) -> torch.Tensor:
    total = torch.zeros(1, dtype=D.dtype)
    for j in range(len(anchor_indices) - 1):
        i0, i1 = anchor_indices[j], anchor_indices[j + 1]
        D_j = D[i0:i1]
        integral = D_j.sum()
        diff = deaths_tgt[j] - integral
        total = total + diff ** 2 / (i1 - i0)
    return total.squeeze()


def curvature_squared_array(
    D: torch.Tensor,
    d_year_2_before: float,
    d_year_1_before: float,
    d_year_1_after: float,
    d_year_2_after: float,
    eps_frac: float = 0.1,
    debug: bool = False,
) -> torch.Tensor:
    """Curvature² at each interior point: (D2/D)². Same shape as D.
    """
    padded_D = torch.cat([
        torch.tensor([d_year_2_before, d_year_1_before], dtype=D.dtype, device=D.device),
        D,
        torch.tensor([d_year_1_after, d_year_2_after], dtype=D.dtype, device=D.device),
    ])
    
    numeric_D2 = padded_D[2:] - 2 * padded_D[1:-1] + padded_D[:-2]
    curvature = numeric_D2 / padded_D[1:-1]
    return curvature.pow(2)


def total_curvature_squared_loss(
    D: torch.Tensor,
    d_year_2_before: float,
    d_year_1_before: float,
    d_year_1_after: float,
    d_year_2_after: float,
) -> torch.Tensor:
    """Sum of squared curvature of D (discrete second derivative / value).
    """
    curv = curvature_squared_array(
        D, d_year_2_before, d_year_1_before, d_year_1_after, d_year_2_after
    )
    return curv.sum()


def compute_grad_optim_grads(
    D_param: torch.Tensor,
    deaths_tgt: torch.Tensor,
    anchor_indices: list[int],
    relative_curvature_importance: float = 1.0,
    d_year_2_before: float | None = None,
    d_year_1_before: float | None = None,
    d_year_1_after: float | None = None,
    d_year_2_after: float | None = None,
) -> float:
    """Compute gradients for grad optim. Backward in place; returns loss value."""
    loss = period_cumulative_deaths_loss(D_param, deaths_tgt, anchor_indices)
    loss += relative_curvature_importance * total_curvature_squared_loss(
        D_param,
        d_year_2_before, d_year_1_before,
        d_year_1_after, d_year_2_after,
    )
    loss.backward()
    return loss.item()


def compute_separate_grads(
    D_param: torch.Tensor,
    deaths_tgt: torch.Tensor,
    anchor_indices: list[int],
    d_year_2_before: float,
    d_year_1_before: float,
    d_year_1_after: float,
    d_year_2_after: float,
) -> tuple:
    """Return (grad_integrals, grad_curvature) as numpy arrays (same shape as D_param)."""
    p1 = D_param.detach().clone().requires_grad_(True)
    period_cumulative_deaths_loss(p1, deaths_tgt, anchor_indices).backward()
    grad_integrals = p1.grad.cpu().numpy().copy()

    p2 = D_param.detach().clone().requires_grad_(True)
    total_curvature_squared_loss(
        p2,
        d_year_2_before, d_year_1_before,
        d_year_1_after, d_year_2_after,
    ).backward()
    grad_curvature = p2.grad.cpu().numpy().copy()

    return grad_integrals, grad_curvature
