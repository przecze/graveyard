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
    yr: torch.Tensor,
    deaths_tgt: torch.Tensor,
    anchor_years: list[int],
) -> torch.Tensor:
    """Loss = sum over periods of ((desired − integral) / T_j)² where T_j is period length.

    Dividing by T_j (period length in years) normalises the gradient magnitude so it
    is O(D_error) rather than O(D_error · T_j).  Without this, the -13k→-8k period
    (T_j ≈ 5000 yr) produces gradients ~10⁹× too large for lr=0.1, causing explosion.
    The stability condition for gradient descent on this quadratic is lr < 1/2 after
    normalisation, vs lr < 1/(2·T_j) ≈ 1e-4 without it.
    """
    total = torch.zeros(1, dtype=D.dtype)
    for j in range(len(anchor_years) - 1):
        a, b = anchor_years[j], anchor_years[j + 1]
        mask = (yr >= a) & (yr < b)
        if mask.sum() < 2:
            continue
        yr_j = yr[mask]
        D_j = D[mask]
        T_j = yr_j[-1] - yr_j[0]  # period length in years (torch scalar)
        if T_j == 0:
            continue
        integral = ((D_j[:-1] + D_j[1:]) * (yr_j[1:] - yr_j[:-1]) / 2.0).sum()
        diff = deaths_tgt[j] - integral
        total = total + diff ** 2 / T_j
    return total.squeeze()


def total_curvature_squared_loss(
    D: torch.Tensor,
    yr: torch.Tensor,
    d_year_1_before: float,
    d_year_2_before: float,
    d_year_1_after: float,
    d_year_2_after: float,
) -> torch.Tensor:
    """Sum of squared curvature of D (discrete second derivative / value)."""
    padded_D = torch.cat([
        torch.tensor([d_year_2_before, d_year_1_before]),
        D,
        torch.tensor([d_year_1_after, d_year_2_after]),
    ])
    numeric_D2 = padded_D[2:] - 2 * padded_D[1:-1] + padded_D[:-2]
    curvature = numeric_D2 / (padded_D[1:-1])
    loss = curvature.pow(2).sum()
    return loss.squeeze()


def compute_grad_optim_grads(
    D_param: torch.Tensor,
    yr: torch.Tensor,
    deaths_tgt: torch.Tensor,
    anchor_years: list[int],
    relative_curvature_importance: float = 1.0,
    d_year_1_before: float | None = None,
    d_year_2_before: float | None = None,
    d_year_1_after: float | None = None,
    d_year_2_after: float | None = None,
) -> float:
    """Compute gradients for grad optim. Backward in place; returns loss value."""
    loss = period_cumulative_deaths_loss(D_param, yr, deaths_tgt, anchor_years)
    loss += relative_curvature_importance * total_curvature_squared_loss(
        D_param, yr,
        d_year_1_before, d_year_2_before,
        d_year_1_after, d_year_2_after,
    )
    loss.backward()
    return loss.item()


def compute_separate_grads(
    D_param: torch.Tensor,
    yr: torch.Tensor,
    deaths_tgt: torch.Tensor,
    anchor_years: list[int],
    d_year_1_before: float,
    d_year_2_before: float,
    d_year_1_after: float,
    d_year_2_after: float,
) -> tuple:
    """Return (grad_integrals, grad_curvature) as numpy arrays (same shape as D_param)."""
    p1 = D_param.detach().clone().requires_grad_(True)
    period_cumulative_deaths_loss(p1, yr, deaths_tgt, anchor_years).backward()
    grad_integrals = p1.grad.cpu().numpy().copy()

    p2 = D_param.detach().clone().requires_grad_(True)
    total_curvature_squared_loss(
        p2, yr,
        d_year_1_before, d_year_2_before,
        d_year_1_after, d_year_2_after,
    ).backward()
    grad_curvature = p2.grad.cpu().numpy().copy()

    return grad_integrals, grad_curvature
