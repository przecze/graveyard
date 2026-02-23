# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""
Piecewise polynomial (cubic/quartic) fitting for deaths/yr.

No UI dependencies. Used by death_model for the main fit and by standalone scripts.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq, null_space
from scipy.optimize import minimize


X_MODE_LABELS: dict[str, str] = {
    "log_before_2026": "log(years before 2026)  [current]",
    "year":            "bare year  (no log)",
}
X_MODE_K_REF: dict[str, float | None] = {
    "log_before_2026": 2026.0,
    "year":            None,
}


def _u_transform(year_arr: np.ndarray, logarithm_reference_year: float | None) -> np.ndarray:
    """Map year-count values to the fitting x-variable u.

    logarithm_reference_year is None : u = t  (identity).
    logarithm_reference_year is float: u = ln(ref − t), clamped to 1e-9.
    """
    if logarithm_reference_year is None:
        return year_arr.astype(float)
    return np.log(np.maximum(logarithm_reference_year - year_arr, 1e-9))


def fit_piecewise_polynomial(
    cumulated_D: dict[tuple[int, int], float],
    quartic_period_ids: list[int],
    total_years: int,
    D_start: float | None = None,
    D_end: float | None = None,
    dDdy_start: float | None = None,
    d2Ddy2_start: float | None = None,
    dDdy_end: float | None = None,
    d2Ddy2_end: float | None = None,
    logarithm_reference_year: float | None = None,
) -> dict:
    """Fit deaths/yr as piecewise polynomial (cubic/quartic).

    Calendar-agnostic: works in year-count space (integers starting from 0).

    Parameters
    ----------
    cumulated_D : mapping (start_t, end_t) → deaths
        Cumulative death constraints per period in year-count space.
    quartic_period_ids : list of int
        Indices of cumulated_D periods (0, 1, 2, …) that get quartic (degree 4)
        polynomials; others get cubic.
    total_years : int
        Length of the output array.
    logarithm_reference_year : float or None
        None → u = t (identity); float → u = ln(ref − t).

    Returns
    -------
    dict with D_fitted (ndarray, length total_years, NaN before fit start).
    """

    anchor_years = sorted({t for pair in cumulated_D for t in pair})
    n = len(anchor_years) - 1
    deaths_tgt = np.array([cumulated_D[(anchor_years[j], anchor_years[j + 1])] for j in range(n)])

    t0 = anchor_years[0]
    T_cumul = np.array([y - t0 for y in anchor_years])
    T_periods = np.diff(T_cumul)
    _anc_yr = np.array(anchor_years)
    u_bnd = _u_transform(_anc_yr, logarithm_reference_year)

    is_log_mode = logarithm_reference_year is not None
    mono_factor = 1.0 if is_log_mode else -1.0

    quartic_period_idx = set(quartic_period_ids)
    period_degree = np.array([4 if j in quartic_period_idx else 3 for j in range(n)], dtype=int)

    period_offsets = np.zeros(n + 1, dtype=int)
    for j in range(n):
        period_offsets[j + 1] = period_offsets[j] + int(period_degree[j] + 1)
    n_params = int(period_offsets[-1])

    # Moment integrals IU[j, k] = ∫ u(t)^k dt
    max_deg = int(period_degree.max())
    IU = np.zeros((n, max_deg + 1))
    for j in range(n):
        yr_j  = np.arange(anchor_years[j], anchor_years[j + 1] + 1)
        t_j   = yr_j - t0
        u_j   = _u_transform(yr_j, logarithm_reference_year)
        IU[j, 0] = T_periods[j]
        for k in range(1, int(period_degree[j]) + 1):
            IU[j, k] = np.trapezoid(u_j ** k, t_j)

    n_extra = (
        (1 if D_start is not None else 0)
        + (1 if D_end is not None else 0)
        + (1 if dDdy_start is not None else 0)
        + (1 if d2Ddy2_start is not None else 0)
        + (1 if dDdy_end is not None else 0)
        + (1 if d2Ddy2_end is not None else 0)
    )
    n_constr = 4 * n - 3 + n_extra
    A = np.zeros((n_constr, n_params))
    b = np.zeros(n_constr)
    row = 0

    def _sl(j: int) -> slice:
        return slice(int(period_offsets[j]), int(period_offsets[j + 1]))

    def _basis_u(u: float, degree: int, order: int) -> np.ndarray:
        coeff = np.zeros(degree + 1, dtype=float)
        for k in range(order, degree + 1):
            f = 1.0
            for p in range(order):
                f *= (k - p)
            coeff[k] = f * (u ** (k - order))
        return coeff

    def _du_dy(y: int) -> tuple[float, float]:
        if logarithm_reference_year is None:
            return 1.0, 0.0
        rem = max(logarithm_reference_year - y, 1e-9)
        return -1.0 / rem, -1.0 / (rem * rem)

    # n integral constraints
    for j in range(n):
        deg = int(period_degree[j])
        A[row, _sl(j)] = IU[j, : deg + 1]
        b[row] = deaths_tgt[j]
        row += 1

    # C0, C1, C2 at interior seams
    for order in range(3):
        for j in range(1, n):
            u = float(u_bnd[j])
            A[row, _sl(j - 1)] = _basis_u(u, int(period_degree[j - 1]), order)
            A[row, _sl(j)] = -_basis_u(u, int(period_degree[j]), order)
            row += 1

    # Endpoint value constraints
    if D_start is not None:
        A[row, _sl(0)] = _basis_u(float(u_bnd[0]), int(period_degree[0]), 0)
        b[row] = float(D_start)
        row += 1
    if D_end is not None:
        A[row, _sl(n - 1)] = _basis_u(float(u_bnd[n]), int(period_degree[n - 1]), 0)
        b[row] = float(D_end)
        row += 1

    # Endpoint d1 constraints (year-space → u-space via chain rule)
    if dDdy_start is not None:
        u0, y0 = float(u_bnd[0]), anchor_years[0]
        du, _ = _du_dy(y0)
        A[row, _sl(0)] = du * _basis_u(u0, int(period_degree[0]), 1)
        b[row] = float(dDdy_start)
        row += 1
    if dDdy_end is not None:
        un, yn = float(u_bnd[n]), anchor_years[n]
        du, _ = _du_dy(yn)
        A[row, _sl(n - 1)] = du * _basis_u(un, int(period_degree[n - 1]), 1)
        b[row] = float(dDdy_end)
        row += 1

    # Endpoint d2 constraints (year-space → u-space via chain rule)
    if d2Ddy2_start is not None:
        u0, y0 = float(u_bnd[0]), anchor_years[0]
        du, d2u = _du_dy(y0)
        A[row, _sl(0)] = (du ** 2) * _basis_u(u0, int(period_degree[0]), 2) \
                        + d2u * _basis_u(u0, int(period_degree[0]), 1)
        b[row] = float(d2Ddy2_start)
        row += 1
    if d2Ddy2_end is not None:
        un, yn = float(u_bnd[n]), anchor_years[n]
        du, d2u = _du_dy(yn)
        A[row, _sl(n - 1)] = (du ** 2) * _basis_u(un, int(period_degree[n - 1]), 2) \
                            + d2u * _basis_u(un, int(period_degree[n - 1]), 1)
        b[row] = float(d2Ddy2_end)
        row += 1

    assert row == n_constr, f"row={row} != n_constr={n_constr}"

    # Scale rows for conditioning
    D_scale = np.median(deaths_tgt / T_periods)
    A_sc, b_sc = A.copy(), b.copy()
    for j in range(n):
        A_sc[j] /= deaths_tgt[j]
        b_sc[j] /= deaths_tgt[j]
    for j in range(n, n_constr):
        A_sc[j] /= D_scale
        b_sc[j] /= D_scale

    theta_p, _res, rank, sv = lstsq(A_sc, b_sc)
    N_mat = null_space(A_sc)
    null_dim = N_mat.shape[1]

    def _poly_eval(theta_vec, j, u, order):
        deg = int(period_degree[j])
        c = theta_vec[_sl(j)]
        out = np.zeros_like(u, dtype=float) if isinstance(u, np.ndarray) else 0.0
        for k in range(order, deg + 1):
            f = 1.0
            for p in range(order):
                f *= (k - p)
            out = out + c[k] * f * (u ** (k - order))
        return out

    def _D_theta(tv, j, u):
        return _poly_eval(tv, j, u, 0)

    def _dDdu_theta(tv, j, u):
        return _poly_eval(tv, j, u, 1)

    MONO_TOL = 1_000.0

    def _u_j(j: int) -> np.ndarray:
        return _u_transform(np.arange(anchor_years[j], anchor_years[j + 1] + 1), logarithm_reference_year)

    def _basis_matrix(u_arr: np.ndarray, degree: int, order: int) -> np.ndarray:
        """Return (len(u_arr), degree+1) basis matrix for given derivative order."""
        P = len(u_arr)
        B = np.zeros((P, degree + 1))
        for k in range(order, degree + 1):
            f = 1.0
            for p in range(order):
                f *= k - p
            B[:, k] = f * (u_arr ** (k - order))
        return B

    if null_dim == 0:
        theta = theta_p
    else:
        MAX_OPT_PTS = 50
        G_blocks, h_blocks = [], []
        G_full_blocks, h_full_blocks = [], []
        for j in range(n):
            u_full = _u_j(j)
            stride = max(1, len(u_full) // MAX_OPT_PTS)
            u_opt = u_full[::stride]
            deg = int(period_degree[j])
            B1_opt  = _basis_matrix(u_opt,  deg, 1)
            B1_full = _basis_matrix(u_full, deg, 1)
            G_blocks.append(mono_factor * (B1_opt  @ N_mat[_sl(j), :]))
            h_blocks.append(mono_factor * -_dDdu_theta(theta_p, j, u_opt))
            G_full_blocks.append(mono_factor * (B1_full @ N_mat[_sl(j), :]))
            h_full_blocks.append(mono_factor * -_dDdu_theta(theta_p, j, u_full))
        for j in range(n):
            u_full = _u_j(j)
            stride = max(1, len(u_full) // MAX_OPT_PTS)
            u_opt = u_full[::stride]
            deg = int(period_degree[j])
            B0_opt  = _basis_matrix(u_opt,  deg, 0)
            B0_full = _basis_matrix(u_full, deg, 0)
            G_blocks.append(-(B0_opt  @ N_mat[_sl(j), :]))
            h_blocks.append(_D_theta(theta_p, j, u_opt))
            G_full_blocks.append(-(B0_full @ N_mat[_sl(j), :]))
            h_full_blocks.append(_D_theta(theta_p, j, u_full))
        G_opt,  h_opt  = np.vstack(G_blocks),      np.concatenate(h_blocks)
        G_full, h_full = np.vstack(G_full_blocks),  np.concatenate(h_full_blocks)

        sol = minimize(lambda x: float(x @ x), np.zeros(null_dim), jac=lambda x: 2.0 * x,
                       method="SLSQP",
                       constraints={"type": "ineq", "fun": lambda x: h_opt + 1.0 - G_opt @ x, "jac": lambda x: -G_opt},
                       options={"ftol": 1e-15, "maxiter": 3_000})
        x_opt = sol.x
        if not (sol.success and np.all(G_full @ sol.x <= h_full + MONO_TOL)):
            lam = 1e8
            sol2 = minimize(
                lambda x: float(lam * np.sum(np.maximum(0.0, G_opt @ x - h_opt) ** 2) + x @ x),
                np.zeros(null_dim), method="SLSQP", options={"ftol": 1e-20, "maxiter": 5_000})
            x_opt = sol2.x
        theta = theta_p + N_mat @ x_opt

    D_fitted = np.full(total_years, np.nan)
    for j in range(n):
        yr_arr = np.arange(anchor_years[j], anchor_years[j + 1] + 1, dtype=int)
        u_arr  = _u_transform(yr_arr, logarithm_reference_year)
        D_arr  = np.maximum(0.0, _D_theta(theta, j, u_arr))
        D_fitted[anchor_years[j]:anchor_years[j + 1] + 1] = D_arr

    return dict(D_fitted=D_fitted)
