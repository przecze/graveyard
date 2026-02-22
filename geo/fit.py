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


def u_from_year(year_arr: np.ndarray, x_mode: str, K_ref: float | None) -> np.ndarray:
    """Map calendar years to the fitting x-variable u.

    log_before_2026: u = ln(K_ref − year), clamped to 1e-9.
    year:            u = year (bare calendar year).
    """
    if x_mode == "year":
        return year_arr.astype(float)
    assert K_ref is not None
    return np.log(np.maximum(K_ref - year_arr, 1e-9))


def period_cumulative_deaths(data: dict, anchor_years: list[int]) -> np.ndarray:
    """Target cumulative deaths per period from data anchors."""
    n = len(anchor_years) - 1
    deaths_tgt = np.zeros(n, dtype=float)
    for j in range(n):
        start_year = anchor_years[j]
        end_row = data[anchor_years[j + 1]]
        pop_start = 0.0 if anchor_years[j + 1] == -8_000 else float(data[start_year].get("pop", 0.0))
        deaths_tgt[j] = float(end_row["cumulative_births"]) - float(end_row.get("pop", 0.0)) + pop_start
    return deaths_tgt


def fit_deaths_direct(
    data: dict,
    D_start: float | None = None,
    D_end: float | None = None,
    dDdy_start: float | None = None,
    d2Ddy2_start: float | None = None,
    dDdy_end: float | None = None,
    d2Ddy2_end: float | None = None,
    x_mode: str = "log_before_2026",
) -> dict:
    """Fit deaths/yr as piecewise polynomial (cubic/quartic) in year-space.

    Cubic (4-param, degree 3) on most periods.
    Quartic (5-param, degree 4) on 1200→1650 and the last two periods.
    C2 seam continuity + 6 endpoint constraints → 4n+3 = n_params.
    Solved as a linear system (instant).
    """
    if x_mode not in X_MODE_K_REF:
        raise ValueError(f"x_mode must be one of {list(X_MODE_K_REF)}; got {x_mode!r}")
    years = sorted(data.keys())
    n = len(years) - 1
    anchor_years = years

    t0 = anchor_years[0]
    K_ref = X_MODE_K_REF[x_mode]
    K = (K_ref - t0) if K_ref is not None else None
    T_cumul = np.array([y - t0 for y in anchor_years])
    T_periods = np.diff(T_cumul)
    deaths_tgt = period_cumulative_deaths(data, anchor_years)
    _anc_yr = np.array(anchor_years)
    u_bnd = u_from_year(_anc_yr, x_mode, K_ref)

    is_log_mode = x_mode != "year"
    mono_factor = 1.0 if is_log_mode else -1.0

    # Quartic periods: 1200→1650 and the last two
    quartic_period_idx: set[int] = set()
    for j in range(n):
        if anchor_years[j] == 1200 and anchor_years[j + 1] == 1650:
            quartic_period_idx.add(j)
    if n >= 2:
        quartic_period_idx.add(n - 2)
        quartic_period_idx.add(n - 1)
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
        u_j   = u_from_year(yr_j, x_mode, K_ref)
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
        if x_mode == "year":
            return 1.0, 0.0
        assert K_ref is not None
        rem = max(K_ref - y, 1e-9)
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
        return u_from_year(np.arange(anchor_years[j], anchor_years[j + 1] + 1), x_mode, K_ref)

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
        mono_ok = all(
            not np.any(mono_factor * _dDdu_theta(theta, j, _u_j(j)) > MONO_TOL)
            for j in range(n)
        )
    else:
        # Build constraint matrices from integer years.
        # Degree-3/4 polynomials need only ~30 pts/period to bound monotonicity for
        # the optimizer; stride long periods so G rows stay tractable.
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
        mono_ok = sol.success and np.all(G_full @ sol.x <= h_full + MONO_TOL)
        if not mono_ok:
            lam = 1e8
            sol2 = minimize(
                lambda x: float(lam * np.sum(np.maximum(0.0, G_opt @ x - h_opt) ** 2) + x @ x),
                np.zeros(null_dim), method="SLSQP", options={"ftol": 1e-20, "maxiter": 5_000})
            x_opt = sol2.x
            mono_ok = bool(np.maximum(0.0, G_full @ x_opt - h_full).max() <= MONO_TOL)
        theta = theta_p + N_mat @ x_opt

    periods = []
    deaths_model = np.zeros(n)
    for j in range(n):
        yr_arr = np.arange(anchor_years[j], anchor_years[j + 1] + 1, dtype=int)
        u_arr    = u_from_year(yr_arr, x_mode, K_ref)
        D_arr    = np.maximum(0.0, _D_theta(theta, j, u_arr))
        deaths_model[j] = np.trapezoid(_D_theta(theta, j, u_arr), yr_arr)
        periods.append({"yr_arr": yr_arr, "u_arr": u_arr, "D_arr": D_arr})

    return dict(
        periods=periods,
        deaths_tgt=deaths_tgt,
        deaths_model=deaths_model,
        anchor_years=anchor_years,
        theta=theta,
        mono_ok=mono_ok,
        K=K,
        K_ref=K_ref,
        x_mode=x_mode,
        T_cumul=T_cumul,
        u_bnd=u_bnd,
        n=n,
        quartic_periods=[(anchor_years[j], anchor_years[j + 1]) for j in sorted(quartic_period_idx)],
    )
