# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Piecewise constant-curvature initialization for deaths/yr.

Each period is either:
  - 3-parameter: a * exp(c * x) + b * exp(-c * x)
  - 4-parameter: a * exp(c * x) + b * exp(-d * x)

No UI dependencies. Used by death_model. Fitting (nonlinear solve) is not implemented.
"""

from __future__ import annotations

import numpy as np

_Z_CLIP = 150.0


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


def _du_dy(y: int, logarithm_reference_year: float | None) -> tuple[float, float]:
    if logarithm_reference_year is None:
        return 1.0, 0.0
    rem = max(logarithm_reference_year - y, 1e-9)
    return -1.0 / rem, -1.0 / (rem * rem)


def _param_size_for_period(j: int, four_param_period_idx: set[int]) -> int:
    return 4 if j in four_param_period_idx else 3


def _unpack_period_params(theta_vec: np.ndarray, sl: slice, n_params: int) -> tuple[float, float, float, float]:
    p = theta_vec[sl]
    if n_params == 3:
        # 3 params: [a, b, log_c], with d == c.
        a, b_amp, log_c = float(p[0]), float(p[1]), float(p[2])
        log_d = log_c
    else:
        # 4 params: [a, b, log_c, log_d]
        a, b_amp, log_c, log_d = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    log_c = float(np.clip(log_c, -16.0, 2.0))
    log_d = float(np.clip(log_d, -16.0, 2.0))
    c = float(np.exp(log_c))
    d = float(np.exp(log_d))
    return a, b_amp, c, d


def _D_period(theta_vec: np.ndarray, sl: slice, n_params: int, u: np.ndarray | float) -> np.ndarray | float:
    a, b_amp, c, d = _unpack_period_params(theta_vec, sl, n_params)
    z_pos = np.clip(c * u, -_Z_CLIP, _Z_CLIP)
    z_neg = np.clip(-d * u, -_Z_CLIP, _Z_CLIP)
    return a * np.exp(z_pos) + b_amp * np.exp(z_neg)


def _dDdu_period(theta_vec: np.ndarray, sl: slice, n_params: int, u: np.ndarray | float) -> np.ndarray | float:
    a, b_amp, c, d = _unpack_period_params(theta_vec, sl, n_params)
    z_pos = np.clip(c * u, -_Z_CLIP, _Z_CLIP)
    z_neg = np.clip(-d * u, -_Z_CLIP, _Z_CLIP)
    return a * c * np.exp(z_pos) - b_amp * d * np.exp(z_neg)


def _d2Ddu2_period(theta_vec: np.ndarray, sl: slice, n_params: int, u: np.ndarray | float) -> np.ndarray | float:
    a, b_amp, c, d = _unpack_period_params(theta_vec, sl, n_params)
    z_pos = np.clip(c * u, -_Z_CLIP, _Z_CLIP)
    z_neg = np.clip(-d * u, -_Z_CLIP, _Z_CLIP)
    return a * (c ** 2) * np.exp(z_pos) + b_amp * (d ** 2) * np.exp(z_neg)


def _fit_piecewise_constant_curvature(
    cumulated_D: dict[tuple[int, int], float],
    four_param_period_ids: list[int],
    total_years: int,
    D_start: float | None = None,
    D_end: float | None = None,
    dDdy_start: float | None = None,
    d2Ddy2_start: float | None = None,
    dDdy_end: float | None = None,
    d2Ddy2_end: float | None = None,
    logarithm_reference_year: float | None = None,
    init_edge_values: list[tuple[float, float]] | None = None,
    init_x0_t: float = 0.0,
) -> dict:
    anchor_years = sorted({t for pair in cumulated_D for t in pair})
    n = len(anchor_years) - 1
    deaths_tgt = np.array([cumulated_D[(anchor_years[j], anchor_years[j + 1])] for j in range(n)], dtype=float)

    t0 = anchor_years[0]
    _anc_yr = np.array(anchor_years, dtype=int)
    u_bnd = _u_transform(_anc_yr, logarithm_reference_year)
    four_param_period_idx = set(four_param_period_ids)
    period_param_size = np.array(
        [_param_size_for_period(j, four_param_period_idx) for j in range(n)],
        dtype=int,
    )

    period_offsets = np.zeros(n + 1, dtype=int)
    for j in range(n):
        period_offsets[j + 1] = period_offsets[j] + int(period_param_size[j])
    n_params = int(period_offsets[-1])

    period_grids: list[tuple[np.ndarray, np.ndarray]] = []
    period_year_counts = np.array(
        [max(anchor_years[j + 1] - anchor_years[j], 1) for j in range(n)],
        dtype=float,
    )
    for j in range(n):
        # Discrete period "integral": include first year, exclude last year.
        # Example: 1200→1650 uses years 1200..1649.
        yr_j = np.arange(anchor_years[j], anchor_years[j + 1], dtype=int)
        t_j = yr_j - t0
        u_j = _u_transform(yr_j, logarithm_reference_year)
        period_grids.append((t_j, u_j))

    n_extra = (
        (1 if D_start is not None else 0)
        + (1 if D_end is not None else 0)
        + (1 if dDdy_start is not None else 0)
        + (1 if d2Ddy2_start is not None else 0)
        + (1 if dDdy_end is not None else 0)
        + (1 if d2Ddy2_end is not None else 0)
    )
    # Constraints:
    #   n period integrals
    #   (n-1) C0 seam equalities
    #   (n-1) C1 seam equalities
    #   + endpoint constraints
    n_constr = n + (n - 1) + (n - 1) + n_extra

    def _sl(j: int) -> slice:
        return slice(int(period_offsets[j]), int(period_offsets[j + 1]))

    # Numerical stabilization: evaluate each period in local u-coordinates
    # (u_rel = u_abs - u_ref_j) to avoid huge exponent arguments/amplitudes.
    period_u_ref = np.array([float(u_bnd[j]) for j in range(n)], dtype=float)

    def _u_rel(j: int, u_abs: np.ndarray | float) -> np.ndarray:
        return np.asarray(u_abs, dtype=float) - period_u_ref[j]

    def _D_j(theta_vec: np.ndarray, j: int, u_abs: np.ndarray | float) -> np.ndarray | float:
        return _D_period(theta_vec, _sl(j), int(period_param_size[j]), _u_rel(j, u_abs))

    def _dDdu_j(theta_vec: np.ndarray, j: int, u_abs: np.ndarray | float) -> np.ndarray | float:
        return _dDdu_period(theta_vec, _sl(j), int(period_param_size[j]), _u_rel(j, u_abs))

    def _d2Ddu2_j(theta_vec: np.ndarray, j: int, u_abs: np.ndarray | float) -> np.ndarray | float:
        return _d2Ddu2_period(theta_vec, _sl(j), int(period_param_size[j]), _u_rel(j, u_abs))

    if n_params != n_constr:
        raise ValueError(
            f"Underdetermined/overdetermined system: {n_params} params vs {n_constr} constraints. "
            f"Adjust four_param_period_ids (currently {sorted(four_param_period_idx)})."
        )

    lower = np.full(n_params, -1e9, dtype=float)
    upper = np.full(n_params, 1e9, dtype=float)

    def _init_theta(log_b_init: float) -> np.ndarray:
        theta0 = np.zeros(n_params, dtype=float)

        for j in range(n):
            sl = _sl(j)
            n_p = int(period_param_size[j])
            _t_j, u_j = period_grids[j]
            mean_level = max(deaths_tgt[j] / period_year_counts[j], 1e-6)
            k0 = float(np.exp(log_b_init))
            u_rel_j = _u_rel(j, u_j)
            max_x = max(float(np.max(np.abs(u_rel_j))), 1e-9)
            # Keep initialization rates inside runtime clip range to avoid
            # flatlined exp terms on large absolute-u periods.
            k_cap = max(1e-9, min(np.exp(2.0), (_Z_CLIP - 1.0) / max_x))
            k0 = min(k0, k_cap)
            # Initialize from exact boundary coordinates so edge targets map correctly.
            x_left = 0.0
            x_right = float(u_bnd[j + 1] - u_bnd[j])
            D_left_tgt, D_right_tgt = (mean_level, mean_level)
            if init_edge_values is not None and j < len(init_edge_values):
                D_left_tgt = float(init_edge_values[j][0])
                D_right_tgt = float(init_edge_values[j][1])
            eps = max(1e-9, 1e-9 * max(abs(D_left_tgt), abs(D_right_tgt), 1.0))
            yL = max(D_left_tgt, eps)
            yR = max(D_right_tgt, eps)
            dx = max(x_right - x_left, 1e-9)
            inc = yR > yL * (1.0 + 1e-12)
            dec = yL > yR * (1.0 + 1e-12)
            if inc:
                k = float(np.clip(np.log(yR / yL) / dx, 1e-9, k_cap))
                k_pos, k_neg = k, k
            elif dec:
                k = float(np.clip(np.log(yL / yR) / dx, 1e-9, k_cap))
                k_pos, k_neg = k, k
            else:
                # Near-flat case: symmetric small-slope initialization.
                k_pos = k0
                k_neg = k0

            # For 4-parameter periods, solve amplitudes from both edge targets
            # so left/right anchors are met even on large absolute-u segments.
            if n_p == 4:
                Epl = float(np.exp(np.clip(k_pos * x_left, -_Z_CLIP, _Z_CLIP)))
                Epr = float(np.exp(np.clip(k_pos * x_right, -_Z_CLIP, _Z_CLIP)))
                Enl = float(np.exp(np.clip(-k_neg * x_left, -_Z_CLIP, _Z_CLIP)))
                Enr = float(np.exp(np.clip(-k_neg * x_right, -_Z_CLIP, _Z_CLIP)))
                det = Epl * Enr - Epr * Enl
                if abs(det) > 1e-18:
                    amp_pos = (yL * Enr - yR * Enl) / det
                    amp_neg = (Epl * yR - Epr * yL) / det
                else:
                    amp_pos = 0.5 * yL * np.exp(-k_pos * x_left)
                    amp_neg = 0.5 * yL * np.exp(k_neg * x_left)
            else:
                if inc:
                    amp_pos = yL * np.exp(-k_pos * x_left)
                    amp_neg = 0.0
                elif dec:
                    amp_pos = 0.0
                    amp_neg = yL * np.exp(k_neg * x_left)
                else:
                    amp_pos = 0.5 * yL * np.exp(-k_pos * x_left)
                    amp_neg = 0.5 * yL * np.exp(k_neg * x_left)

            if n_p == 3:
                theta0[sl] = np.array([amp_pos, amp_neg, np.log(max(k_pos, 1e-12))], dtype=float)
                lower[sl] = np.array([-1e12, -1e12, -16.0], dtype=float)
                upper[sl] = np.array([1e12, 1e12, 2.0], dtype=float)
            else:
                theta0[sl] = np.array(
                    [amp_pos, amp_neg, np.log(max(k_pos, 1e-12)), np.log(max(k_neg, 1e-12))],
                    dtype=float,
                )
                lower[sl] = np.array([-1e12, -1e12, -16.0, -16.0], dtype=float)
                upper[sl] = np.array([1e12, 1e12, 2.0, 2.0], dtype=float)
        return theta0

    theta0 = _init_theta(-5.0)
    theta0 = np.minimum(np.maximum(theta0, lower + 1e-9), upper - 1e-9)

    # Initialization diagnostics (before nonlinear solve).
    D_init = np.full(total_years, np.nan)
    init_edge_report: list[dict[str, float | int]] = []
    init_seam_report: list[dict[str, float | int]] = []
    for j in range(n):
        yr_arr = np.arange(anchor_years[j], anchor_years[j + 1] + 1, dtype=int)
        u_arr = _u_transform(yr_arr, logarithm_reference_year)
        D_init_arr = _D_j(theta0, j, u_arr)
        D_init[anchor_years[j]:anchor_years[j + 1] + 1] = D_init_arr

        u_left = float(u_arr[0])
        u_right = float(u_arr[-1])
        init_left = float(_D_j(theta0, j, u_left))
        init_right = float(_D_j(theta0, j, u_right))
        left_tgt = float("nan")
        right_tgt = float("nan")
        if init_edge_values is not None and j < len(init_edge_values):
            left_tgt = float(init_edge_values[j][0])
            right_tgt = float(init_edge_values[j][1])
        init_edge_report.append({
            "period_idx": int(j),
            "start_t": int(anchor_years[j]),
            "end_t": int(anchor_years[j + 1]),
            "n_params": int(period_param_size[j]),
            "target_left": left_tgt,
            "target_right": right_tgt,
            "init_left": init_left,
            "init_right": init_right,
            "init_min": float(np.nanmin(D_init_arr)),
            "init_max": float(np.nanmax(D_init_arr)),
            "err_left": float(init_left - left_tgt) if np.isfinite(left_tgt) else float("nan"),
            "err_right": float(init_right - right_tgt) if np.isfinite(right_tgt) else float("nan"),
        })
    for j in range(1, n):
        seam_u = float(u_bnd[j])
        left_val = float(_D_j(theta0, j - 1, seam_u))
        right_val = float(_D_j(theta0, j, seam_u))
        init_seam_report.append({
            "seam_idx": int(j),
            "seam_t": int(anchor_years[j]),
            "left_period_idx": int(j - 1),
            "right_period_idx": int(j),
            "left_val": left_val,
            "right_val": right_val,
            "jump": float(right_val - left_val),
        })

    return dict(
        D_init=D_init,
        D_fitted=D_init.copy(),
        n_params=n_params,
        n_constraints=n_constr,
        max_abs_residual=0.0,
        max_abs_scaled_residual=0.0,
        max_abs_equation_residual=0.0,
        max_abs_equation_scaled_residual=0.0,
        max_abs_positivity_violation=0.0,
        max_abs_edge_anchor_error=0.0,
        max_abs_c0_raw=0.0,
        max_abs_c0_clipped=0.0,
        max_abs_c1_raw=0.0,
        init_edge_report=init_edge_report,
        init_seam_report=init_seam_report,
        solver_success=False,
        solver_message="Init only (fitting not implemented).",
    )


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
    init_edge_values: list[tuple[float, float]] | None = None,
    init_x0_t: float = 0.0,
) -> dict:
    """Init-only entrypoint for piecewise constant-curvature basis.

    `quartic_period_ids` means periods with 4-parameter sinh+cosh basis.
    Other periods use 3-parameter sinh basis.
    Fitting (nonlinear solve) is not implemented.
    """
    return _fit_piecewise_constant_curvature(
        cumulated_D=cumulated_D,
        four_param_period_ids=quartic_period_ids,
        total_years=total_years,
        D_start=D_start,
        D_end=D_end,
        dDdy_start=dDdy_start,
        d2Ddy2_start=d2Ddy2_start,
        dDdy_end=dDdy_end,
        d2Ddy2_end=d2Ddy2_end,
        logarithm_reference_year=logarithm_reference_year,
        init_edge_values=init_edge_values,
        init_x0_t=init_x0_t,
    )


def solve_piecewise_polynomial(
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
    init_edge_values: list[tuple[float, float]] | None = None,
    init_x0_t: float = 0.0,
) -> dict:
    """Compatibility wrapper around the constant-curvature init."""
    return fit_piecewise_polynomial(
        cumulated_D=cumulated_D,
        quartic_period_ids=quartic_period_ids,
        total_years=total_years,
        D_start=D_start,
        D_end=D_end,
        dDdy_start=dDdy_start,
        d2Ddy2_start=d2Ddy2_start,
        dDdy_end=dDdy_end,
        d2Ddy2_end=d2Ddy2_end,
        logarithm_reference_year=logarithm_reference_year,
        init_edge_values=init_edge_values,
        init_x0_t=init_x0_t,
    )
