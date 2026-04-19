# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""
Piecewise constant-curvature fit for deaths/yr.

Each period is either:
  - 3-parameter: a * exp(c * x) + b * exp(-c * x)
  - 4-parameter: a * exp(c * x) + b * exp(-d * x)

No UI dependencies. Used by death_model.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_Z_CLIP = 150.0


def _u_transform(year_arr: np.ndarray, logarithm_reference_year: float | None) -> np.ndarray:
    """u = t (identity) when logarithm_reference_year is None, else u = ln(ref − t)."""
    if logarithm_reference_year is None:
        return year_arr.astype(float)
    return np.log(np.maximum(logarithm_reference_year - year_arr, 1e-9))


def _param_size_for_period(j: int, four_param_period_idx: set[int]) -> int:
    return 4 if j in four_param_period_idx else 3


def _unpack_period_params(theta_vec: np.ndarray, sl: slice, n_params: int) -> tuple[float, float, float, float]:
    p = theta_vec[sl]
    if n_params == 3:
        a, b_amp, log_c = float(p[0]), float(p[1]), float(p[2])
        log_d = log_c
    else:
        a, b_amp, log_c, log_d = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    log_c = float(np.clip(log_c, -16.0, 2.0))
    log_d = float(np.clip(log_d, -16.0, 2.0))
    return a, b_amp, float(np.exp(log_c)), float(np.exp(log_d))


def _D_period(theta_vec: np.ndarray, sl: slice, n_params: int, u: np.ndarray | float) -> np.ndarray | float:
    a, b_amp, c, d = _unpack_period_params(theta_vec, sl, n_params)
    return a * np.exp(np.clip(c * u, -_Z_CLIP, _Z_CLIP)) + b_amp * np.exp(np.clip(-d * u, -_Z_CLIP, _Z_CLIP))


def _dDdu_period(theta_vec: np.ndarray, sl: slice, n_params: int, u: np.ndarray | float) -> np.ndarray | float:
    a, b_amp, c, d = _unpack_period_params(theta_vec, sl, n_params)
    return a * c * np.exp(np.clip(c * u, -_Z_CLIP, _Z_CLIP)) - b_amp * d * np.exp(np.clip(-d * u, -_Z_CLIP, _Z_CLIP))


def _d2Ddu2_period(theta_vec: np.ndarray, sl: slice, n_params: int, u: np.ndarray | float) -> np.ndarray | float:
    a, b_amp, c, d = _unpack_period_params(theta_vec, sl, n_params)
    return (a * (c ** 2) * np.exp(np.clip(c * u, -_Z_CLIP, _Z_CLIP))
            + b_amp * (d ** 2) * np.exp(np.clip(-d * u, -_Z_CLIP, _Z_CLIP)))


_PROGRESS_LOG_EVERY = 50


def fit_piecewise_constant_curvature(
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
    run_solver: bool = True,
    max_nfev: int = 500,
    progress_callback: Callable[[int, float], None] | None = None,
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
        # Half-open: include first year, exclude last year. Example: 1200→1650 uses 1200..1649.
        yr_j = np.arange(anchor_years[j], anchor_years[j + 1], dtype=int)
        t_j = yr_j - t0
        u_j = _u_transform(yr_j, logarithm_reference_year)
        period_grids.append((t_j, u_j))

    n_extra = sum(v is not None for v in [D_start, D_end, dDdy_start, d2Ddy2_start, dDdy_end, d2Ddy2_end])
    # n period integrals + (n-1) C0 seams + (n-1) C1 seams + endpoint constraints
    n_constr = n + (n - 1) + (n - 1) + n_extra

    def _sl(j: int) -> slice:
        return slice(int(period_offsets[j]), int(period_offsets[j + 1]))

    # Evaluate each period in local u-coordinates to avoid huge exponent arguments.
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
            # Cap rates to keep exp terms within clip range on large-u periods.
            k_cap = max(1e-9, min(np.exp(2.0), (_Z_CLIP - 1.0) / max_x))
            k0 = min(k0, k_cap)
            x_left = 0.0
            x_right = float(u_bnd[j + 1] - u_bnd[j])
            D_left_tgt, D_right_tgt = mean_level, mean_level
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
                k_pos = k0
                k_neg = k0

            if n_p == 4:
                # Solve amplitudes from both edge targets so anchors are met on large-u segments.
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

    # ── nonlinear solver ──────────────────────────────────────────────────────
    # Residual scales derived from data: average deaths/yr per period.
    mean_D = deaths_tgt / np.maximum(period_year_counts, 1.0)

    def _residuals(theta: np.ndarray) -> np.ndarray:
        res = np.empty(n_constr, dtype=float)
        k = 0

        # Integral constraints: ∫D_j dt = deaths_tgt[j]
        for j in range(n):
            _, u_j = period_grids[j]
            integral = float(np.sum(_D_j(theta, j, u_j)))
            res[k] = (integral - deaths_tgt[j]) / deaths_tgt[j]
            k += 1

        # C0 seam continuity: D_j(right) = D_{j+1}(left)
        for j in range(n - 1):
            seam_u  = float(u_bnd[j + 1])
            D_left  = float(_D_j(theta, j,     seam_u))
            D_right = float(_D_j(theta, j + 1, seam_u))
            scale   = max(0.5 * (mean_D[j] + mean_D[j + 1]), 1.0)
            res[k]  = (D_left - D_right) / scale
            k += 1

        # C1 seam continuity: D'_j(right) = D'_{j+1}(left)
        for j in range(n - 1):
            seam_u   = float(u_bnd[j + 1])
            dD_left  = float(_dDdu_j(theta, j,     seam_u))
            dD_right = float(_dDdu_j(theta, j + 1, seam_u))
            T_avg    = 0.5 * (period_year_counts[j] + period_year_counts[j + 1])
            scale    = max(0.5 * (mean_D[j] + mean_D[j + 1]) / T_avg, 1.0)
            res[k]   = (dD_left - dD_right) / scale
            k += 1

        # Endpoint constraints
        u0, uN = float(u_bnd[0]), float(u_bnd[n])
        T0, TN = period_year_counts[0], period_year_counts[n - 1]
        if D_start is not None:
            res[k] = (float(_D_j(theta, 0, u0)) - D_start) / max(abs(D_start), 1.0)
            k += 1
        if dDdy_start is not None:
            scale  = max(abs(dDdy_start), mean_D[0] / T0, 1.0)
            res[k] = (float(_dDdu_j(theta, 0, u0)) - dDdy_start) / scale
            k += 1
        if d2Ddy2_start is not None:
            scale  = max(abs(d2Ddy2_start), mean_D[0] / T0 ** 2, 1.0)
            res[k] = (float(_d2Ddu2_j(theta, 0, u0)) - d2Ddy2_start) / scale
            k += 1
        if D_end is not None:
            res[k] = (float(_D_j(theta, n - 1, uN)) - D_end) / max(abs(D_end), 1.0)
            k += 1
        if dDdy_end is not None:
            scale  = max(abs(dDdy_end), mean_D[n - 1] / TN, 1.0)
            res[k] = (float(_dDdu_j(theta, n - 1, uN)) - dDdy_end) / scale
            k += 1
        if d2Ddy2_end is not None:
            scale  = max(abs(d2Ddy2_end), mean_D[n - 1] / TN ** 2, 1.0)
            res[k] = (float(_d2Ddu2_j(theta, n - 1, uN)) - d2Ddy2_end) / scale
            k += 1

        return res

    if run_solver:
        from scipy.optimize import least_squares
        _nfev = [0]
        _last_logged = [0]

        def _residuals_logged(theta: np.ndarray) -> np.ndarray:
            res = _residuals(theta)
            _nfev[0] += 1
            if progress_callback is not None and (_nfev[0] - _last_logged[0]) >= _PROGRESS_LOG_EVERY:
                progress_callback(_nfev[0], float(0.5 * np.dot(res, res)))
                _last_logged[0] = _nfev[0]
            return res

        sol = least_squares(
            _residuals_logged,
            theta0,
            bounds=(lower, upper),
            method="trf",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-10,
            max_nfev=max_nfev,
            verbose=0,
        )
        theta_fitted   = sol.x
        solver_success = bool(sol.cost < 1e-8)
        solver_message = sol.message
        solver_cost    = float(sol.cost)
    else:
        theta_fitted   = theta0
        solver_success = False
        solver_message = "Solver not run (init only)."
        solver_cost    = float(np.sum(_residuals(theta0) ** 2)) / 2.0

    # ── build output arrays ───────────────────────────────────────────────────
    D_init   = np.full(total_years, np.nan)
    D_fitted = np.full(total_years, np.nan)
    init_edge_report: list[dict[str, float | int]] = []
    init_seam_report: list[dict[str, float | int]] = []

    for j in range(n):
        yr_arr = np.arange(anchor_years[j], anchor_years[j + 1] + 1, dtype=int)
        u_arr  = _u_transform(yr_arr, logarithm_reference_year)
        D_init_arr   = _D_period(theta0,       _sl(j), int(period_param_size[j]), _u_rel(j, u_arr))
        D_fitted_arr = _D_period(theta_fitted, _sl(j), int(period_param_size[j]), _u_rel(j, u_arr))
        D_init  [anchor_years[j]:anchor_years[j + 1] + 1] = D_init_arr
        D_fitted[anchor_years[j]:anchor_years[j + 1] + 1] = D_fitted_arr

        u_left  = float(u_arr[0])
        u_right = float(u_arr[-1])
        init_left  = float(_D_j(theta0, j, u_left))
        init_right = float(_D_j(theta0, j, u_right))
        left_tgt = right_tgt = float("nan")
        if init_edge_values is not None and j < len(init_edge_values):
            left_tgt  = float(init_edge_values[j][0])
            right_tgt = float(init_edge_values[j][1])
        init_edge_report.append({
            "period_idx": int(j),
            "start_t":    int(anchor_years[j]),
            "end_t":      int(anchor_years[j + 1]),
            "n_params":   int(period_param_size[j]),
            "target_left":  left_tgt,
            "target_right": right_tgt,
            "init_left":    init_left,
            "init_right":   init_right,
            "init_min":  float(np.nanmin(D_init_arr)),
            "init_max":  float(np.nanmax(D_init_arr)),
            "err_left":  float(init_left  - left_tgt)  if np.isfinite(left_tgt)  else float("nan"),
            "err_right": float(init_right - right_tgt) if np.isfinite(right_tgt) else float("nan"),
        })

    for j in range(1, n):
        seam_u    = float(u_bnd[j])
        left_val  = float(_D_j(theta0, j - 1, seam_u))
        right_val = float(_D_j(theta0, j,     seam_u))
        init_seam_report.append({
            "seam_idx":        int(j),
            "seam_t":          int(anchor_years[j]),
            "left_period_idx": int(j - 1),
            "right_period_idx":int(j),
            "left_val":  left_val,
            "right_val": right_val,
            "jump":      float(right_val - left_val),
        })

    return dict(
        D_init=D_init,
        D_fitted=D_fitted,
        n_params=n_params,
        n_constraints=n_constr,
        solver_success=solver_success,
        solver_message=solver_message,
        solver_cost=solver_cost,
        init_edge_report=init_edge_report,
        init_seam_report=init_seam_report,
    )
