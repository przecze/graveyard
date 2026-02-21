# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""
Deaths/yr model — piecewise quadratic in log-time.

For n consecutive periods defined by the anchor years in data.json we fit

    D_j(t) = a_j + b_j·u(t) + c_j·u(t)²      deaths / yr

where  u(t) = ln(K − t),  t measured from the first anchor year,
K = 2026 − t₀  (years remaining until "now").

3n free parameters, 3n − 2 linear equality constraints
─────────────────────────────────────────────────────
  n   integral constraints : ∫ D_j dt = cumulative_deaths[j]
  n−1 value continuity     : D_{j−1}(u_j) = D_j(u_j)
  n−1 derivative continuity: dD_{j−1}/du  = dD_j/du   at each boundary

The 2 remaining degrees of freedom are resolved by minimising ‖x‖²
subject to G·x ≤ h, where the linear inequalities encode

    dD_j/du(u) ≤ 0   for all u in period j
                      ↔ dD_j/dt ≥ 0  (deaths/yr non-decreasing)

because du/dt = −1/(K−t) < 0.  dD_j/du is linear in u so checking at the
two boundary u-values of each period is both necessary and sufficient.

Period cumulative death constraints are derived from per-period births and
population carry-over:

    cumulative_deaths[j] = cumulative_births[end_j] - pop[end_j] + pop[start_j]

with a special-case baseline pop[start_j] = 0 when end_j is -8000.

Always run this module via `uv run`.
Standalone : uv run death_model.py [path/to/data.json] [--verbose]
Streamlit  : import death_model; death_model.render()
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq, null_space
from scipy.optimize import minimize

_DOCKER_PATH = Path("/data/data.json")
_LOCAL_PATH  = Path(__file__).parent.parent / "frontend" / "src" / "data.json"

N_GRID = 4_001    # output grid points per period
N_FINE = 40_001   # fine grid for moment integrals and cumulative check

# ── x-variable modes ───────────────────────────────────────────────────────────

_X_MODE_LABELS: dict[str, str] = {
    "log_before_2026": "log(years before 2026)  [current]",
    "year":            "bare year  (no log)",
    "log_before_1950": "log(years before 1950)",
}
_X_MODE_K_REF: dict[str, float | None] = {
    "log_before_2026": 2026.0,
    "log_before_1950": 1950.0,
    "year":            None,
}


def _u_from_year(year_arr: np.ndarray, x_mode: str, K_ref: float | None) -> np.ndarray:
    """Map calendar years to the fitting x-variable u.

    log_before_2026 / log_before_1950: u = ln(K_ref − year), clamped to 1e-9.
    year:                               u = year (bare calendar year).
    """
    if x_mode == "year":
        return year_arr.astype(float)
    assert K_ref is not None
    return np.log(np.maximum(K_ref - year_arr, 1e-9))


# OWID actual deaths/yr (UN World Population Prospects 2024), 1950–2023
_OWID_DEATHS: dict[int, int] = {
    1950: 48_486_892, 1951: 48_176_160, 1952: 47_383_364, 1953: 47_239_576,
    1954: 46_662_428, 1955: 46_635_656, 1956: 46_479_064, 1957: 46_880_776,
    1958: 46_518_264, 1959: 50_724_510, 1960: 54_612_440, 1961: 49_918_924,
    1962: 46_061_450, 1963: 46_913_210, 1964: 46_822_880, 1965: 48_213_548,
    1966: 47_841_524, 1967: 47_568_790, 1968: 47_629_250, 1969: 47_815_950,
    1970: 48_163_468, 1971: 49_384_108, 1972: 47_770_224, 1973: 47_573_276,
    1974: 47_478_096, 1975: 47_622_990, 1976: 47_707_256, 1977: 47_166_790,
    1978: 47_293_144, 1979: 46_963_884, 1980: 47_349_924, 1981: 47_470_790,
    1982: 47_659_344, 1983: 48_206_576, 1984: 48_430_624, 1985: 48_772_812,
    1986: 48_643_612, 1987: 48_657_012, 1988: 49_183_184, 1989: 49_178_264,
    1990: 49_794_344, 1991: 50_108_430, 1992: 50_281_870, 1993: 50_831_804,
    1994: 51_930_610, 1995: 51_406_890, 1996: 51_410_956, 1997: 51_432_010,
    1998: 51_853_504, 1999: 52_183_584, 2000: 52_240_370, 2001: 52_431_244,
    2002: 52_757_536, 2003: 53_077_070, 2004: 53_187_800, 2005: 53_390_436,
    2006: 53_281_776, 2007: 53_410_050, 2008: 53_987_020, 2009: 54_122_624,
    2010: 54_268_588, 2011: 54_581_440, 2012: 54_794_788, 2013: 55_092_620,
    2014: 55_545_320, 2015: 56_305_970, 2016: 56_756_910, 2017: 57_572_256,
    2018: 57_792_804, 2019: 58_354_932, 2020: 63_546_316, 2021: 69_728_100,
    2022: 62_278_628, 2023: 61_651_610,
}
# Sorted list of (year, deaths/yr) for convenient iteration
_OWID_SERIES: list[tuple[int, int]] = sorted(_OWID_DEATHS.items())


# ── data loading ───────────────────────────────────────────────────────────────

def _load_data(data_path: Path | str | None) -> tuple[dict, list[int]]:
    if data_path is None:
        data_path = _DOCKER_PATH if _DOCKER_PATH.exists() else _LOCAL_PATH
    with open(data_path) as f:
        raw = json.load(f)
    data = {int(k): v for k, v in raw.items()}
    return data, sorted(data.keys())


def _yfmt(y: int) -> str:
    return f"{-y} BCE" if y <= 0 else f"{y} CE"


def _period_cumulative_deaths(data: dict, anchor_years: list[int]) -> np.ndarray:
    """Build period cumulative death targets from data entries.

    Priority by period end-year:
      1) ``cumulative_deaths_estimated`` (explicit override)
      2) ``cumulative_births`` and population carry-over:
           deaths = births - pop_end + pop_start
      3) legacy ``cumulative`` (backward compatibility)
    """
    n = len(anchor_years) - 1
    deaths_tgt = np.zeros(n, dtype=float)

    for j in range(n):
        start_year = anchor_years[j]
        end_year = anchor_years[j + 1]
        end_row = data[end_year]

        if "cumulative_deaths_estimated" in end_row:
            deaths_tgt[j] = float(end_row["cumulative_deaths_estimated"])
            continue

        if "cumulative_births" in end_row:
            pop_start = 0.0 if end_year == -8000 else float(data[start_year].get("pop", 0.0))
            deaths_tgt[j] = float(end_row["cumulative_births"]) - float(end_row.get("pop", 0.0)) + pop_start
            continue

        if "cumulative" in end_row:
            deaths_tgt[j] = float(end_row["cumulative"])
            continue

        raise KeyError(
            f"Missing cumulative deaths source for year {end_year}. "
            "Expected one of: cumulative_deaths_estimated, cumulative_births, cumulative."
        )

    return deaths_tgt


# ── fitting ────────────────────────────────────────────────────────────────────

def fit_deaths_direct(
    data_path: Path | str | None = None,
    verbose: bool = False,
    data: dict | None = None,
    D_start: float | None = None,
    D_end: float | None = None,
    x_mode: str = "log_before_2026",
    mono_soft_slack_1200_1650: float = 0.0,
    cubic_1200_1650: bool = True,
    cubic_1900_1950: bool = False,
) -> dict:
    """
    Fit deaths/yr as piecewise quadratic in u (x-variable).

    Three x-variable modes are supported (``x_mode``):

    ``"log_before_2026"`` (default / current)
        u(year) = ln(2026 − year).  Decreasing as year increases, so
        dD/du ≤ 0 ↔ D non-decreasing.

    ``"year"``
        u(year) = year (bare calendar year).  Increasing, so
        dD/du ≥ 0 ↔ D non-decreasing.

    ``"log_before_1950"``
        u(year) = ln(1950 − year).  Like the 2026 variant but the
        singularity is at 1950 (clamped to 1e-9 at the boundary).

    Uses all anchor years present in data.json. For each period end-year, the
    fitter uses ``cumulative_deaths_estimated`` when present. Otherwise it
    derives cumulative deaths from ``cumulative_births`` and population
    carry-over (births - end-pop + start-pop), with start-pop forced to 0 when
    the period end-year is -8000.

    Parameters
    ----------
    data_path : path to data.json (ignored when ``data`` is supplied)
    verbose   : print diagnostics
    data      : pre-built {year: {...}} dict; if given, skips
                file loading entirely (useful when prepending synthetic periods)
    D_start   : if given, pin D = D_start at the first anchor year (consumes 1 DOF)
    D_end     : if given, pin D = D_end   at the last  anchor year (consumes 1 DOF)
    x_mode    : one of "log_before_2026", "year", "log_before_1950"
    mono_soft_slack_1200_1650
              : extra allowed monotonicity slack for the 1200→1650 period
                only (same units as dD/du constraints). 0 = default behavior.
    cubic_1200_1650
              : if True and anchors include 1200→1650 as a single period,
                add a u^3 term only in that period (adds one global DOF).
    cubic_1900_1950
              : if True and anchors include 1900→1950 as a single period,
                add a u^3 term only in that period (adds one global DOF).

    Returns
    -------
    dict with keys
        periods       list of per-period dicts: t_arr, yr_arr, u_arr, D_arr
        deaths_tgt    (n,) array — cumulative deaths targets
        deaths_model  (n,) array — fitted cumulative deaths
        anchor_years  list of anchor years (length n+1)
        theta         (3n,) coefficient vector [a0,b0,c0, a1,b1,c1, …]
        mono_ok       bool — was hard monotonicity achieved?
        K, t0         log-time basis constants (K = K_ref − t0; None for year mode)
        K_ref         reference year for log modes (None for year mode)
        x_mode        the x_mode used
        T_cumul       (n+1,) cumulative times from t0
        u_bnd         (n+1,) u-values at each anchor year
    """
    if x_mode not in _X_MODE_K_REF:
        raise ValueError(f"x_mode must be one of {list(_X_MODE_K_REF)}; got {x_mode!r}")
    if data is None:
        data, years = _load_data(data_path)
    else:
        years = sorted(data.keys())
    n            = len(years) - 1
    anchor_years = years

    t0        = float(anchor_years[0])
    K_ref     = _X_MODE_K_REF[x_mode]
    K         = (K_ref - t0) if K_ref is not None else None
    T_cumul   = np.array([float(y - t0) for y in anchor_years])   # shape (n+1,)
    T_periods = np.diff(T_cumul)                                   # shape (n,)

    # cumulative values per period (used directly as the curve targets)
    deaths_tgt = _period_cumulative_deaths(data, anchor_years)

    # u at each anchor year
    _anc_yr = np.array(anchor_years, dtype=float)
    u_bnd   = _u_from_year(_anc_yr, x_mode, K_ref)   # shape (n+1,)

    # For log modes u decreases as year increases (du/dt < 0); for year mode it increases.
    # monotonicity factor: mono_factor * dD/du ≤ 0  ↔  D non-decreasing in both cases.
    is_log_mode  = x_mode != "year"
    mono_factor  = 1.0 if is_log_mode else -1.0

    if verbose:
        print(f"n={n} periods,  t0={t0},  x_mode={x_mode!r},  K_ref={K_ref}")
        print(f"u range: {u_bnd[-1]:.4f} … {u_bnd[0]:.4f}\n")

    if cubic_1200_1650 and cubic_1900_1950:
        raise ValueError(
            "Choose at most one cubic segment: 1200→1650 or 1900→1950."
        )

    cubic_segment = None
    if cubic_1200_1650:
        cubic_segment = (1200, 1650)
    elif cubic_1900_1950:
        cubic_segment = (1900, 1950)

    # Optional mixed-order period: cubic only for one selected segment.
    cubic_period_idx = None
    if cubic_segment is not None:
        seg_start, seg_end = cubic_segment
        for j in range(n):
            if anchor_years[j] == seg_start and anchor_years[j + 1] == seg_end:
                cubic_period_idx = j
                break
    has_cubic = cubic_period_idx is not None

    # ── moment integrals ─────────────────────────────────────────────────────
    # IU[j, k] = ∫_{T_cumul[j]}^{T_cumul[j+1]}  u(t)^k  dt
    IU = np.zeros((n, 4))
    for j in range(n):
        t_arr   = np.linspace(T_cumul[j], T_cumul[j + 1], N_FINE)
        u_arr   = _u_from_year(t_arr + t0, x_mode, K_ref)
        IU[j, 0] = T_periods[j]
        IU[j, 1] = np.trapezoid(u_arr,       t_arr)
        IU[j, 2] = np.trapezoid(u_arr ** 2,  t_arr)
        IU[j, 3] = np.trapezoid(u_arr ** 3,  t_arr)

    # ── linear constraint system A @ theta = b ───────────────────────────────
    # theta = [a_0, b_0, c_0,  a_1, b_1, c_1,  …,  a_{n-1}, b_{n-1}, c_{n-1}]
    n_params  = 3 * n + (1 if has_cubic else 0)
    d_idx     = (3 * n) if has_cubic else None
    n_extra   = (1 if D_start is not None else 0) + (1 if D_end is not None else 0)
    n_constr  = 3 * n - 2 + n_extra
    expected_null_dim = 2 - n_extra + (1 if has_cubic else 0)

    A = np.zeros((n_constr, n_params))
    b = np.zeros(n_constr)

    row = 0

    # n integral constraints
    for j in range(n):
        A[row, 3 * j : 3 * j + 3] = IU[j, :3]
        if has_cubic and j == cubic_period_idx:
            A[row, d_idx] = IU[j, 3]
        b[row] = deaths_tgt[j]
        row += 1

    # n−1 value-continuity at interior boundaries
    for j in range(1, n):
        u = u_bnd[j]
        A[row, 3 * (j - 1) : 3 * (j - 1) + 3]  = [ 1.0,  u,  u ** 2]
        A[row, 3 * j        : 3 * j        + 3] += [-1.0, -u, -u ** 2]
        if has_cubic and (j - 1) == cubic_period_idx:
            A[row, d_idx] += u ** 3
        if has_cubic and j == cubic_period_idx:
            A[row, d_idx] += -u ** 3
        b[row] = 0.0
        row += 1

    # n−1 derivative-continuity (dD/du) at interior boundaries
    for j in range(1, n):
        u = u_bnd[j]
        A[row, 3 * (j - 1) + 1 : 3 * (j - 1) + 3]  = [ 1.0,  2.0 * u]
        A[row, 3 * j        + 1 : 3 * j        + 3] += [-1.0, -2.0 * u]
        if has_cubic and (j - 1) == cubic_period_idx:
            A[row, d_idx] += 3.0 * u ** 2
        if has_cubic and j == cubic_period_idx:
            A[row, d_idx] += -3.0 * u ** 2
        b[row] = 0.0
        row += 1

    # optional: pin D at the first anchor year
    if D_start is not None:
        u = u_bnd[0]
        A[row, 0:3] = [1.0, u, u ** 2]
        if has_cubic and cubic_period_idx == 0:
            A[row, d_idx] = u ** 3
        b[row] = D_start
        row += 1

    # optional: pin D at the last anchor year
    if D_end is not None:
        u = u_bnd[n]
        A[row, 3 * (n - 1) : 3 * n] = [1.0, u, u ** 2]
        if has_cubic and cubic_period_idx == (n - 1):
            A[row, d_idx] = u ** 3
        b[row] = D_end
        row += 1

    assert row == n_constr, f"row={row} != n_constr={n_constr}"

    # ── scale rows to improve conditioning ───────────────────────────────────
    D_scale = np.median(deaths_tgt / T_periods)
    A_sc = A.copy()
    b_sc = b.copy()
    for j in range(n):                          # integral rows
        A_sc[j] /= deaths_tgt[j]
        b_sc[j] /= deaths_tgt[j]
    for j in range(n, n_constr):                # continuity + endpoint rows
        A_sc[j] /= D_scale
        b_sc[j] /= D_scale

    # ── particular solution + null space ─────────────────────────────────────
    theta_p, _res, rank, sv = lstsq(A_sc, b_sc)
    N_mat = null_space(A_sc)   # numeric null space (dimension can vary with conditioning)
    null_dim = N_mat.shape[1]

    if verbose:
        print(f"Constraint rank: {rank}/{n_constr}  null-space dim: {null_dim}")
        print(f"Condition (σ_min={sv[-1]:.3e}, σ_max={sv[0]:.3e})  "
              f"κ={sv[0]/sv[-1]:.2e}\n")
        if null_dim != expected_null_dim:
            print(
                f"Note: numeric null-space dim ({null_dim}) differs from expected "
                f"model DOF ({expected_null_dim}); proceeding with numeric basis."
            )

    def _D_theta(theta_vec: np.ndarray, j: int, u: np.ndarray | float) -> np.ndarray | float:
        out = theta_vec[3*j] + theta_vec[3*j+1] * u + theta_vec[3*j+2] * (u ** 2)
        if has_cubic and j == cubic_period_idx:
            out = out + theta_vec[d_idx] * (u ** 3)
        return out

    def _dDdu_theta(theta_vec: np.ndarray, j: int, u: np.ndarray | float) -> np.ndarray | float:
        out = theta_vec[3*j+1] + 2.0 * theta_vec[3*j+2] * u
        if has_cubic and j == cubic_period_idx:
            out = out + 3.0 * theta_vec[d_idx] * (u ** 2)
        return out

    def _D_null_vec(j: int, u: float) -> np.ndarray:
        out = N_mat[3*j, :] + N_mat[3*j+1, :] * u + N_mat[3*j+2, :] * (u ** 2)
        if has_cubic and j == cubic_period_idx:
            out = out + N_mat[d_idx, :] * (u ** 3)
        return out

    def _dDdu_null_vec(j: int, u: float) -> np.ndarray:
        out = N_mat[3*j+1, :] + 2.0 * N_mat[3*j+2, :] * u
        if has_cubic and j == cubic_period_idx:
            out = out + 3.0 * N_mat[d_idx, :] * (u ** 2)
        return out

    # ── monotonicity QP (only when free DOF remain) ───────────────────────────
    # D_j(t) non-decreasing  ↔  dD_j/du(u) ≤ 0  on [u_{j+1}, u_j]
    # Since dD_j/du(u) = b_j + 2·c_j·u  is linear in u,
    # checking at the two boundary u-values is both necessary and sufficient.

    MONO_TOL = 1000.0

    if null_dim == 0:
        # System fully determined — no free DOF to optimise
        theta   = theta_p
        x_opt   = np.zeros(0)
        # Still evaluate monotonicity as a diagnostic
        mono_ok = True
        for j in range(n):
            for u in (u_bnd[j], u_bnd[j + 1]):
                dDdu = float(_dDdu_theta(theta, j, u))
                if mono_factor * dDdu > MONO_TOL:
                    mono_ok = False
        if verbose and not mono_ok:
            print("Note: fully-determined solution is not monotone.\n")
    else:
        # Build G·x ≤ h: mono_factor * dD/du ≤ 0 at each period endpoint,
        # and positivity at each period endpoint.
        # For log modes mono_factor=+1: dD/du ≤ 0 (u decreasing → D rising).
        # For year mode mono_factor=−1: −dD/du ≤ 0 i.e. dD/du ≥ 0 (u increasing → D rising).
        G_rows, h_rows, slack_rows = [], [], []

        for j in range(n):
            period_slack = (
                float(mono_soft_slack_1200_1650)
                if anchor_years[j] == 1200 and anchor_years[j + 1] == 1650
                else 0.0
            )
            for u in (u_bnd[j], u_bnd[j + 1]):
                g   = mono_factor * _dDdu_null_vec(j, u)
                hv  = mono_factor * -float(_dDdu_theta(theta_p, j, u))
                G_rows.append(g); h_rows.append(hv); slack_rows.append(period_slack)

        for j in range(n):
            for u in (u_bnd[j], u_bnd[j + 1]):
                g   = -_D_null_vec(j, u)
                hv  = float(_D_theta(theta_p, j, u))
                G_rows.append(g); h_rows.append(hv); slack_rows.append(0.0)

        G       = np.array(G_rows)
        h       = np.array(h_rows)
        h_slack = h + np.array(slack_rows)

        EPS_HARD = 1.0
        sol = minimize(
            lambda x: float(np.dot(x, x)),
            x0=np.zeros(null_dim),
            jac=lambda x: 2.0 * x,
            method="SLSQP",
            constraints={"type": "ineq",
                         "fun": lambda x: h_slack + EPS_HARD - G @ x,
                         "jac": lambda x: -G},
            options={"ftol": 1e-15, "maxiter": 2000},
        )

        mono_ok = sol.success and np.all(G @ sol.x <= h_slack + MONO_TOL)
        x_opt   = sol.x

        if not mono_ok:
            lam = 1e8

            def soft_obj(x: np.ndarray) -> float:
                viol = np.maximum(0.0, G @ x - h_slack)
                return float(lam * np.dot(viol, viol) + np.dot(x, x))

            def soft_jac(x: np.ndarray) -> np.ndarray:
                viol = np.maximum(0.0, G @ x - h_slack)
                return 2.0 * lam * (G.T @ viol) + 2.0 * x

            sol2  = minimize(soft_obj, np.zeros(null_dim), jac=soft_jac,
                             method="SLSQP",
                             options={"ftol": 1e-20, "maxiter": 5000})
            x_opt = sol2.x
            viol  = np.maximum(0.0, G @ x_opt - h_slack)
            mono_ok = bool(viol.max() <= MONO_TOL)
            if verbose:
                print(f"Hard mono failed → soft penalty.  "
                      f"Max violation: {viol.max():.3e}  mono_ok={mono_ok}\n")

        theta = theta_p + N_mat @ x_opt

    if verbose:
        print(f"mono_ok={mono_ok}  x_opt={x_opt}")

    # ── per-period output curves ──────────────────────────────────────────────
    periods      = []
    deaths_model = np.zeros(n)

    for j in range(n):
        t_arr = np.linspace(T_cumul[j], T_cumul[j + 1], N_GRID)
        u_arr = _u_from_year(t_arr + t0, x_mode, K_ref)
        D_arr = np.maximum(0.0, _D_theta(theta, j, u_arr))

        # Cumulative integral on fine grid
        t_fine = np.linspace(T_cumul[j], T_cumul[j + 1], N_FINE)
        u_fine = _u_from_year(t_fine + t0, x_mode, K_ref)
        D_fine = _D_theta(theta, j, u_fine)
        deaths_model[j] = np.trapezoid(D_fine, t_fine)

        periods.append({
            "t_arr" : t_arr,
            "yr_arr": t_arr + t0,
            "u_arr" : u_arr,
            "D_arr" : D_arr,
        })

    if verbose:
        print(f"\n{'Period':<32}  {'Target':>14}  {'Model':>14}  {'Err%':>9}")
        for j in range(n):
            pct = (deaths_model[j] - deaths_tgt[j]) / deaths_tgt[j] * 100
            lbl = f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j + 1])}"
            print(f"  {lbl:<30}  {deaths_tgt[j]:>14.5e}  "
                  f"{deaths_model[j]:>14.5e}  {pct:>+9.4f}%")
        print()
        print("Deaths/yr at anchor years:")
        for j in range(n + 1):
            if j == 0:
                D_anc = _D_theta(theta, 0, u_bnd[j])
            elif j == n:
                D_anc = _D_theta(theta, n - 1, u_bnd[j])
            else:
                # use the left-side polynomial (continuity ensures same value)
                D_anc = _D_theta(theta, j - 1, u_bnd[j])
            print(f"  {_yfmt(anchor_years[j]):<12}: {D_anc:>14,.0f}  deaths/yr")
        print()
        # Derivative continuity sanity check
        print("dD/du continuity check at interior boundaries:")
        for j in range(1, n):
            u = u_bnd[j]
            dDdu_L = _dDdu_theta(theta, j - 1, u)
            dDdu_R = _dDdu_theta(theta, j, u)
            print(f"  boundary {_yfmt(anchor_years[j])}: "
                  f"left={dDdu_L:.4e}  right={dDdu_R:.4e}  "
                  f"diff={dDdu_R - dDdu_L:.2e}")
        print()
        # Monotonicity check
        mono_dir = "≤ 0" if is_log_mode else "≥ 0"
        print(f"Monotonicity (dD/du {mono_dir}; tol={MONO_TOL:.0f} → deaths/yr non-decreasing):")
        all_mono = True
        for j in range(n):
            t_chk = np.linspace(T_cumul[j], T_cumul[j+1], 1001)
            u_chk = _u_from_year(t_chk + t0, x_mode, K_ref)
            dDdu  = _dDdu_theta(theta, j, u_chk)
            ok    = bool((mono_factor * dDdu).max() <= MONO_TOL)
            if not ok:
                all_mono = False
            print(f"  period {j}  max(mono_factor·dD/du)={(mono_factor*dDdu).max():.3e}  ok={ok}")
        print(f"  → overall monotone: {all_mono}\n")

    return dict(
        periods=periods,
        deaths_tgt=deaths_tgt,
        deaths_model=deaths_model,
        anchor_years=anchor_years,
        theta=theta,
        mono_ok=mono_ok,
        K=K,
        K_ref=K_ref,
        t0=t0,
        x_mode=x_mode,
        T_cumul=T_cumul,
        u_bnd=u_bnd,
        n=n,
        cubic_period_idx=cubic_period_idx,
    )


# ── Streamlit colours ──────────────────────────────────────────────────────────

_COLOURS = [
    "#e31a1c", "#ff7f00", "#6a3d9a", "#33a02c", "#1f78b4",
    "#b15928", "#fb9a99", "#fdbf6f",
]


# ── Streamlit render ───────────────────────────────────────────────────────────

def render() -> None:
    import plotly.graph_objects as go
    import streamlit as st

    data_path = _DOCKER_PATH if _DOCKER_PATH.exists() else _LOCAL_PATH

    # ── x-variable mode ───────────────────────────────────────────────────────
    x_mode = st.radio(
        "X-variable (fitting basis)",
        options=list(_X_MODE_LABELS),
        format_func=_X_MODE_LABELS.__getitem__,
        horizontal=True,
        help=(
            "log_before_2026: u = ln(2026 − year) — the default; singularity at 2026.  "
            "year: u = year — bare calendar year, no transform.  "
            "log_before_1950: u = ln(1950 − year) — singularity pulled back to 1950."
        ),
    )

    # ── ancient period slider ─────────────────────────────────────────────────
    ancient_length = st.slider(
        "Ancient period length (years)",
        min_value=1_000,
        max_value=20_000,
        value=6_000,
        step=500,
        help="Duration of the pre-history period prepended before the first data anchor (-8000).",
    )
    data, _ = _load_data(data_path)
    yc             = ancient_length
    ancient_start  = -8000 - yc

    density_window_years = st.slider(
        "Ancient density window X (years)",
        min_value=1_000,
        max_value=min(6_000, yc),
        value=min(1_000, yc),
        step=100,
        help="Use only the first X years of the ancient era to compute ancient_density.",
    )
    mono_soft_slack_1200_1650 = st.slider(
        "1200 -> 1650 monotonicity softness",
        min_value=0.0,
        max_value=20_000.0,
        value=0.0,
        step=100.0,
        help=(
            "Extra slack applied only to monotonicity constraints in the 1200 -> 1650 period. "
            "Higher = softer penalty in that segment."
        ),
    )
    cubic_segment_option = st.radio(
        "Cubic-fit segment",
        options=["1200 -> 1650", "1900 -> 1950", "None"],
        index=0,
        horizontal=True,
        help=(
            "Adds a u^3 term only in the selected segment, giving one extra DOF "
            "while keeping D_start and D_end pinned."
        ),
    )
    cubic_1200_1650 = cubic_segment_option == "1200 -> 1650"
    cubic_1900_1950 = cubic_segment_option == "1900 -> 1950"

    # Prepend the ancient anchor year with zero baseline population.
    extended_data = {ancient_start: {"pop": 0, "cbr": 0}, **data}

    with st.spinner("Fitting deaths/yr piecewise quadratic…"):
        r = fit_deaths_direct(
            data=extended_data,
            D_start=0.0,
            D_end=float(_OWID_DEATHS[1950]),
            x_mode=x_mode,
            mono_soft_slack_1200_1650=mono_soft_slack_1200_1650,
            cubic_1200_1650=cubic_1200_1650,
            cubic_1900_1950=cubic_1900_1950,
        )

    anchor_years = r["anchor_years"]
    n            = r["n"]

    if not r["mono_ok"]:
        st.warning(
            "Hard monotonicity constraint infeasible for this dataset — "
            "soft-penalty solution used; deaths/yr may not be strictly non-decreasing."
        )

    _u_desc = {
        "log_before_2026": "u = ln(2026 − yr)",
        "year":            "u = yr  (bare year)",
        "log_before_1950": "u = ln(1950 − yr)",
    }[x_mode]
    cubic_desc = (
        "plus optional u³ only in 1200→1650"
        if cubic_1200_1650
        else "plus optional u³ only in 1900→1950"
        if cubic_1900_1950
        else "quadratic only (no optional u³ segment)"
    )
    st.subheader("Deaths / yr — piecewise quadratic")
    st.caption(
        f"D_j(t) = a_j + b_j·u + c_j·u² ({cubic_desc}),  {_u_desc}.  "
        "C¹ continuity at period boundaries.  "
        "D=0 pinned at start, D=OWID(1950) pinned at end.  "
        "Period constraints use cumulative_births and anchor populations."
    )

    # ── deaths/yr curve ───────────────────────────────────────────────────────
    col_logy, col_logx, col_norm = st.columns(3)
    with col_logy:
        log_y = st.toggle("Log Y axis", value=False)
    with col_logx:
        log_x = st.toggle("Log X axis", value=False)
    with col_norm:
        norm_circ = st.toggle("Normalise D", value=False)

    # ancient_density from first X years of fitted ancient D:
    # sum(first X yearly D values) / (π · X²)
    ancient_pr = r["periods"][0]
    ancient_yr = ancient_pr["yr_arr"]
    ancient_D = ancient_pr["D_arr"]
    x_years = density_window_years
    x_end_year = ancient_start + x_years
    x_mask = (ancient_yr >= ancient_start) & (ancient_yr < x_end_year)
    D_first_x = ancient_D[x_mask]
    if D_first_x.size == 0:
        D_first_x = ancient_D[:1]
        x_years = 1
    ancient_density = D_first_x.sum() / (np.pi * (x_years ** 2))

    # OWID extension arrays (1950 included for a seamless join with the model)
    _owid_yr = np.array([float(y) for y, _ in _OWID_SERIES])
    _owid_D  = np.array([float(d) for _, d in _OWID_SERIES])

    fig = go.Figure()
    for j, pr in enumerate(r["periods"]):
        label = f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j + 1])}"
        yr    = pr["yr_arr"][1:]                      # drop shared/pinned first point
        age   = yr - ancient_start
        D     = pr["D_arr"][1:] / (2.0 * np.pi * np.maximum(age, 1.0)) if norm_circ else pr["D_arr"][1:]
        y_vals = np.maximum(D, 1e-6) if log_y else D
        # log X: plot years-before-present (always positive) on log scale,
        # then reverse the axis so recent time stays on the right.
        x_vals   = 2026.0 - yr if log_x else yr
        hover_yr = [_yfmt(int(round(y))) for y in yr]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines",
            line=dict(color=_COLOURS[j % len(_COLOURS)], width=2.5),
            name=label,
            customdata=hover_yr,
            hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
        ))

    # OWID actual deaths trace (1950–2023)
    _owid_age  = _owid_yr - ancient_start
    _owid_plot = _owid_D / (2.0 * np.pi * np.maximum(_owid_age, 1.0)) if norm_circ else _owid_D
    fig.add_trace(go.Scatter(
        x=2026.0 - _owid_yr if log_x else _owid_yr,
        y=np.maximum(_owid_plot, 1e-6) if log_y else _owid_plot,
        mode="lines",
        line=dict(color="#555555", width=2.5),
        name="OWID actual 1950–2023",
        customdata=[_yfmt(int(y)) for y in _owid_yr],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))

    # dashed reference: 2π · age · ancient_density
    # when norm_circ is on (÷ 2π·age) this collapses to the constant ancient_density
    _owid_last_yr = float(_OWID_SERIES[-1][0])
    ref_yr  = np.linspace(float(ancient_start), _owid_last_yr, 8_000)
    ref_age = np.maximum(ref_yr - ancient_start, 1.0)
    if norm_circ:
        # main plot always normalises by 2π·age → reference collapses to constant
        ref_y = np.full_like(ref_yr, ancient_density)
    else:
        ref_y = 2.0 * np.pi * ref_age * ancient_density
    ref_y = np.maximum(ref_y, 1e-6) if log_y else ref_y
    ref_x        = 2026.0 - ref_yr if log_x else ref_yr
    ref_hover_yr = [_yfmt(int(round(y))) for y in ref_yr]
    fig.add_trace(go.Scatter(
        x=ref_x, y=ref_y,
        mode="lines",
        line=dict(color="black", width=1.5, dash="dash"),
        name="2π · age · ancient_density",
        customdata=ref_hover_yr,
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))

    y_title = "D / (2π·age)" if norm_circ else "Deaths / year"
    fig.update_layout(
        title="Deaths per year (fitted)",
        xaxis_title="Years before 2026 (log)" if log_x else "Year (−ve = BCE)",
        yaxis_title=y_title,
        xaxis=dict(
            type="log" if log_x else "linear",
            autorange="reversed" if log_x else True,
        ),
        yaxis_type="log" if log_y else "linear",
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ) if (not log_x and not log_y) else dict(
            x=1.0, y=0.0, xanchor="right", yanchor="bottom",
        ),
        margin=dict(t=40),
    )
    st.plotly_chart(fig, width="stretch")

    years_sorted = sorted(data.keys())
    cm_deaths_rows = []
    for idx, year in enumerate(years_sorted):
        row = data[year]
        cm_births = float(row.get("cumulative_births", 0.0))
        pop = float(row.get("pop", 0.0))
        pop_before = 0.0 if year == -8000 else float(data[years_sorted[idx - 1]].get("pop", 0.0))
        cm_deaths = cm_births - pop + pop_before
        cm_deaths_rows.append({
            "year": year,
            "cm_births": cm_births,
            "pop": pop,
            "pop_before": pop_before,
            "cm_deaths": cm_deaths,
        })

    with st.expander("Period cumulative inputs/derived deaths", expanded=False):
        st.dataframe(cm_deaths_rows, width="stretch")

    st.metric(
        "ancient_density  (deaths / yr²)",
        f"{ancient_density:,.1f}",
        help=f"Computed from first {x_years:,} years of ancient D, divided by π·X².",
    )

    # ── C = D / ancient_density ───────────────────────────────────────────────
    # D = ancient_density · C · 1yr  →  C = D / ancient_density  (units: yr)
    st.subheader("C = D / ancient_density  (yr)")
    fig_c = go.Figure()
    for j, pr in enumerate(r["periods"]):
        label = f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j + 1])}"
        yr    = pr["yr_arr"][1:]
        C_arr = pr["D_arr"][1:] / ancient_density          # C = D / ancient_density
        x_vals   = 2026.0 - yr if log_x else yr
        y_vals   = np.maximum(C_arr, 1e-30) if log_y else C_arr
        hover_yr = [_yfmt(int(round(y))) for y in yr]
        fig_c.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines",
            line=dict(color=_COLOURS[j % len(_COLOURS)], width=2.5),
            name=label,
            customdata=hover_yr,
            hovertemplate="%{customdata}  —  %{y:,.4g}<extra></extra>",
        ))

    # OWID actual C trace (1950–2023)
    _owid_C = _owid_D / ancient_density
    fig_c.add_trace(go.Scatter(
        x=2026.0 - _owid_yr if log_x else _owid_yr,
        y=np.maximum(_owid_C, 1e-30) if log_y else _owid_C,
        mode="lines",
        line=dict(color="#555555", width=2.5),
        name="OWID actual 1950–2023",
        customdata=[_yfmt(int(y)) for y in _owid_yr],
        hovertemplate="%{customdata}  —  %{y:,.4g}<extra></extra>",
    ))

    # dashed reference: C_ref = 2π · age
    ref_yr    = np.linspace(float(ancient_start), _owid_last_yr, 8_000)
    ref_age   = np.maximum(ref_yr - ancient_start, 1.0)
    ref_C     = 2.0 * np.pi * ref_age
    ref_x     = 2026.0 - ref_yr if log_x else ref_yr
    ref_C_plt = np.maximum(ref_C, 1e-30) if log_y else ref_C
    ref_hover = [_yfmt(int(round(y))) for y in ref_yr]
    fig_c.add_trace(go.Scatter(
        x=ref_x, y=ref_C_plt,
        mode="lines",
        line=dict(color="black", width=1.5, dash="dash"),
        name="2π · age",
        customdata=ref_hover,
        hovertemplate="%{customdata}  —  %{y:,.4g}<extra></extra>",
    ))

    fig_c.update_layout(
        title="C = D / ancient_density  vs  2π · age",
        xaxis_title="Years before 2026 (log)" if log_x else "Year (−ve = BCE)",
        yaxis_title="C  (yr)",
        xaxis=dict(
            type="log" if log_x else "linear",
            autorange="reversed" if log_x else True,
        ),
        yaxis_type="log" if log_y else "linear",
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
        margin=dict(t=40),
    )
    st.plotly_chart(fig_c, width="stretch")

    # ── f = C / 2π = D / (ancient_density · 2π) ──────────────────────────────
    from plotly.subplots import make_subplots

    # Build combined model + OWID series; OWID starts at 1951 (1950 already in model)
    yr_model = np.concatenate([pr["yr_arr"][1:] for pr in r["periods"]])
    D_model  = np.concatenate([pr["D_arr"][1:]  for pr in r["periods"]])
    yr_owid_ext = np.array([float(y) for y, _ in _OWID_SERIES if y > 1950])
    D_owid_ext  = np.array([float(d) for y, d in _OWID_SERIES if y > 1950])
    yr_all = np.concatenate([yr_model, yr_owid_ext])
    D_all  = np.concatenate([D_model,  D_owid_ext])

    f_all   = D_all / (ancient_density * 2.0 * np.pi)   # f = C/2π

    f1 = np.gradient(f_all, yr_all)
    f2 = np.gradient(f1,    yr_all)
    safe_f1 = np.where(np.abs(f1) > 1e-30, f1, np.nan)
    kappa   = -f2 / safe_f1                              # −f″/f′

    # NaN-out ±1 window at model period boundaries to suppress gradient spikes
    # (the model-OWID join at 1950 is a natural data boundary, not NaN'd)
    period_size  = N_GRID - 1
    boundary_idx = [k * period_size for k in range(1, n)]
    nan_mask = np.zeros(len(yr_all), dtype=bool)
    for bi in boundary_idx:
        nan_mask[max(0, bi - 1) : min(len(yr_all), bi + 2)] = True
    for arr in (f1, f2, kappa):
        arr[nan_mask] = np.nan

    x_all     = 2026.0 - yr_all if log_x else yr_all
    hover_all = [_yfmt(int(round(y))) for y in yr_all]
    xaxis_cfg = dict(
        title="Years before 2026 (log)" if log_x else "Year (−ve = BCE)",
        type="log" if log_x else "linear",
        autorange="reversed" if log_x else True,
    )

    # f, f′, f″ subplots
    deriv_series = [
        (f_all, "f",   "f = C / 2π  =  D / (2π · ancient_density)"),
        (f1,    "f′",  "f′  (df/dyr)"),
        (f2,    "f″",  "f″  (d²f/dyr²)"),
    ]
    fig_f = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[s[2] for s in deriv_series],
        vertical_spacing=0.08,
    )
    for row, (y_data, ytitle, _) in enumerate(deriv_series, start=1):
        y_plt = np.where(np.isfinite(y_data), np.maximum(np.abs(y_data), 1e-30) * np.sign(y_data), np.nan) if log_y else y_data
        fig_f.add_trace(
            go.Scatter(
                x=x_all, y=y_plt, mode="lines",
                line=dict(color="#1f78b4", width=1.8), showlegend=False,
                customdata=hover_all,
                hovertemplate="%{customdata}  —  %{y:.4g}<extra></extra>",
            ),
            row=row, col=1,
        )
        fig_f.update_yaxes(title_text=ytitle, row=row, col=1)

    fig_f.update_xaxes(**xaxis_cfg)
    fig_f.update_layout(title="f, f′, f″", height=700, margin=dict(t=50))
    st.plotly_chart(fig_f, width="stretch")

    # −f″/f′  (curvature)
    fig_k = go.Figure(go.Scatter(
        x=x_all, y=kappa, mode="lines",
        line=dict(color="#e31a1c", width=1.8), showlegend=False,
        customdata=hover_all,
        hovertemplate="%{customdata}  —  %{y:.4g}<extra></extra>",
    ))
    fig_k.update_xaxes(**xaxis_cfg)
    fig_k.update_yaxes(title_text="−f″/f′")
    fig_k.update_layout(title="Curvature K = −f″/f′", height=350, margin=dict(t=50))
    st.plotly_chart(fig_k, width="stretch")

    st.stop()


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args      = sys.argv[1:]
    verbose   = "--verbose" in args
    path_args = [a for a in args if not a.startswith("--")]
    path      = Path(path_args[0]) if path_args else None

    _yc_default = 6_000
    _data, _    = _load_data(path)
    _ancient_start = -8000 - _yc_default
    _extended_data = {_ancient_start: {"pop": 0, "cbr": 0}, **_data}

    fit_deaths_direct(
        data=_extended_data,
        verbose=True,
        D_start=0.0,
        D_end=float(_OWID_DEATHS[1950]),
    )
