# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy"]
# ///
"""
Deaths/yr model — piecewise polynomial (cubic / quartic).

For n consecutive periods defined by the anchor years in data.json we fit
polynomials of degree 3 (cubic, 4 coefficients) on most periods, and degree 4
(quartic, 5 coefficients) on 1200→1650 and the final two periods.

C2-continuous at seam points + 6 endpoint constraints (value, d1, d2 at each
end).  Solved as a linear system (instant).

4n+3 equality constraints
─────────────────────────
  n   integral constraints : ∫ D_j dt = cumulative_deaths[j]
  n−1 C0 continuity        : D_{j−1}(seam) = D_j(seam)
  n−1 C1 continuity        : D′_{j−1}(seam) = D′_j(seam)
  n−1 C2 continuity        : D″_{j−1}(seam) = D″_j(seam)
  6   endpoint constraints : D, D′, D″ at both edges

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
}
_X_MODE_K_REF: dict[str, float | None] = {
    "log_before_2026": 2026.0,
    "year":            None,
}


def _u_from_year(year_arr: np.ndarray, x_mode: str, K_ref: float | None) -> np.ndarray:
    """Map calendar years to the fitting x-variable u.

    log_before_2026: u = ln(K_ref − year), clamped to 1e-9.
    year:            u = year (bare calendar year).
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


def _owid_smoothed(window: int = 5) -> dict[int, float]:
    """Mirror-padded moving average of OWID deaths series."""
    years_sorted = sorted(_OWID_DEATHS)
    vals = [float(_OWID_DEATHS[y]) for y in years_sorted]
    n = len(vals)
    half = window // 2
    padded = vals[:half][::-1] + vals + vals[-half:][::-1]
    ma = [sum(padded[i:i + window]) / window for i in range(n)]
    return dict(zip(years_sorted, ma))


def _owid_endpoint_constraints(window: int = 5) -> tuple[float, float, float]:
    """Return (D_1950, dD/dy_1950, d²D/dy²_1950) from smoothed OWID."""
    sm = _owid_smoothed(window)
    D0 = sm[1950]
    D1 = sm[1951]
    D2 = sm[1952]
    dDdy = (-3.0 * D0 + 4.0 * D1 - D2) / 2.0
    d2Ddy2 = D0 - 2.0 * D1 + D2
    return D0, dDdy, d2Ddy2


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
    n = len(anchor_years) - 1
    deaths_tgt = np.zeros(n, dtype=float)
    for j in range(n):
        start_year = anchor_years[j]
        end_row = data[anchor_years[j + 1]]
        pop_start = 0.0 if anchor_years[j + 1] == -8000 else float(data[start_year].get("pop", 0.0))
        deaths_tgt[j] = float(end_row["cumulative_births"]) - float(end_row.get("pop", 0.0)) + pop_start
    return deaths_tgt


# ── fitting ────────────────────────────────────────────────────────────────────

def fit_deaths_direct(
    data_path: Path | str | None = None,
    verbose: bool = False,
    data: dict | None = None,
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
    if x_mode not in _X_MODE_K_REF:
        raise ValueError(f"x_mode must be one of {list(_X_MODE_K_REF)}; got {x_mode!r}")
    if data is None:
        data, years = _load_data(data_path)
    else:
        years = sorted(data.keys())
    n = len(years) - 1
    anchor_years = years

    t0 = float(anchor_years[0])
    K_ref = _X_MODE_K_REF[x_mode]
    K = (K_ref - t0) if K_ref is not None else None
    T_cumul = np.array([float(y - t0) for y in anchor_years])
    T_periods = np.diff(T_cumul)
    deaths_tgt = _period_cumulative_deaths(data, anchor_years)
    _anc_yr = np.array(anchor_years, dtype=float)
    u_bnd = _u_from_year(_anc_yr, x_mode, K_ref)

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
        t_arr = np.linspace(T_cumul[j], T_cumul[j + 1], N_FINE)
        u_arr = _u_from_year(t_arr + t0, x_mode, K_ref)
        IU[j, 0] = T_periods[j]
        for k in range(1, int(period_degree[j]) + 1):
            IU[j, k] = np.trapezoid(u_arr ** k, t_arr)

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

    def _du_dy(y: float) -> tuple[float, float]:
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
        u0, y0 = float(u_bnd[0]), float(anchor_years[0])
        du, _ = _du_dy(y0)
        A[row, _sl(0)] = du * _basis_u(u0, int(period_degree[0]), 1)
        b[row] = float(dDdy_start)
        row += 1
    if dDdy_end is not None:
        un, yn = float(u_bnd[n]), float(anchor_years[n])
        du, _ = _du_dy(yn)
        A[row, _sl(n - 1)] = du * _basis_u(un, int(period_degree[n - 1]), 1)
        b[row] = float(dDdy_end)
        row += 1

    # Endpoint d2 constraints (year-space → u-space via chain rule)
    if d2Ddy2_start is not None:
        u0, y0 = float(u_bnd[0]), float(anchor_years[0])
        du, d2u = _du_dy(y0)
        A[row, _sl(0)] = (du ** 2) * _basis_u(u0, int(period_degree[0]), 2) \
                        + d2u * _basis_u(u0, int(period_degree[0]), 1)
        b[row] = float(d2Ddy2_start)
        row += 1
    if d2Ddy2_end is not None:
        un, yn = float(u_bnd[n]), float(anchor_years[n])
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
    expected_null_dim = n_params - n_constr

    if verbose:
        print(f"n={n}, n_params={n_params}, n_constr={n_constr}, expected_null={expected_null_dim}")
        print(f"rank={rank}, null_dim={null_dim}, κ={sv[0]/sv[-1]:.2e}")

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

    MONO_TOL = 1000.0
    MONO_PTS = 17

    if null_dim == 0:
        theta = theta_p
        mono_ok = True
        for j in range(n):
            ulo, uhi = sorted([float(u_bnd[j]), float(u_bnd[j + 1])])
            for u in np.linspace(ulo, uhi, MONO_PTS):
                if mono_factor * float(_dDdu_theta(theta, j, u)) > MONO_TOL:
                    mono_ok = False
    else:
        G_rows, h_rows = [], []
        for j in range(n):
            ulo, uhi = sorted([float(u_bnd[j]), float(u_bnd[j + 1])])
            for u in np.linspace(ulo, uhi, MONO_PTS):
                nv = N_mat[_sl(j), :].T @ _basis_u(u, int(period_degree[j]), 1)
                G_rows.append(mono_factor * nv)
                h_rows.append(mono_factor * -float(_dDdu_theta(theta_p, j, u)))
        for j in range(n):
            ulo, uhi = sorted([float(u_bnd[j]), float(u_bnd[j + 1])])
            for u in np.linspace(ulo, uhi, MONO_PTS):
                nv = N_mat[_sl(j), :].T @ _basis_u(u, int(period_degree[j]), 0)
                G_rows.append(-nv)
                h_rows.append(float(_D_theta(theta_p, j, u)))
        G, h = np.array(G_rows), np.array(h_rows)
        sol = minimize(lambda x: float(x @ x), np.zeros(null_dim), jac=lambda x: 2.0 * x,
                       method="SLSQP",
                       constraints={"type": "ineq", "fun": lambda x: h + 1.0 - G @ x, "jac": lambda x: -G},
                       options={"ftol": 1e-15, "maxiter": 3000})
        mono_ok = sol.success and np.all(G @ sol.x <= h + MONO_TOL)
        x_opt = sol.x
        if not mono_ok:
            lam = 1e8
            sol2 = minimize(
                lambda x: float(lam * np.sum(np.maximum(0.0, G @ x - h) ** 2) + x @ x),
                np.zeros(null_dim), method="SLSQP", options={"ftol": 1e-20, "maxiter": 5000})
            x_opt = sol2.x
            mono_ok = bool(np.maximum(0.0, G @ x_opt - h).max() <= MONO_TOL)
        theta = theta_p + N_mat @ x_opt

    periods = []
    deaths_model = np.zeros(n)
    for j in range(n):
        t_arr = np.linspace(T_cumul[j], T_cumul[j + 1], N_GRID)
        u_arr = _u_from_year(t_arr + t0, x_mode, K_ref)
        D_arr = np.maximum(0.0, _D_theta(theta, j, u_arr))
        t_fine = np.linspace(T_cumul[j], T_cumul[j + 1], N_FINE)
        u_fine = _u_from_year(t_fine + t0, x_mode, K_ref)
        deaths_model[j] = np.trapezoid(_D_theta(theta, j, u_fine), t_fine)
        periods.append({"t_arr": t_arr, "yr_arr": t_arr + t0, "u_arr": u_arr, "D_arr": D_arr})

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
        quartic_periods=[(anchor_years[j], anchor_years[j + 1]) for j in sorted(quartic_period_idx)],
    )


# ── Streamlit colours ──────────────────────────────────────────────────────────

_COLOURS = [
    "#e31a1c", "#ff7f00", "#6a3d9a", "#33a02c", "#1f78b4",
    "#b15928", "#fb9a99", "#fdbf6f",
]


# ── Streamlit render ───────────────────────────────────────────────────────────

def render() -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import streamlit as st

    data_path = _DOCKER_PATH if _DOCKER_PATH.exists() else _LOCAL_PATH

    # ── x-variable mode ───────────────────────────────────────────────────────
    x_mode = st.sidebar.radio(
        "X-variable (fitting basis)",
        options=list(_X_MODE_LABELS),
        format_func=_X_MODE_LABELS.__getitem__,
        help=(
            "log_before_2026: u = ln(2026 − year) — the default; singularity at 2026.  "
            "year: u = year — bare calendar year, no transform."
        ),
    )

    # ── ancient period slider ─────────────────────────────────────────────────
    ancient_length = st.sidebar.slider(
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

    ancient_flat_length = st.sidebar.slider(
        "Ancient 'flat' length",
        min_value=1,
        max_value=max(1, yc),
        value=min(1_000, max(1, yc)),
        step=100,
        help="Length of analytic pre-fit ancient segment to exclude from fitting.",
    )
    density_slider = st.sidebar.slider(
        "Density (graves/year**2)",
        min_value=1,
        max_value=200,
        value=60,
        step=1,
        help="Density used for analytic flat-ancient calculations.",
    )
    density = float(density_slider)
    owid_smooth_window = st.sidebar.slider(
        "OWID smoothing window (years)",
        min_value=0,
        max_value=50,
        value=10,
        step=1,
        help="Moving-average window for OWID deaths series. 0 = raw (no smoothing).",
    )

    # Flat ancient segment is analytic (not fitted), then fit starts at
    # the post-flat ancient anchor and continues through the historical anchors.
    flat_len_fit = min(max(1, int(ancient_flat_length)), yc - 1)
    fit_ancient_start = ancient_start + flat_len_fit

    row_minus_8000 = data[-8000]
    if "cumulative_deaths_estimated" in row_minus_8000:
        ancient_total_deaths = float(row_minus_8000["cumulative_deaths_estimated"])
    elif "cumulative_births" in row_minus_8000:
        ancient_total_deaths = float(row_minus_8000["cumulative_births"]) - float(row_minus_8000.get("pop", 0.0))
    elif "cumulative" in row_minus_8000:
        ancient_total_deaths = float(row_minus_8000["cumulative"])
    else:
        raise KeyError(
            "Missing cumulative deaths source for year -8000. "
            "Expected one of: cumulative_deaths_estimated, cumulative_births, cumulative."
        )

    flat_deaths = np.pi * (flat_len_fit ** 2) * density
    late_ancient_deaths = ancient_total_deaths - flat_deaths
    late_ancient_start_D = 2.0 * np.pi * flat_len_fit * density
    dDdy_start = 2.0 * np.pi * density
    d2Ddy2_start = 0.0
    D_end_val, dDdy_end, d2Ddy2_end = _owid_endpoint_constraints(window=max(1, owid_smooth_window))

    print(f"[pre-fit] flat_len_years={flat_len_fit}")
    print(f"[pre-fit] flat_density={density:.6g} graves/yr^2")
    print(f"[pre-fit] flat_deaths=pi*L^2*density={flat_deaths:.6e}")
    print(f"[pre-fit] late_ancient_deaths=ancient_total-flat={late_ancient_deaths:.6e}")
    print(f"[pre-fit] late_ancient_start_D=2*pi*L*density={late_ancient_start_D:.6e} deaths/yr")

    extended_data = {
        fit_ancient_start: {"pop": 0, "cbr": 0},
        **data,
    }
    extended_data[-8000] = {
        **extended_data[-8000],
        "cumulative_deaths_estimated": late_ancient_deaths,
    }

    with st.spinner("Fitting deaths/yr piecewise polynomial…"):
        r = fit_deaths_direct(
            data=extended_data,
            D_start=late_ancient_start_D,
            D_end=D_end_val,
            dDdy_start=dDdy_start,
            d2Ddy2_start=d2Ddy2_start,
            dDdy_end=dDdy_end,
            d2Ddy2_end=d2Ddy2_end,
            x_mode=x_mode,
        )

    anchor_years = r["anchor_years"]
    n            = r["n"]

    if not r["mono_ok"]:
        st.warning(
            "Hard monotonicity constraint infeasible for this dataset — "
            "soft-penalty solution used; deaths/yr may not be strictly non-decreasing."
        )

    st.subheader("Deaths / yr — piecewise polynomial")

    # ── deaths/yr curve ───────────────────────────────────────────────────────
    col_logy, col_logx = st.columns(2)
    with col_logy:
        log_y = st.toggle("Log Y axis", value=False)
    with col_logx:
        log_x = st.toggle("Log X axis", value=False)
    show_legend = st.toggle("Show legend", value=True)

    # Analytic flat ancient segment (included in plotting, excluded from fit).
    flat_yr = np.linspace(float(ancient_start), float(fit_ancient_start), N_GRID)
    flat_age = flat_yr - ancient_start
    flat_D = 2.0 * np.pi * density * flat_age

    # Dashed/reference density comes directly from the density slider.

    yr_model_fit = np.concatenate([pr["yr_arr"][1:] for pr in r["periods"]])
    D_model_fit = np.concatenate([pr["D_arr"][1:] for pr in r["periods"]])

    # OWID extension arrays (smoothed; 1950 included for a seamless join with the model)
    if owid_smooth_window >= 1:
        _owid_sm = _owid_smoothed(window=owid_smooth_window)
    else:
        _owid_sm = {y: float(d) for y, d in _OWID_DEATHS.items()}
    _owid_yr = np.array([float(y) for y, _ in _OWID_SERIES])
    _owid_D  = np.array([float(d) for _, d in _OWID_SERIES])
    _owid_D_plot = np.array([_owid_sm[int(y)] for y, _ in _OWID_SERIES])

    # Build combined model + OWID series (OWID starts at 1951; 1950 already pinned in model)
    yr_model = np.concatenate([flat_yr, yr_model_fit])
    D_model  = np.concatenate([flat_D, D_model_fit])
    yr_owid_ext = np.array([float(y) for y, _ in _OWID_SERIES if y > 1950])
    D_owid_ext  = np.array([float(d) for y, d in _OWID_SERIES if y > 1950])
    yr_all = np.concatenate([yr_model, yr_owid_ext])
    D_all  = np.concatenate([D_model,  D_owid_ext])

    _owid_last_yr = float(_OWID_SERIES[-1][0])
    ref_yr  = np.linspace(float(ancient_start), _owid_last_yr, 8_000)
    ref_age = np.maximum(ref_yr - ancient_start, 1.0)
    ref_2pi_age = 2.0 * np.pi * ref_age
    ref_hover_yr = [_yfmt(int(round(y))) for y in ref_yr]

    def _hex_to_rgba(hex_colour: str, alpha: float) -> str:
        h = hex_colour.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"

    def _yearly_series(
        years: np.ndarray,
        values: np.ndarray,
        start_year: float | None = None,
        end_year: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        y0 = float(np.min(years) if start_year is None else start_year)
        y1 = float(np.max(years) if end_year is None else end_year)
        i0 = int(np.ceil(y0))
        i1 = int(np.floor(y1))
        if i1 < i0:
            return np.array([], dtype=float), np.array([], dtype=float)
        yr_int = np.arange(i0, i1 + 1, dtype=float)
        val_int = np.interp(yr_int, years, values)
        return yr_int, val_int

    def _integral_on_period(
        years: np.ndarray,
        values: np.ndarray,
        start_year: float,
        end_year: float,
    ) -> float:
        if end_year <= start_year:
            return 0.0
        interior = years[(years > start_year) & (years < end_year)]
        x = np.concatenate(([start_year], interior, [end_year]))
        y = np.interp(x, years, values)
        return float(np.trapezoid(y, x))

    tab_rate, tab_cumul = st.tabs(["deaths/yr", "cumulative deaths"])

    with tab_rate:
        fig = go.Figure()
        flat_yr_plot, flat_D_plot_src = _yearly_series(
            yr_model, D_model, start_year=float(flat_yr[0]), end_year=float(flat_yr[-1])
        )
        fig.add_trace(go.Scatter(
            x=2026.0 - flat_yr_plot if log_x else flat_yr_plot,
            y=np.maximum(flat_D_plot_src, 1e-6) if log_y else flat_D_plot_src,
            mode="lines",
            line=dict(color="#222222", width=2.5),
            name=f"{_yfmt(int(round(flat_yr_plot[0])))} → {_yfmt(int(round(flat_yr_plot[-1])))} (flat analytic)",
            customdata=[_yfmt(int(round(y))) for y in flat_yr_plot],
            hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
        ))
        for j, pr in enumerate(r["periods"]):
            label = f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j + 1])}"
            yr, D_src = _yearly_series(
                yr_model,
                D_model,
                start_year=float(anchor_years[j] + 1),
                end_year=float(anchor_years[j + 1]),
            )
            if len(yr) == 0:
                continue
            y_vals = np.maximum(D_src, 1e-6) if log_y else D_src
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

        # OWID smoothed deaths trace (1950–2023)
        fig.add_trace(go.Scatter(
            x=2026.0 - _owid_yr if log_x else _owid_yr,
            y=np.maximum(_owid_D_plot, 1e-6) if log_y else _owid_D_plot,
            mode="lines",
            line=dict(color="#555555", width=2.5),
            name=f"OWID {'raw' if owid_smooth_window < 1 else f'smooth({owid_smooth_window}yr)'} 1950–2023",
            customdata=[_yfmt(int(y)) for y in _owid_yr],
            hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
        ))

        ref_yr_plot, ref_y_base = _yearly_series(ref_yr, ref_2pi_age)
        ref_y = np.maximum(ref_y_base, 1e-6) if log_y else ref_y_base
        ref_x = 2026.0 - ref_yr_plot if log_x else ref_yr_plot
        fig.add_trace(go.Scatter(
            x=ref_x, y=ref_y,
            mode="lines",
            line=dict(color="black", width=1.5, dash="dash"),
            name="2π · age",
            customdata=[_yfmt(int(round(y))) for y in ref_yr_plot],
            hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
        ))

        fig.update_layout(
            title="Deaths per year (fitted)",
            xaxis_title="Years before 2026 (log)" if log_x else "Year (−ve = BCE)",
            yaxis_title="Deaths / year",
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
            showlegend=show_legend,
            margin=dict(t=40),
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("Deaths/yr numeric derivatives", expanded=False):
            yr_all_plot = np.arange(int(np.ceil(yr_model[0])), int(_owid_last_yr) + 1, dtype=float)
            D_all_plot = np.interp(yr_all_plot, yr_all, D_all)
            for idx, yr in enumerate(yr_all_plot):
                if yr > 1950 and int(yr) in _owid_sm:
                    D_all_plot[idx] = _owid_sm[int(yr)]

            d1 = np.gradient(D_all_plot, yr_all_plot)
            d2 = np.gradient(d1, yr_all_plot)

            x_vals = 2026.0 - yr_all_plot if log_x else yr_all_plot
            hover_yr = [_yfmt(int(round(y))) for y in yr_all_plot]
            d1_plot = (
                np.where(np.isfinite(d1), np.maximum(np.abs(d1), 1e-30) * np.sign(d1), np.nan)
                if log_y else d1
            )
            d2_plot = (
                np.where(np.isfinite(d2), np.maximum(np.abs(d2), 1e-30) * np.sign(d2), np.nan)
                if log_y else d2
            )

            fig_deriv = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                subplot_titles=["dD/dyear", "d²D/dyear²"],
                vertical_spacing=0.08,
            )
            fig_deriv.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=d1_plot,
                    mode="lines",
                    line=dict(color="#1f78b4", width=1.8),
                    name="dD/dyear",
                    customdata=hover_yr,
                    hovertemplate="%{customdata}  —  %{y:.4g}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig_deriv.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=d2_plot,
                    mode="lines",
                    line=dict(color="#e31a1c", width=1.8),
                    name="d²D/dyear²",
                    customdata=hover_yr,
                    hovertemplate="%{customdata}  —  %{y:.4g}<extra></extra>",
                ),
                row=2,
                col=1,
            )
            fig_deriv.update_xaxes(
                title="Years before 2026 (log)" if log_x else "Year (−ve = BCE)",
                type="log" if log_x else "linear",
                autorange="reversed" if log_x else True,
            )
            fig_deriv.update_yaxes(title_text="dD/dyear", row=1, col=1)
            fig_deriv.update_yaxes(title_text="d²D/dyear²", row=2, col=1)
            fig_deriv.update_layout(
                title="Numeric derivatives of deaths/year",
                height=560,
                showlegend=show_legend,
                legend=dict(
                    x=0.01, y=0.99, xanchor="left", yanchor="top",
                ) if (not log_x and not log_y) else dict(
                    x=1.0, y=0.0, xanchor="right", yanchor="bottom",
                ),
                margin=dict(t=50),
            )
            st.plotly_chart(fig_deriv, width="stretch")

    with tab_cumul:
        yr_all_plot = np.arange(int(np.ceil(yr_model[0])), int(_owid_last_yr) + 1, dtype=float)
        D_all_plot = np.interp(yr_all_plot, yr_all, D_all)
        for idx, yr in enumerate(yr_all_plot):
            if yr > 1950 and int(yr) in _owid_sm:
                D_all_plot[idx] = _owid_sm[int(yr)]

        cumul_all = np.zeros_like(D_all_plot)
        for i in range(1, len(D_all_plot)):
            cumul_all[i] = cumul_all[i - 1] + 0.5 * (D_all_plot[i] + D_all_plot[i - 1])

        # PRB checkpoints: cumulative sum of per-row cm_deaths from raw data.json.
        prb_years_sorted = sorted(data.keys())
        prb_cm_deaths = []
        for idx, year in enumerate(prb_years_sorted):
            row = data[year]
            cm_births = float(row.get("cumulative_births", 0.0))
            pop = float(row.get("pop", 0.0))
            pop_before = 0.0 if year == -8000 else float(data[prb_years_sorted[idx - 1]].get("pop", 0.0))
            prb_cm_deaths.append(cm_births - pop + pop_before)
        prb_cumul = np.cumsum(np.array(prb_cm_deaths, dtype=float))
        prb_x = 2026.0 - np.array(prb_years_sorted, dtype=float) if log_x else np.array(prb_years_sorted, dtype=float)
        prb_y = np.maximum(prb_cumul, 1e-6) if log_y else prb_cumul

        ref_cumul = np.pi * (ref_age ** 2) * density
        ref_x = 2026.0 - ref_yr if log_x else ref_yr
        ref_y = np.maximum(ref_cumul, 1e-6) if log_y else ref_cumul
        fig_cumul = go.Figure()

        flat_mask = (yr_all_plot >= int(np.ceil(flat_yr[0]))) & (yr_all_plot <= int(np.floor(flat_yr[-1])))
        x_seg = 2026.0 - yr_all_plot[flat_mask] if log_x else yr_all_plot[flat_mask]
        y_seg = cumul_all[flat_mask]
        y_seg = np.maximum(y_seg, 1e-6) if log_y else y_seg
        fig_cumul.add_trace(go.Scatter(
            x=x_seg,
            y=y_seg,
            mode="lines",
            line=dict(color="#222222", width=2.5),
            fill="tozeroy",
            fillcolor=_hex_to_rgba("#222222", 0.12),
            name=f"{_yfmt(int(round(flat_yr[0])))} → {_yfmt(int(round(flat_yr[-1])))} (flat analytic)",
            customdata=[_yfmt(int(round(y))) for y in yr_all_plot[flat_mask]],
            hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
        ))

        # Fitted period segments (same labels/colours as deaths/yr tab)
        for j, pr in enumerate(r["periods"]):
            seg_mask = (
                (yr_all_plot >= float(anchor_years[j] + 1))
                & (yr_all_plot <= float(anchor_years[j + 1]))
            )
            x_seg = 2026.0 - yr_all_plot[seg_mask] if log_x else yr_all_plot[seg_mask]
            y_seg = cumul_all[seg_mask]
            if len(x_seg) == 0:
                continue
            y_seg = np.maximum(y_seg, 1e-6) if log_y else y_seg
            seg_colour = _COLOURS[j % len(_COLOURS)]
            fig_cumul.add_trace(go.Scatter(
                x=x_seg,
                y=y_seg,
                mode="lines",
                line=dict(color=seg_colour, width=2.5),
                fill="tozeroy",
                fillcolor=_hex_to_rgba(seg_colour, 0.12),
                name=f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j + 1])}",
                customdata=[_yfmt(int(round(y))) for y in yr_all_plot[seg_mask]],
                hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
            ))

        # OWID extension segment
        seg_mask = yr_all_plot > 1950.0
        x_seg = 2026.0 - yr_all_plot[seg_mask] if log_x else yr_all_plot[seg_mask]
        y_seg = cumul_all[seg_mask]
        y_seg = np.maximum(y_seg, 1e-6) if log_y else y_seg
        fig_cumul.add_trace(go.Scatter(
            x=x_seg,
            y=y_seg,
            mode="lines",
            line=dict(color="#555555", width=2.5),
            fill="tozeroy",
            fillcolor=_hex_to_rgba("#555555", 0.12),
            name=f"OWID {'raw' if owid_smooth_window < 1 else f'smooth({owid_smooth_window}yr)'} 1950–2023",
            customdata=[_yfmt(int(round(y))) for y in yr_all_plot[seg_mask]],
            hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
        ))

        fig_cumul.add_trace(go.Scatter(
            x=ref_x,
            y=ref_y,
            mode="lines",
            line=dict(color="black", width=1.5, dash="dash"),
            name="π · age² · density",
            customdata=ref_hover_yr,
            hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
        ))
        fig_cumul.add_trace(go.Scatter(
            x=prb_x,
            y=prb_y,
            mode="markers",
            marker=dict(color="#111111", size=7, symbol="circle"),
            name="PRB data points",
            customdata=[_yfmt(int(y)) for y in prb_years_sorted],
            hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
        ))
        fig_cumul.update_layout(
            title="Cumulative deaths",
            xaxis_title="Years before 2026 (log)" if log_x else "Year (−ve = BCE)",
            yaxis_title="Cumulative deaths",
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
            showlegend=show_legend,
            margin=dict(t=40),
        )
        st.plotly_chart(fig_cumul, width="stretch")

    years_sorted = sorted(data.keys())
    cm_deaths_rows = []
    for idx, year in enumerate(years_sorted):
        row = data[year]
        cm_births = float(row.get("cumulative_births", 0.0))
        pop = float(row.get("pop", 0.0))
        pop_before = 0.0 if year == -8000 else float(data[years_sorted[idx - 1]].get("pop", 0.0))
        cm_deaths = cm_births - pop + pop_before
        period_start = float(ancient_start) if idx == 0 else float(years_sorted[idx - 1])
        period_end = float(year)
        period_label = f"{_yfmt(int(round(period_start)))} → {_yfmt(int(year))}"
        cm_reconstructed = _integral_on_period(yr_model, D_model, period_start, period_end)
        cm_deaths_rows.append({
            "period": period_label,
            "year": year,
            "cm_births": cm_births,
            "pop": pop,
            "pop_before": pop_before,
            "cm_deaths": cm_deaths,
            "d_cm_deaths_reconstructed": cm_reconstructed - cm_deaths,
        })

    fit_quality = float(
        np.sum(
            np.abs(
                [
                    r["d_cm_deaths_reconstructed"]
                    for r in cm_deaths_rows
                ]
            )
        )
    )
    st.metric(
        "Fit quality (sum |reconstructed per-period death diff|)",
        f"{fit_quality:.6e}",
    )

    with st.expander("Period cumulative inputs/derived deaths", expanded=False):
        transposed_rows = []
        if cm_deaths_rows:
            metric_keys = [k for k in cm_deaths_rows[0].keys() if k not in ("period", "year")]
            for metric in metric_keys:
                row_out = {"metric": metric}
                for row in cm_deaths_rows:
                    row_out[row["period"]] = row[metric]
                transposed_rows.append(row_out)
        st.dataframe(transposed_rows, width="stretch")

    # ── C = D / density ───────────────────────────────────────────────────────
    # D = density · C · 1yr  →  C = D / density  (units: yr)
    fig_c = go.Figure()
    flat_yr_c, flat_D_c = _yearly_series(
        yr_model, D_model, start_year=float(flat_yr[0]), end_year=float(flat_yr[-1])
    )
    flat_C = flat_D_c / density
    fig_c.add_trace(go.Scatter(
        x=2026.0 - flat_yr_c if log_x else flat_yr_c,
        y=np.maximum(flat_C, 1e-30) if log_y else flat_C,
        mode="lines",
        line=dict(color="#222222", width=2.5),
        name=f"{_yfmt(int(round(flat_yr_c[0])))} → {_yfmt(int(round(flat_yr_c[-1])))} (flat analytic)",
        customdata=[_yfmt(int(round(y))) for y in flat_yr_c],
        hovertemplate="%{customdata}  —  %{y:,.4g}<extra></extra>",
    ))
    for j, pr in enumerate(r["periods"]):
        label = f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j + 1])}"
        yr, D_src = _yearly_series(
            yr_model,
            D_model,
            start_year=float(anchor_years[j] + 1),
            end_year=float(anchor_years[j + 1]),
        )
        if len(yr) == 0:
            continue
        C_arr = D_src / density          # C = D / density
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
    _owid_C = _owid_D_plot / density
    fig_c.add_trace(go.Scatter(
        x=2026.0 - _owid_yr if log_x else _owid_yr,
        y=np.maximum(_owid_C, 1e-30) if log_y else _owid_C,
        mode="lines",
        line=dict(color="#555555", width=2.5),
        name=f"OWID {'raw' if owid_smooth_window < 1 else f'smooth({owid_smooth_window}yr)'} 1950–2023",
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
        title="C = D / density",
        xaxis_title="Years before 2026 (log)" if log_x else "Year (−ve = BCE)",
        yaxis_title="C  (yr)",
        xaxis=dict(
            type="log" if log_x else "linear",
            autorange="reversed" if log_x else True,
        ),
        yaxis_type="log" if log_y else "linear",
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
        showlegend=show_legend,
        margin=dict(t=40),
    )
    st.plotly_chart(fig_c, width="stretch")

    # ── f = C / 2π = D / (density · 2π) ──────────────────────────────────────

    yr_all_plot = np.arange(int(np.ceil(yr_model[0])), int(_owid_last_yr) + 1, dtype=float)
    D_all_plot = np.interp(yr_all_plot, yr_all, D_all)
    for idx, yr in enumerate(yr_all_plot):
        if yr > 1950 and int(yr) in _owid_sm:
            D_all_plot[idx] = _owid_sm[int(yr)]

    f_all   = D_all_plot / (density * 2.0 * np.pi)   # f = C/2π

    f1 = np.gradient(f_all, yr_all_plot)
    f2 = np.gradient(f1,    yr_all_plot)
    kappa   = -f2 / f_all                              # −f″


    x_all     = 2026.0 - yr_all_plot if log_x else yr_all_plot
    hover_all = [_yfmt(int(round(y))) for y in yr_all_plot]
    xaxis_cfg = dict(
        title="Years before 2026 (log)" if log_x else "Year (−ve = BCE)",
        type="log" if log_x else "linear",
        autorange="reversed" if log_x else True,
    )

    # f, f′, f″ subplots
    deriv_series = [
        (f_all, "f",   "f = C / 2π  =  D / (2π · density)"),
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
    fig_f.update_layout(title="f, f′, f″", height=700, showlegend=show_legend, margin=dict(t=50))
    st.plotly_chart(fig_f, width="stretch")

    # −f″/f′  (curvature)
    fig_k = go.Figure(go.Scatter(
        x=x_all, y=kappa, mode="lines",
        line=dict(color="#e31a1c", width=1.8), showlegend=False,
        customdata=hover_all,
        hovertemplate="%{customdata}  —  %{y:.4g}<extra></extra>",
    ))
    fig_k.update_xaxes(**xaxis_cfg)
    fig_k.update_yaxes(title_text="−f″/f")
    fig_k.update_layout(title="Curvature K = −f″/f", height=350, showlegend=show_legend, margin=dict(t=50))
    st.plotly_chart(fig_k, width="stretch")


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
    _D_end_val, _dDdy_end, _d2Ddy2_end = _owid_endpoint_constraints(window=5)

    r = fit_deaths_direct(
        data=_extended_data,
        verbose=verbose,
        D_start=0.0,
        D_end=_D_end_val,
        dDdy_start=0.0,
        d2Ddy2_start=0.0,
        dDdy_end=_dDdy_end,
        d2Ddy2_end=_d2Ddy2_end,
    )

    _years = np.array(sorted(r["anchor_years"]), dtype=float)
    _cm_deaths = _period_cumulative_deaths(_extended_data, sorted(r["anchor_years"]))
    _yr_model = np.concatenate(
        [r["periods"][0]["yr_arr"]] + [pr["yr_arr"][1:] for pr in r["periods"][1:]]
    )
    _D_model = np.concatenate(
        [r["periods"][0]["D_arr"]] + [pr["D_arr"][1:] for pr in r["periods"][1:]]
    )

    def _period_integral(years: np.ndarray, values: np.ndarray, y0: float, y1: float) -> float:
        if y1 <= y0:
            return 0.0
        mid = years[(years > y0) & (years < y1)]
        x = np.concatenate(([y0], mid, [y1]))
        y = np.interp(x, years, values)
        return float(np.trapezoid(y, x))

    print(f"\n[reconstructed diffs]")
    print(f"{'Period':<30} {'d_cm_deaths':>16}")
    fit_quality = 0.0
    for j in range(len(_years) - 1):
        y0 = float(_years[j])
        y1 = float(_years[j + 1])
        recon = _period_integral(_yr_model, _D_model, y0, y1)
        dcm = recon - float(_cm_deaths[j])
        fit_quality += abs(dcm)
        print(f"{_yfmt(int(y0)) + ' -> ' + _yfmt(int(y1)):<30} {dcm:>+16.6e}")
    print(f"{'fit_quality = sum(|d_cm_deaths|)':<30} {fit_quality:>+16.6e}")
