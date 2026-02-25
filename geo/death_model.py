# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "plotly"]
# ///
"""
Deaths/yr model — piecewise constant-curvature basis.

For n consecutive periods defined by the anchor years in data.json we fit:
  - 3-parameter segments: a * sinh(b * (u - u0))
  - 4-parameter segments: a * sinh(b * (u - u0)) + c * cosh(b * (u - u0))

Constraints:
  n   integral constraints : ∫ D_j dt = cumulative_deaths[j]
  n−1 C0 seam continuity   : D_{j−1}(seam) = D_j(seam)
  n−1 C1 seam continuity   : D′_{j−1}(seam) = D′_j(seam)
  6   endpoint constraints : D, D′, D″ at both edges

The code checks parameter count vs constraint count before fitting so the
system is fully determined.

Period cumulative death constraints are derived from per-period births and
population carry-over:

    cumulative_deaths[j] = cumulative_births[end_j] - pop[end_j] + pop[start_j]

with a special-case baseline pop[start_j] = 0 when end_j is -8_000.

Always run this module via `uv run`.
Standalone : uv run death_model.py

Important instruction for coding assistant: run this module as standalone after EVERY CHANGE as a test!!!

Streamlit  : import death_model; death_model.render()
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import ui
ui.IS_MAIN = __name__ == "__main__"

import fit

_DOCKER_PATH = Path("/data/data.json")
_LOCAL_PATH  = Path(__file__).parent.parent / "frontend" / "src" / "data.json"

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── OWID data ───────────────────────────────────────────────────────────────────

# OWID actual deaths/yr (UN World Population Prospects 2024), 1950–2023
_OWID_CSV_PATH = Path(__file__).parent / "owid_deaths.csv"


def _load_owid_deaths() -> dict[int, int]:
    """Load OWID deaths series from CSV (year, deaths)."""
    out: dict[int, int] = {}
    with open(_OWID_CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["year"])] = int(row["deaths"])
    return out


_OWID_DEATHS: dict[int, int] = _load_owid_deaths()
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

def _load_data() -> tuple[dict, list[int]]:
    path = _DOCKER_PATH if _DOCKER_PATH.exists() else _LOCAL_PATH
    with open(path) as f:
        raw = json.load(f)
    data = {int(k): v for k, v in raw.items()}
    return data


def _yfmt(y: int) -> str:
    return f"{-y} BCE" if y <= 0 else f"{y} CE"


# ── Streamlit colours ──────────────────────────────────────────────────────────

_COLOURS = [
    "#e31a1c", "#ff7f00", "#6a3d9a", "#33a02c", "#1f78b4",
    "#b15928", "#fb9a99", "#fdbf6f",
]



# ── Render (single computation flow) ────────────────────────────────────────────

def render() -> None:

    data = _load_data()

    # ── x-variable mode ───────────────────────────────────────────────────────
    # Log mode is disabled; use bare-year fitting only.
    x_mode = "year"

    # ── ancient period slider ─────────────────────────────────────────────────
    ancient_length = ui.sidebar_slider(
        "Ancient period length (years)",
        min_value=1_000,
        max_value=20_000,
        value=6_000,
        step=500,
        help="Duration of the pre-history period prepended before the first data anchor (-8_000).",
    )
    yc             = ancient_length
    ancient_start  = -8_000 - yc

    ancient_flat_length = ui.sidebar_slider(
        "Ancient 'flat' length",
        min_value=1,
        max_value=max(1, yc),
        value=min(1_000, max(1, yc)),
        step=100,
        help="Length of analytic pre-fit ancient segment to exclude from fitting.",
    )
    density_slider = ui.sidebar_slider(
        "Density (graves/year**2)",
        min_value=1,
        max_value=200,
        value=5,
        step=1,
        help="Density used for analytic flat-ancient calculations.",
    )
    density = float(density_slider)
    owid_smooth_window = ui.sidebar_slider(
        "OWID smoothing window (years)",
        min_value=0,
        max_value=50,
        value=10,
        step=1,
        help="Moving-average window for OWID deaths series. 0 = raw (no smoothing).",
    )
    run_solver = ui.sidebar_toggle(
        "Run nonlinear fit",
        value=False,
        help="Off = initialization only (fast). On = run nonlinear least-squares fit.",
    )
    init_multiplier = ui.sidebar_slider(
        "Init edge multiplier",
        min_value=0.1,
        max_value=5.0,
        value=1.09,
        step=0.01,
        help="Multiplier for init edge targets: pop * cbr/1000 * multiplier (first/last edges use fixed targets).",
    )
    # Flat ancient segment is analytic (not fitted), then fit starts at
    # the post-flat ancient anchor and continues through the historical anchors.
    fit_ancient_start = ancient_start + ancient_flat_length

    ancient_total_deaths = data[-8_000]["cumulative_births"] - data[-8_000]["pop"]

    flat_deaths = round(np.pi * (ancient_flat_length ** 2) * density)
    late_ancient_deaths = ancient_total_deaths - flat_deaths
    D_late_ancient_start = 2.0 * np.pi * (ancient_flat_length + 1) * density
    dDdy_late_ancient_start = 2.0 * np.pi * density
    d2Ddy2_late_ancient_start = 0.0
    D_1950, dDdy_1950, d2Ddy2_1950 = _owid_endpoint_constraints(window=owid_smooth_window)


    data_years = sorted(data.keys())
    for j in range(len(data_years) - 1):
        cal_start = data_years[j]
        cal_end = data_years[j + 1]
        t_start = cal_start - ancient_start
        t_end = cal_end - ancient_start
        if j == 0:
            deaths = late_ancient_deaths
        else:
            end_row = data[cal_end]
            pop_start = data[cal_start]["pop"]
            data[cal_end]["cumulative_deaths"] = end_row["cumulative_births"] - end_row["pop"] + pop_start

    extended_data = {
        fit_ancient_start: {"pop": 0, "cbr": 0, "cumulative_deaths": flat_deaths},
        **data,
    }
    extended_data[-8_000]["cumulative_deaths"] = late_ancient_deaths
    extended_data_years = sorted(extended_data.keys())

    # ── Build cumulated_D in year-count space (t = cal_year − ancient_start) ──
    anchor_cal_years = extended_data_years
    cumulated_D: dict[tuple[int, int], float] = {}
    init_edge_values: list[tuple[float, float]] = []

    # Edge heuristic targets: use cbr averaged from bordering periods so seam
    # targets are shared across adjacent periods (C0-consistent initialization).
    # period_cbr[j] is the cbr associated with period j (start_j -> end_j).
    period_cbr = np.array(
        [float(extended_data[anchor_cal_years[j + 1]].get("cbr", 0.0)) for j in range(len(anchor_cal_years) - 1)],
        dtype=float,
    )
    edge_targets: list[float] = []
    n_periods_local = len(anchor_cal_years) - 1
    for k, cal_edge in enumerate(anchor_cal_years):
        pop_edge = float(extended_data[cal_edge].get("pop", 0.0))
        if n_periods_local <= 0:
            cbr_edge = 0.0
        elif k == 0:
            cbr_edge = float(period_cbr[0])
        elif k == n_periods_local:
            cbr_edge = float(period_cbr[-1])
        else:
            cbr_edge = 0.5 * float(period_cbr[k - 1] + period_cbr[k])
        edge_targets.append(cbr_edge / 1_000.0 * pop_edge * float(init_multiplier))
    # First fitted edge is synthetic (pop=0); use analytic-flat boundary instead.
    if edge_targets:
        edge_targets[0] = float(D_late_ancient_start)
        # Rightmost initialization edge should match OWID 1950 endpoint target.
        edge_targets[-1] = float(D_1950)

    for j in range(len(anchor_cal_years) - 1):
        cal_start = anchor_cal_years[j]
        cal_end = anchor_cal_years[j + 1]
        t_start = cal_start - ancient_start
        t_end = cal_end - ancient_start
        if j == 0:
            deaths = late_ancient_deaths
        else:
            end_row = extended_data[cal_end]
            pop_start = extended_data[cal_start]["pop"]
            deaths = end_row["cumulative_births"] - end_row["pop"] + pop_start
        cumulated_D[(t_start, t_end)] = deaths
        left_target = float(edge_targets[j])
        right_target = float(edge_targets[j + 1])
        init_edge_values.append((left_target, right_target))

    n_periods = len(anchor_cal_years) - 1
    four_param_period_ids: list[int] = []
    for j in range(n_periods):
        if anchor_cal_years[j] == 1200:
            four_param_period_ids.append(j)
            break
    four_param_period_ids.append(n_periods - 2)
    four_param_period_ids.append(n_periods - 1)
    # With C0/C1 seams (no C2 seam continuity), square system condition is:
    #   (3*n + n_four_param) params == (3*n - 2 + n_endpoint_constraints)
    n_endpoint_constraints = 6  # D, dD/dy, d2D/dy2 at both ends
    required_four_param = max(0, n_endpoint_constraints - 2)
    dedup_ids: list[int] = []
    seen = set()
    for pid in four_param_period_ids:
        if 0 <= pid < n_periods and pid not in seen:
            dedup_ids.append(pid)
            seen.add(pid)
    four_param_period_ids = dedup_ids
    if len(four_param_period_ids) < required_four_param:
        for pid in range(n_periods - 3, -1, -1):
            if pid not in seen:
                four_param_period_ids.append(pid)
                seen.add(pid)
            if len(four_param_period_ids) == required_four_param:
                break
    if len(four_param_period_ids) != required_four_param:
        raise ValueError(
            "Could not build a fully determined constant-curvature fit setup: "
            f"need {required_four_param} four-parameter periods, got {len(four_param_period_ids)}."
        )

    total_years = anchor_cal_years[-1] - ancient_start + 1

    n_params = 3 * n_periods + len(four_param_period_ids)
    n_constraints = (3 * n_periods - 2) + n_endpoint_constraints

    logarithm_reference_year = None

    if run_solver:
        raise NotImplementedError("Nonlinear fitting is not implemented.")

    with ui.spinner("Initializing deaths/yr basis…"):
        r = fit.fit_piecewise_polynomial(
            cumulated_D=cumulated_D,
            quartic_period_ids=four_param_period_ids,
            total_years=total_years,
            D_start=D_late_ancient_start,
            D_end=D_1950,
            dDdy_start=dDdy_late_ancient_start,
            d2Ddy2_start=d2Ddy2_late_ancient_start,
            dDdy_end=dDdy_1950,
            d2Ddy2_end=d2Ddy2_1950,
            logarithm_reference_year=logarithm_reference_year,
            init_edge_values=init_edge_values,
            init_x0_t=0.0,
        )
    if n_params != n_constraints:
        raise ValueError(f"Fit is not square: params={n_params}, constraints={n_constraints}")

    D_init = r["D_init"]
    D_fitted = r["D_fitted"]
    anchor_years = anchor_cal_years
    n = n_periods

    ui.subheader("Deaths / yr — piecewise constant-curvature")

    # ── deaths/yr curve ───────────────────────────────────────────────────────
    col_logy, col_logx, col_legend = ui.columns(3)
    with col_logy:
        log_y = ui.toggle("Log Y axis", value=False)
    with col_logx:
        log_x = ui.toggle("Log X axis", value=False)
    with col_legend:
        show_legend = ui.toggle("Show legend", value=True)

    with ui.expander("Edge targets vs initialized edge values", expanded=False):
        edge_rows = []
        for row in r.get("init_edge_report", []):
            cal_s = int(row["start_t"]) + ancient_start
            cal_e = int(row["end_t"]) + ancient_start
            edge_rows.append({
                "period": f"{_yfmt(cal_s)} → {_yfmt(cal_e)}",
                "n_params": row["n_params"],
                "target_left": row["target_left"],
                "target_right": row["target_right"],
                "init_left": row["init_left"],
                "init_right": row["init_right"],
                "err_left": row["err_left"],
                "err_right": row["err_right"],
                "init_min": row["init_min"],
                "init_max": row["init_max"],
            })
        ui.dataframe(edge_rows, width="stretch")

    # Analytic flat ancient segment (included in plotting, excluded from fit).
    flat_yr = np.arange(ancient_start, fit_ancient_start + 1)
    flat_age = flat_yr - ancient_start
    flat_D = 2.0 * np.pi * density * flat_age

    # Dashed/reference density comes directly from the density slider.

    fit_start_t = ancient_flat_length
    fit_end_t = total_years - 1
    yr_model_fit = np.arange(fit_ancient_start + 1, anchor_cal_years[-1] + 1)
    D_model_init_fit = D_init[fit_start_t + 1 : fit_end_t + 1]
    D_model_fit = D_fitted[fit_start_t + 1 : fit_end_t + 1]

    # OWID extension arrays (smoothed; 1950 included for a seamless join with the model)
    if owid_smooth_window >= 1:
        _owid_sm = _owid_smoothed(window=owid_smooth_window)
    else:
        _owid_sm = {y: float(d) for y, d in _OWID_DEATHS.items()}
    _owid_yr = np.array([y for y, _ in _OWID_SERIES])
    _owid_D  = np.array([float(d) for _, d in _OWID_SERIES])
    _owid_D_plot = np.array([_owid_sm[int(y)] for y, _ in _OWID_SERIES])

    # Build combined model + OWID series (OWID starts at 1951; 1950 already pinned in model)
    yr_model = np.concatenate([flat_yr, yr_model_fit])
    D_model_init = np.concatenate([flat_D, D_model_init_fit])
    D_model = np.concatenate([flat_D, D_model_fit])
    yr_owid_ext = np.array([y for y, _ in _OWID_SERIES if y > 1950])
    D_owid_ext  = np.array([float(d) for y, d in _OWID_SERIES if y > 1950])
    yr_all = np.concatenate([yr_model, yr_owid_ext])
    D_all  = np.concatenate([D_model,  D_owid_ext])

    # Show initialization curve first (before nonlinear fit).
    fig_init = go.Figure()
    fig_init.add_trace(go.Scatter(
        x=2026.0 - flat_yr if log_x else flat_yr,
        y=np.maximum(flat_D, 1e-6) if log_y else flat_D,
        mode="lines",
        line=dict(color="#222222", width=2.5),
        name=f"{_yfmt(int(flat_yr[0]))} → {_yfmt(int(flat_yr[-1]))} (flat analytic)",
        customdata=[_yfmt(int(y)) for y in flat_yr],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))
    for j in range(n):
        label = f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j + 1])}"
        t_s = anchor_years[j] - ancient_start
        t_e = anchor_years[j + 1] - ancient_start
        yr = np.arange(anchor_years[j] + 1, anchor_years[j + 1] + 1)
        D_src = D_fitted[t_s + 1 : t_e + 1]
        if len(yr) == 0:
            continue
        y_vals = np.maximum(D_src, 1e-6) if log_y else D_src
        x_vals = 2026.0 - yr if log_x else yr
        hover_yr = [_yfmt(int(round(y))) for y in yr]
        D_src_init = D_init[t_s + 1 : t_e + 1]
        fig_init.add_trace(go.Scatter(
            x=x_vals, y=(np.maximum(D_src_init, 1e-6) if log_y else D_src_init),
            mode="lines",
            line=dict(color=_COLOURS[j % len(_COLOURS)], width=2.0, dash="dot"),
            name=f"{label} (init)",
            customdata=hover_yr,
            hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
        ))
        fig_init.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines",
            line=dict(color=_COLOURS[j % len(_COLOURS)], width=2.5),
            name=label,
            customdata=hover_yr,
            hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
        ))

    fig_init.add_trace(go.Scatter(
        x=2026.0 - _owid_yr if log_x else _owid_yr,
        y=np.maximum(_owid_D_plot, 1e-6) if log_y else _owid_D_plot,
        mode="lines",
        line=dict(color="#555555", width=2.5),
        name=f"OWID {'raw' if owid_smooth_window < 1 else f'smooth({owid_smooth_window}yr)'} 1950–2023",
        customdata=[_yfmt(int(y)) for y in _owid_yr],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))

    _owid_last_yr = _OWID_SERIES[-1][0]
    ref_yr = np.arange(ancient_start, _owid_last_yr + 1)
    ref_age = np.maximum(ref_yr - ancient_start, 1.0)
    ref_D_ref = 2.0 * np.pi * ref_age * density
    ref_y = np.maximum(ref_D_ref, 1e-6) if log_y else ref_D_ref
    ref_x = 2026.0 - ref_yr if log_x else ref_yr
    fig_init.add_trace(go.Scatter(
        x=ref_x, y=ref_y,
        mode="lines",
        line=dict(color="black", width=1.5, dash="dash"),
        name="2π · age · density",
        customdata=[_yfmt(int(y)) for y in ref_yr],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))

    fig_init.update_layout(
        title="Deaths per year — initialization (dotted) vs fitted (solid)",
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

    # Fit-debug tabs.
    tab_rate, tab_cumul = ui.tabs(["deaths/yr", "cumulative deaths"])
    with tab_rate:
        ui.plotly_chart(fig_init, width="stretch")

    def _reconstruct_period_sum(
        years: np.ndarray, values: np.ndarray, start_year: int, end_year: int
    ) -> float:
        if end_year <= start_year:
            return 0.0
        x = np.arange(start_year, end_year, dtype=float)
        y = np.interp(x, years, values)
        return float(np.sum(y))

    period_rows: list[dict[str, float | str]] = []
    cumul_expected = [0.0]
    cumul_reconstructed = [0.0]
    cumul_years = [anchor_years[0]]
    for j in range(n):
        cal_s = anchor_years[j]
        cal_e = anchor_years[j + 1]
        t_s = cal_s - ancient_start
        t_e = cal_e - ancient_start
        expected = float(cumulated_D[(t_s, t_e)])
        reconstructed = _reconstruct_period_sum(yr_model, D_model, cal_s, cal_e)
        diff = reconstructed - expected
        period_rows.append({
            "period": f"{_yfmt(cal_s)} → {_yfmt(cal_e)}",
            "expected_deaths": expected,
            "reconstructed_deaths": reconstructed,
            "diff": diff,
        })
        cumul_expected.append(cumul_expected[-1] + expected)
        cumul_reconstructed.append(cumul_reconstructed[-1] + reconstructed)
        cumul_years.append(cal_e)

    period_rows.append({
        "period": "Total",
        "expected_deaths": cumul_expected[-1],
        "reconstructed_deaths": cumul_reconstructed[-1],
        "diff": cumul_reconstructed[-1] - cumul_expected[-1],
    })

    def _hex_to_rgba(hex_colour: str, alpha: float) -> str:
        h = hex_colour.lstrip("#")
        r_, g_, b_ = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r_}, {g_}, {b_}, {alpha})"

    _owid_last_yr = _OWID_SERIES[-1][0]
    yr_all_plot = np.arange(int(np.ceil(yr_model[0])), int(_owid_last_yr) + 1)
    D_all_plot = np.interp(yr_all_plot, yr_all, D_all)
    for idx, yr in enumerate(yr_all_plot):
        if yr > 1950 and int(yr) in _owid_sm:
            D_all_plot[idx] = _owid_sm[int(yr)]

    cumul_all = np.zeros_like(D_all_plot, dtype=float)
    for i in range(1, len(D_all_plot)):
        cumul_all[i] = cumul_all[i - 1] + 0.5 * (D_all_plot[i] + D_all_plot[i - 1])

    prb_years_sorted = sorted(data.keys())
    prb_cm_deaths = []
    for idx, year in enumerate(prb_years_sorted):
        row = data[year]
        cm_births = float(row.get("cumulative_births", 0.0))
        pop = float(row.get("pop", 0.0))
        pop_before = 0.0 if year == -8_000 else float(data[prb_years_sorted[idx - 1]].get("pop", 0.0))
        prb_cm_deaths.append(cm_births - pop + pop_before)
    prb_cumul = np.cumsum(np.array(prb_cm_deaths, dtype=float))
    prb_x = 2026.0 - np.array(prb_years_sorted, dtype=float) if log_x else np.array(prb_years_sorted, dtype=float)
    prb_y = np.maximum(prb_cumul, 1e-6) if log_y else prb_cumul

    _owid_last_yr = _OWID_SERIES[-1][0]
    ref_yr = np.arange(ancient_start, _owid_last_yr + 1)
    ref_age = np.maximum(ref_yr - ancient_start, 1.0)
    ref_cumul = np.pi * (ref_age ** 2) * density
    ref_x = 2026.0 - ref_yr if log_x else ref_yr
    ref_y = np.maximum(ref_cumul, 1e-6) if log_y else ref_cumul

    fig_cumul = go.Figure()
    flat_mask = (yr_all_plot >= int(np.ceil(flat_yr[0]))) & (yr_all_plot <= int(np.floor(flat_yr[-1])))
    x_seg = 2026.0 - yr_all_plot[flat_mask] if log_x else yr_all_plot[flat_mask]
    y_seg = np.maximum(cumul_all[flat_mask], 1e-6) if log_y else cumul_all[flat_mask]
    fig_cumul.add_trace(go.Scatter(
        x=x_seg, y=y_seg,
        mode="lines",
        line=dict(color="#222222", width=2.5),
        fill="tozeroy",
        fillcolor=_hex_to_rgba("#222222", 0.12),
        name=f"{_yfmt(int(round(flat_yr[0])))} → {_yfmt(int(round(flat_yr[-1])))} (flat analytic)",
        customdata=[_yfmt(int(round(y))) for y in yr_all_plot[flat_mask]],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))

    for j in range(n):
        seg_mask = ((yr_all_plot >= anchor_years[j] + 1) & (yr_all_plot <= anchor_years[j + 1]))
        x_seg = 2026.0 - yr_all_plot[seg_mask] if log_x else yr_all_plot[seg_mask]
        if len(x_seg) == 0:
            continue
        y_seg = np.maximum(cumul_all[seg_mask], 1e-6) if log_y else cumul_all[seg_mask]
        seg_colour = _COLOURS[j % len(_COLOURS)]
        fig_cumul.add_trace(go.Scatter(
            x=x_seg, y=y_seg,
            mode="lines",
            line=dict(color=seg_colour, width=2.5),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(seg_colour, 0.12),
            name=f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j + 1])}",
            customdata=[_yfmt(int(round(y))) for y in yr_all_plot[seg_mask]],
            hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
        ))

    seg_mask = yr_all_plot > 1950.0
    x_seg = 2026.0 - yr_all_plot[seg_mask] if log_x else yr_all_plot[seg_mask]
    y_seg = np.maximum(cumul_all[seg_mask], 1e-6) if log_y else cumul_all[seg_mask]
    fig_cumul.add_trace(go.Scatter(
        x=x_seg, y=y_seg,
        mode="lines",
        line=dict(color="#555555", width=2.5),
        fill="tozeroy",
        fillcolor=_hex_to_rgba("#555555", 0.12),
        name=f"OWID {'raw' if owid_smooth_window < 1 else f'smooth({owid_smooth_window}yr)'} 1950–2023",
        customdata=[_yfmt(int(round(y))) for y in yr_all_plot[seg_mask]],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))
    fig_cumul.add_trace(go.Scatter(
        x=ref_x, y=ref_y,
        mode="lines",
        line=dict(color="black", width=1.5, dash="dash"),
        name="π · age² · density",
        customdata=[_yfmt(int(round(y))) for y in ref_yr],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))
    fig_cumul.add_trace(go.Scatter(
        x=prb_x, y=prb_y,
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
        xaxis=dict(type="log" if log_x else "linear", autorange="reversed" if log_x else True),
        yaxis_type="log" if log_y else "linear",
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ) if (not log_x and not log_y) else dict(
            x=1.0, y=0.0, xanchor="right", yanchor="bottom",
        ),
        showlegend=show_legend,
        margin=dict(t=40),
    )

    with tab_cumul:
        ui.plotly_chart(fig_cumul, width="stretch")
        ui.dataframe(period_rows, width="stretch")

    if ui.IS_MAIN:
        return
    ui.st.stop()


# ── CLI entry point (single computation flow via render) ───────────────────────

if __name__ == "__main__":
    render()
