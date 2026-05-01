# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "plotly"]
# ///
"""
Always run this module via `uv run`.
Standalone : uv run death_model.py

Important instruction for coding assistant: run this module as standalone after EVERY CHANGE as a test!!!
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import ui
ui.IS_MAIN = __name__ == "__main__"

import fit

_DATA_PATH  = Path("/data/prb.json") if Path("/data/prb.json").exists() else Path(__file__).parent.parent / "data" / "prb.json"
_OUTPUT_DIR = Path("/output") if Path("/output").is_dir() else Path(__file__).parent.parent / "data" / "precomputed"

import plotly.graph_objects as go


# ── OWID data ───────────────────────────────────────────────────────────────────

_OWID_CSV_PATH = Path(__file__).parent / "owid_deaths.csv"


def _load_owid_deaths() -> dict[int, int]:
    out: dict[int, int] = {}
    with open(_OWID_CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["year"])] = int(row["deaths"])
    return out


_OWID_DEATHS: dict[int, int] = _load_owid_deaths()
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
    D0, D1, D2 = sm[1950], sm[1951], sm[1952]
    dDdy = (-3.0 * D0 + 4.0 * D1 - D2) / 2.0
    d2Ddy2 = D0 - 2.0 * D1 + D2
    return D0, dDdy, d2Ddy2


# ── data loading ───────────────────────────────────────────────────────────────

def _load_data() -> dict:
    with open(_DATA_PATH) as f:
        return {int(k): v for k, v in json.load(f).items()}


def _yfmt(y: int) -> str:
    return f"{-y} BCE" if y <= 0 else f"{y} CE"


# ── colours ────────────────────────────────────────────────────────────────────

_COLOURS = [
    "#e31a1c", "#ff7f00", "#6a3d9a", "#33a02c", "#1f78b4",
    "#b15928", "#fb9a99", "#fdbf6f",
]


# ── Output writer ───────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    f = float(v)
    if np.isinf(f):
        return "Infinity"
    f = f + 0.0  # eliminate -0.0
    return f"{f:.2f}" if f != 0.0 else "0.00"


def _write_outputs(
    *,
    ancient_start: int,
    ancient_length: int,
    radius: np.ndarray,
    deaths: np.ndarray,
    deaths_end_year: int,
) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUTPUT_DIR / "info.json").write_text(json.dumps({
        "start_year":      ancient_start,
        "ancient_length":  ancient_length,
        "deaths_end_year": deaths_end_year,
    }, indent=2))
    (_OUTPUT_DIR / "radius.txt").write_text("\n".join(_fmt(v) for v in radius))
    (_OUTPUT_DIR / "deaths.txt").write_text("\n".join(_fmt(v) for v in deaths))


# ── Render ───────────────────────────────────────────────────────────────────────

def render() -> None:

    data = _load_data()

    # ── controls ──────────────────────────────────────────────────────────────
    ancient_length = ui.sidebar_slider(
        "Ancient period length (years)",
        min_value=1_000, max_value=20_000, value=6_000, step=500,
        help="Duration prepended before the first data anchor (-8 000).",
    )
    yc            = ancient_length
    ancient_start = -8_000 - yc

    ancient_flat_length = ui.sidebar_slider(
        "Ancient 'flat' length",
        min_value=1, max_value=max(1, yc), value=min(1_000, max(1, yc)), step=100,
        help="Length of the analytic pre-fit ancient segment excluded from fitting.",
    )
    density = float(ui.sidebar_slider(
        "Density (graves/year²)",
        min_value=1, max_value=200, value=5, step=1,
        help="Density used for analytic flat-ancient calculations.",
    ))
    owid_smooth_window = ui.sidebar_slider(
        "OWID smoothing window (years)",
        min_value=0, max_value=50, value=10, step=1,
        help="Moving-average window for OWID deaths series. 0 = raw.",
    )
    ring_width = ui.sidebar_slider(
        "Ring width (years)",
        min_value=1, max_value=100, value=10, step=1,
        help="Width of each time-ring bin in years.",
    )
    max_graves_per_segment = ui.sidebar_slider(
        "Max graves per segment",
        min_value=100_000, max_value=10_000_000, value=1_000_000, step=100_000,
        help="Maximum graves per segment; determines number of segments per ring.",
    )
    fit_ancient_start = ancient_start + ancient_flat_length

    # ── ancient period quantities ──────────────────────────────────────────────
    ancient_total_deaths = data[-8_000]["cumulative_births"] - data[-8_000]["pop"]
    flat_deaths          = round(np.pi * (ancient_flat_length ** 2) * density)
    late_ancient_deaths  = ancient_total_deaths - flat_deaths
    D_late_ancient_start    = 2.0 * np.pi * (ancient_flat_length + 1) * density
    dDdy_late_ancient_start = 2.0 * np.pi * density
    d2Ddy2_late_ancient_start = 0.0
    D_1950, dDdy_1950, d2Ddy2_1950 = _owid_endpoint_constraints(window=owid_smooth_window)

    # ── build anchor grid and cumulated_D ─────────────────────────────────────
    extended_data = {
        fit_ancient_start: {"pop": 0, "cbr": 0, "cumulative_deaths": flat_deaths},
        **data,
    }
    extended_data[-8_000]["cumulative_deaths"] = late_ancient_deaths
    anchor_cal_years = sorted(extended_data.keys())
    n_periods = len(anchor_cal_years) - 1

    cumulated_D: dict[tuple[int, int], float] = {}
    for j in range(n_periods):
        cal_start = anchor_cal_years[j]
        cal_end   = anchor_cal_years[j + 1]
        t_start   = cal_start - ancient_start
        t_end     = cal_end   - ancient_start
        if j == 0:
            deaths = late_ancient_deaths
        else:
            end_row   = extended_data[cal_end]
            pop_start = extended_data[cal_start]["pop"]
            deaths    = end_row["cumulative_births"] - end_row["pop"] + pop_start
        cumulated_D[(t_start, t_end)] = deaths

    # ── edge initialization targets ───────────────────────────────────────────
    # Interior edges: average CBR of the two bordering periods × pop (C0-consistent).
    # First/last edges: pinned to analytic-flat boundary and OWID 1950 endpoint.
    def _init_edge_mult(cal_year: int) -> float:
        if cal_year <= 1650:
            return 1.09
        if cal_year >= 1900:
            return 0.7
        return 1.09 + (0.7 - 1.09) * (cal_year - 1650) / (1900 - 1650)

    period_cbr = np.array(
        [float(extended_data[anchor_cal_years[j + 1]].get("cbr", 0.0)) for j in range(n_periods)],
        dtype=float,
    )
    edge_targets: list[float] = []
    for k, cal_edge in enumerate(anchor_cal_years):
        pop_edge = float(extended_data[cal_edge].get("pop", 0.0))
        if k == 0:
            cbr_edge = float(period_cbr[0])
        elif k == n_periods:
            cbr_edge = float(period_cbr[-1])
        else:
            cbr_edge = 0.5 * float(period_cbr[k - 1] + period_cbr[k])
        edge_targets.append(cbr_edge / 1_000.0 * pop_edge * _init_edge_mult(cal_edge))
    edge_targets[0]  = float(D_late_ancient_start)
    edge_targets[-1] = float(D_1950)
    init_edge_values = [(edge_targets[j], edge_targets[j + 1]) for j in range(n_periods)]

    # ── four-parameter period selection ───────────────────────────────────────
    # With C0/C1 seams and 6 endpoint constraints, we need 4 four-parameter periods.
    n_endpoint_constraints = 6
    required_four_param    = max(0, n_endpoint_constraints - 2)

    four_param_period_ids: list[int] = []
    seen: set[int] = set()
    for j in range(n_periods):
        if anchor_cal_years[j] == 1200:
            four_param_period_ids.append(j)
            seen.add(j)
            break
    for pid in [n_periods - 2, n_periods - 1]:
        if 0 <= pid < n_periods and pid not in seen:
            four_param_period_ids.append(pid)
            seen.add(pid)
    if len(four_param_period_ids) < required_four_param:
        for pid in range(n_periods - 3, -1, -1):
            if pid not in seen:
                four_param_period_ids.append(pid)
                seen.add(pid)
            if len(four_param_period_ids) == required_four_param:
                break
    if len(four_param_period_ids) != required_four_param:
        raise ValueError(
            f"Need {required_four_param} four-parameter periods, got {len(four_param_period_ids)}."
        )

    total_years = anchor_cal_years[-1] - ancient_start + 1

    _fit_kwargs = dict(
        cumulated_D=cumulated_D,
        four_param_period_ids=four_param_period_ids,
        total_years=total_years,
        D_start=D_late_ancient_start,
        D_end=D_1950,
        dDdy_start=dDdy_late_ancient_start,
        d2Ddy2_start=d2Ddy2_late_ancient_start,
        dDdy_end=dDdy_1950,
        d2Ddy2_end=d2Ddy2_1950,
        init_edge_values=init_edge_values,
    )

    # ── init fit (always) ─────────────────────────────────────────────────────
    r_init = fit.fit_piecewise_constant_curvature(**_fit_kwargs, run_solver=False)

    def _constr_rows(report: list[dict]) -> list[dict]:
        rows = []
        for row in report:
            kind = row["kind"]
            if kind == "integral":
                cal_s = int(row["start_t"]) + ancient_start
                cal_e = int(row["end_t"]) + ancient_start
                label = f"∫D  {_yfmt(cal_s)} → {_yfmt(cal_e)}"
            elif kind == "C0_seam":
                label = f"C0 seam  {_yfmt(int(row['seam_t']) + ancient_start)}"
            elif kind == "C1_seam":
                label = f"C1 seam  {_yfmt(int(row['seam_t']) + ancient_start)}"
            else:
                label = row["name"]
            rows.append({
                "constraint": label,
                "target":     row["target"],
                "value":      row["value"],
                "error":      row["error"],
            })
        return rows

    with ui.expander("Initialized constraint residuals", expanded=False):
        ui.dataframe(_constr_rows(r_init.get("constraint_report", [])), width="stretch")

    # ── run solver button ─────────────────────────────────────────────────────
    run_solver = ui.button(
        "Run nonlinear fit",
        help="Run nonlinear least-squares fit.",
    )

    # ── fit ───────────────────────────────────────────────────────────────────
    if run_solver:
        with ui.solver_status("Running nonlinear fit…") as _log:
            r = fit.fit_piecewise_constant_curvature(
                **_fit_kwargs,
                run_solver=True,
                progress_callback=lambda nfev, cost: _log(f"nfev={nfev:5d}  cost={cost:.6e}"),
            )
    else:
        r = r_init

    ui.write(
        f"**Solver:** {'✓ converged' if r['solver_success'] else '✗ not converged'}"
        f" — cost={r['solver_cost']:.3e} — {r['solver_message']}"
    )

    # ── solver constraints table ──────────────────────────────────────────────
    if run_solver:
        ui.dataframe(_constr_rows(r.get("constraint_report", [])), width="stretch")

    D_init       = r["D_init"]
    D_fitted     = r["D_fitted"]
    anchor_years = anchor_cal_years
    n            = n_periods

    # ── plot controls ─────────────────────────────────────────────────────────
    col_logy, col_logx, col_legend = ui.columns(3)
    with col_logy:
        log_y = ui.toggle("Log Y axis", value=False)
    with col_logx:
        log_x = ui.toggle("Log X axis", value=False)
    with col_legend:
        show_legend = ui.toggle("Show legend", value=True)

    # ── build combined model + OWID arrays ────────────────────────────────────
    # Flat analytic segment covers [ancient_start, fit_ancient_start) exactly.
    flat_yr  = np.arange(ancient_start, fit_ancient_start)
    flat_age = flat_yr - ancient_start
    flat_D   = 2.0 * np.pi * density * flat_age

    fit_start_t = ancient_flat_length
    fit_end_t   = total_years - 1
    yr_model_fit = np.arange(fit_ancient_start, anchor_cal_years[-1])
    D_model_fit  = D_fitted[fit_start_t:fit_end_t]

    if owid_smooth_window >= 1:
        _owid_sm = _owid_smoothed(window=owid_smooth_window)
    else:
        _owid_sm = {y: float(d) for y, d in _OWID_DEATHS.items()}
    _owid_yr     = np.array([y for y, _ in _OWID_SERIES])
    _owid_D_plot = np.array([_owid_sm[int(y)] for y, _ in _OWID_SERIES])

    yr_model     = np.concatenate([flat_yr, yr_model_fit])
    D_model      = np.concatenate([flat_D,  D_model_fit])
    yr_owid_ext  = np.array([y for y, _ in _OWID_SERIES if y >= 1950])
    D_owid_ext   = np.array([float(d) for y, d in _OWID_SERIES if y >= 1950])
    yr_all       = np.concatenate([yr_model, yr_owid_ext])
    D_all        = np.concatenate([D_model,  D_owid_ext])

    # ── helper: shared layout update ─────────────────────────────────────────
    def _apply_layout(fig: go.Figure, title: str, yaxis_title: str) -> None:
        fig.update_layout(
            title=title,
            xaxis_title="Years before 2026 (log)" if log_x else "Year (−ve = BCE)",
            yaxis_title=yaxis_title,
            xaxis=dict(type="log" if log_x else "linear", autorange="reversed" if log_x else True),
            yaxis_type="log" if log_y else "linear",
            legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top")
            if (not log_x and not log_y)
            else dict(x=1.0, y=0.0, xanchor="right", yanchor="bottom"),
            showlegend=show_legend,
            margin=dict(t=40),
        )

    def _xv(yr: np.ndarray) -> np.ndarray:
        return 2026.0 - yr if log_x else yr

    def _yv(vals: np.ndarray) -> np.ndarray:
        return np.maximum(vals, 1e-6) if log_y else vals

    # ── deaths/yr figure ──────────────────────────────────────────────────────
    _owid_last_yr = _OWID_SERIES[-1][0]

    fig_D = go.Figure()
    fig_D.add_trace(go.Scatter(
        x=_xv(flat_yr), y=_yv(flat_D),
        mode="lines", line=dict(color="#222222", width=2.5),
        name=f"{_yfmt(int(flat_yr[0]))} → {_yfmt(int(flat_yr[-1]))} (flat analytic)",
        customdata=[_yfmt(int(y)) for y in flat_yr],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))
    for j in range(n):
        t_s = anchor_years[j] - ancient_start
        t_e = anchor_years[j + 1] - ancient_start
        yr  = np.arange(anchor_years[j], anchor_years[j + 1])
        if len(yr) == 0:
            continue
        colour = _COLOURS[j % len(_COLOURS)]
        label  = f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j + 1])}"
        D_init_seg = D_init[t_s:t_e]
        if run_solver:
            fig_D.add_trace(go.Scatter(
                x=_xv(yr), y=_yv(D_init_seg),
                mode="lines", line=dict(color=colour, width=1.5, dash="dot"),
                name=f"{label} (init)", showlegend=False,
                customdata=[_yfmt(int(round(y))) for y in yr],
                hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
            ))
        D_seg = D_fitted[t_s:t_e]
        fig_D.add_trace(go.Scatter(
            x=_xv(yr), y=_yv(D_seg),
            mode="lines", line=dict(color=colour, width=2.5),
            name=label,
            customdata=[_yfmt(int(round(y))) for y in yr],
            hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
        ))
    fig_D.add_trace(go.Scatter(
        x=_xv(_owid_yr), y=_yv(_owid_D_plot),
        mode="lines", line=dict(color="#555555", width=2.5),
        name=f"OWID {'raw' if owid_smooth_window < 1 else f'smooth({owid_smooth_window}yr)'} 1950–2023",
        customdata=[_yfmt(int(y)) for y in _owid_yr],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))
    ref_yr  = np.arange(ancient_start, _owid_last_yr + 1)
    ref_age = np.maximum(ref_yr - ancient_start, 1.0)
    fig_D.add_trace(go.Scatter(
        x=_xv(ref_yr), y=_yv(2.0 * np.pi * ref_age * density),
        mode="lines", line=dict(color="black", width=1.5, dash="dash"),
        name="2π · age · density",
        customdata=[_yfmt(int(y)) for y in ref_yr],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))
    _apply_layout(fig_D, "Deaths per year", "Deaths / year")

    # ── derivatives and curvature ──────────────────────────────────────────────
    dD1 = np.gradient(D_model, yr_model)

    # 2nd derivative: central everywhere; one-sided at period edges to stay within
    # one polynomial segment (avoiding false seam discontinuities in the stencil).
    _last_years_set  = {int(fit_ancient_start) - 1} | {int(anchor_years[j + 1]) - 1 for j in range(n - 1)}
    _first_years_set = {int(fit_ancient_start)}     | {int(anchor_years[j + 1])     for j in range(n - 1)}
    yr_model_int = yr_model.astype(int)
    _at_last  = np.isin(yr_model_int, list(_last_years_set))
    _at_first = np.isin(yr_model_int, list(_first_years_set))

    N   = len(D_model)
    idx = np.arange(N)
    d2D = np.empty_like(D_model)
    d2D[1:-1] = D_model[2:] - 2.0 * D_model[1:-1] + D_model[:-2]   # central
    d2D[0]    = D_model[2]  - 2.0 * D_model[1]  + D_model[0]        # forward
    d2D[-1]   = D_model[-1] - 2.0 * D_model[-2] + D_model[-3]       # backward
    bi = idx[_at_last  & (idx >= 2)]
    d2D[bi] = D_model[bi] - 2.0 * D_model[bi - 1] + D_model[bi - 2]
    fi = idx[_at_first & (idx + 2 < N)]
    d2D[fi] = D_model[fi + 2] - 2.0 * D_model[fi + 1] + D_model[fi]

    D_safe   = np.maximum(np.abs(D_model), 1e-9)
    curvature = d2D / D_safe
    curv_safe = np.maximum(np.abs(curvature), 1e-12)
    radius    = np.sign(curvature) / curv_safe
    radius[yr_model < fit_ancient_start] = np.inf  # flat segment is exactly linear

    def _seg_figure(d_vals: np.ndarray, title: str, yaxis_title: str, *, include_flat: bool = True) -> go.Figure:
        fig = go.Figure()
        if include_flat:
            flat_mask = yr_model < fit_ancient_start
            fig.add_trace(go.Scatter(
                x=_xv(yr_model[flat_mask]), y=d_vals[flat_mask],
                mode="lines", line=dict(color="#222222", width=2.5),
                name=f"{_yfmt(int(flat_yr[0]))} → {_yfmt(int(flat_yr[-1]))} (flat analytic)",
                customdata=[_yfmt(int(y)) for y in yr_model[flat_mask]],
                hovertemplate="%{customdata}  —  %{y:,.3g}<extra></extra>",
            ))
        for j in range(n):
            seg_mask = (yr_model >= anchor_years[j]) & (yr_model < anchor_years[j + 1])
            if not np.any(seg_mask):
                continue
            fig.add_trace(go.Scatter(
                x=_xv(yr_model[seg_mask]), y=d_vals[seg_mask],
                mode="lines", line=dict(color=_COLOURS[j % len(_COLOURS)], width=2.5),
                name=f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j + 1])}",
                customdata=[_yfmt(int(round(y))) for y in yr_model[seg_mask]],
                hovertemplate="%{customdata}  —  %{y:,.3g}<extra></extra>",
            ))
        _apply_layout(fig, title, yaxis_title)
        return fig

    fig_d1     = _seg_figure(dD1, "1st derivative  dD/dt", "dD/dt  [deaths/yr²]")
    fig_d2     = _seg_figure(d2D, "2nd derivative  d²D/dt²  (one-sided at period edges)", "d²D/dt²  [deaths/yr³]")
    fig_curv   = _seg_figure(curvature, "Curvature  D″/D", "D″/D  [1/yr²]", include_flat=False)
    fig_radius = _seg_figure(
        np.where(np.isinf(radius), np.nan, radius),
        "Radius  1/(D″/D)", "1/(D″/D)  [yr²]", include_flat=False,
    )

    # ── cumulative deaths figure ───────────────────────────────────────────────
    yr_all_plot = np.arange(int(np.ceil(yr_model[0])), int(_owid_last_yr) + 1)
    D_all_plot  = np.interp(yr_all_plot, yr_all, D_all)
    for i, yr in enumerate(yr_all_plot):
        if yr >= 1950 and int(yr) in _owid_sm:
            D_all_plot[i] = _owid_sm[int(yr)]

    cumul_all = np.zeros_like(D_all_plot, dtype=float)
    for i in range(1, len(D_all_plot)):
        cumul_all[i] = cumul_all[i - 1] + 0.5 * (D_all_plot[i] + D_all_plot[i - 1])

    def _hex_to_rgba(hex_colour: str, alpha: float) -> str:
        h = hex_colour.lstrip("#")
        r_, g_, b_ = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r_}, {g_}, {b_}, {alpha})"

    fig_cumul = go.Figure()
    flat_mask = (yr_all_plot >= int(np.ceil(flat_yr[0]))) & (yr_all_plot <= int(np.floor(flat_yr[-1])))
    fig_cumul.add_trace(go.Scatter(
        x=_xv(yr_all_plot[flat_mask]), y=_yv(cumul_all[flat_mask]),
        mode="lines", line=dict(color="#222222", width=2.5),
        fill="tozeroy", fillcolor=_hex_to_rgba("#222222", 0.12),
        name=f"{_yfmt(int(round(flat_yr[0])))} → {_yfmt(int(round(flat_yr[-1])))} (flat analytic)",
        customdata=[_yfmt(int(round(y))) for y in yr_all_plot[flat_mask]],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))
    for j in range(n):
        seg_mask = (yr_all_plot >= anchor_years[j]) & (yr_all_plot < anchor_years[j + 1])
        if not np.any(seg_mask):
            continue
        seg_colour = _COLOURS[j % len(_COLOURS)]
        fig_cumul.add_trace(go.Scatter(
            x=_xv(yr_all_plot[seg_mask]), y=_yv(cumul_all[seg_mask]),
            mode="lines", line=dict(color=seg_colour, width=2.5),
            fill="tozeroy", fillcolor=_hex_to_rgba(seg_colour, 0.12),
            name=f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j + 1])}",
            customdata=[_yfmt(int(round(y))) for y in yr_all_plot[seg_mask]],
            hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
        ))
    seg_mask = yr_all_plot >= 1950.0
    fig_cumul.add_trace(go.Scatter(
        x=_xv(yr_all_plot[seg_mask]), y=_yv(cumul_all[seg_mask]),
        mode="lines", line=dict(color="#555555", width=2.5),
        fill="tozeroy", fillcolor=_hex_to_rgba("#555555", 0.12),
        name=f"OWID {'raw' if owid_smooth_window < 1 else f'smooth({owid_smooth_window}yr)'} 1950–2023",
        customdata=[_yfmt(int(round(y))) for y in yr_all_plot[seg_mask]],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))
    ref_cumul = np.pi * (np.maximum(ref_yr - ancient_start, 1.0) ** 2) * density
    fig_cumul.add_trace(go.Scatter(
        x=_xv(ref_yr), y=_yv(ref_cumul),
        mode="lines", line=dict(color="black", width=1.5, dash="dash"),
        name="π · age² · density",
        customdata=[_yfmt(int(round(y))) for y in ref_yr],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))

    # PRB control points
    prb_years_sorted = sorted(data.keys())
    prb_cumul_vals: list[float] = []
    running = 0.0
    for idx_p, year in enumerate(prb_years_sorted):
        row      = data[year]
        cm_births = float(row.get("cumulative_births", 0.0))
        pop       = float(row.get("pop", 0.0))
        pop_before = 0.0 if year == -8_000 else float(data[prb_years_sorted[idx_p - 1]].get("pop", 0.0))
        running += cm_births - pop + pop_before
        prb_cumul_vals.append(running)
    prb_x = _xv(np.array(prb_years_sorted, dtype=float))
    prb_y = _yv(np.array(prb_cumul_vals, dtype=float))
    fig_cumul.add_trace(go.Scatter(
        x=prb_x, y=prb_y,
        mode="markers", marker=dict(color="#111111", size=7, symbol="circle"),
        name="PRB data points",
        customdata=[_yfmt(int(y)) for y in prb_years_sorted],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))
    _apply_layout(fig_cumul, "Cumulative deaths", "Cumulative deaths")

    # ── cumulative reconstruction check ───────────────────────────────────────
    def _period_sum(years: np.ndarray, values: np.ndarray, start_year: int, end_year: int) -> float:
        if end_year <= start_year:
            return 0.0
        return float(np.sum(np.interp(np.arange(start_year, end_year, dtype=float), years, values)))

    period_rows: list[dict] = []
    cumul_exp = cumul_rec = 0.0
    for j in range(n):
        cal_s = anchor_years[j]
        cal_e = anchor_years[j + 1]
        t_s   = cal_s - ancient_start
        t_e   = cal_e - ancient_start
        expected     = float(cumulated_D[(t_s, t_e)])
        reconstructed = _period_sum(yr_model, D_model, cal_s, cal_e)
        period_rows.append({
            "period":               f"{_yfmt(cal_s)} → {_yfmt(cal_e)}",
            "expected_deaths":      expected,
            "reconstructed_deaths": reconstructed,
            "diff":                 reconstructed - expected,
        })
        cumul_exp += expected
        cumul_rec += reconstructed
    period_rows.append({
        "period":               "Total",
        "expected_deaths":      cumul_exp,
        "reconstructed_deaths": cumul_rec,
        "diff":                 cumul_rec - cumul_exp,
    })

    # ── write outputs ─────────────────────────────────────────────────────────
    _write_outputs(
        ancient_start=int(ancient_start),
        ancient_length=int(ancient_length),
        radius=radius,
        deaths=D_all_plot,
        deaths_end_year=int(_owid_last_yr),
    )

    # ── display ───────────────────────────────────────────────────────────────
    tab_rate, tab_cumul = ui.tabs(["deaths/yr", "cumulative deaths"])
    with tab_rate:
        ui.plotly_chart(fig_D,      width="stretch")
        ui.plotly_chart(fig_d1,     width="stretch")
        ui.plotly_chart(fig_d2,     width="stretch")
        ui.plotly_chart(fig_curv,   width="stretch")
        ui.plotly_chart(fig_radius, width="stretch")
    with tab_cumul:
        ui.plotly_chart(fig_cumul, width="stretch")
        ui.dataframe(period_rows, width="stretch")

    if ui.IS_MAIN:
        return
    ui.st.stop()


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    render()
