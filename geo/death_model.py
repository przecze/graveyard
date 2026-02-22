# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "plotly", "torch"]
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
import torch

import ui
ui.IS_MAIN = __name__ == "__main__"

import fit
import grad

_DOCKER_PATH = Path("/data/data.json")
_LOCAL_PATH  = Path(__file__).parent.parent / "frontend" / "src" / "data.json"

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── x-variable modes (labels for UI; fit.X_MODE_K_REF used inside fit) ──────────

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
    x_mode = ui.sidebar_radio(
        "X-variable (fitting basis)",
        options=list(fit.X_MODE_LABELS),
        format_func=fit.X_MODE_LABELS.__getitem__,
        help=(
            "log_before_2026: u = ln(2026 − year) — the default; singularity at 2026.  "
            "year: u = year — bare calendar year, no transform."
        ),
    )

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
        value=60,
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
    learning_rate = ui.sidebar_select_slider(
            "Learning rate (grad optim)",
            options=[0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000],
            value=0.1,
            format_func=lambda v: "0" if v == 0 else f"{v:g}",
            help="Learning rate for one SGD step on sum(deaths array) loss.",
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

    ui.write(f"[pre-fit] flat_len_years={ancient_flat_length}")
    ui.write(f"[pre-fit] flat_density={density:.6g} graves/yr^2")
    ui.write(f"[pre-fit] flat_deaths=pi*L^2*density={flat_deaths:_.0f}")
    ui.write(f"[pre-fit] late_ancient_deaths=ancient_total-flat={late_ancient_deaths:_.0f}")
    ui.write(f"[pre-fit] D_late_ancient_start=2*pi*L*density={D_late_ancient_start:_.1f} deaths/yr")

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

    quartic_period_ids: list[tuple[int, int]] = []
    n_periods = len(anchor_cal_years) - 1
    quartic_period_ids.append((1200 - ancient_start, 1650 - ancient_start))
    quartic_period_ids.append((anchor_cal_years[-3] - ancient_start, anchor_cal_years[-2] - ancient_start))
    quartic_period_ids.append((anchor_cal_years[-2] - ancient_start, anchor_cal_years[-1] - ancient_start))

    total_years = anchor_cal_years[-1] - ancient_start + 1

    ui.write("---")
    ui.write(f"**ancient_start={ancient_start}, fit_ancient_start={fit_ancient_start}**")
    ui.write(f"**total_years={total_years}, flat_length={ancient_flat_length}**")
    ui.write(f"**cumulated_D** (year-count space, t=0 is {_yfmt(ancient_start)}):")
    for (ts, te), d in cumulated_D.items():
        cal_s = ts + ancient_start
        cal_e = te + ancient_start
        ui.write(f"  ({ts:>6}, {te:>6}) → {d:>20,.0f}  [{_yfmt(cal_s)} → {_yfmt(cal_e)}]")
    ui.write(f"**quartic_period_ids:** {quartic_period_ids}")
    ui.write("---")

    logarithm_reference_year = fit.X_MODE_K_REF[x_mode]
    if logarithm_reference_year is not None:
        logarithm_reference_year -= ancient_start

    with ui.spinner("Fitting deaths/yr piecewise polynomial…"):
        r = fit.fit_piecewise_polynomial(
            cumulated_D=cumulated_D,
            quartic_period_ids=quartic_period_ids,
            total_years=total_years,
            D_start=D_late_ancient_start,
            D_end=D_1950,
            dDdy_start=dDdy_late_ancient_start,
            d2Ddy2_start=d2Ddy2_late_ancient_start,
            dDdy_end=dDdy_1950,
            d2Ddy2_end=d2Ddy2_1950,
            logarithm_reference_year=logarithm_reference_year,
        )

    D_fitted = r["D_fitted"]
    anchor_years = anchor_cal_years
    n = n_periods

    ui.subheader("Deaths / yr — piecewise polynomial")

    # ── deaths/yr curve ───────────────────────────────────────────────────────
    col_logy, col_logx, col_legend = ui.columns(3)
    with col_logy:
        log_y = ui.toggle("Log Y axis", value=False)
    with col_logx:
        log_x = ui.toggle("Log X axis", value=False)
    with col_legend:
        show_legend = ui.toggle("Show legend", value=True)

    # Analytic flat ancient segment (included in plotting, excluded from fit).
    flat_yr = np.arange(ancient_start, fit_ancient_start + 1)
    flat_age = flat_yr - ancient_start
    flat_D = 2.0 * np.pi * density * flat_age

    # Dashed/reference density comes directly from the density slider.

    fit_start_t = ancient_flat_length
    fit_end_t = total_years - 1
    yr_model_fit = np.arange(fit_ancient_start + 1, anchor_cal_years[-1] + 1)
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
    D_model  = np.concatenate([flat_D, D_model_fit])
    yr_owid_ext = np.array([y for y, _ in _OWID_SERIES if y > 1950])
    D_owid_ext  = np.array([float(d) for y, d in _OWID_SERIES if y > 1950])
    yr_all = np.concatenate([yr_model, yr_owid_ext])
    D_all  = np.concatenate([D_model,  D_owid_ext])

    # Extract deaths per year from late ancient (start + flat length) through 1949
    grad_optim_start_yr = fit_ancient_start
    mask = (yr_model >= grad_optim_start_yr) & (yr_model <= 1949)
    yr_for_grad_optim = torch.tensor(yr_model[mask], dtype=torch.float64)
    D_for_grad_optim = torch.tensor(D_model[mask], dtype=torch.float64)
    first_yr_included = int(yr_for_grad_optim[0].item())
    last_yr_included = int(yr_for_grad_optim[-1].item())
    ui.write(f"**Deaths (late ancient → 1949)** first year included: **{first_yr_included}** · last year included: **{last_yr_included}**")
    ui.write(f"Shape: `{D_for_grad_optim.shape}` · First 3: `{D_for_grad_optim[:3].tolist()}` · Last 3: `{D_for_grad_optim[-3:].tolist()}`")
    relative_curvature_importance = ui.sidebar_slider(
        "Relative curvature importance",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.1,
        help="Relative importance of curvature loss compared to cumulative deaths loss.",
    )
    grad_steps = ui.sidebar_slider(
        "Gradient steps",
        min_value=0,
        max_value=100,
        value=3,
        step=1,
        help="Number of gradient steps to take.",
    )

    # One torch optim step: loss = sum_j (desired_j - integral_j)^2
    lr = float(learning_rate)
    _deaths_tgt_t = torch.tensor(list(cumulated_D.values()), dtype=torch.float64)

    D_param = D_for_grad_optim.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([D_param], lr=lr)
    for _ in range(grad_steps):
        optimizer.zero_grad()
        grad.compute_grad_optim_grads(D_param, yr_for_grad_optim, _deaths_tgt_t, anchor_years,
        relative_curvature_importance=relative_curvature_importance,
        d_year_2_before=D_all[yr_all == grad_optim_start_yr-2].item(),
        d_year_1_before=D_all[yr_all == grad_optim_start_yr-1].item(),
        d_year_1_after=D_all[yr_all == 1950].item(),
        d_year_2_after=D_all[yr_all == 1951].item(),
        )

        optimizer.step()
    D_after_step = D_param.detach().clone()

    with torch.no_grad():
        loss_before = grad.period_cumulative_deaths_loss(D_for_grad_optim, yr_for_grad_optim, _deaths_tgt_t, anchor_years).item()
        loss_after  = grad.period_cumulative_deaths_loss(D_after_step,     yr_for_grad_optim, _deaths_tgt_t, anchor_years).item()
    max_abs_diff = (D_for_grad_optim - D_after_step).abs().max().item()
    ui.write(f"**Grad optim** loss Σ(desired−integral)² before step: `{loss_before:.6e}` → after one step: `{loss_after:.6e}`")
    ui.write(f"Max |before − after| per element: `{max_abs_diff:.6g}` (lr={lr:.6g})")

    _yr_np = yr_for_grad_optim.cpu().numpy()
    _D_np = D_for_grad_optim.cpu().numpy()
    _D_after_np = D_after_step.cpu().numpy()

    # D_model with grad-stepped values substituted in (for per-period table diff)
    D_model_after = D_model.copy()
    D_model_after[mask] = _D_after_np

    fig_for_grad_optim = go.Figure()
    fig_for_grad_optim.add_trace(go.Scatter(
        x=(2026.0 - _yr_np if log_x else _yr_np),
        y=np.maximum(_D_np, 1e-6) if log_y else _D_np,
        mode="lines",
        line=dict(color="#1f78b4", width=2),
        name=f"Deaths/yr {first_yr_included}→{last_yr_included} (before step)",
        customdata=[_yfmt(int(y)) for y in _yr_np],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))
    fig_for_grad_optim.add_trace(go.Scatter(
        x=(2026.0 - _yr_np if log_x else _yr_np),
        y=np.maximum(_D_after_np, 1e-6) if log_y else _D_after_np,
        mode="lines",
        line=dict(color="#e6550d", width=2, dash="dash"),
        name=f"Deaths/yr {first_yr_included}→{last_yr_included} (after step)",
        customdata=[_yfmt(int(y)) for y in _yr_np],
        hovertemplate="%{customdata}  —  %{y:,.0f}<extra></extra>",
    ))
    fig_for_grad_optim.update_layout(
        title=f"Deaths per year ({first_yr_included} to {last_yr_included})",
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
    ui.plotly_chart(fig_for_grad_optim, width="stretch")

    # Difference (after − before) so the step is visible despite scale
    delta_np = _D_after_np - _D_np

    _sep_kwargs = dict(
        d_year_2_before=D_all[yr_all == grad_optim_start_yr-2].item(),
        d_year_1_before=D_all[yr_all == grad_optim_start_yr-1].item(),
        d_year_1_after=D_all[yr_all == 1950].item(),
        d_year_2_after=D_all[yr_all == 1951].item(),
    )
    grad_integrals, grad_curvature = grad.compute_separate_grads(
        D_for_grad_optim, yr_for_grad_optim, _deaths_tgt_t, anchor_years,
        **_sep_kwargs,
    )
    _x_delta = 2026.0 - _yr_np if log_x else _yr_np
    _cd = [_yfmt(int(y)) for y in _yr_np]

    def _delta_fig(title, y, color):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=_x_delta, y=y,
            mode="lines",
            line=dict(color=color, width=1.5),
            customdata=_cd,
            hovertemplate="%{customdata}  —  %{y:.6g}<extra></extra>",
            showlegend=False,
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Years before 2026 (log)" if log_x else "Year (−ve = BCE)",
            yaxis_title="Δ (deaths/yr)",
            xaxis=dict(
                type="log" if log_x else "linear",
                autorange="reversed" if log_x else True,
            ),
            margin=dict(t=40),
            height=220,
        )
        return fig

    ui.plotly_chart(_delta_fig("Δ total (after − before)", delta_np, "#2ca02c"), width="stretch")
    ui.plotly_chart(_delta_fig("−lr · ∇ integrals", -lr * grad_integrals, "#1f78b4"), width="stretch")
    ui.plotly_chart(_delta_fig("−lr · ∇ curvature", -lr * grad_curvature, "#e6550d"), width="stretch")

    _owid_last_yr = _OWID_SERIES[-1][0]
    ref_yr  = np.arange(ancient_start, _owid_last_yr + 1)
    ref_age = np.maximum(ref_yr - ancient_start, 1.0)
    ref_2pi_age = 2.0 * np.pi * ref_age
    ref_hover_yr = [_yfmt(y) for y in ref_yr]

    def _hex_to_rgba(hex_colour: str, alpha: float) -> str:
        h = hex_colour.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"

    def _integral_on_period(
        years: np.ndarray,
        values: np.ndarray,
        start_year: int,
        end_year: int,
    ) -> float:
        if end_year <= start_year:
            return 0.0
        interior = years[(years > start_year) & (years < end_year)]
        x = np.concatenate(([start_year], interior, [end_year]))
        y = np.interp(x, years, values)
        return float(np.trapezoid(y, x))

    tab_rate, tab_cumul = ui.tabs(["deaths/yr", "cumulative deaths"])

    with tab_rate:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
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

        ref_D_ref = ref_2pi_age * density
        ref_y = np.maximum(ref_D_ref, 1e-6) if log_y else ref_D_ref
        ref_x = 2026.0 - ref_yr if log_x else ref_yr
        fig.add_trace(go.Scatter(
            x=ref_x, y=ref_y,
            mode="lines",
            line=dict(color="black", width=1.5, dash="dash"),
            name="2π · age · density",
            customdata=[_yfmt(int(y)) for y in ref_yr],
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
        ui.plotly_chart(fig, width="stretch")

        with ui.expander("Deaths/yr numeric derivatives", expanded=False):
            yr_all_plot = np.arange(int(np.ceil(yr_model[0])), int(_owid_last_yr) + 1)
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
            ui.plotly_chart(fig_deriv, width="stretch")

    with tab_cumul:
        yr_all_plot = np.arange(int(np.ceil(yr_model[0])), int(_owid_last_yr) + 1)
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
            pop_before = 0.0 if year == -8_000 else float(data[prb_years_sorted[idx - 1]].get("pop", 0.0))
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

        for j in range(n):
            seg_mask = (
                (yr_all_plot >= anchor_years[j] + 1)
                & (yr_all_plot <= anchor_years[j + 1])
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
        ui.plotly_chart(fig_cumul, width="stretch")

    years_sorted = sorted(data.keys())
    cm_deaths_rows = []
    for idx, year in enumerate(years_sorted):
        row = data[year]
        cm_births = float(row.get("cumulative_births", 0.0))
        pop = float(row.get("pop", 0.0))
        pop_before = 0.0 if year == -8_000 else float(data[years_sorted[idx - 1]].get("pop", 0.0))
        cm_deaths = cm_births - pop + pop_before
        period_start = ancient_start if idx == 0 else years_sorted[idx - 1]
        period_end = year
        period_label = f"{_yfmt(period_start)} → {_yfmt(year)}"
        cm_reconstructed = _integral_on_period(yr_model, D_model, period_start, period_end)
        cm_reconstructed_after = _integral_on_period(yr_model, D_model_after, period_start, period_end)
        cm_deaths_rows.append({
            "period": period_label,
            "year": year,
            "cm_births": cm_births,
            "pop": pop,
            "pop_before": pop_before,
            "cm_deaths": cm_deaths,
            "diff_cm_deaths_fit": cm_reconstructed - cm_deaths,
            "diff_cm_deaths_after_grad_descent": cm_reconstructed_after - cm_deaths,
        })

    fit_quality = float(
        np.sum(
            np.abs(
                [
                    r["diff_cm_deaths_fit"]
                    for r in cm_deaths_rows
                ]
            )
        )
    )
    ui.metric(
        "Fit quality (sum |reconstructed per-period death diff|)",
        f"{fit_quality:.6e}",
    )

    with ui.expander("Period cumulative inputs/derived deaths", expanded=False):
        transposed_rows = []
        if cm_deaths_rows:
            metric_keys = [k for k in cm_deaths_rows[0].keys() if k not in ("period", "year")]
            for key in metric_keys:
                row_out = {"metric": key}
                for row in cm_deaths_rows:
                    row_out[row["period"]] = row[key]
                transposed_rows.append(row_out)
        ui.dataframe(transposed_rows, width="stretch")

    # ── C = D / density ───────────────────────────────────────────────────────
    # D = density · C · 1yr  →  C = D / density  (units: yr)
    fig_c = go.Figure()
    flat_C = flat_D / density
    fig_c.add_trace(go.Scatter(
        x=2026.0 - flat_yr if log_x else flat_yr,
        y=np.maximum(flat_C, 1e-30) if log_y else flat_C,
        mode="lines",
        line=dict(color="#222222", width=2.5),
        name=f"{_yfmt(int(flat_yr[0]))} → {_yfmt(int(flat_yr[-1]))} (flat analytic)",
        customdata=[_yfmt(int(y)) for y in flat_yr],
        hovertemplate="%{customdata}  —  %{y:,.4g}<extra></extra>",
    ))
    for j in range(n):
        label = f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j + 1])}"
        t_s = anchor_years[j] - ancient_start
        t_e = anchor_years[j + 1] - ancient_start
        yr = np.arange(anchor_years[j] + 1, anchor_years[j + 1] + 1)
        D_src = D_fitted[t_s + 1 : t_e + 1]
        if len(yr) == 0:
            continue
        C_arr = D_src / density
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
    ref_yr    = np.arange(ancient_start, _owid_last_yr + 1)
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
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ) if (not log_x and not log_y) else dict(
            x=1.0, y=0.0, xanchor="right", yanchor="bottom",
        ),
        showlegend=show_legend,
        margin=dict(t=40),
    )
    ui.plotly_chart(fig_c, width="stretch")

    # ── f = C / 2π = D / (density · 2π) ──────────────────────────────────────

    yr_all_plot = np.arange(int(np.ceil(yr_model[0])), int(_owid_last_yr) + 1)
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
    ui.plotly_chart(fig_f, width="stretch")

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
    ui.plotly_chart(fig_k, width="stretch")


# ── CLI entry point (single computation flow via render) ───────────────────────

if __name__ == "__main__":
    render()
