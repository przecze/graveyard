import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter1d


def render():
    st.markdown(
        "Given a circumference function C(r), reconstruct the curvature of the surface."
    )
    st.latex(
        r"f(r) = \frac{C(r)}{2\pi}, \qquad K(r) = -\frac{f''(r)}{f(r)}, "
        r"\qquad \kappa(r) = \sqrt{|K|}\cdot\operatorname{sign}(K)"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        slope = st.slider(
            "dC/dr at r=0",
            min_value=0.1,
            max_value=20.0,
            value=2 * np.pi,
            step=0.1,
            format="%.2f",
        )
    with col2:
        yc = st.slider(
            "ancient circle radius (yrs)",
            min_value=100,
            max_value=50000,
            value=20000,
            step=100,
        )
    with col3:
        curv = st.slider(
            "Inner −K₀·yc²  (r < yc)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.05,
            format="%.2f",
            help=(
                "Slider up = more negative Gaussian curvature in the inner "
                "region. 0 = flat (C(r)=slope·r).  K₀ = −curv/yc²."
            ),
        )
    with col4:
        dq = st.slider(
            "Outer Δq  (extra curvature beyond C² match)",
            min_value=-0.010,
            max_value=0.010,
            value=0.0,
            step=0.0001,
            format="%.4f",
            help=(
                "0 = C²-continuous at yc (curvature matches inner at boundary). "
                "Positive = outer region curves more than inner."
            ),
        )

    r_c = yc
    r_end = yc + 10026
    r = np.linspace(0, r_end, 30001)
    dr = r[1] - r[0]
    r_max = r[-1]

    # Inner region r ≤ r_c: constant Gaussian curvature K₀ = −curv / r_c²
    # Jacobi ODE: f'' + K₀·f = 0,  f(0)=0,  f'(0) = slope/(2π)
    #   K₀ < 0 → C(r) = slope · sinh(k·r) / k,  k = √curv / r_c
    #   K₀ = 0 → C(r) = slope · r
    if curv > 0:
        k = np.sqrt(curv) / r_c
        C_inner = slope * np.sinh(k * r) / k
        C_yc = float(slope * np.sinh(k * r_c) / k)
        dC_yc = float(slope * np.cosh(k * r_c))
        # C''(yc⁻) = k² · C_yc  →  q_c2 = ½ · k² · C_yc
        q_c2 = 0.5 * k ** 2 * C_yc
    else:
        C_inner = slope * r
        C_yc = float(slope * r_c)
        dC_yc = float(slope)
        q_c2 = 0.0  # flat inner → C'' = 0 at yc

    # Outer region r > r_c: C²-continuous base (q_c2) + free extra curvature (dq)
    # C²-continuity: C''(yc⁺) = 2·q_c2  matches  C''(yc⁻) = k²·C_yc
    q = q_c2 + dq
    C_outer = C_yc + dC_yc * (r - r_c) + q * (r - r_c) ** 2

    C_in = np.where(r <= r_c, C_inner, C_outer)

    f_in = C_in / (2 * np.pi)
    sigma_smooth = 20
    f_smooth = gaussian_filter1d(f_in, sigma=sigma_smooth)
    f2 = np.gradient(np.gradient(f_smooth, dr), dr)
    safe_f = np.where(np.abs(f_smooth) > 1e-6, f_smooth, np.nan)
    K_inv = -f2 / safe_f
    kappa_inv = np.sqrt(np.abs(K_inv)) * np.sign(K_inv)
    R_max = 5 * r_max
    R_kappa = np.where(np.abs(kappa_inv) > 1.0 / R_max, 1.0 / kappa_inv, np.nan)
    R_gauss = np.where(
        np.abs(K_inv) > (1.0 / R_max) ** 2,
        np.sign(K_inv) / np.sqrt(np.abs(K_inv)),
        np.nan,
    )

    trim = max(1, len(r) // 100)
    r_p = r[trim:-trim]
    C_p = C_in[trim:-trim]
    Rk_p = R_kappa[trim:-trim]
    Rg_p = R_gauss[trim:-trim]

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        fig_c = go.Figure()
        fig_c.add_trace(
            go.Scatter(
                x=r_p,
                y=2 * np.pi * r_p,
                mode="lines",
                line=dict(color="grey", dash="dash"),
                name="flat 2πr",
            )
        )
        fig_c.add_trace(
            go.Scatter(
                x=r_p,
                y=C_p,
                mode="lines",
                line=dict(color="steelblue"),
                name="C(r)",
            )
        )
        fig_c.update_layout(
            title="C(r)",
            xaxis_title="r",
            yaxis_title="circumference",
            legend=dict(x=0.01, y=0.99),
            margin=dict(t=40),
        )
        st.plotly_chart(fig_c, width="stretch")

    with col_b:
        fig_rk = go.Figure(
            go.Scatter(x=r_p, y=Rk_p, mode="lines", line=dict(color="tomato"))
        )
        fig_rk.update_layout(
            title="1/κ(r) — meridian radius",
            xaxis_title="r",
            yaxis_title="1/κ",
            margin=dict(t=40),
        )
        st.plotly_chart(fig_rk, width="stretch")

    with col_c:
        fig_rg = go.Figure(
            go.Scatter(x=r_p, y=Rg_p, mode="lines", line=dict(color="mediumpurple"))
        )
        fig_rg.update_layout(
            title="1/√|K| — Gaussian radius",
            xaxis_title="r",
            yaxis_title="1/√K",
            margin=dict(t=40),
        )
        st.plotly_chart(fig_rg, width="stretch")
