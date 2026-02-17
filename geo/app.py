import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter1d

st.title("hello geo")
st.markdown("Given a circumference function C(r), reconstruct the curvature of the surface.")
st.latex(r"f(r) = \frac{C(r)}{2\pi}, \qquad K(r) = -\frac{f''(r)}{f(r)}, \qquad \kappa(r) = \sqrt{|K|}\cdot\operatorname{sign}(K)")

# ── sliders (before grid so r can depend on yc) ──────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    slope = st.slider("Linear slope  (dC/dr for r ≤ yc)", min_value=0.1, max_value=20.0,
                      value=2 * np.pi, step=0.1, format="%.2f")
with col2:
    yc = st.slider("ancient circle radius (yrs)", min_value=100, max_value=50000,
                   value=20000, step=100)
with col3:
    q = st.slider("Quadratic param  q  (for r > yc)", min_value=-0.010, max_value=0.010,
                  value=0.0005, step=0.0001, format="%.4f")

r_c = yc
r_end = yc + 10026
r = np.linspace(0, r_end, 30001)
dr = r[1] - r[0]
r_max = r[-1]

# C(r) = slope·r                                    for r ≤ r_c
#       = slope·r_c + slope·(r−r_c) + q·(r−r_c)²  for r > r_c
# C1-continuous at r_c, then accelerates by q
C_in = np.where(
    r <= r_c,
    slope * r,
    slope * r_c + slope * (r - r_c) + q * (r - r_c) ** 2,
)

# ── reconstruction ───────────────────────────────────────────────────────────
f_in = C_in / (2 * np.pi)
sigma_smooth = 20
f_smooth = gaussian_filter1d(f_in, sigma=sigma_smooth)
f2 = np.gradient(np.gradient(f_smooth, dr), dr)
safe_f = np.where(np.abs(f_smooth) > 1e-6, f_smooth, np.nan)
K_inv = -f2 / safe_f
kappa_inv = np.sqrt(np.abs(K_inv)) * np.sign(K_inv)
R_max = 5 * r_max  # cap display at 5× the geodesic range
R_kappa = np.where(np.abs(kappa_inv) > 1.0 / R_max, 1.0 / kappa_inv, np.nan)
R_gauss = np.where(np.abs(K_inv) > (1.0 / R_max) ** 2, np.sign(K_inv) / np.sqrt(np.abs(K_inv)), np.nan)

trim = max(1, len(r) // 100)
r_p = r[trim:-trim]
C_p = C_in[trim:-trim]
Rk_p = R_kappa[trim:-trim]
Rg_p = R_gauss[trim:-trim]

# ── plots ────────────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns(3)

with col_a:
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=r_p, y=2 * np.pi * r_p, mode="lines",
                                line=dict(color="grey", dash="dash"), name="flat 2πr"))
    fig_c.add_trace(go.Scatter(x=r_p, y=C_p, mode="lines",
                                line=dict(color="steelblue"), name="C(r)"))
    fig_c.update_layout(title="C(r)", xaxis_title="r", yaxis_title="circumference",
                         legend=dict(x=0.01, y=0.99), margin=dict(t=40))
    st.plotly_chart(fig_c, use_container_width=True)

with col_b:
    fig_rk = go.Figure(go.Scatter(x=r_p, y=Rk_p, mode="lines", line=dict(color="tomato")))
    fig_rk.update_layout(title="1/κ(r) — meridian radius", xaxis_title="r", yaxis_title="1/κ",
                          margin=dict(t=40))
    st.plotly_chart(fig_rk, use_container_width=True)

with col_c:
    fig_rg = go.Figure(go.Scatter(x=r_p, y=Rg_p, mode="lines", line=dict(color="mediumpurple")))
    fig_rg.update_layout(title="1/√|K| — Gaussian radius", xaxis_title="r", yaxis_title="1/√K",
                          margin=dict(t=40))
    st.plotly_chart(fig_rg, use_container_width=True)
