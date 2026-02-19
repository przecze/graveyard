import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Scale so that at K=0, ro=1 → 6 points. C_flat(1) = 2π, so scale = 3/π.
_SCALE = 3 / np.pi


def _circumference(ro, K):
    """Circumference of a geodesic circle at distance ro from the pole."""
    if abs(K) < 1e-9:
        return 2 * np.pi * ro
    elif K > 0:
        return 2 * np.pi * np.sin(ro * np.sqrt(K)) / np.sqrt(K)
    else:
        return 2 * np.pi * np.sinh(ro * np.sqrt(-K)) / np.sqrt(-K)


def _generate_points(K):
    """Return (rs, phis_rad, ring_ids) arrays for all spawned points."""
    rs, phis, ring_ids = [], [], []
    for ro in range(1, 26):
        n = max(0, round(_SCALE * _circumference(ro, K)))
        if n == 0:
            continue
        rs.extend([ro] * n)
        phis.extend(np.linspace(0, 2 * np.pi, n, endpoint=False).tolist())
        ring_ids.extend([ro] * n)
    return np.array(rs), np.array(phis), np.array(ring_ids)


def _log_map_sphere(ro0, phi0, rs, phis, R):
    """Log map at (ro0, phi0) on sphere of radius R. Returns (x_tan, y_tan)."""
    sin0, cos0 = np.sin(ro0 / R), np.cos(ro0 / R)
    P = R * np.array([sin0 * np.cos(phi0), sin0 * np.sin(phi0), cos0])
    e_rho = np.array([cos0 * np.cos(phi0), cos0 * np.sin(phi0), -sin0])
    e_phi = np.array([-np.sin(phi0), np.cos(phi0), 0.0])

    Q = R * np.column_stack([
        np.sin(rs / R) * np.cos(phis),
        np.sin(rs / R) * np.sin(phis),
        np.cos(rs / R),
    ])
    cos_angle = np.clip(Q @ P / R ** 2, -1, 1)
    d = R * np.arccos(cos_angle)

    v = Q - cos_angle[:, None] * P
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    mask = v_norm.ravel() > 1e-9
    v[mask] = v[mask] / v_norm[mask] * d[mask, None]
    v[~mask] = 0.0

    return v @ e_phi, v @ e_rho


def _log_map_flat(ro0, phi0, rs, phis):
    """Log map at (ro0, phi0) in flat space. Returns (x_tan, y_tan)."""
    dx = rs * np.cos(phis) - ro0 * np.cos(phi0)
    dy = rs * np.sin(phis) - ro0 * np.sin(phi0)
    # Project onto local tangent frame (e_phi=east, e_rho=away-from-pole)
    e_phi = np.array([-np.sin(phi0), np.cos(phi0)])
    e_rho = np.array([np.cos(phi0), np.sin(phi0)])
    return dx * e_phi[0] + dy * e_phi[1], dx * e_rho[0] + dy * e_rho[1]


def _exp_map_sphere(ro0, phi0, vx, vy, R):
    """Exp map: tangent vectors (vx, vy) at (ro0,phi0) → (rs, phis_deg) on sphere."""
    sin0, cos0 = np.sin(ro0 / R), np.cos(ro0 / R)
    P = R * np.array([sin0 * np.cos(phi0), sin0 * np.sin(phi0), cos0])
    e_rho = np.array([cos0 * np.cos(phi0), cos0 * np.sin(phi0), -sin0])
    e_phi = np.array([-np.sin(phi0), np.cos(phi0), 0.0])

    V = np.outer(vx, e_phi) + np.outer(vy, e_rho)
    d = np.sqrt(vx ** 2 + vy ** 2)
    mask = d > 1e-9
    V_hat = np.where(mask[:, None], V / np.where(mask[:, None], d[:, None], 1), 0.0)

    Q = np.cos(d[:, None] / R) * P + np.sin(d[:, None] / R) * R * V_hat
    rs = R * np.arccos(np.clip(Q[:, 2] / R, -1, 1))
    phis_deg = np.rad2deg(np.arctan2(Q[:, 1], Q[:, 0])) % 360
    return rs, phis_deg


def _exp_map_flat(ro0, phi0, vx, vy):
    """Exp map: tangent vectors (vx, vy) at (ro0,phi0) → (rs, phis_deg) in flat space."""
    e_phi = np.array([-np.sin(phi0), np.cos(phi0)])
    e_rho = np.array([np.cos(phi0), np.sin(phi0)])
    x0, y0 = ro0 * np.cos(phi0), ro0 * np.sin(phi0)
    pts = np.outer(vx, e_phi) + np.outer(vy, e_rho)
    xq, yq = x0 + pts[:, 0], y0 + pts[:, 1]
    rs = np.sqrt(xq ** 2 + yq ** 2)
    phis_deg = np.rad2deg(np.arctan2(yq, xq)) % 360
    return rs, phis_deg


def _exp_map_hyperbolic(ro0, phi0, vx, vy, R):
    """Exp map: tangent vectors (vx, vy) at (ro0,phi0) → (rs, phis_deg) in hyperbolic space."""
    sinh0, cosh0 = np.sinh(ro0 / R), np.cosh(ro0 / R)
    P = np.array([sinh0 * np.cos(phi0), sinh0 * np.sin(phi0), cosh0])
    e_rho = np.array([cosh0 * np.cos(phi0), cosh0 * np.sin(phi0), sinh0])
    e_phi = np.array([-np.sin(phi0), np.cos(phi0), 0.0])

    V = np.outer(vx, e_phi) + np.outer(vy, e_rho)
    d = np.sqrt(vx ** 2 + vy ** 2)
    mask = d > 1e-9
    V_hat = np.where(mask[:, None], V / np.where(mask[:, None], d[:, None], 1), 0.0)

    Q = np.cosh(d[:, None] / R) * P + np.sinh(d[:, None] / R) * V_hat
    rs = R * np.arccosh(np.clip(Q[:, 2], 1.0, None))
    phis_deg = np.rad2deg(np.arctan2(Q[:, 1], Q[:, 0])) % 360
    return rs, phis_deg


def _viewport_boundary(dro, n_per_edge=80):
    """Sample the 4 edges of the viewport square in (vx, vy) tangent coords."""
    t = np.linspace(-dro, dro, n_per_edge)
    vx = np.concatenate([t,  [dro] * n_per_edge, t[::-1], [-dro] * n_per_edge, [-dro]])
    vy = np.concatenate([[-dro] * n_per_edge, t,  [dro] * n_per_edge, t[::-1],  [-dro]])
    return vx, vy


def _log_map_hyperbolic(ro0, phi0, rs, phis, R):
    """Log map at (ro0, phi0) in hyperbolic space (hyperboloid model). Returns (x_tan, y_tan)."""
    sinh0, cosh0 = np.sinh(ro0 / R), np.cosh(ro0 / R)
    P = np.array([sinh0 * np.cos(phi0), sinh0 * np.sin(phi0), cosh0])
    # Unit tangent basis vectors (Minkowski-normalised)
    e_rho = np.array([cosh0 * np.cos(phi0), cosh0 * np.sin(phi0), sinh0])
    e_phi = np.array([-np.sin(phi0), np.cos(phi0), 0.0])

    Q = np.column_stack([
        np.sinh(rs / R) * np.cos(phis),
        np.sinh(rs / R) * np.sin(phis),
        np.cosh(rs / R),
    ])
    # Minkowski inner product: x·x + y·y - z·z
    PQ = Q[:, 0] * P[0] + Q[:, 1] * P[1] - Q[:, 2] * P[2]
    d = R * np.arccosh(np.clip(-PQ, 1.0, None))

    v = Q + PQ[:, None] * P   # Minkowski rejection: ⟨P,P⟩_M = -1, so proj = -PQ·P
    v_mink_sq = v[:, 0] ** 2 + v[:, 1] ** 2 - v[:, 2] ** 2
    v_mink_norm = np.sqrt(np.clip(v_mink_sq, 0, None))
    mask = v_mink_norm > 1e-9
    v[mask] = v[mask] / v_mink_norm[mask, None] * d[mask, None]
    v[~mask] = 0.0

    # Project via Minkowski inner product (e_phi has z=0, so same as Euclidean there)
    x_tan = v[:, 0] * e_phi[0] + v[:, 1] * e_phi[1]
    y_tan = v[:, 0] * e_rho[0] + v[:, 1] * e_rho[1] - v[:, 2] * e_rho[2]
    return x_tan, y_tan


def render():
    st.write("hello view port")

    st.latex(
        r"C(\rho) = 2\pi R \sin\!\left(\frac{\rho}{R}\right) = "
        r"\frac{2\pi \sin(\rho\sqrt{K})}{\sqrt{K}}"
    )
    st.caption(
        "Circumference of a geodesic circle at distance ρ from the pole. "
        "Flat: C = 2πρ. Sphere (K>0): shrinks. Hyperbolic (K<0): grows faster."
    )

    K = st.slider("Curvature K", min_value=-0.016, max_value=0.016, value=0.0,
                  step=0.0002, format="%.4f")

    col1, col2, col3 = st.columns(3)
    with col1:
        ro0 = st.slider("ρ₀ — observer geodesic distance", 0.0, 24.5, 3.0, 0.1)
    with col2:
        phi0_deg = st.slider("φ₀ — observer angle (°)", 0, 355, 0, 5)
    with col3:
        dro = st.slider("Δρ — viewport half-size", 0.5, 25.0, 3.0, 0.5)
    phi0 = np.deg2rad(phi0_deg)

    rs, phis, ring_ids = _generate_points(K)

    # ── Polar plot ────────────────────────────────────────────────────────────
    fig_polar = go.Figure()
    fig_polar.add_trace(go.Scatterpolar(
        r=rs, theta=np.rad2deg(phis),
        mode="markers",
        marker=dict(size=5, opacity=0.6, color=ring_ids, colorscale="Viridis"),
        name="points",
    ))
    fig_polar.add_trace(go.Scatterpolar(
        r=[ro0], theta=[phi0_deg],
        mode="markers",
        marker=dict(size=14, color="red", symbol="star"),
        name="observer",
    ))
    if abs(K) > 1e-9:
        ro_anti = np.pi / np.sqrt(abs(K))
        ct = np.linspace(0, 360, 360, endpoint=False)
        fig_polar.add_trace(go.Scatterpolar(
            r=[ro_anti] * 360, theta=ct,
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name=f"ρ = πR = {ro_anti:.2f}",
        ))

    # Viewport boundary mapped back to geodesic space
    bvx, bvy = _viewport_boundary(dro)
    if abs(K) < 1e-9:
        b_rs, b_thetas = _exp_map_flat(ro0, phi0, bvx, bvy)
    elif K > 0:
        b_rs, b_thetas = _exp_map_sphere(ro0, phi0, bvx, bvy, R=1 / np.sqrt(K))
    else:
        b_rs, b_thetas = _exp_map_hyperbolic(ro0, phi0, bvx, bvy, R=1 / np.sqrt(-K))
    fig_polar.add_trace(go.Scatterpolar(
        r=b_rs, theta=b_thetas,
        mode="lines",
        line=dict(color="red", width=2),
        name="viewport",
    ))

    fig_polar.update_layout(
        title="Geodesic rings (pole view)",
        polar=dict(radialaxis=dict(range=[0, 25])),
    )
    st.plotly_chart(fig_polar, use_container_width=True)

    # ── Tangent plane (log map) ───────────────────────────────────────────────
    if abs(K) < 1e-9:
        x_tan, y_tan = _log_map_flat(ro0, phi0, rs, phis)
    elif K > 0:
        x_tan, y_tan = _log_map_sphere(ro0, phi0, rs, phis, R=1 / np.sqrt(K))
    else:
        x_tan, y_tan = _log_map_hyperbolic(ro0, phi0, rs, phis, R=1 / np.sqrt(-K))

    fig_tan = go.Figure()
    fig_tan.add_trace(go.Scatter(
        x=x_tan, y=y_tan,
        mode="markers",
        marker=dict(size=5, opacity=0.6, color=ring_ids, colorscale="Viridis"),
        name="points",
    ))
    fig_tan.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        marker=dict(size=14, color="red", symbol="star"),
        name="observer (ρ₀, φ₀)",
    ))
    fig_tan.update_layout(
        title="Tangent plane at observer — log map",
        xaxis=dict(title="← west / east →  (ê_φ)", range=[-dro, dro],
                   constrain="domain"),
        yaxis=dict(title="← pole / away →  (ê_ρ)", range=[-dro, dro],
                   scaleanchor="x", scaleratio=1, constrain="domain"),
        width=600,
        height=600,
    )
    st.plotly_chart(fig_tan, use_container_width=False)
