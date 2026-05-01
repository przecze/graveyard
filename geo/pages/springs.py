import numpy as np
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(page_title="Spring-Ball Surface", layout="wide")
st.title("Spring-Ball Surface Relaxation")

# ── sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Grid")
    n_layers = st.slider("Layers", 1, 20, 4)
    growth_exp = st.slider("Ring growth exponent", 1.0, 2.5, 1.5, 0.01,
                           help="Nodes per ring ≈ 6·k^exp  (exp=1 → linear, exp>1 → super-linear)")

    st.header("Springs")
    base_rest = st.slider("Base rest length", 0.5, 2.0, 1.0, 0.05)
    alpha = st.slider("α  (outer growth amplitude)", 0.0, 1.5, 0.4, 0.05,
                      help="How much rest length grows toward the edge")
    beta = st.slider("β  (rest-length exponent)", 0.5, 3.0, 1.8, 0.1,
                     help="β > 1 → outer springs longer → surface domes up")

    st.header("Initial shape")
    dome_h = st.slider("Dome seed height", 0.0, 4.0, 1.5, 0.1,
                       help="Centre starts this high; outer ring at z = 0")
    jitter = st.slider("Noise", 0.0, 0.3, 0.05, 0.01)

    st.header("Solver")
    fix_center = st.checkbox("Fix centre node", value=True)
    fix_boundary = st.checkbox("Fix boundary ring", value=False)
    solver = st.selectbox("Method", ["L-BFGS-B", "CG", "BFGS"])
    maxiter = st.slider("Max iterations", 50, 2000, 600, 50)

    run = st.button("Relax ↩", type="primary")

# ── graph builder ─────────────────────────────────────────────────────────────


def _rest(k, n_layers, base_rest, alpha, beta):
    return base_rest * (1.0 + alpha * (k / max(n_layers, 1)) ** beta)


def _angle_diff(arr, ref):
    d = (arr - ref) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def _hex_graph(n_layers, base_rest, alpha, beta):
    """Standard axial hex grid — exact positions and connectivity."""
    G = nx.Graph()
    coord_to_id = {}
    nid = 0
    for q in range(-n_layers, n_layers + 1):
        for r in range(-n_layers, n_layers + 1):
            layer = max(abs(q), abs(r), abs(q + r))
            if layer > n_layers:
                continue
            G.add_node(nid, layer=layer,
                       x2d=q + r * 0.5, y2d=r * (3 ** 0.5) / 2)
            coord_to_id[(q, r)] = nid
            nid += 1
    for (q, r), u in coord_to_id.items():
        for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
            nb = (q + dq, r + dr)
            if nb not in coord_to_id:
                continue
            v = coord_to_id[nb]
            if v <= u:
                continue
            k_avg = (G.nodes[u]["layer"] + G.nodes[v]["layer"]) / 2.0
            G.add_edge(u, v, rest=_rest(k_avg, n_layers, base_rest, alpha, beta))
    return G


def _ring_graph(n_layers, growth_exp, base_rest, alpha, beta):
    """Circular rings with n_k = round(6·k^exp) nodes, zipper-triangulated."""
    G = nx.Graph()
    ring_nodes = [[0]]
    G.add_node(0, layer=0, x2d=0.0, y2d=0.0, angle=0.0)
    nid = 1

    for k in range(1, n_layers + 1):
        n_ring = max(3, round(6 * k ** growth_exp))
        # radius so arc-length between tangential neighbours ≈ base_rest
        # (excess over linear growth compresses radial springs → hyperbolic folds)
        R_k = n_ring * base_rest / (2 * np.pi)
        angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
        ring = []
        for theta in angles:
            G.add_node(nid, layer=k,
                       x2d=R_k * np.cos(theta), y2d=R_k * np.sin(theta), angle=float(theta))
            ring.append(nid)
            nid += 1
        ring_nodes.append(ring)

        rest_t = _rest(k, n_layers, base_rest, alpha, beta)
        for i in range(n_ring):
            G.add_edge(ring[i], ring[(i + 1) % n_ring], rest=rest_t)

        prev = ring_nodes[k - 1]
        rest_r = _rest(k - 0.5, n_layers, base_rest, alpha, beta)

        if k == 1:
            for v in ring:
                G.add_edge(0, v, rest=rest_r)
        else:
            n_in, n_out = len(prev), n_ring
            ia = np.array([G.nodes[n]["angle"] for n in prev])
            oa = np.array([G.nodes[n]["angle"] for n in ring])
            start_j = int(np.argmin(_angle_diff(ia, oa[0])))
            for i in range(n_out):
                j = (start_j + (i * n_in) // n_out) % n_in
                j1 = (start_j + ((i + 1) * n_in) // n_out) % n_in
                if not G.has_edge(ring[i], prev[j]):
                    G.add_edge(ring[i], prev[j], rest=rest_r)
                if j1 != j and not G.has_edge(ring[i], prev[j1]):
                    G.add_edge(ring[i], prev[j1], rest=rest_r)

    return G


@st.cache_data
def build_graph(n_layers, growth_exp, base_rest, alpha, beta):
    if growth_exp <= 1.0:
        return _hex_graph(n_layers, base_rest, alpha, beta)
    return _ring_graph(n_layers, growth_exp, base_rest, alpha, beta)


G = build_graph(n_layers, growth_exp, base_rest, alpha, beta)
N = G.number_of_nodes()

# ── fixed-node mask ───────────────────────────────────────────────────────────

fixed = np.zeros(N, dtype=bool)
for n in G.nodes():
    if fix_center and G.nodes[n]["layer"] == 0:
        fixed[n] = True
    if fix_boundary and G.nodes[n]["layer"] == n_layers:
        fixed[n] = True

# ── initial positions ─────────────────────────────────────────────────────────

rng = np.random.default_rng(42)


def make_initial_pos():
    pos = np.zeros((N, 3))
    for n in G.nodes():
        x, y = G.nodes[n]["x2d"], G.nodes[n]["y2d"]
        t = 1.0 - G.nodes[n]["layer"] / max(n_layers, 1)
        pos[n] = [x, y, dome_h * t]
    pos += rng.standard_normal((N, 3)) * jitter
    return pos


# ── spring energy + gradient ──────────────────────────────────────────────────

edges_u = np.array([u for u, v in G.edges()])
edges_v = np.array([v for u, v in G.edges()])
rest_lens = np.array([d["rest"] for _, _, d in G.edges(data=True)])
layers_arr = np.array([G.nodes[n]["layer"] for n in G.nodes()])


def energy_and_grad(flat_free, free_idx, base_pos):
    pos = base_pos.copy()
    pos[free_idx] = flat_free.reshape(-1, 3)
    d = pos[edges_v] - pos[edges_u]
    dist = np.maximum(np.linalg.norm(d, axis=1), 1e-10)
    stretch = dist - rest_lens
    E = 0.5 * np.sum(stretch ** 2)
    f = (stretch / dist)[:, None] * d
    grad = np.zeros_like(pos)
    np.add.at(grad, edges_u, -f)
    np.add.at(grad, edges_v,  f)
    return E, grad[free_idx].ravel()


# ── session state ─────────────────────────────────────────────────────────────

cache_key = (n_layers, growth_exp)
if "pos" not in st.session_state or st.session_state.get("cache_key") != cache_key:
    st.session_state.pos = make_initial_pos()
    st.session_state.cache_key = cache_key

if run:
    base_pos = make_initial_pos()
    free_idx = np.where(~fixed)[0]
    result = minimize(
        energy_and_grad, base_pos[free_idx].ravel(),
        args=(free_idx, base_pos), method=solver, jac=True,
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
    )
    final_pos = base_pos.copy()
    final_pos[free_idx] = result.x.reshape(-1, 3)
    st.session_state.pos = final_pos

pos = st.session_state.pos

# ── 3d plot ───────────────────────────────────────────────────────────────────

xe, ye, ze = [], [], []
for u, v in G.edges():
    xe += [pos[u, 0], pos[v, 0], None]
    ye += [pos[u, 1], pos[v, 1], None]
    ze += [pos[u, 2], pos[v, 2], None]

fig3d = go.Figure()
fig3d.add_trace(go.Scatter3d(
    x=xe, y=ye, z=ze, mode="lines",
    line=dict(color="rgba(80,130,200,0.45)", width=2),
    hoverinfo="none", showlegend=False,
))
fig3d.add_trace(go.Scatter3d(
    x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers",
    marker=dict(size=5, color=layers_arr, colorscale="Plasma",
                showscale=True, colorbar=dict(title="Layer", thickness=14)),
    text=[f"layer {layers_arr[n]}" for n in range(N)],
    hovertemplate="%{text}<br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
    showlegend=False,
))
fig3d.update_layout(
    scene=dict(
        xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
        aspectmode="data", bgcolor="rgb(15,17,22)",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        zaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
    ),
    paper_bgcolor="rgb(15,17,22)", font=dict(color="white"),
    margin=dict(l=0, r=0, t=30, b=0), height=680,
)
st.plotly_chart(fig3d, use_container_width=True)

# ── stats ─────────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Nodes", N)
c2.metric("Edges", G.number_of_edges())
c3.metric("Fixed nodes", int(fixed.sum()))
dists = np.linalg.norm(pos[edges_v] - pos[edges_u], axis=1)
c4.metric("Spring energy", f"{0.5 * np.sum((dists - rest_lens) ** 2):.4f}")

# ── nodes per layer plot ──────────────────────────────────────────────────────

st.subheader("Nodes per layer")

ks = list(range(n_layers + 1))
actual = [len([n for n in G.nodes() if G.nodes[n]["layer"] == k]) for k in ks]
linear = [actual[1] * k if k > 0 else 1 for k in ks]

fig2d = go.Figure()
fig2d.add_trace(go.Bar(
    x=ks, y=actual, name=f"actual (exp={growth_exp:.1f})",
    marker_color="#a78bfa",
))
fig2d.add_trace(go.Scatter(
    x=ks, y=linear, mode="lines", name="linear reference",
    line=dict(color="#6ee7b7", width=2, dash="dot"),
))
fig2d.update_layout(
    xaxis=dict(title="layer", tickmode="linear", dtick=1),
    yaxis=dict(title="nodes per layer"),
    legend=dict(orientation="h", y=1.08),
    margin=dict(l=0, r=0, t=10, b=0),
    height=280,
    paper_bgcolor="rgb(15,17,22)",
    plot_bgcolor="rgb(15,17,22)",
    font=dict(color="white"),
)
st.plotly_chart(fig2d, use_container_width=True)
