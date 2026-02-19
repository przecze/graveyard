"""
Death rate model: joint CDR fit over one or more periods.

Fit modes
─────────
quadratic  (original)
    Two separate quadratic CDR polynomials joined with value + slope
    continuity at the period boundary.  2 free params (b1, c1).

unified  (n=2 specialisation)
    Single cubic CDR(t) = A + B·u + C·u² + D·u³ over both periods,
    where u(t) = ln(2026 − yr).
    Pop endpoints → linear system for (A,B) given (C,D).
    Birth integrals → fsolve for (C, D).

fit_n_unified  (general, n ≥ 2 periods)
    Single degree-(2n−1) CDR polynomial over n periods.
        CDR(t) = Σ_{k=0}^{2n-1} coef_k · u(t)^k,  u = ln(K − t)
    2n constraints:
      · n pop-endpoint constraints  → linear system → coef[0..n-1]
      · n cumulative-birth constraints → fsolve   → coef[n..2n-1]
    Initialisation: sequential brentq per free coef, then joint fsolve.

Standalone:  python death_model.py [path] [--verbose] [--unified] [--periods N]
Streamlit:   import death_model; death_model.render()
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq, fsolve

_DOCKER_PATH = Path("/data/data.json")
_LOCAL_PATH  = Path(__file__).parent.parent / "frontend" / "src" / "data.json"

N_GRID = 10_001   # grid points per period


# ── pure computation: original quadratic model ─────────────────────────────────

def fit(data_path: Path | str | None = None, verbose: bool = False,
        max_iter: int = 200, start_year: int | None = None) -> dict:
    """
    Fit joint quadratic CDR curves for two periods.

    The three anchor years (t0_1, t1_1=t0_2, t1_2) are the three consecutive
    entries in data.json beginning at *start_year* (default: the earliest year).

    Constraint reduction (all 4 derived analytically from 2 free params b1,c1):
      a1  ← period-1 end-pop  (linear)
      a2  ← CDR value continuity at boundary
      b2  ← CDR slope continuity at boundary  [d(cdr·P)/dt matches]
      c2  ← period-2 end-pop  (linear)

    Two-step initialisation, then refined with fsolve:
      Step 1 — brentq on births1 residual with c1=0  → b1₀
      Step 2 — brentq on births2 residual with b1 fixed → c1₀
      Step 3 — fsolve([b1, c1]) from (b1₀, c1₀)
    """
    if data_path is None:
        data_path = _DOCKER_PATH if _DOCKER_PATH.exists() else _LOCAL_PATH
    with open(data_path) as f:
        raw = json.load(f)
    data = {int(k): v for k, v in raw.items()}

    # ── resolve anchor years from data keys ───────────────────────────────────
    years = sorted(data.keys())
    if start_year is None:
        start_year = years[0]
    if start_year not in years:
        raise ValueError(f"start_year={start_year} not in data ({years})")
    idx = years.index(start_year)
    if idx + 2 >= len(years):
        raise ValueError(
            f"start_year={start_year} needs at least 2 more data points after it"
        )
    t0_1, t1_1, t1_2 = years[idx], years[idx + 1], years[idx + 2]

    # ── constants ─────────────────────────────────────────────────────────────
    P0_1 = float(data[t0_1]["pop"])
    P1_1 = float(data[t1_1]["pop"])
    CBR1  = data[t1_1]["cbr"] / 1000.0
    births1_tgt = float(data[t1_1]["cumulative"])
    T1 = float(t1_1 - t0_1)
    R1 = np.log(P1_1 / P0_1)

    t0_2, t1_2 = t1_1, t1_2
    P0_2 = float(data[t0_2]["pop"])
    P1_2 = float(data[t1_2]["pop"])
    cbr2_0 = data[t0_2]["cbr"] / 1000.0
    cbr2_1 = data[t1_2]["cbr"] / 1000.0
    births2_tgt = float(data[t1_2]["cumulative"])
    T2 = float(t1_2 - t0_2)
    R2 = np.log(P1_2 / P0_2)
    alpha2 = (cbr2_1 - cbr2_0) / T2

    N = N_GRID

    # ── parameter reduction ────────────────────────────────────────────────────
    def get_params(b1: float, c1: float) -> tuple[float, float, float, float]:
        a1 = R1 / T1 - CBR1 - b1 * T1 / 2.0 - c1 * T1 ** 2 / 3.0
        a2 = a1 + b1 * T1 + c1 * T1 ** 2
        b2 = b1 + 2.0 * c1 * T1
        c2 = (R2 - (cbr2_0 + a2) * T2 - (alpha2 + b2) * T2 ** 2 / 2.0) * 3.0 / T2 ** 3
        return a1, a2, b2, c2

    # ── simulation ────────────────────────────────────────────────────────────
    def simulate(b1: float, c1: float) -> dict:
        a1, a2, b2, c2 = get_params(b1, c1)

        s1 = np.linspace(0.0, T1, N)
        P1c = P0_1 * np.exp(
            (CBR1 + a1) * s1 + b1 * s1 ** 2 / 2.0 + c1 * s1 ** 3 / 3.0
        )
        births1 = CBR1 * np.trapezoid(P1c, s1)

        s2 = np.linspace(0.0, T2, N)
        cbr2_s = cbr2_0 + alpha2 * s2
        P2c = P0_2 * np.exp(
            (cbr2_0 + a2) * s2
            + (alpha2 + b2) * s2 ** 2 / 2.0
            + c2 * s2 ** 3 / 3.0
        )
        births2 = np.trapezoid(cbr2_s * P2c, s2)

        return dict(
            a1=a1, b1=b1, c1=c1, a2=a2, b2=b2, c2=c2,
            s1=s1, P1c=P1c, births1=births1,
            s2=s2, P2c=P2c, cbr2_s=cbr2_s, births2=births2,
        )

    # ── verbose landscape scans ───────────────────────────────────────────────
    if verbose:
        print("=== scan: c1=0, varying b1 ===")
        b1_vals = np.linspace(-1e-6, 1e-6, 21)
        print(f"{'b1':>14}  {'births1/tgt':>12}  {'a2*1000':>10}  {'births2/tgt':>12}")
        for b1v in b1_vals:
            r = simulate(b1v, 0.0)
            print(f"{b1v:14.3e}  {r['births1']/births1_tgt:12.6f}  "
                  f"{r['a2']*1000:10.5f}  {r['births2']/births2_tgt:12.6f}")
        print()

    # ── step 1: brentq on births1 with c1=0 ──────────────────────────────────
    def res1(b1):
        return simulate(b1, 0.0)["births1"] - births1_tgt

    b1_step1 = 0.0
    for scale in [1e-8, 1e-7, 1e-6, 1e-5]:
        if res1(-scale) * res1(scale) < 0:
            b1_step1 = brentq(res1, -scale, scale, xtol=1e-25, rtol=1e-12)
            break

    if verbose:
        r = simulate(b1_step1, 0.0)
        print(f"Step 1: b1={b1_step1:.6e}")
        print(f"  births1 err: {r['births1']-births1_tgt:+.3e}")
        print(f"  births2 err: {r['births2']-births2_tgt:+.3e}  "
              f"({r['births2']/births2_tgt:.4f}× target)\n")

    # ── step 2: brentq on births2 with b1 fixed ───────────────────────────────
    # From scan: births2 is monotone increasing in c1 for c1 > 0.
    # births2(b1_step1, c1=0) < target  →  need c1 > 0.
    # Find upper bracket by doubling until births2 > target.
    def res2(c1):
        return simulate(b1_step1, c1)["births2"] - births2_tgt

    c_scale = R1 / T1 ** 3
    c1_step2 = 0.0
    c_hi = c_scale
    for _ in range(60):
        if res2(c_hi) > 0:
            break
        c_hi *= 2.0

    if res2(0.0) * res2(c_hi) < 0:
        c1_step2 = brentq(res2, 0.0, c_hi, xtol=1e-30, rtol=1e-12)

    if verbose:
        r = simulate(b1_step1, c1_step2)
        print(f"Step 2: c1={c1_step2:.6e}")
        print(f"  births1 err: {r['births1']-births1_tgt:+.3e}")
        print(f"  births2 err: {r['births2']-births2_tgt:+.3e}\n")

    # ── step 3: joint fsolve from (b1_step1, c1_step2) ───────────────────────
    b1_ref = max(abs(b1_step1), 1e-10)
    c1_ref = max(abs(c1_step2), 1e-15)

    def residuals_norm(pn):
        r = simulate(pn[0] * b1_ref, pn[1] * c1_ref)
        return [
            (r["births1"] - births1_tgt) / births1_tgt,
            (r["births2"] - births2_tgt) / births2_tgt,
        ]

    x0 = np.array([b1_step1 / b1_ref, c1_step2 / c1_ref])
    sol, _, ier, msg = fsolve(residuals_norm, x0, full_output=True, factor=0.1,
                              maxfev=max_iter)
    b1_opt = sol[0] * b1_ref
    c1_opt = sol[1] * c1_ref

    if verbose:
        r = simulate(b1_opt, c1_opt)
        print(f"Step 3 (fsolve) ier={ier}: {msg.strip()}")
        print(f"  b1={b1_opt:.6e}, c1={c1_opt:.6e}")
        print(f"  births1: {r['births1']:.6e} vs {births1_tgt:.6e}  "
              f"({(r['births1']-births1_tgt)/births1_tgt*100:+.6f}%)")
        print(f"  births2: {r['births2']:.6e} vs {births2_tgt:.6e}  "
              f"({(r['births2']-births2_tgt)/births2_tgt*100:+.6f}%)")
        a1, a2, b2, c2 = get_params(b1_opt, c1_opt)
        print(f"  CDR at -8000: {-a1*1000:.4f}‰")
        print(f"  CDR at  1 CE: {-(a1+b1_opt*T1+c1_opt*T1**2)*1000:.4f}‰  (=a2={-a2*1000:.4f}‰)")
        print(f"  CDR at 1200: {-(a2+b2*T2+c2*T2**2)*1000:.4f}‰")

    result = simulate(b1_opt, c1_opt)
    result.update(
        converged=(ier == 1), ier=ier, msg=msg,
        t0_1=t0_1, t1_1=t1_1, t0_2=t0_2, t1_2=t1_2,
        P0_1=P0_1, P1_1=P1_1, P0_2=P0_2, P1_2=P1_2,
        CBR1=CBR1, cbr2_0=cbr2_0, cbr2_1=cbr2_1,
        births1_tgt=births1_tgt, births2_tgt=births2_tgt,
        T1=T1, T2=T2,
    )
    return result


# ── pure computation: unified cubic model ──────────────────────────────────────

def fit_unified(data_path: Path | str | None = None, verbose: bool = False,
                max_iter: int = 200, start_year: int | None = None) -> dict:
    """
    Fit a single cubic CDR polynomial over both periods joined.

    Basis: u(t) = ln(2026 − yr) = ln(K − t),  K = 2026 − t0_1
    (logarithm of "years before 2026", compressed timescale).

        CDR(t) = A + B·u(t) + C·u(t)² + D·u(t)³  [/yr]

    Four constraints
    ----------------
    (1) Pop at t_mid:  ∫₀^{T1} (CBR+CDR) dt = log(P_mid/P0)  — linear in A,B
    (2) Pop at t_end:  ∫₀^{T_tot}(CBR+CDR) dt = log(P_end/P0) — linear in A,B
        → moment integrals IU1,IU2,IU3 precomputed; 2×2 system for A,B given C,D.
    (3) ∫ births over period 1 = births1_tgt  — nonlinear → free param C
    (4) ∫ births over period 2 = births2_tgt  — nonlinear → free param D

    Three-step initialisation (mirrors fit()):
      Step 1 — brentq C (D=0) → births1 residual
      Step 2 — brentq D (C fixed) → births2 residual
      Step 3 — joint fsolve(C, D)
    """
    if data_path is None:
        data_path = _DOCKER_PATH if _DOCKER_PATH.exists() else _LOCAL_PATH
    with open(data_path) as f:
        raw = json.load(f)
    data = {int(k): v for k, v in raw.items()}

    years = sorted(data.keys())
    if start_year is None:
        start_year = years[0]
    if start_year not in years:
        raise ValueError(f"start_year={start_year} not in data ({years})")
    idx = years.index(start_year)
    if idx + 2 >= len(years):
        raise ValueError(
            f"start_year={start_year} needs at least 2 more data points after it"
        )
    t0_1, t1_1, t1_2 = years[idx], years[idx + 1], years[idx + 2]

    # ── constants ─────────────────────────────────────────────────────────────
    P0_1 = float(data[t0_1]["pop"])
    P1_1 = float(data[t1_1]["pop"])
    P1_2 = float(data[t1_2]["pop"])
    CBR1        = data[t1_1]["cbr"] / 1000.0
    births1_tgt = float(data[t1_1]["cumulative"])
    births2_tgt = float(data[t1_2]["cumulative"])
    T1    = float(t1_1 - t0_1)
    T2    = float(t1_2 - t1_1)
    T_tot = T1 + T2

    R1      = np.log(P1_1 / P0_1)
    R_total = np.log(P1_2 / P0_1)

    cbr2_0 = data[t1_1]["cbr"] / 1000.0
    cbr2_1 = data[t1_2]["cbr"] / 1000.0
    alpha2 = (cbr2_1 - cbr2_0) / T2

    GCBR_T1  = CBR1 * T1
    GCBR_tot = CBR1 * T1 + cbr2_0 * T2 + alpha2 * T2 ** 2 / 2.0

    # ── log-time basis: u(t) = ln(K − t), K = 2026 − t0_1 ───────────────────
    # t is measured from t0_1; yr = t0_1 + t  →  u = ln(2026 − yr) = ln(K − t)
    K = float(2026 - t0_1)

    # precompute u-moment integrals on a fine grid for the linear system
    _N = max(N_GRID * 4, 40_001)
    _t1  = np.linspace(0.0, T1,    _N)
    _tot = np.linspace(0.0, T_tot, _N)
    _u1  = np.log(K - _t1)
    _ut  = np.log(K - _tot)

    IU1_T1  = np.trapezoid(_u1,      _t1)
    IU2_T1  = np.trapezoid(_u1 ** 2, _t1)
    IU3_T1  = np.trapezoid(_u1 ** 3, _t1)
    IU1_tot = np.trapezoid(_ut,      _tot)
    IU2_tot = np.trapezoid(_ut ** 2, _tot)
    IU3_tot = np.trapezoid(_ut ** 3, _tot)

    # ── linear system for (A,B) given (C,D) ───────────────────────────────────
    # ∫₀^{T} CDR dt = A·T + B·IU1_T + C·IU2_T + D·IU3_T  (linear in A,B,C,D)
    # Pop constraints:
    #   A·T1 + B·IU1_T1 = R1      − GCBR_T1  − C·IU2_T1  − D·IU3_T1
    #   A·T_tot+ B·IU1_tot= R_total − GCBR_tot − C·IU2_tot − D·IU3_tot
    RHS1 = R1      - GCBR_T1
    RHS2 = R_total - GCBR_tot

    M_lin = np.array([[T1,    IU1_T1 ],
                      [T_tot, IU1_tot]])

    def get_AB(C: float, D: float) -> tuple[float, float]:
        rhs = np.array([
            RHS1 - C * IU2_T1  - D * IU3_T1,
            RHS2 - C * IU2_tot - D * IU3_tot,
        ])
        AB = np.linalg.solve(M_lin, rhs)
        return float(AB[0]), float(AB[1])

    N = N_GRID

    # ── simulation ────────────────────────────────────────────────────────────
    def simulate(C: float, D: float) -> dict:
        A, B = get_AB(C, D)

        # Period 1: t ∈ [0, T1]
        t_arr1 = np.linspace(0.0, T1, N)
        u1     = np.log(K - t_arr1)
        cdr1   = A + B * u1 + C * u1 ** 2 + D * u1 ** 3
        GCBR1  = CBR1 * t_arr1
        CDI1   = np.concatenate([[0.0], cumulative_trapezoid(cdr1, t_arr1)])
        P1     = P0_1 * np.exp(GCBR1 + CDI1)
        births1 = CBR1 * np.trapezoid(P1, t_arr1)

        # Period 2: t ∈ [T1, T_tot]  (CDI continues from CDI1[-1])
        t_arr2    = np.linspace(T1, T_tot, N)
        u2        = np.log(K - t_arr2)
        tau_local = t_arr2 - T1
        cdr2      = A + B * u2 + C * u2 ** 2 + D * u2 ** 3
        cbr2_vals = cbr2_0 + alpha2 * tau_local
        GCBR2     = GCBR_T1 + cbr2_0 * tau_local + alpha2 * tau_local ** 2 / 2.0
        CDI2_incr = np.concatenate([[0.0], cumulative_trapezoid(cdr2, t_arr2)])
        CDI2      = CDI1[-1] + CDI2_incr
        P2        = P0_1 * np.exp(GCBR2 + CDI2)
        births2   = np.trapezoid(cbr2_vals * P2, t_arr2)

        return dict(
            A=A, B=B, C=C, D=D,
            t_arr1=t_arr1, P1=P1, cdr1=cdr1, births1=births1,
            t_arr2=t_arr2, P2=P2, cbr2_vals=cbr2_vals, cdr2=cdr2, births2=births2,
        )

    # ── verbose landscape scan ────────────────────────────────────────────────
    if verbose:
        u0 = np.log(K)
        C_scale = 1.0 / u0 ** 2   # 1‰ effect at t=0 per this unit of C
        C_vals = np.linspace(-5 * C_scale, 5 * C_scale, 21)
        print(f"=== scan: D=0, varying C  (u0={u0:.3f}, C_scale={C_scale:.3e}) ===")
        print(f"{'C':>14}  {'births1/tgt':>12}  {'births2/tgt':>12}")
        for Cv in C_vals:
            with np.errstate(over="ignore", invalid="ignore"):
                r = simulate(Cv, 0.0)
            b1r = r["births1"] / births1_tgt if np.isfinite(r["births1"]) else float("inf")
            b2r = r["births2"] / births2_tgt if np.isfinite(r["births2"]) else float("inf")
            print(f"{Cv:14.4e}  {b1r:12.6f}  {b2r:12.6f}")
        print()

    # ── step 1: brentq C (D=0) → births1 ────────────────────────────────────
    def res1(C: float) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            b1 = simulate(C, 0.0)["births1"]
        return (b1 - births1_tgt) if np.isfinite(b1) else np.sign(-C) * 1e30

    C_step1 = 0.0
    u0 = np.log(K)
    # Search for bracket: double from a small scale in each direction
    _C_try = 1e-6 / u0 ** 2
    for _ in range(80):
        if res1(-_C_try) * res1(_C_try) < 0:
            C_step1 = brentq(res1, -_C_try, _C_try, xtol=1e-20, rtol=1e-12)
            break
        _C_try *= 2.0

    if verbose:
        r = simulate(C_step1, 0.0)
        print(f"Step 1: C={C_step1:.6e}")
        print(f"  births1 err: {r['births1']-births1_tgt:+.3e}")
        print(f"  births2 err: {r['births2']-births2_tgt:+.3e}\n")

    # ── step 2: brentq D (C fixed) → births2 ─────────────────────────────────
    # births2 > target at C_step1 → need D < 0 (more deaths reduce population).
    # Positive D causes population explosion and overflow, so search only the
    # side whose residual has opposite sign from res2(0).
    def res2(D: float) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            b2 = simulate(C_step1, D)["births2"]
        return (b2 - births2_tgt) if np.isfinite(b2) else np.sign(-D) * 1e30

    D_step2 = 0.0
    r2_at_0 = simulate(C_step1, 0.0)["births2"] - births2_tgt

    # Determine search direction: births2 > target → need D < 0
    direction = -1.0 if r2_at_0 > 0 else 1.0
    D_trial = direction * 1e-5
    for _ in range(80):
        r_trial = res2(D_trial)
        if r_trial * r2_at_0 < 0:
            break
        D_trial *= 2.0

    lo = min(D_trial, 0.0)
    hi = max(D_trial, 0.0)
    if res2(lo) * res2(hi) < 0:
        D_step2 = brentq(res2, lo, hi, xtol=1e-20, rtol=1e-12)

    if verbose:
        r = simulate(C_step1, D_step2)
        print(f"Step 2: D={D_step2:.6e}")
        print(f"  births1 err: {r['births1']-births1_tgt:+.3e}")
        print(f"  births2 err: {r['births2']-births2_tgt:+.3e}\n")

    # ── step 3: joint fsolve from (C_step1, D_step2) ─────────────────────────
    C_ref = max(abs(C_step1), 1e-6)
    D_ref = max(abs(D_step2), 1e-6)

    def residuals_norm(xn):
        r = simulate(xn[0] * C_ref, xn[1] * D_ref)
        return [
            (r["births1"] - births1_tgt) / births1_tgt,
            (r["births2"] - births2_tgt) / births2_tgt,
        ]

    x0 = np.array([C_step1 / C_ref, D_step2 / D_ref])
    sol, _, ier, msg = fsolve(residuals_norm, x0, full_output=True, factor=0.1,
                              maxfev=max_iter)
    C_opt = sol[0] * C_ref
    D_opt = sol[1] * D_ref

    if verbose:
        r = simulate(C_opt, D_opt)
        print(f"Step 3 (fsolve) ier={ier}: {msg.strip()}")
        print(f"  C={C_opt:.6e}, D={D_opt:.6e}")
        print(f"  A={r['A']:.6e}  B={r['B']:.6e}  "
              f"C={r['C']:.6e}  D={r['D']:.6e}")
        print(f"  births1: {r['births1']:.6e} vs {births1_tgt:.6e}  "
              f"({(r['births1']-births1_tgt)/births1_tgt*100:+.6f}%)")
        print(f"  births2: {r['births2']:.6e} vs {births2_tgt:.6e}  "
              f"({(r['births2']-births2_tgt)/births2_tgt*100:+.6f}%)")
        print(f"  CDR at t0_1 ({t0_1}):  {-r['cdr1'][0]*1000:.4f}‰")
        print(f"  CDR at t1_1 ({t1_1}):  {-r['cdr1'][-1]*1000:.4f}‰")
        print(f"  CDR at t1_2 ({t1_2}):  {-r['cdr2'][-1]*1000:.4f}‰")

    result = simulate(C_opt, D_opt)
    result.update(
        converged=(ier == 1), ier=ier, msg=msg,
        t0_1=t0_1, t1_1=t1_1, t1_2=t1_2,
        P0_1=P0_1, P1_1=P1_1, P1_2=P1_2,
        CBR1=CBR1, cbr2_0=cbr2_0, cbr2_1=cbr2_1,
        births1_tgt=births1_tgt, births2_tgt=births2_tgt,
        T1=T1, T2=T2, T_tot=T_tot,
    )
    return result


# ── generalised n-period unified model ────────────────────────────────────────

def fit_n_unified(
    data_path: Path | str | None = None,
    verbose: bool = False,
    max_iter: int = 200,
    start_year: int | None = None,
    n_periods: int = 2,
) -> dict:
    """
    Fit a single degree-(2n−1) CDR polynomial over n consecutive periods.

        CDR(t) = Σ_{k=0}^{2n-1} coef_k · u(t)^k   where  u(t) = ln(K − t),  K = 2026 − t₀

    2n constraints
    ──────────────
    n  pop-endpoint constraints  (t₁…tₙ)  → linear system → coef[0..n-1]
    n  cumulative-birth constraints        → fsolve        → coef[n..2n-1]  (free params)

    CBR within each period j is piecewise linear between the data's CBR values at
    anchor_years[j] and anchor_years[j+1].

    Initialisation (mirrors fit_unified)
    ─────────────────────────────────────
    For k = 0…n-1 : brentq free_coef[k] (others held fixed) → zeroes births[k] residual.
    Then joint fsolve from that seed.
    """
    n = n_periods
    if data_path is None:
        data_path = _DOCKER_PATH if _DOCKER_PATH.exists() else _LOCAL_PATH
    with open(data_path) as f:
        raw = json.load(f)
    data = {int(k): v for k, v in raw.items()}

    years = sorted(data.keys())
    if start_year is None:
        start_year = years[0]
    if start_year not in years:
        raise ValueError(f"start_year={start_year} not in data ({years})")
    idx = years.index(start_year)
    if idx + n >= len(years):
        raise ValueError(
            f"start_year={start_year} needs at least {n} more data points after it"
        )
    anchor_years = [years[idx + i] for i in range(n + 1)]

    def _yfmt(y: int) -> str:
        return f"{-y} BCE" if y <= 0 else f"{y} CE"

    P0 = float(data[anchor_years[0]]["pop"])
    T_periods = [float(anchor_years[j + 1] - anchor_years[j]) for j in range(n)]
    T_cumul   = np.concatenate([[0.0], np.cumsum(T_periods)])   # length n+1
    cbr_anchors = [data[y]["cbr"] / 1000.0 for y in anchor_years]  # length n+1

    # ── cumulative CBR integrals ───────────────────────────────────────────────
    # GCBR_arr[i] = ∫₀^{T_cumul[i]} CBR(t) dt  (piecewise-linear trapezoid)
    GCBR_arr = np.zeros(n + 1)
    for j in range(n):
        GCBR_arr[j + 1] = (
            GCBR_arr[j]
            + (cbr_anchors[j] + cbr_anchors[j + 1]) / 2.0 * T_periods[j]
        )

    # Pop ratios: R_arr[i] = log(P_{i+1} / P0)
    R_arr = np.array([
        np.log(float(data[anchor_years[i + 1]]["pop"]) / P0) for i in range(n)
    ])

    # Target cumulative births per period
    births_tgt = [float(data[anchor_years[j + 1]]["cumulative"]) for j in range(n)]

    # Log-time basis: u(t) = ln(K − t),  K = 2026 − anchor_years[0]
    K = float(2026 - anchor_years[0])

    # ── moment integrals ───────────────────────────────────────────────────────
    # IU[k, i] = ∫₀^{T_cumul[i+1]} u(t)^k dt   (k = 0..2n-1, i = 0..n-1)
    n_coef  = 2 * n
    _N_fine = max(N_GRID * 4, 40_001)
    IU = np.zeros((n_coef, n))
    for i in range(n):
        _t = np.linspace(0.0, T_cumul[i + 1], _N_fine)
        _u = np.log(K - _t)
        for k in range(n_coef):
            IU[k, i] = np.trapezoid(_u ** k, _t)

    # Linear system:
    #   M_low  @ low_coef  +  M_high @ high_coef  =  RHS_base
    #   low_coef  = solve(M_low,  RHS_base − M_high @ high_coef)
    M_low    = IU[:n, :].T   # shape (n, n)  row i, col k = IU[k,i] k∈0..n-1
    M_high   = IU[n:, :].T   # shape (n, n)  row i, col k = IU[k+n,i]
    RHS_base = R_arr - GCBR_arr[1:]  # shape (n,)

    def get_low_coef(high_coef: np.ndarray) -> np.ndarray:
        rhs = RHS_base - M_high @ high_coef
        return np.linalg.solve(M_low, rhs)

    # ── simulation ────────────────────────────────────────────────────────────
    def simulate(high_coef: np.ndarray) -> dict:
        low_coef = get_low_coef(high_coef)
        coef     = np.concatenate([low_coef, high_coef])

        period_results: list[dict] = []
        CDI_prev = 0.0

        for j in range(n):
            t_arr   = np.linspace(T_cumul[j], T_cumul[j + 1], N_GRID)
            tau     = t_arr - T_cumul[j]
            alpha_j = (cbr_anchors[j + 1] - cbr_anchors[j]) / T_periods[j]

            u   = np.log(K - t_arr)
            cdr = sum(coef[k] * u ** k for k in range(n_coef))

            cbr_arr    = cbr_anchors[j] + alpha_j * tau
            gcbr_local = cbr_anchors[j] * tau + alpha_j * tau ** 2 / 2.0
            gcbr_total = GCBR_arr[j] + gcbr_local

            cdi_local = np.concatenate([[0.0], cumulative_trapezoid(cdr, t_arr)])
            cdi_total = CDI_prev + cdi_local

            P      = P0 * np.exp(gcbr_total + cdi_total)
            births = np.trapezoid(cbr_arr * P, t_arr)

            CDI_prev += cdi_local[-1]

            period_results.append({
                "t_arr":   t_arr,
                "u":       u,
                "cdr":     cdr,
                "cbr_arr": cbr_arr,
                "P":       P,
                "births":  births,
            })

        return dict(
            coef=coef,
            low_coef=low_coef,
            high_coef=high_coef,
            periods=period_results,
            births_list=[pr["births"] for pr in period_results],
        )

    # ── sequential brentq initialisation ──────────────────────────────────────
    u0 = np.log(K)
    high_init = np.zeros(n)

    if verbose:
        print(f"=== fit_n_unified  n={n}  degree={n_coef - 1}  K={K:.1f}  u0={u0:.4f} ===")
        print(f"Anchor years: {[_yfmt(y) for y in anchor_years]}\n")

    for step in range(n):
        k_power = n + step   # polynomial power for this free coef

        def res_step(x: float, _s: int = step) -> float:
            trial = high_init.copy()
            trial[_s] = x
            with np.errstate(over="ignore", invalid="ignore"):
                r = simulate(trial)
            b = r["births_list"][_s]
            return (b - births_tgt[_s]) if np.isfinite(b) else np.sign(-x) * 1e30

        scale = 1e-6 / max(u0 ** k_power, 1.0)
        lo, hi = -scale, scale
        found  = False
        for _ in range(100):
            if res_step(lo) * res_step(hi) < 0:
                found = True
                break
            lo *= 2.0
            hi *= 2.0

        if found:
            high_init[step] = brentq(res_step, lo, hi, xtol=1e-20, rtol=1e-12)

        if verbose:
            r = simulate(high_init)
            print(f"Step {step + 1}: free_coef[{step}] (u^{k_power}) = {high_init[step]:.6e}"
                  f"  bracket found={found}")
            for j in range(n):
                err = r["births_list"][j] - births_tgt[j]
                pct = err / births_tgt[j] * 100
                print(f"  births[{j}] ({_yfmt(anchor_years[j])}→{_yfmt(anchor_years[j+1])}): "
                      f"{r['births_list'][j]:.5e} vs {births_tgt[j]:.5e}  "
                      f"err {err:+.3e} ({pct:+.4f}%)")
            print()

    # ── joint fsolve ──────────────────────────────────────────────────────────
    refs = np.array([max(abs(x), 1e-15) for x in high_init])

    def residuals_norm(xn: np.ndarray) -> list[float]:
        r = simulate(xn * refs)
        return [
            (r["births_list"][j] - births_tgt[j]) / births_tgt[j] for j in range(n)
        ]

    x0  = high_init / refs
    sol, _, ier, msg = fsolve(residuals_norm, x0, full_output=True, factor=0.1,
                              maxfev=max_iter)
    high_opt = sol * refs
    r_final  = simulate(high_opt)

    if verbose:
        print(f"fsolve ier={ier}: {msg.strip()}")
        coef = r_final["coef"]
        print(f"  coef = [{', '.join(f'{c:.6e}' for c in coef)}]")
        print()
        print(f"{'Period':<30}  {'Births model':>14}  {'Births target':>14}  "
              f"{'Err %':>9}  {'P_end model':>14}  {'P_end data':>14}  "
              f"{'CDR start':>10}  {'CDR end':>10}")
        for j in range(n):
            pr      = r_final["periods"][j]
            b_mod   = r_final["births_list"][j]
            b_tgt   = births_tgt[j]
            pct     = (b_mod - b_tgt) / b_tgt * 100
            P_end   = pr["P"][-1]
            P_data  = float(data[anchor_years[j + 1]]["pop"])
            cdr_s   = -pr["cdr"][0]  * 1000
            cdr_e   = -pr["cdr"][-1] * 1000
            label   = f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j+1])}"
            print(f"  {label:<28}  {b_mod:>14.5e}  {b_tgt:>14.5e}  "
                  f"{pct:>+9.4f}%  {P_end:>14.4e}  {P_data:>14.4e}  "
                  f"{cdr_s:>10.4f}‰  {cdr_e:>10.4f}‰")
        print()
        print(f"  CDR values at anchor years:")
        for j in range(n + 1):
            if j == 0:
                cdr_val = -r_final["periods"][0]["cdr"][0] * 1000
            elif j == n:
                cdr_val = -r_final["periods"][-1]["cdr"][-1] * 1000
            else:
                cdr_val = -r_final["periods"][j - 1]["cdr"][-1] * 1000
            print(f"    {_yfmt(anchor_years[j])}: {cdr_val:.4f}‰")

    result = dict(r_final)
    result.update(
        converged=(ier == 1),
        ier=ier,
        msg=msg,
        anchor_years=anchor_years,
        P0=P0,
        births_tgt=births_tgt,
        T_periods=T_periods,
        T_cumul=T_cumul,
        cbr_anchors=cbr_anchors,
        n_periods=n,
        K=K,
        u0=u0,
    )
    return result


# ── streamlit rendering ────────────────────────────────────────────────────────

# Colour palettes — one entry per period, cycling if n > len
_POP_COLOURS = [
    "#1f78b4", "#33a02c", "#ff7f00", "#6a3d9a", "#e31a1c",
    "#b15928", "#a6cee3", "#b2df8a",
]
_CDR_COLOURS = [
    "#e31a1c", "#ff7f00", "#6a3d9a", "#33a02c", "#1f78b4",
    "#b15928", "#fb9a99", "#fdbf6f",
]


def render():
    import plotly.graph_objects as go
    import streamlit as st

    def _yfmt(y: int) -> str:
        return f"{-y} BCE" if y <= 0 else f"{y} CE"

    data_path = _DOCKER_PATH if _DOCKER_PATH.exists() else _LOCAL_PATH
    with open(data_path) as _f:
        _raw = json.load(_f)
    _years = sorted(int(k) for k in _raw)
    max_n = len(_years) - 1   # maximum number of periods coverable

    # ── controls ──────────────────────────────────────────────────────────────
    col_n, col_start, col_iter = st.columns([1, 2, 1])
    with col_n:
        n_periods = st.slider("Number of periods", min_value=2, max_value=max_n,
                              value=3, step=1)

    valid_starts = _years[: len(_years) - n_periods]

    def _start_label(y: int) -> str:
        end = _years[_years.index(y) + n_periods]
        mid_list = [_yfmt(_years[_years.index(y) + k]) for k in range(1, n_periods)]
        inner = " → ".join(mid_list)
        return f"{_yfmt(y)} → {inner} → {_yfmt(end)}" if inner else f"{_yfmt(y)} → {_yfmt(end)}"

    with col_start:
        start_year = st.selectbox(
            "Starting year",
            options=valid_starts,
            format_func=_start_label,
            index=0,
        )
    with col_iter:
        max_iter = st.slider("Max solver iterations",
                             min_value=10, max_value=2000, value=200, step=10)

    # ── fit ───────────────────────────────────────────────────────────────────
    with st.spinner(f"Fitting degree-{2 * n_periods - 1} CDR over {n_periods} periods…"):
        r = fit_n_unified(data_path=data_path, max_iter=max_iter,
                          start_year=start_year, n_periods=n_periods)

    if not r["converged"]:
        st.warning(f"fsolve did not fully converge (ier={r['ier']}); result may be approximate.")

    anchor_years = r["anchor_years"]
    n            = r["n_periods"]

    st.subheader(f"Unified CDR fit — {n} period{'s' if n > 1 else ''}, degree {2*n-1} polynomial")
    st.caption(
        f"CDR(t) = Σ_{{k=0}}^{{{2*n-1}}} coef_k · u^k,   u = ln(2026 − yr).  "
        f"CBR piecewise linear per period.  "
        f"{n} pop endpoints → coef[0..{n-1}] (linear);  "
        f"{n} birth integrals → coef[{n}..{2*n-1}] (fsolve)."
    )

    # ── population plot ───────────────────────────────────────────────────────
    P_anchors = [r["P0"]] + [pr["P"][-1] for pr in r["periods"]]
    fig_pop = go.Figure()
    for j, pr in enumerate(r["periods"]):
        yr_arr = anchor_years[0] + pr["t_arr"]
        fig_pop.add_trace(go.Scatter(
            x=yr_arr, y=pr["P"], mode="lines",
            line=dict(color=_POP_COLOURS[j % len(_POP_COLOURS)]),
            name=f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j+1])}",
        ))
    fig_pop.add_trace(go.Scatter(
        x=anchor_years, y=P_anchors, mode="markers",
        marker=dict(size=12, color="tomato", symbol="diamond"),
        name="PRB benchmarks",
    ))
    fig_pop.update_layout(
        title="Reconstructed population",
        xaxis_title="Year (−ve = BCE)", yaxis_title="Population",
        legend=dict(x=0.01, y=0.99), margin=dict(t=40),
    )
    st.plotly_chart(fig_pop, width="stretch")

    # ── CDR / CBR plot ────────────────────────────────────────────────────────
    fig_cdr = go.Figure()
    for j, pr in enumerate(r["periods"]):
        yr_arr = anchor_years[0] + pr["t_arr"]
        col_p  = _POP_COLOURS[j % len(_POP_COLOURS)]
        col_d  = _CDR_COLOURS[j % len(_CDR_COLOURS)]
        label  = f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j+1])}"
        fig_cdr.add_trace(go.Scatter(
            x=yr_arr, y=-pr["cdr"] * 1000, mode="lines",
            line=dict(color=col_d), name=f"CDR {label}",
        ))
        fig_cdr.add_trace(go.Scatter(
            x=yr_arr, y=pr["cbr_arr"] * 1000, mode="lines",
            line=dict(color=col_p, dash="dash"), name=f"CBR {label}",
        ))
    fig_cdr.update_layout(
        title="Fitted CDR vs CBR (per 1 000)",
        xaxis_title="Year (−ve = BCE)", yaxis_title="Rate (per 1 000)",
        legend=dict(x=0.01, y=0.99), margin=dict(t=40),
    )
    st.plotly_chart(fig_cdr, width="stretch")

    # ── deaths per year plot ──────────────────────────────────────────────────
    fig_deaths = go.Figure()
    for j, pr in enumerate(r["periods"]):
        yr_arr = anchor_years[0] + pr["t_arr"]
        label  = f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j+1])}"
        fig_deaths.add_trace(go.Scatter(
            x=yr_arr, y=-pr["cdr"] * pr["P"], mode="lines",
            line=dict(color=_CDR_COLOURS[j % len(_CDR_COLOURS)]),
            name=label,
        ))
    fig_deaths.update_layout(
        title="Deaths per year",
        xaxis_title="Year (−ve = BCE)", yaxis_title="Deaths / year",
        legend=dict(x=0.01, y=0.99), margin=dict(t=40),
    )
    st.plotly_chart(fig_deaths, width="stretch")

    # ── growth rate plot ──────────────────────────────────────────────────────
    fig_gr = go.Figure()
    for j, pr in enumerate(r["periods"]):
        yr_arr = anchor_years[0] + pr["t_arr"]
        gr_pm  = (pr["cbr_arr"] + pr["cdr"]) * 1000   # net growth rate ‰/yr
        label  = f"{_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j+1])}"
        fig_gr.add_trace(go.Scatter(
            x=yr_arr, y=gr_pm, mode="lines",
            line=dict(color=_POP_COLOURS[j % len(_POP_COLOURS)]),
            name=label,
        ))
    fig_gr.add_hline(y=0, line=dict(color="black", dash="dot", width=1))
    fig_gr.update_layout(
        title="Net population growth rate (CBR + CDR, per 1 000)",
        xaxis_title="Year (−ve = BCE)", yaxis_title="Growth rate (‰/yr)",
        legend=dict(x=0.01, y=0.99), margin=dict(t=40),
    )
    st.plotly_chart(fig_gr, width="stretch")

    # ── births check table ────────────────────────────────────────────────────
    st.markdown("### Births check")
    rows = ("| Period | PRB target | Model | Diff | Diff % |\n"
            "|--------|-----------|-------|------|--------|\n")
    for j in range(n):
        b_tgt = r["births_tgt"][j]
        b_mod = r["births_list"][j]
        rows += (
            f"| {_yfmt(anchor_years[j])} → {_yfmt(anchor_years[j+1])} | "
            f"{b_tgt:,.0f} | {b_mod:,.0f} | "
            f"{b_mod - b_tgt:+,.0f} | {(b_mod - b_tgt) / b_tgt * 100:+.4f}% |\n"
        )
    st.markdown(rows)

    # ── model parameters expander ─────────────────────────────────────────────
    with st.expander("Model parameters"):
        coef  = r["coef"]
        K_val = r["K"]
        u_hi  = np.log(K_val)
        u_lo  = np.log(K_val - r["T_cumul"][-1])
        terms = "  \n".join(
            f"coef[{k}] = `{c:.6e}`" for k, c in enumerate(coef)
        )
        st.markdown(
            f"**CDR(t) = Σ coef_k · u^k**   u = ln({int(K_val)} − t from {_yfmt(anchor_years[0])})  \n"
            f"u ∈ [{u_lo:.3f}, {u_hi:.3f}]  \n\n"
            f"{terms}\n\n"
        )
        cdr_rows = ("| Anchor year | CDR (‰) | CBR (‰) |\n"
                    "|-------------|---------|----------|\n")
        for j in range(n + 1):
            if j == 0:
                cdr_val = -r["periods"][0]["cdr"][0] * 1000
            elif j == n:
                cdr_val = -r["periods"][-1]["cdr"][-1] * 1000
            else:
                cdr_val = -r["periods"][j - 1]["cdr"][-1] * 1000
            cbr_val = r["cbr_anchors"][j] * 1000
            cdr_rows += f"| {_yfmt(anchor_years[j])} | {cdr_val:.2f} | {cbr_val:.1f} |\n"
        st.markdown(cdr_rows)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    verbose = "--verbose" in args
    unified = "--unified" in args

    # --periods N  →  run fit_n_unified with n_periods=N
    n_periods: int | None = None
    for i, a in enumerate(args):
        if a == "--periods" and i + 1 < len(args):
            n_periods = int(args[i + 1])
            break

    path_args = [
        a for a in args
        if not a.startswith("--") and not (
            len(args) > args.index(a) - 1 >= 0
            and args[args.index(a) - 1] == "--periods"
        )
    ]
    # Simpler: filter out flag tokens and their integer argument
    _skip_next = False
    path_args = []
    for a in args:
        if _skip_next:
            _skip_next = False
            continue
        if a == "--periods":
            _skip_next = True
            continue
        if not a.startswith("--"):
            path_args.append(a)
    path = Path(path_args[0]) if path_args else None

    if n_periods is not None:
        print(f"=== fit_n_unified  n_periods={n_periods}  degree={2*n_periods-1} ===\n")
        fit_n_unified(path, verbose=verbose, n_periods=n_periods)
    elif unified:
        print("=== fit_unified (single cubic CDR, n=2) ===\n")
        fit_unified(path, verbose=verbose)
    else:
        print("=== fit (joint quadratic CDR) ===\n")
        fit(path, verbose=verbose)
