// The ancient zone as one smooth curve: ring radius f(ρ) from the centre to
// the ancient radius R, with no pieces.
//
// Unknown: the growth rate of the rings, f'(ρ) = (f0/R)·p(ρ/R), p a quintic
// B-spline on x ∈ [0, 1]; f is its integral. Every constraint is linear in p:
//   centre   f'(0) = 1, f''(0) = 0, f''''(0) = 0     (f ≈ ρ: a regular point, K smooth)
//   edge     f, f', f'', f''' at R match the history ring at the switch
//            (so K is C¹ across the rim)
//   area     2π ∫ f dρ = 2π ∫ (R − ρ) f'(ρ) dρ = graves before the switch / σ
// Among those, the smoothest: min ∫ x² (f'''/f)² dx — roughly how fast the
// curvature changes, with f in the weight taken from the previous pass
// (3 passes). One linear (KKT) solve per pass.
//
// Rings may grow past the history ring and narrow back to it at the rim: a
// sphere-like bulb. That is what lets the ancient zone be short: with rings
// capped at the history ring it could not be shorter than N_before / D(switch)
// years (~a third of the walk). The price of a short zone is curvature.

import { BSpline, DEG, quadrature, solveLinear } from './spline';

const NSEG = 40;
const PASSES = 3;
export const ANCIENT_CELLS = 8192;

export type AncientFit = {
  ok: boolean;
  problem?: string;
  sp: BSpline;
  p: Float64Array;      // coefficients of p (f' in units of f0/R)
  F: Float64Array;      // f/f0 on x = k / ANCIENT_CELLS
};

/**
 * @param R        ancient radius (m)
 * @param plazaR   radius of the empty plaza (m); graves start there
 * @param area     area the graves need, plaza → R (m²)
 * @param edge     history ring at ρ = R: [f, f', f'', f'''] (m, per m)
 */
export function fitAncient(R: number, plazaR: number, area: number, edge: number[]): AncientFit {
  const breaks = Array.from({ length: NSEG + 1 }, (_, i) => i / NSEG);
  const sp = new BSpline(breaks);
  const n = sp.n;
  const quad = quadrature(sp, breaks);
  const [f0, f1, f2, f3] = edge;
  const S = f0 / R; // f' unit

  const row = (x: number, d: number) => {
    const { span, ders } = sp.basis(x, 3);
    const r = new Array(n).fill(0);
    for (let j = 0; j <= DEG; j++) r[span - DEG + j] = ders[d][j];
    return r;
  };
  // ∫ B dx and ∫ (1 − x) B dx
  const intB = new Array(n).fill(0), momB = new Array(n).fill(0);
  for (const q of quad) for (let j = 0; j <= DEG; j++) {
    intB[q.span - DEG + j] += q.w * q.b[0][j];
    momB[q.span - DEG + j] += q.w * (1 - q.x) * q.b[0][j];
  }
  // f'(ρ) = S·p(x): f'' = S p_x / R, f''' = S p_xx / R²
  const E = [row(0, 0), row(0, 1), row(0, 3), intB, row(1, 0), row(1, 1), row(1, 2), momB];
  const e = [1 / S, 0, 0, 1, f1 / S, f2 * R / S, f3 * R * R / S,
    (area + Math.PI * plazaR * plazaR) / (2 * Math.PI * R * R * S)];
  const m = E.length;

  const F = new Float64Array(ANCIENT_CELLS + 1);
  const tabulate = (p: Float64Array) => { // F(x) = ∫₀ˣ p, on the fine grid
    const h = 1 / ANCIENT_CELLS;
    for (let k = 1; k <= ANCIENT_CELLS; k++) {
      let acc = 0;
      for (const [x, w] of GL4) acc += w * sp.value(p, (k - 1 + x) * h);
      F[k] = F[k - 1] + acc * h;
    }
  };
  let p: Float64Array | null = null;
  for (let pass = 0; pass < PASSES; pass++) {
    // weight x²/F² from the previous pass (first pass: a smooth guess)
    const Fref = (x: number) => {
      if (!p) return Math.max(x / S, x ** 4);
      const k = Math.min(ANCIENT_CELLS, Math.round(x * ANCIENT_CELLS));
      return Math.max(F[k], 0.5 * x / S);
    };
    const K = Array.from({ length: n + m }, () => new Array(n + m).fill(0)), y = new Array(n + m).fill(0);
    for (const q of quad) {
      const w = q.w * q.x * q.x / Fref(q.x) ** 2;
      for (let i = 0; i <= DEG; i++) for (let j = 0; j <= DEG; j++)
        K[q.span - DEG + i][q.span - DEG + j] += w * q.b[2][i] * q.b[2][j];
    }
    let mx = 0;
    for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs(K[i][i]));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i][j] /= mx;
    for (let r = 0; r < m; r++) { y[n + r] = e[r]; for (let j = 0; j < n; j++) K[n + r][j] = K[j][n + r] = E[r][j]; }
    const sol = solveLinear(K, y);
    if (!sol.every(Number.isFinite)) return { ok: false, problem: 'ancient fit failed', sp, p: new Float64Array(n), F };
    p = Float64Array.from(sol.slice(0, n));
    tabulate(p);
  }
  for (let k = 1; k <= ANCIENT_CELLS; k++)
    if (!(F[k] > 0)) return { ok: false, problem: 'the ancient era is too short: its rings would collapse', sp, p: p!, F };
  return { ok: true, sp, p: p!, F };
}

const GL4: [number, number][] = [
  [0.0694318442029737, 0.1739274225687269], [0.3300094782075719, 0.3260725774312731],
  [0.6699905217924281, 0.3260725774312731], [0.9305681557970263, 0.1739274225687269],
];
