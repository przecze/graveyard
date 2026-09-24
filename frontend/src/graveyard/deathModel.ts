// Smooth deaths-per-year model D(t), 50 000 BCE → 2030.
//
// Everything is a quintic B-spline (C⁴), so the surface curvature K ∝ −D''/D
// is C²: no kinks, no jumps, no yearly spikes for the moving projection to
// trip on. Smoothness is measured as ∫(s''')² dt; ρ is linear in t, so this is
// smoothness along the walking direction.
//
// D = P · m:
//   P  prior, exp of the smoothest spline through PRB-style point estimates
//      (population × (birth rate − growth)) and OWID decade means. P > 0.
//   m  multiplier (≈ 1) minimising ∫ m'² + α m'''² such that ∫_period D dt
//      equals the data exactly, for every PRB benchmark period (including the single
//      50 000 → 8000 BCE total) and every OWID decade.
// Both are linear least-norm problems: one KKT solve each, no iteration,
// nothing to converge.

import { OWID_DEATHS, OWID_FIRST_YEAR, PRB } from './data';
import { BSpline, DEG, GL8, quadrature, solveLinear, type Quad } from './spline';

export const T0 = -8000;
export const T_END = 2030;       // data after 2023 is extrapolated
export const ANCIENT_START = -50000;
const OWID_BIN = 10;
const EXTRAPOLATION_GROWTH = 0.008;
const MULT_SMOOTHING = 20 ** 4; // α in years⁴: multiplier bends over ≳ 20 years
const PRIOR_SMOOTHING = 1e3; // λ in years⁶: smooths slope breaks over ~3 years

export type Period = { a: number; b: number; target: number; fitted: number; source: string };

// knot spacing per data period (years): coarse data → coarse knots
const KNOT_STEP: [number, number][] = [
  [T0, 2000], [1, 200], [1200, 100], [1650, 50], [1750, 25], [1850, 20], [1900, 10], [1950, 5], [T_END, 2.5],
];

// ── data periods ───────────────────────────────────────────────────────────

type RawPeriod = { a: number; b: number; target: number; source: string };

function rawPeriods(): { periods: RawPeriod[]; ancientDeaths: number } {
  const periods: RawPeriod[] = [];
  let ancientDeaths = 0;
  for (let i = 1; i < PRB.length; i++) {
    const prev = PRB[i - 1], cur = PRB[i];
    const deaths = cur.birthsSincePrev - cur.pop + prev.pop;
    if (cur.year <= T0) ancientDeaths += deaths;
    periods.push({ a: prev.year, b: cur.year, target: deaths, source: `PRB ${prev.year}→${cur.year}` });
  }
  const owidLast = OWID_FIRST_YEAR + OWID_DEATHS.length - 1;
  const deathsInYear = (y: number) => y <= owidLast
    ? OWID_DEATHS[y - OWID_FIRST_YEAR]
    : OWID_DEATHS[OWID_DEATHS.length - 1] * Math.pow(1 + EXTRAPOLATION_GROWTH, y - owidLast);
  for (let a = OWID_FIRST_YEAR; a < T_END; a += OWID_BIN) {
    const b = Math.min(a + OWID_BIN, T_END);
    let target = 0;
    for (let y = a; y < b; y++) target += deathsInYear(y);
    periods.push({ a, b, target, source: `${b - 1 > owidLast ? 'OWID + extrap.' : 'OWID'} ${a}→${b}` });
  }
  return { periods, ancientDeaths };
}

/** PRB-style point estimates of deaths/yr (t, D) used only to shape the prior */
function priorPoints(): [number, number][] {
  const pts: [number, number][] = [];
  const rows = PRB.filter(r => r.year >= T0);
  const rate = (i: number) => { // (birth rate − growth) of the period ending at rows[i]
    const prev = rows[i - 1], cur = rows[i];
    return cur.cbr / 1000 - Math.log(cur.pop / prev.pop) / (cur.year - prev.year);
  };
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].year >= OWID_FIRST_YEAR) continue;
    const rs = [i > 0 ? rate(i) : NaN, i + 1 < rows.length ? rate(i + 1) : NaN].filter(isFinite);
    pts.push([rows[i].year, rows[i].pop * rs.reduce((a, b) => a + b, 0) / rs.length]);
  }
  // before T0 PRB only gives a total: assume one exponential ramp D8·e^{r(t−T0)}
  // from 50 000 BCE carrying it, i.e. D8/r · (1 − e^{−r·L}) = N
  const { ancientDeaths } = rawPeriods();
  const D8 = pts[0][1], L = T0 - ANCIENT_START;
  let r = D8 / ancientDeaths;
  for (let i = 0; i < 200; i++) r = D8 / ancientDeaths * (1 - Math.exp(-r * L));
  pts.unshift([ANCIENT_START, D8 * Math.exp(-r * L)]);
  for (let a = OWID_FIRST_YEAR; a + OWID_BIN <= OWID_FIRST_YEAR + OWID_DEATHS.length; a += OWID_BIN) {
    const ys = OWID_DEATHS.slice(a - OWID_FIRST_YEAR, a - OWID_FIRST_YEAR + OWID_BIN);
    pts.push([a + OWID_BIN / 2, ys.reduce((x, y) => x + y, 0) / ys.length]);
  }
  return pts;
}

/** argmin ∫ (s − L)² dt + λ ∫ (s''')² dt   (normal equations) */
function smoothProjection(sp: BSpline, quad: Quad[], L: (t: number) => number, lambda: number): Float64Array {
  const nc = sp.n;
  const M = Array.from({ length: nc }, () => new Array(nc).fill(0));
  const rhs = new Array(nc).fill(0);
  for (const q of quad) {
    const l = L(q.x);
    for (let i = 0; i <= DEG; i++) {
      const bi = q.b[0][i], I = q.span - DEG + i;
      rhs[I] += q.w * bi * l;
      for (let j = 0; j <= DEG; j++) M[I][q.span - DEG + j] += q.w * (bi * q.b[0][j] + lambda * q.b[3][i] * q.b[3][j]);
    }
  }
  return Float64Array.from(solveLinear(M, rhs));
}

/** argmin ∫ (s')² + α (s''')² dt  subject to  A·c = y   (one KKT solve) */
function minRoughness(sp: BSpline, quad: Quad[], A: number[][], y: number[], alpha: number): Float64Array {
  const nc = sp.n, m = A.length, n = nc + m;
  const K = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const q of quad)
    for (let i = 0; i <= DEG; i++) for (let j = 0; j <= DEG; j++)
      K[q.span - DEG + i][q.span - DEG + j] += q.w * (q.b[1][i] * q.b[1][j] + alpha * q.b[3][i] * q.b[3][j]);
  let rMax = 0;
  for (let i = 0; i < nc; i++) for (let j = 0; j < nc; j++) rMax = Math.max(rMax, Math.abs(K[i][j]));
  for (let i = 0; i < nc; i++) for (let j = 0; j < nc; j++) K[i][j] /= rMax;
  const rhs = new Array(n).fill(0);
  for (let r = 0; r < m; r++) {
    for (let j = 0; j < nc; j++) K[nc + r][j] = K[j][nc + r] = A[r][j];
    rhs[nc + r] = y[r];
  }
  return Float64Array.from(solveLinear(K, rhs).slice(0, nc));
}

// ── model ──────────────────────────────────────────────────────────────────

export class DeathModel {
  readonly spline: BSpline;
  readonly lnPrior: Float64Array;
  readonly mult: Float64Array;
  readonly periods: Period[];
  /** deaths before T0 (8000 BCE) */
  readonly ancientDeaths: number;
  // yearly tables for exact-ish integrals (cubic Hermite in between)
  private readonly cumTab: Float64Array; // ∫_{T_START}^{T_START+k} D
  private readonly Dtab: Float64Array;

  constructor() {
    const { periods: raw, ancientDeaths } = rawPeriods();

    const breaks = [ANCIENT_START];
    for (const [until, step] of KNOT_STEP) {
      const from = breaks[breaks.length - 1];
      const n = Math.max(1, Math.round((until - from) / step));
      for (let i = 1; i <= n; i++) breaks.push(from + (until - from) * i / n);
    }
    const sp = this.spline = new BSpline(breaks);
    const nc = sp.n, np = raw.length;

    const quad = quadrature(sp, breaks, mid => raw.findIndex(q => mid >= q.a && mid < q.b));

    // Two linear solves, both "smoothest curve subject to linear constraints":
    //  1. prior  ln P(t): least-squares spline fit of the piecewise-linear
    //     ln D through PRB-style point estimates (population × (birth rate −
    //     growth), i.e. PRB's own exponential-per-period assumption) and the
    //     OWID decade means. Log space → P > 0 by construction.
    //  2. D = P · m, with m ≈ 1 the smoothest multiplier that makes every
    //     period total exact.
    const points = priorPoints();
    const lnL = (t: number) => { // piecewise-linear ln D through the points
      if (t <= points[0][0]) return Math.log(points[0][1]);
      for (let i = 1; i < points.length; i++) if (t <= points[i][0]) {
        const [t0, d0] = points[i - 1], [t1, d1] = points[i];
        return Math.log(d0) + (Math.log(d1) - Math.log(d0)) * (t - t0) / (t1 - t0);
      }
      return Math.log(points[points.length - 1][1]);
    };
    const lnP = this.lnPrior = smoothProjection(sp, quad, lnL, PRIOR_SMOOTHING);
    const rows = raw.map(() => new Array(nc).fill(0));
    for (const q of quad) {
      const P = Math.exp(q.b[0].reduce((a, b, j) => a + b * lnP[q.span - DEG + j], 0));
      for (let j = 0; j <= DEG; j++) rows[q.p][q.span - DEG + j] += q.w * P * q.b[0][j] / raw[q.p].target;
    }
    this.mult = minRoughness(sp, quad, rows, raw.map(() => 1), MULT_SMOOTHING);
    for (let y = ANCIENT_START; y <= T_END; y += 0.5)
      if (!(this.D(y) > 0)) throw new Error(`death model non-positive at ${y}`);

    // yearly table of ∫D (cubic Hermite in between, exact derivative D)
    const K = T_END - ANCIENT_START;
    this.Dtab = new Float64Array(K + 1);
    this.cumTab = new Float64Array(K + 1);
    for (let k = 0; k <= K; k++) this.Dtab[k] = this.D(ANCIENT_START + k);
    for (let k = 1; k <= K; k++) {
      let acc = 0;
      for (const [x, w] of GL8) acc += w * this.D(ANCIENT_START + k - 1 + x);
      this.cumTab[k] = this.cumTab[k - 1] + acc;
    }
    this.periods = raw.map(p => ({ ...p, fitted: this.cum(p.b) - this.cum(p.a) }));
    this.ancientDeaths = ancientDeaths;
  }

  private clampT(t: number) { return Math.min(Math.max(t, ANCIENT_START), T_END); }

  /** [D, D', D'', D'''] at t */
  Dd(t: number): [number, number, number, number] {
    const x = this.clampT(t);
    const [l, l1, l2, l3] = this.spline.eval(this.lnPrior, x, 3);
    const [m, m1, m2, m3] = this.spline.eval(this.mult, x, 3);
    const P = Math.exp(l), P1 = P * l1, P2 = P * (l2 + l1 * l1), P3 = P * (l3 + 3 * l1 * l2 + l1 * l1 * l1);
    return [P * m, P1 * m + P * m1, P2 * m + 2 * P1 * m1 + P * m2, P3 * m + 3 * P2 * m1 + 3 * P1 * m2 + P * m3];
  }
  D(t: number): number {
    const sp = this.spline, i = sp.basisValues(this.clampT(t)), N = sp.N;
    let l = 0, m = 0;
    for (let j = 0; j <= DEG; j++) { l += N[j] * this.lnPrior[i - DEG + j]; m += N[j] * this.mult[i - DEG + j]; }
    return Math.exp(l) * m;
  }

  /** ∫_{50 000 BCE}^{t} D dt: graves of everyone who died before t */
  cum(t: number): number {
    const tab = this.cumTab, K = tab.length - 1;
    const x = Math.min(Math.max(t - ANCIENT_START, 0), K);
    const k = Math.min(Math.floor(x), K - 1), s = x - k;
    const m0 = this.Dtab[k], m1 = this.Dtab[k + 1], y0 = tab[k], y1 = tab[k + 1];
    const s2 = s * s, s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * m1;
  }
}

let cached: DeathModel | null = null;
export function deathModel(): DeathModel {
  return cached ??= new DeathModel();
}
