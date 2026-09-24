// Continuous deaths-per-year model D(t), faithful to the PRB period totals
// and OWID yearly data.
//
// Approach (no nonlinear fitting):
//   1. Each data period j gets a fixed *shape* b_j(t): PRB's own assumption
//      (deaths ∝ exponentially growing population) for old periods, flat for
//      the late 19th/20th century and the 5-year OWID bins.
//   2. The sum Σ c_j b_j is smoothed with a Gaussian whose width scales with
//      how coarse the data is at that time (200 yr in the neolithic, 2.5 yr
//      after 1950). This removes the jumps between periods.
//   3. Smoothing is linear, so the period totals of S(Σ c_j b_j) are linear in
//      c. We solve the small linear system so every period total is matched
//      exactly (up to float error).
//
// Before T0 (-8000) the "ancient" era is modelled as a single exponential
// ramp D0·e^{r(t−T0)} from -50 000, with r chosen to match PRB's total. Inside
// the flat ancient circle time is not linear in radius, so this only matters
// for labels.

import { OWID_DEATHS, OWID_FIRST_YEAR, PRB } from './data';

export const T0 = -8000;
export const T_END = 2030;       // table end; data after 2023 is extrapolated
export const ANCIENT_START = -50000;
const OWID_BIN = 5;
const EXTRAPOLATION_GROWTH = 0.008; // deaths/yr growth after last OWID year

export type Period = { a: number; b: number; target: number; fitted: number; source: string };

export type DeathModel = {
  /** D at integer years T0..T_END (index k ↔ year T0+k); linear in between */
  D: Float64Array;
  /** ∫_{T0}^{T0+k} D dt */
  cum: Float64Array;
  ancient: { N: number; D0: number; r: number };
  periods: Period[];
  total: (t: number) => number;
};

// smoothing width (years) at time t, linear between knots
const SIGMA_KNOTS: [number, number][] = [
  [-8000, 200], [1, 150], [1200, 60], [1650, 20], [1750, 12],
  [1850, 8], [1900, 5], [1950, 2.5], [T_END, 2.5],
];

function sigmaAt(t: number): number {
  for (let i = 1; i < SIGMA_KNOTS.length; i++) {
    const [t1, s1] = SIGMA_KNOTS[i];
    if (t <= t1) {
      const [t0, s0] = SIGMA_KNOTS[i - 1];
      return s0 + (s1 - s0) * (t - t0) / (t1 - t0);
    }
  }
  return SIGMA_KNOTS[SIGMA_KNOTS.length - 1][1];
}

function solveLinear(A: number[][], y: number[]): number[] {
  const n = y.length;
  const M = A.map((row, i) => [...row, y[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

type RawPeriod = { a: number; b: number; target: number; source: string; shape: (t: number) => number };

function rawPeriods(): { periods: RawPeriod[]; ancientDeaths: number } {
  const periods: RawPeriod[] = [];
  let ancientDeaths = 0;
  for (let i = 1; i < PRB.length; i++) {
    const prev = PRB[i - 1], cur = PRB[i];
    const deaths = cur.birthsSincePrev - cur.pop + prev.pop;
    if (cur.year <= T0) { ancientDeaths += deaths; continue; }
    const a = prev.year, b = cur.year;
    const growth = Math.log(cur.pop / prev.pop) / (b - a);
    const shape = b <= 1850 ? (t: number) => Math.exp(growth * (t - a)) : () => 1;
    periods.push({ a, b, target: deaths, source: `PRB ${a}→${b}`, shape });
  }
  const owidLast = OWID_FIRST_YEAR + OWID_DEATHS.length - 1;
  const deathsInYear = (y: number) => y <= owidLast
    ? OWID_DEATHS[y - OWID_FIRST_YEAR]
    : OWID_DEATHS[OWID_DEATHS.length - 1] * Math.pow(1 + EXTRAPOLATION_GROWTH, y - owidLast);
  for (let a = OWID_FIRST_YEAR; a < T_END; a += OWID_BIN) {
    const b = Math.min(a + OWID_BIN, T_END);
    let target = 0;
    for (let y = a; y < b; y++) target += deathsInYear(y);
    const src = b - 1 > owidLast ? (a > owidLast ? 'extrapolated' : 'OWID + extrapolated') : 'OWID';
    periods.push({ a, b, target, source: `${src} ${a}→${b}`, shape: () => 1 });
  }
  return { periods, ancientDeaths };
}

export function buildDeathModel(): DeathModel {
  const K = T_END - T0;
  const { periods: raw, ancientDeaths } = rawPeriods();
  const n = raw.length;

  // smoothing weights, gather form, per output sample
  const taps: { lo: number; w: Float64Array; norm: number }[] = [];
  for (let k = 0; k <= K; k++) {
    const s = sigmaAt(T0 + k);
    const half = Math.ceil(4 * s);
    const lo = Math.max(0, k - half), hi = Math.min(K, k + half);
    const w = new Float64Array(hi - lo + 1);
    let norm = 0;
    for (let m = lo; m <= hi; m++) {
      const x = (m - k) / s;
      norm += (w[m - lo] = Math.exp(-0.5 * x * x));
    }
    taps.push({ lo, w, norm });
  }
  const maxHalf = Math.ceil(4 * Math.max(...SIGMA_KNOTS.map(([, s]) => s)));
  const smooth = (x: Float64Array, from: number, to: number): Float64Array => {
    const out = new Float64Array(K + 1);
    for (let k = Math.max(0, from - maxHalf); k <= Math.min(K, to + maxHalf); k++) {
      const { lo, w, norm } = taps[k];
      let acc = 0;
      for (let i = 0; i < w.length; i++) acc += x[lo + i] * w[i];
      out[k] = acc / norm;
    }
    return out;
  };
  const periodIntegral = (x: Float64Array, a: number, b: number) => {
    let acc = 0;
    for (let y = a; y < b; y++) acc += 0.5 * (x[y - T0] + x[y + 1 - T0]);
    return acc;
  };

  // basis j: shape on the samples inside [a, b), scaled to unit raw integral.
  // The endpoint sample b belongs to the next period.
  const smoothed: Float64Array[] = raw.map(p => {
    const x = new Float64Array(K + 1);
    for (let y = p.a; y < p.b; y++) x[y - T0] = p.shape(y);
    if (p.b === T_END) x[K] = p.shape(T_END);
    const s = periodIntegral(x, p.a, p.b);
    for (let i = 0; i <= K; i++) x[i] /= s;
    return smooth(x, p.a - T0, p.b - T0);
  });

  const A = raw.map(pi => smoothed.map(sj => periodIntegral(sj, pi.a, pi.b)));
  const c = solveLinear(A, raw.map(p => p.target));

  const D = new Float64Array(K + 1);
  for (let j = 0; j < n; j++) for (let i = 0; i <= K; i++) D[i] += c[j] * smoothed[j][i];
  for (let i = 0; i <= K; i++) if (!(D[i] > 0)) throw new Error(`death model non-positive at ${T0 + i}`);

  const cum = new Float64Array(K + 1);
  for (let k = 1; k <= K; k++) cum[k] = cum[k - 1] + 0.5 * (D[k - 1] + D[k]);

  // ancient ramp: D0/r · (1 − e^{−r·L}) = N
  const D0 = D[0], L = T0 - ANCIENT_START;
  let r = D0 / ancientDeaths;
  for (let i = 0; i < 200; i++) r = D0 / ancientDeaths * (1 - Math.exp(-r * L));

  const periods = raw.map(p => ({
    a: p.a, b: p.b, target: p.target, source: p.source,
    fitted: periodIntegral(D, p.a, p.b),
  }));

  return {
    D, cum, periods,
    ancient: { N: ancientDeaths, D0, r },
    total: (t: number) => ancientDeaths + cumAt(D, cum, t),
  };
}

/** ∫_{T0}^{t} D, t clamped to the table */
export function cumAt(D: Float64Array, cum: Float64Array, t: number): number {
  const K = D.length - 1;
  const x = Math.min(Math.max(t - T0, 0), K);
  const k = Math.min(Math.floor(x), K - 1), s = x - k;
  return cum[k] + D[k] * s + 0.5 * (D[k + 1] - D[k]) * s * s;
}

export function deathsAt(D: Float64Array, t: number): number {
  const K = D.length - 1;
  const x = Math.min(Math.max(t - T0, 0), K);
  const k = Math.min(Math.floor(x), K - 1), s = x - k;
  return D[k] + (D[k + 1] - D[k]) * s;
}
