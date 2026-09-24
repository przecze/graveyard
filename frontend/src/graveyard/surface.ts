// Geometry of the graveyard surface.
//
// Surface of revolution, metric  ds² = dρ² + f(ρ)² dφ²  (ρ = walking distance
// from the centre, φ = angle, a ring at ρ is 2π·f(ρ) long), one uniform grave
// density σ everywhere.
//
// Time is the master coordinate. Pick a smooth "speed of time" v(t) (metres
// walked per year). Then
//     ρ(t) = ∫ v dt,        f = D(t) / (2πσ v(t))
// and the ring between t and t+dt holds exactly D(t)·dt graves, so the density
// is uniform by construction. Gaussian curvature K = −f_ρρ / f.
//
//   flat core   v = D / (2πσ ρ_flat) with ρ_flat = √(graves so far / πσ): this
//               is exactly the flat disc, f = ρ.
//   ancient     the ring profile is set directly: ln f blends (C∞ step) from
//               the flat disc to  f0·(D(t)/D0)^β  and β blends to 1 before
//               8000 BCE, where f0 = D0/(2πσ v_hist) is the history ring. β is
//               found by bisection so the zone ends at the requested radius.
//               β = 1: time linear here too (longest); β = 0: a flat cylinder
//               (shortest without a neck); β < 0: rings shrink outward (neck).
//               For 0 ≤ β ≤ 1 rings never shrink (all blended terms increase).
//               Then v = D/(2πσ f).
//   history     v = v_hist constant: time is linear with distance.
//
// D(t) is C⁴ and the blends are C∞, so f is C⁴ and K is C² everywhere: no
// creases, no jumps. At uniform density a neck-free ancient zone needs a
// radius of at least ≈ N_ancient / D(8000 BCE) ≈ 18 000 "history years"
// (the cylinder), i.e. ≳ 1.8× the whole history zone, whatever v is.
//
// Rendering uses the isothermal coordinate u = ∫ dρ/f = ∫ 2πσ v²/D dt
// (u = ln ρ in the flat core): w = u + iφ is conformal and the view is
// z = f(ρp)·(e^{w − wp} − 1), an exact isometry at the player.

import { ANCIENT_START, deathModel, T0, T_END, type DeathModel } from './deathModel';

export type Layout = {
  rowH: number;         // radial row pitch (m): grave + walkway
  pitch: number;        // nominal spacing along a row (m)
  plazaR: number;       // central plaza radius (m), no graves
  aisleW: number;       // radial aisle width (m)
  aisleSpacing: number; // aisles split so neighbours are aisleSpacing..2×aisleSpacing apart
};

export const DEFAULT_LAYOUT: Layout = { rowH: 2.6, pitch: 1.3, plazaR: 14, aisleW: 2.4, aisleSpacing: 36 };

export type Shape = {
  coreR: number;    // flat core radius (m)
  ancientR: number; // radius of the whole ancient zone, reached at 8000 BCE (m)
  v: number;        // metres per year in the history zone
};

export const DEFAULT_SHAPE: Shape = { coreR: 5000, ancientR: 95000, v: 3 }; // smallest neck-free zone for v = 3

export type ShapeReport = {
  ok: boolean;
  problem?: string;
  shrinks: boolean;      // rings get shorter outward somewhere (a neck)
  neckRatio: number;     // min over the ancient zone of f / (max f so far), 1 = no neck
  minKRadius: number;    // tightest curvature radius in the ancient zone (m)
  rimRing: number;       // ring length at 8000 BCE (m)
  beta: number;          // profile exponent, see above
};

const ANCIENT_STEP = 5;   // table step before T0 (years)
const CORE_BLEND = 3000;  // years to blend out of the flat core
const RIM_BLEND = 4000;   // years to blend β → 1 before 8000 BCE
const GL: [number, number][] = [
  [0.0694318442029737, 0.1739274225687269], [0.3300094782075719, 0.3260725774312731],
  [0.6699905217924281, 0.3260725774312731], [0.9305681557970263, 0.1739274225687269],
];

/** C∞ smooth step 0→1 on [0,1], all derivatives 0 at both ends */
function smoothStep(s: number): number {
  if (s <= 0) return 0;
  if (s >= 1) return 1;
  const a = Math.exp(-1 / s), b = Math.exp(-1 / (1 - s));
  return a / (a + b);
}
/** C∞ bump on [0,1], 1 at s = ½, all derivatives 0 at both ends */
function bump(s: number): number {
  return s <= 0 || s >= 1 ? 0 : Math.exp(4 - 1 / (s * (1 - s)));
}

function hermite(y0: number, y1: number, m0: number, m1: number, s: number): number {
  const s2 = s * s, s3 = s2 * s;
  return (2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * m1;
}

export class Surface {
  readonly layout: Layout;
  readonly shape: Shape;
  readonly model: DeathModel;
  readonly sigma: number;
  readonly coreR: number;
  readonly rho0: number;       // radius at T0 (8000 BCE)
  readonly v: number;          // history speed of time
  readonly tCore: number;      // time the flat core ends
  readonly report: ShapeReport;
  readonly rhoEndTable: number;
  readonly nRows: number;
  readonly rowStart: Float64Array;   // graves before row i
  readonly rowAisles: Float64Array;  // radial aisles in row i (power of two)
  private readonly aisleRuns = new Map<number, [number, number][]>();
  private readonly beta: number;
  private readonly f0: number;
  private readonly D0: number;
  // ancient tables on t = aT0 + k·aH
  private readonly aT0: number;
  private readonly aH: number;
  private readonly aRho: Float64Array;
  private readonly aU: Float64Array;
  // history table on t = T0 + k
  private readonly hU: Float64Array;
  private readonly uCore: number;

  constructor(shape: Shape = DEFAULT_SHAPE, layout: Layout = DEFAULT_LAYOUT, model: DeathModel = deathModel()) {
    this.layout = layout;
    this.shape = shape;
    this.model = model;
    const sigma = this.sigma = 1 / (layout.rowH * layout.pitch);
    this.v = shape.v;
    const coreR = this.coreR = Math.max(layout.plazaR + 10, shape.coreR);
    this.rho0 = shape.ancientR;

    // end of the flat core: graves so far fill the disc of radius coreR
    const coreGraves = sigma * Math.PI * (coreR ** 2 - layout.plazaR ** 2);
    let lo = ANCIENT_START, hi = T0;
    for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (model.cum(m) < coreGraves) lo = m; else hi = m; }
    let problem = model.cum(T0) <= coreGraves ? 'the flat core alone holds more than all ancient graves'
      : this.rho0 <= coreR ? 'the ancient zone must be larger than the flat core' : undefined;
    this.tCore = Math.min(lo, T0 - 100);

    this.D0 = model.D(T0);
    this.f0 = this.D0 / (2 * Math.PI * sigma * this.v);

    // β by bisection on ρ(T0) = ancientR (ρ(T0) grows with β)
    const n = 800, span = T0 - this.tCore;
    const radiusFor = (beta: number) => {
      let acc = coreR;
      for (let k = 0; k < n; k++) for (const [x, w] of GL) acc += w * span / n * this.vAncient(this.tCore + span * (k + x) / n, beta);
      return acc;
    };
    let bLo = -3, bHi = 1.5;
    for (let i = 0; i < 60; i++) { const m = (bLo + bHi) / 2; if (radiusFor(m) < this.rho0) bLo = m; else bHi = m; }
    this.beta = (bLo + bHi) / 2;
    if (!problem && Math.abs(radiusFor(this.beta) - this.rho0) > 1) problem = 'that ancient radius is out of reach for this core and v';

    // ancient tables: ρ(t) and u(t)
    const na = Math.max(200, Math.ceil(span / ANCIENT_STEP));
    this.aT0 = this.tCore; this.aH = span / na;
    this.aRho = new Float64Array(na + 1);
    this.aU = new Float64Array(na + 1);
    this.aRho[0] = coreR;
    this.uCore = Math.log(coreR);
    for (let k = 1; k <= na; k++) {
      let dr = 0, du = 0;
      for (const [x, w] of GL) {
        const t = this.aT0 + this.aH * (k - 1 + x), v = this.vAt(t);
        dr += w * v; du += w * 2 * Math.PI * sigma * v * v / model.D(t);
      }
      this.aRho[k] = this.aRho[k - 1] + dr * this.aH;
      this.aU[k] = this.aU[k - 1] + du * this.aH;
    }
    this.rho0 = this.aRho[na]; // equals ancientR up to quadrature error
    // history table: u(t)
    const nh = T_END - T0;
    this.hU = new Float64Array(nh + 1);
    this.hU[0] = this.uCore + this.aU[na];
    const kU = 2 * Math.PI * sigma * this.v * this.v;
    for (let k = 1; k <= nh; k++) {
      let du = 0;
      for (const [x, w] of GL) du += w / model.D(T0 + k - 1 + x);
      this.hU[k] = this.hU[k - 1] + kU * du;
    }

    // report
    let shrinks = false, fMax = 0, neck = 1, minKR = Infinity;
    for (let k = 0; k <= na; k += 2) {
      const t = this.aT0 + k * this.aH, fv = this.fAt(t);
      fMax = Math.max(fMax, fv);
      neck = Math.min(neck, fv / fMax);
      if (fv < fMax * (1 - 1e-9)) shrinks = true;
      const K = this.gaussKAt(t);
      if (K) minKR = Math.min(minKR, 1 / Math.sqrt(Math.abs(K)));
    }
    this.report = {
      ok: !problem, problem, shrinks, neckRatio: neck, minKRadius: minKR,
      rimRing: 2 * Math.PI * this.fAt(T0), beta: this.beta,
    };

    // rows
    this.rhoEndTable = this.rhoAtTime(T_END);
    this.nRows = Math.ceil((this.rhoEndTable - layout.plazaR) / layout.rowH);
    this.rowStart = new Float64Array(this.nRows + 1);
    this.rowAisles = new Float64Array(this.nRows);
    for (let i = 0; i <= this.nRows; i++) {
      this.rowStart[i] = Math.round(this.cum(this.rowInner(i)));
      if (i < this.nRows) {
        const m = 2 * Math.PI * this.f(this.rowInner(i) + layout.rowH / 2) / layout.aisleSpacing;
        this.rowAisles[i] = Math.max(4, Math.pow(2, Math.floor(Math.log2(Math.max(m, 1)))));
      }
    }
    let maxL = 4;
    for (let i = 0; i < this.nRows; i++) maxL = Math.max(maxL, this.rowAisles[i]);
    for (let lvl = 4; lvl <= maxL; lvl *= 2) {
      const runs: [number, number][] = [];
      for (let i = 0; i < this.nRows; i++) {
        if (this.rowAisles[i] < lvl) continue;
        if (runs.length && runs[runs.length - 1][1] === i) runs[runs.length - 1][1] = i + 1;
        else runs.push([i, i + 1]);
      }
      this.aisleRuns.set(lvl, runs);
    }
  }

  // ── time-parametrised quantities ───────────────────────────────────────────

  private rhoFlat(t: number): number {
    return Math.sqrt(this.model.cum(t) / (Math.PI * this.sigma) + this.layout.plazaR ** 2);
  }

  private vAncient(t: number, beta: number): number {
    const S1 = smoothStep((t - this.tCore) / CORE_BLEND);
    const b = beta + (1 - beta) * smoothStep((t - (T0 - RIM_BLEND)) / RIM_BLEND);
    const D = this.model.D(t);
    const lnF = (1 - S1) * Math.log(this.rhoFlat(t)) + S1 * (Math.log(this.f0) + b * Math.log(D / this.D0));
    return D / (2 * Math.PI * this.sigma * Math.exp(lnF));
  }

  /** metres walked per year at time t */
  vAt(t: number): number {
    if (t >= T0) return this.v;
    if (t <= this.tCore) return this.model.D(t) / (2 * Math.PI * this.sigma * this.rhoFlat(t));
    return this.vAncient(t, this.beta);
  }

  /** ring radius f at time t */
  fAt(t: number): number {
    return this.model.D(t) / (2 * Math.PI * this.sigma * this.vAt(t));
  }

  /** Gaussian curvature at time t, K = −f_ρρ/f with d/dρ = (1/v) d/dt */
  gaussKAt(t: number): number {
    if (t <= this.tCore) return 0;
    const h = t < T0 ? 2 : 0.5;
    const f0 = this.fAt(t), fp = this.fAt(t + h), fm = this.fAt(t - h);
    const v0 = this.vAt(t), vp = this.vAt(t + h), vm = this.vAt(t - h);
    const ft = (fp - fm) / (2 * h), ftt = (fp - 2 * f0 + fm) / (h * h), vt = (vp - vm) / (2 * h);
    return -(ftt - ft * vt / v0) / (v0 * v0) / f0;
  }

  /** ρ at time t */
  rhoAtTime(t: number): number {
    if (t >= T0) return this.rho0 + (t - T0) * this.v;
    if (t <= this.tCore) return this.rhoFlat(Math.max(t, ANCIENT_START));
    const x = (t - this.aT0) / this.aH, n = this.aRho.length - 1;
    const k = Math.min(Math.floor(x), n - 1), s = x - k;
    return hermite(this.aRho[k], this.aRho[k + 1],
      this.vAt(this.aT0 + k * this.aH) * this.aH, this.vAt(this.aT0 + (k + 1) * this.aH) * this.aH, s);
  }

  /** calendar year at distance ρ */
  time(rho: number): number {
    if (rho >= this.rho0) return T0 + (rho - this.rho0) / this.v;
    if (rho <= this.coreR) {
      // invert the flat disc: graves so far = σπ(ρ² − plaza²)
      const g = this.sigma * Math.PI * (Math.max(rho, this.layout.plazaR) ** 2 - this.layout.plazaR ** 2);
      let lo = ANCIENT_START, hi = this.tCore;
      for (let i = 0; i < 50; i++) { const m = (lo + hi) / 2; if (this.model.cum(m) < g) lo = m; else hi = m; }
      return (lo + hi) / 2;
    }
    const a = this.aRho;
    let lo = 0, hi = a.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (a[m] <= rho) lo = m; else hi = m; }
    let t = this.aT0 + (lo + (rho - a[lo]) / (a[hi] - a[lo])) * this.aH;
    for (let i = 0; i < 2; i++) t -= (this.rhoAtTime(t) - rho) / this.vAt(t);
    return t;
  }

  f(rho: number): number {
    return rho <= this.coreR ? Math.max(rho, 1e-9) : this.fAt(this.time(rho));
  }

  gaussK(rho: number): number {
    return rho <= this.coreR ? 0 : this.gaussKAt(this.time(rho));
  }

  /** graves closer to the centre than ρ */
  cum(rho: number): number {
    const pr = this.layout.plazaR;
    if (rho <= pr) return 0;
    if (rho <= this.coreR) return this.sigma * Math.PI * (rho * rho - pr * pr);
    return this.model.cum(this.time(rho));
  }

  private uAtTime(t: number): number {
    const c = 2 * Math.PI * this.sigma;
    if (t >= T0) {
      const nh = this.hU.length - 1, k2 = c * this.v * this.v;
      if (t >= T_END) return this.hU[nh] + k2 * (t - T_END) / this.model.D(T_END);
      const x = t - T0, k = Math.min(Math.floor(x), nh - 1), s = x - k;
      return hermite(this.hU[k], this.hU[k + 1], k2 / this.model.D(T0 + k), k2 / this.model.D(T0 + k + 1), s);
    }
    const x = (t - this.aT0) / this.aH, n = this.aU.length - 1;
    const k = Math.min(Math.max(Math.floor(x), 0), n - 1), s = x - k;
    const d = (tt: number) => { const v = this.vAt(tt); return c * v * v / this.model.D(tt) * this.aH; };
    return this.uCore + hermite(this.aU[k], this.aU[k + 1], d(this.aT0 + k * this.aH), d(this.aT0 + (k + 1) * this.aH), s);
  }

  /** isothermal coordinate u(ρ) = ∫ dρ/f, u = ln ρ in the flat core */
  u(rho: number): number {
    return rho <= this.coreR ? Math.log(Math.max(rho, 1e-9)) : this.uAtTime(this.time(rho));
  }

  rhoAtU(u: number): number {
    if (u <= this.uCore) return Math.exp(u);
    const uEnd = this.hU[this.hU.length - 1];
    if (u >= uEnd) return this.rhoEndTable + (u - uEnd) * this.fAt(T_END);
    let lo = this.coreR, hi = this.rhoEndTable;
    for (let i = 0; i < 64; i++) { const m = (lo + hi) / 2; if (this.u(m) < u) lo = m; else hi = m; }
    let rho = (lo + hi) / 2;
    for (let i = 0; i < 2; i++) rho -= (this.u(rho) - u) * this.f(rho);
    return rho;
  }

  // ── rows, aisles, graves ─────────────────────────────────────────────────────

  rowOf(rho: number): number { return Math.floor((rho - this.layout.plazaR) / this.layout.rowH); }
  rowInner(i: number): number { return this.layout.plazaR + i * this.layout.rowH; }

  /** row intervals [a, b) in which rows have at least `count` aisles */
  aisleRowRuns(count: number): [number, number][] {
    return this.aisleRuns.get(Math.max(4, count)) ?? [];
  }

  /**
   * Graves of row i are split evenly among the M segments between aisles;
   * grave j sits at angle  seg·α + γ/2 + (l + ½)·(α − γ)/n_seg,
   * α = 2π/M, γ = aisle width / f.
   */
  rowGeometry(i: number) {
    const count = this.rowStart[i + 1] - this.rowStart[i];
    const M = this.rowAisles[i];
    const fMid = this.f(this.rowInner(i) + this.layout.rowH / 2);
    const alpha = 2 * Math.PI / M;
    const gamma = Math.min(alpha * 0.5, this.layout.aisleW / fMid);
    return { count, M, fMid, alpha, gamma };
  }
}

export { T0, T_END };
