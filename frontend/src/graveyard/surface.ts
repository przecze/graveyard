// Geometry of the graveyard surface.
//
// Surface of revolution, metric  ds² = dρ² + f(ρ)² dφ²  (ρ = walking distance
// from the centre, φ = angle, a ring at ρ is 2π·f(ρ) long), one uniform grave
// density σ everywhere.
//
// The ring between ρ and ρ+dρ holds σ·2πf·dρ graves; graves are laid out in
// order of death, so "graves nearer the centre" = σ·area(ρ). Gaussian
// curvature K = −f_ρρ / f.
//
//   flat core   f = ρ (the flat disc). Graves are placed by area.
//   ancient     one undated bucket: everyone who died before the switch year.
//               Nobody sees time here, so the zone is defined directly in ρ
//               for compactness: ln f rises (C∞) over the flare length to a
//               plateau P, stays there, then blends (C∞) over the rim length
//               onto the history law g(ρ) = f0·D(t(ρ))/D0 continued inward.
//               P = g(ρ0 − rim), so every blend only increases f: rings never
//               shrink. The plateau length is solved from the grave count.
//               The plateau is a cylinder, the most compact neck-free shape:
//               ancient width ≳ N_before·v / D0.
//   history     time is linear, t = switch + (ρ − ρ0)/v, f = D(t)/(2πσv),
//               K = −D''/(D v²). Year markers start here.
//
// D(t) is C⁴ and the blends are C∞, so f is C⁴ and K is C² everywhere.
// Optionally the death rate right after the switch is "flattened" (starts
// higher, rises slower, same period totals) to enlarge f0 and shrink the
// ancient zone further.
//
// Rendering uses the isothermal coordinate u = ∫ dρ/f (u = ln ρ in the flat
// core): w = u + iφ is conformal and the view is
// z = f(ρp)·(e^{w − wp} − 1), an exact isometry at the player.

import { ANCIENT_START, deathModel, T_END, type DeathModel } from './deathModel';

export type Layout = {
  rowH: number;         // radial row pitch (m): grave + walkway
  pitch: number;        // nominal spacing along a row (m)
  plazaR: number;       // central plaza radius (m), no graves
  aisleW: number;       // radial aisle width (m)
  aisleSpacing: number; // aisles split so neighbours are aisleSpacing..2×aisleSpacing apart
};

export const DEFAULT_LAYOUT: Layout = { rowH: 2.6, pitch: 1.3, plazaR: 14, aisleW: 2.4, aisleSpacing: 36 };

export type Shape = {
  coreR: number;      // flat core radius (m)
  flare: number;      // distance over which rings widen from the core to the plateau (m)
  switchYear: number; // history (dated, time-linear) starts here
  flatten: number;    // years after the switch over which the death rate is flattened (0 = off)
  v: number;          // metres per year in the history zone
};

export const DEFAULT_SHAPE: Shape = { coreR: 1500, flare: 2500, switchYear: -3000, flatten: 1000, v: 3 };

export type ShapeReport = {
  ok: boolean;
  problem?: string;
  minKRadius: number;    // tightest curvature radius in the ancient zone (m)
  rimRing: number;       // ring length at the switch (m)
  plateauRing: number;   // ring length on the plateau (m)
  ancientShare: number;  // ancient radius / total radius
  floorShare: number;    // same for the ideal cylinder N·v/D0
};

const RIM = 400;          // rim blend length (m)
const A_CELLS = 8192;     // ancient zone table cells

/** C∞ smooth step 0→1 on [0,1], all derivatives 0 at both ends */
function smoothStep(s: number): number {
  if (s <= 0) return 0;
  if (s >= 1) return 1;
  const a = Math.exp(-1 / s), b = Math.exp(-1 / (1 - s));
  return a / (a + b);
}

function hermite(y0: number, y1: number, m0: number, m1: number, s: number): number {
  const s2 = s * s, s3 = s2 * s;
  return (2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * m0 + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * m1;
}

const GL: [number, number][] = [
  [0.0694318442029737, 0.1739274225687269], [0.3300094782075719, 0.3260725774312731],
  [0.6699905217924281, 0.3260725774312731], [0.9305681557970263, 0.1739274225687269],
];

export class Surface {
  readonly layout: Layout;
  readonly shape: Shape;
  readonly model: DeathModel;
  readonly sigma: number;
  readonly coreR: number;
  readonly t0: number;         // switch year: history starts
  readonly rho0: number;       // radius at the switch
  readonly v: number;
  readonly report: ShapeReport;
  readonly rhoEndTable: number;
  readonly nRows: number;
  readonly rowStart: Float64Array;   // graves before row i
  readonly rowAisles: Float64Array;  // radial aisles in row i (power of two)
  readonly ancientGraves: number;    // graves before the switch
  private readonly aisleRuns = new Map<number, [number, number][]>();
  private readonly f0: number;
  private readonly D0: number;
  private readonly plateau: number;
  private readonly flareEnd: number; // ρ where the plateau starts
  private readonly rimStart: number; // ρ where the rim blend starts
  // ancient tables on ρ = coreR + k·aH: ∫dρ/f and ∫2πf dρ
  private readonly aH: number;
  private readonly aU: Float64Array;
  private readonly aArea: Float64Array;
  private readonly hU: Float64Array; // history: u on t = t0 + k
  private readonly uCore: number;

  constructor(shape: Shape = DEFAULT_SHAPE, layout: Layout = DEFAULT_LAYOUT,
              model: DeathModel = deathModel({ from: shape.switchYear, years: shape.flatten })) {
    this.layout = layout;
    this.shape = shape;
    this.model = model;
    const sigma = this.sigma = 1 / (layout.rowH * layout.pitch);
    const v = this.v = shape.v;
    const T = this.t0 = shape.switchYear;
    const coreR = this.coreR = Math.max(layout.plazaR + 10, shape.coreR);
    this.D0 = model.D(T);
    this.f0 = this.D0 / (2 * Math.PI * sigma * v);
    this.plateau = this.f0 * model.D(T - RIM / v) / this.D0;
    this.ancientGraves = model.cum(T);
    const needArea = this.ancientGraves / sigma - Math.PI * (coreR ** 2 - layout.plazaR ** 2);

    // pieces relative to their start: flare [0, flare], rim [0, RIM]
    const flare = Math.max(100, shape.flare);
    const flareF = (d: number) => {
      const S = smoothStep(d / flare);
      return Math.exp((1 - S) * Math.log(coreR + d) + S * Math.log(this.plateau));
    };
    const rimF = (d: number) => { // d from rimStart; rim ends at ρ0
      const S = smoothStep(d / RIM);
      const g = this.f0 * model.D(T - (RIM - d) / v) / this.D0;
      return Math.exp((1 - S) * Math.log(this.plateau) + S * Math.log(g));
    };
    const integrate = (fn: (d: number) => number, L: number) => {
      let acc = 0; const n = 400;
      for (let k = 0; k < n; k++) for (const [x, w] of GL) acc += w * 2 * Math.PI * fn(L * (k + x) / n) * L / n;
      return acc;
    };
    const plateauLen = (needArea - integrate(flareF, flare) - integrate(rimF, RIM)) / (2 * Math.PI * this.plateau);
    let problem: string | undefined;
    if (needArea <= 0) problem = 'the flat core holds more than all ancient graves';
    else if (this.plateau < coreR) problem = 'the flat core is wider than the ancient rings: make it smaller';
    else if (plateauLen < 0) problem = 'the flare alone holds more than all ancient graves: shorten it';
    this.flareEnd = coreR + flare;
    this.rimStart = this.flareEnd + Math.max(0, plateauLen);
    this.rho0 = this.rimStart + RIM;
    this.fAncientParts = { flareF, rimF };

    // ancient tables
    const L = this.rho0 - coreR;
    this.aH = L / A_CELLS;
    this.aU = new Float64Array(A_CELLS + 1);
    this.aArea = new Float64Array(A_CELLS + 1);
    this.uCore = Math.log(coreR);
    let minKR = Infinity;
    for (let k = 1; k <= A_CELLS; k++) {
      let su = 0, sa = 0;
      for (const [x, w] of GL) { const f = this.fAncient(coreR + (k - 1 + x) * this.aH); su += w / f; sa += w * 2 * Math.PI * f; }
      this.aU[k] = this.aU[k - 1] + su * this.aH;
      this.aArea[k] = this.aArea[k - 1] + sa * this.aH;
      const K = this.gaussK(coreR + k * this.aH);
      if (K) minKR = Math.min(minKR, 1 / Math.sqrt(Math.abs(K)));
    }
    // history table: u(t) = u(ρ0) + 2πσv² ∫ dt/D
    const nh = Math.ceil(T_END - T);
    this.hU = new Float64Array(nh + 1);
    this.hU[0] = this.uCore + this.aU[A_CELLS];
    const kU = 2 * Math.PI * sigma * v * v;
    for (let k = 1; k <= nh; k++) {
      let du = 0;
      for (const [x, w] of GL) du += w / model.D(T + k - 1 + x);
      this.hU[k] = this.hU[k - 1] + kU * du;
    }

    const hist = (T_END - T) * v, floor = this.ancientGraves * v / this.D0;
    this.report = {
      ok: !problem, problem, minKRadius: minKR,
      rimRing: 2 * Math.PI * this.f0, plateauRing: 2 * Math.PI * this.plateau,
      ancientShare: this.rho0 / (this.rho0 + hist), floorShare: floor / (floor + hist),
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

  private readonly fAncientParts: { flareF: (d: number) => number; rimF: (d: number) => number };

  private fAncient(rho: number): number {
    if (rho <= this.flareEnd) return this.fAncientParts.flareF(rho - this.coreR);
    if (rho <= this.rimStart) return this.plateau;
    return this.fAncientParts.rimF(Math.min(rho, this.rho0) - this.rimStart);
  }

  // ── radial functions ───────────────────────────────────────────────────────

  isAncient(rho: number): boolean { return rho < this.rho0; }

  /** ring radius: a ring at ρ is 2π·f(ρ) long */
  f(rho: number): number {
    if (rho <= this.coreR) return Math.max(rho, 1e-9);
    if (rho < this.rho0) return this.fAncient(rho);
    return this.model.D(this.time(rho)) / (2 * Math.PI * this.sigma * this.v);
  }

  /** Gaussian curvature K = −f''/f */
  gaussK(rho: number): number {
    if (rho <= this.coreR) return 0;
    if (rho >= this.rho0) {
      const [D, , D2] = this.model.Dd(this.time(rho));
      return -D2 / (D * this.v * this.v);
    }
    const h = 2;
    const f0 = this.f(rho), fp = this.f(rho + h), fm = this.f(Math.max(this.coreR, rho - h));
    return -(fp - 2 * f0 + fm) / (h * h) / f0;
  }

  /** graves closer to the centre than ρ */
  cum(rho: number): number {
    const pr = this.layout.plazaR, s = this.sigma;
    if (rho <= pr) return 0;
    const core = s * Math.PI * (Math.min(rho, this.coreR) ** 2 - pr * pr);
    if (rho <= this.coreR) return core;
    if (rho < this.rho0) return core + s * this.tableAt(this.aArea, rho, f => 2 * Math.PI * f);
    return this.model.cum(this.time(rho));
  }

  private tableAt(tab: Float64Array, rho: number, deriv: (f: number) => number): number {
    const x = (rho - this.coreR) / this.aH;
    const k = Math.min(Math.max(Math.floor(x), 0), A_CELLS - 1), s = x - k;
    const m0 = deriv(this.fAncient(this.coreR + k * this.aH)) * this.aH;
    const m1 = deriv(this.fAncient(this.coreR + (k + 1) * this.aH)) * this.aH;
    return hermite(tab[k], tab[k + 1], m0, m1, s);
  }

  /** calendar year at ρ. Inside the ancient zone this is only an ordering by graves (not shown). */
  time(rho: number): number {
    if (rho >= this.rho0) return this.t0 + (rho - this.rho0) / this.v;
    const g = this.cum(rho);
    let lo = ANCIENT_START, hi = this.t0;
    for (let i = 0; i < 50; i++) { const m = (lo + hi) / 2; if (this.model.cum(m) < g) lo = m; else hi = m; }
    return (lo + hi) / 2;
  }

  rhoAtTime(t: number): number {
    if (t >= this.t0) return this.rho0 + (t - this.t0) * this.v;
    return this.rhoAtCum(this.model.cum(Math.max(t, ANCIENT_START)));
  }

  rhoAtCum(n: number): number {
    if (n >= this.ancientGraves) { // history: invert the death model in time
      let lo = this.t0, hi = T_END;
      for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (this.model.cum(m) < n) lo = m; else hi = m; }
      return this.rhoAtTime((lo + hi) / 2);
    }
    let lo = 0, hi = this.rho0;
    for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (this.cum(m) < n) lo = m; else hi = m; }
    return (lo + hi) / 2;
  }

  /** metres per year: constant in history, meaningless (undated) in the ancient zone */
  vAt(t: number): number { return t >= this.t0 ? this.v : NaN; }

  /** isothermal coordinate u(ρ) = ∫ dρ/f, u = ln ρ in the flat core */
  u(rho: number): number {
    if (rho <= this.coreR) return Math.log(Math.max(rho, 1e-9));
    if (rho < this.rho0) return this.uCore + this.tableAt(this.aU, rho, f => 1 / f);
    const t = this.t0 + (rho - this.rho0) / this.v;
    const nh = this.hU.length - 1, k2 = 2 * Math.PI * this.sigma * this.v * this.v;
    if (t >= this.t0 + nh) return this.hU[nh] + k2 * (t - this.t0 - nh) / this.model.D(T_END);
    const x = t - this.t0, k = Math.min(Math.floor(x), nh - 1), s = x - k;
    return hermite(this.hU[k], this.hU[k + 1], k2 / this.model.D(this.t0 + k), k2 / this.model.D(this.t0 + k + 1), s);
  }

  rhoAtU(u: number): number {
    if (u <= this.uCore) return Math.exp(u);
    const uEnd = this.hU[this.hU.length - 1];
    if (u >= uEnd) return this.rhoEndTable + (u - uEnd) * this.f(this.rhoEndTable);
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

export { T_END };
