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
//   ancient     one undated bucket: everyone who died before the switch year,
//               from the centre out to the ancient radius R (an input). One
//               smooth curve f = ρ·e^g (ancient.ts): regular at the centre,
//               matching the history ring and its first three derivatives at
//               R, holding exactly the graves before the switch, and otherwise
//               as smooth as possible. It curves however much it has to;
//               positive curvature (a bulge) and shrinking rings (a neck) are
//               reported. Floor: rings never shrink ⇒ R ≳ N_before·v / D0.
//   history     time is linear, t = switch + (ρ − ρ0)/v, f = D(t)/(2πσv),
//               K = −D''/(D v²). Year markers start here.
//
// D(t) is C⁴; the ancient curve is a quintic spline joined C³ to history, so K
// is C¹ at the rim and C² everywhere else.
// Optionally the death rate right after the switch is "flattened" (starts
// higher, rises slower, same period totals) to enlarge f0 and shrink the
// ancient zone further.
//
// Units: the interface measures distance in history-years (1 yr = v metres).
// Uniformly rescaling all lengths (σ → σ/λ², v → λv, every length → λ·) gives
// the same world, so in years the shape has one density knob: graves per yr²
// (σv²). With the plot size fixed in metres (grid.ts, σ graves/m²) that sets
// v = √(density/σ): denser means a year is fewer metres, i.e. plots look
// bigger at the same view width in years. Everything else is given in years.
//
// Rendering uses the isothermal coordinate u = ∫ dρ/f (u ≈ ln ρ at the
// centre): w = u + iφ is conformal and the view is
// z = f(ρp)·(e^{w − wp} − 1), an exact isometry at the player.

import { ANCIENT_CELLS, fitAncient, type AncientFit } from './ancient';
import { DEFAULT_LAYOUT, gridMetrics, Grid, type Layout } from './grid';
import { ANCIENT_START, DEFAULT_MODEL_PARAMS, deathModel, T_END, type DeathModel, type ModelParams } from './deathModel';

export { DEFAULT_LAYOUT, type Layout };

export type Shape = ModelParams & {
  switchYear: number;  // history (dated, time-linear) starts here
  flatten: number;     // years after the switch over which the death rate is flattened (0 = off)
  density: number;     // graves per yr² (σv²): the world-shape knob
  ancientShare: number; // ancient zone's share of the whole walk, centre → edge (0..1)
};

export const DEFAULT_SHAPE: Shape = {
  ...DEFAULT_MODEL_PARAMS, switchYear: -3000, flatten: 3000, density: 2.7, ancientShare: 0.1,
};

export type ShapeReport = {
  ok: boolean;
  problem?: string;
  minKRadius: number;    // tightest curvature radius in the ancient zone (m)
  rimRing: number;       // ring length at the switch (m)
  ancientShare: number;  // ancient radius / total radius
  floorShare: number;    // same for the ideal cylinder N·v/D0
  ancientLen: number;    // centre → switch (m)
  historyLen: number;    // switch → T_END (m)
  floorLen: number;      // below N_before·v/D0 the ancient rings must outgrow the history ring (a bulb) (m)
  maxRing: number;       // longest ring in the ancient zone (m)
  rimK: number;          // curvature of history itself at the switch (1/m²): the fit must match it
  neck?: [number, number];   // ρ range where rings shrink outward in the ancient zone (m)
  bulge?: [number, number];  // ρ range of positive curvature in the ancient zone (m)
  maxBulgeK: number;         // largest positive K there (1/m², 0 if none)
};

const A_CELLS = 8192;     // ancient zone table cells

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
  readonly v: number;            // metres per history year
  readonly report: ShapeReport;
  readonly rhoEndTable: number;
  readonly grid: Grid;
  readonly ancientGraves: number;    // graves before the switch
  private readonly f0: number;
  private readonly D0: number;
  private readonly aFit: AncientFit;  // ancient: f' = (f0/R)·p(ρ/R), f tabulated
  // ancient tables on ρ = coreR + k·aH: ∫dρ/f and ∫2πf dρ
  private readonly aH: number;
  private readonly aU: Float64Array;
  private readonly aArea: Float64Array;
  private readonly hU: Float64Array; // history: u on t = t0 + k
  private readonly uCore: number;

  constructor(shape: Shape = DEFAULT_SHAPE, layout: Layout = DEFAULT_LAYOUT,
              model: DeathModel = deathModel({ from: shape.switchYear, years: shape.flatten }, shape)) {
    this.layout = layout;
    this.shape = shape;
    this.model = model;
    const gm = gridMetrics(layout);
    const sigma = this.sigma = gm.sigma;
    const v = this.v = Math.sqrt(shape.density / sigma);
    const T = this.t0 = shape.switchYear;
    const coreR = this.coreR = layout.plazaR; // f ≈ ρ inside the plaza
    this.D0 = model.D(T);
    this.f0 = this.D0 / (2 * Math.PI * sigma * v);
    this.ancientGraves = model.cum(T);
    const share = Math.min(0.9, Math.max(0.01, shape.ancientShare));
    const R = this.rho0 = Math.max(4 * coreR, share / (1 - share) * (T_END - T) * v);
    // history ring at the switch: ln f and its ρ-derivatives
    const [D, D1, D2, D3] = model.Dd(T);
    // f ∝ D(t), dρ = v dt  ⇒  f^(k) = f0·D^(k)/(D·v^k)
    const f0 = this.f0;
    const fit = this.aFit = fitAncient(R, layout.plazaR, this.ancientGraves / sigma, [f0, f0 * D1 / (D * v), f0 * D2 / (D * v * v), f0 * D3 / (D * v ** 3)]);
    const problem = fit.problem;

    // ancient tables
    const L = this.rho0 - coreR;
    this.aH = L / A_CELLS;
    this.aU = new Float64Array(A_CELLS + 1);
    this.aArea = new Float64Array(A_CELLS + 1);
    this.uCore = Math.log(coreR);
    let minKR = Infinity, maxBulgeK = 0, fPrev = coreR;
    let neck: [number, number] | undefined, bulge: [number, number] | undefined;
    const grow = (range: [number, number] | undefined, r: number): [number, number] => (range ? [range[0], r] : [r, r]);
    for (let k = 1; k <= A_CELLS; k++) {
      const rk = coreR + k * this.aH, fk = this.fAncient(rk);
      if (fk < fPrev * (1 - 1e-12)) neck = grow(neck, rk);
      fPrev = fk;
      let su = 0, sa = 0;
      for (const [x, w] of GL) { const f = this.fAncient(coreR + (k - 1 + x) * this.aH); su += w / f; sa += w * 2 * Math.PI * f; }
      this.aU[k] = this.aU[k - 1] + su * this.aH;
      this.aArea[k] = this.aArea[k - 1] + sa * this.aH;
      const K = this.gaussK(rk);
      if (K) minKR = Math.min(minKR, 1 / Math.sqrt(Math.abs(K)));
      if (K > 1e-12 / (v * v)) { bulge = grow(bulge, rk); maxBulgeK = Math.max(maxBulgeK, K); }
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
      rimRing: 2 * Math.PI * this.f0,
      ancientShare: this.rho0 / (this.rho0 + hist), floorShare: floor / (floor + hist),
      ancientLen: this.rho0, historyLen: hist,
      floorLen: floor, neck, bulge, maxBulgeK, rimK: -D2 / (D * v * v),
      maxRing: 2 * Math.PI * this.f0 * fit.F.reduce((a, b) => Math.max(a, b), 0),
    };

    // rows
    this.rhoEndTable = this.rhoAtTime(T_END);
    this.grid = new Grid(layout, this, this.rhoEndTable);
  }

  get nRows(): number { return this.grid.nRows; }
  get rowStart(): Float64Array { return this.grid.rowStart; }

  private fAncient(rho: number): number {
    const R = this.rho0, x = Math.min(Math.max(rho, 0), R) / R * ANCIENT_CELLS;
    const k = Math.min(Math.floor(x), ANCIENT_CELLS - 1), s = x - k, F = this.aFit.F, h = 1 / ANCIENT_CELLS;
    const sp = this.aFit.sp, p = this.aFit.p;
    const val = hermite(F[k], F[k + 1], sp.value(p, k * h) * h, sp.value(p, (k + 1) * h) * h, s);
    return Math.max(this.f0 * val, 1e-9);
  }

  // ── radial functions ───────────────────────────────────────────────────────

  isAncient(rho: number): boolean { return rho < this.rho0; }

  /** ring radius: a ring at ρ is 2π·f(ρ) long */
  f(rho: number): number {
    if (rho < this.rho0) return this.fAncient(rho);
    return this.model.D(this.time(rho)) / (2 * Math.PI * this.sigma * this.v);
  }

  /** radial grave rows per history year */
  get rowsPerYear(): number { return this.v / this.grid.m.rowH; }

  /** metres → history-years of distance */
  yr(m: number): number { return m / this.v; }

  /** graves per metre walked outward × v: equals D(t) in the history zone */
  dEquiv(rho: number): number { return rho < this.grid.L.plazaR ? 0 : 2 * Math.PI * this.sigma * this.f(rho) * this.v; }

  /** Gaussian curvature K = −f''/f */
  gaussK(rho: number): number {
    if (rho >= this.rho0) {
      const [D, , D2] = this.model.Dd(this.time(rho));
      return -D2 / (D * this.v * this.v);
    }
    // f'' = (f0/R²)·p_x; at the centre K = −f'''(0) = −(f0/R³)·p_xx(0)
    const R = this.rho0, x = Math.max(rho, 0) / R;
    const [, px, pxx] = this.aFit.sp.eval(this.aFit.p, x, 2);
    if (x < 1e-4) return -this.f0 * pxx / R ** 3;
    return -this.f0 * px / (R * R) / this.fAncient(rho);
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

  /** isothermal coordinate u(ρ) = ∫ dρ/f, u = ln ρ inside the plaza */
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
}

export { T_END };
