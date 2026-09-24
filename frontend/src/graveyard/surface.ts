// Geometry of the graveyard surface.
//
// Surface of revolution with metric  ds² = dρ² + f(ρ)² dφ²  (ρ = walking
// distance from the centre, φ = angle). Graves have uniform density σ
// everywhere, so the area between ρ and ρ+dρ must hold exactly the graves
// of the corresponding time span:
//
//   ancient circle (ρ ≤ ρ0):  flat, f = ρ. Holds all N_anc pre-8000 BCE graves:
//                             σ·π·(ρ0² − ρc²) = N_anc   (ρc = central plaza)
//   outer region  (ρ > ρ0):   time is linear, t = T0 + (ρ − ρ0)/v, and
//                             σ · 2π f(ρ) dρ = D(t) dt   ⇒   f = D(t) / (2πσv)
//
// Continuity of the circumference at ρ0 (f(ρ0) = ρ0) fixes the walking speed
// of time:  v = D(T0) / (2πσρ0)  metres per year. Nothing else is free: σ only
// sets the overall length scale. Gaussian curvature K = −f''/f = −D''/(D v²).
// f' jumps at ρ0 (1 inside, D'/(2πσv²) outside): the rim of the ancient
// circle is a crease carrying concentrated negative curvature.
//
// Rendering uses the isothermal coordinate u(ρ) = ∫ dρ/f  (u = ln ρ inside
// the flat circle). (u, φ) are conformal: w = u + iφ maps the whole surface
// onto the plane via e^w, preserving angles; the view around the player is
// z = f(ρp)·(e^{w − wp} − 1), which is an exact isometry at the player.

import { buildDeathModel, cumAt, deathsAt, T0, T_END, type DeathModel } from './deathModel';

export type Layout = {
  rowH: number;       // radial row pitch (m): grave + walkway
  pitch: number;      // nominal spacing along a row (m)
  plazaR: number;     // central plaza radius (m), no graves
  aisleW: number;     // radial aisle width (m)
  aisleSpacing: number; // aisles branch so neighbours are aisleSpacing..2×aisleSpacing apart
};

export const DEFAULT_LAYOUT: Layout = {
  rowH: 2.6,
  pitch: 1.3,
  plazaR: 14,
  aisleW: 2.4,
  aisleSpacing: 36,
};

const LN1P_EPS = 1e-6;
/** ln(1+x)/x, stable near 0 */
function lnRatio(x: number): number {
  return Math.abs(x) < LN1P_EPS ? 1 - x / 2 + x * x / 3 : Math.log1p(x) / x;
}

export class Surface {
  readonly layout: Layout;
  readonly model: DeathModel;
  readonly sigma: number;
  readonly rho0: number;
  readonly v: number;
  readonly rhoEndTable: number;
  readonly nRows: number;
  /** graves before row i (row i spans [plazaR + i·rowH, plazaR + (i+1)·rowH)) */
  readonly rowStart: Float64Array;
  /** number of radial aisles in row i (power of two, non-decreasing) */
  readonly rowAisles: Float64Array;
  private readonly U: Float64Array; // ∫_{T0}^{T0+k} dρ/f
  private readonly kU: number;      // 2πσv²
  private readonly lnRho0: number;

  constructor(layout: Layout = DEFAULT_LAYOUT, model: DeathModel = buildDeathModel()) {
    this.layout = layout;
    this.model = model;
    this.sigma = 1 / (layout.rowH * layout.pitch);
    const { N } = model.ancient;
    this.rho0 = Math.sqrt(N / (Math.PI * this.sigma) + layout.plazaR ** 2);
    this.v = model.D[0] / (2 * Math.PI * this.sigma * this.rho0);
    this.lnRho0 = Math.log(this.rho0);
    this.kU = 2 * Math.PI * this.sigma * this.v * this.v;

    const D = model.D, K = D.length - 1;
    this.U = new Float64Array(K + 1);
    for (let k = 1; k <= K; k++)
      this.U[k] = this.U[k - 1] + this.kU * lnRatio((D[k] - D[k - 1]) / D[k - 1]) / D[k - 1];

    this.rhoEndTable = this.rhoAtTime(T_END);
    this.nRows = Math.ceil((this.rhoEndTable - layout.plazaR) / layout.rowH);
    this.rowStart = new Float64Array(this.nRows + 1);
    this.rowAisles = new Float64Array(this.nRows);
    let fMax = 0;
    for (let i = 0; i <= this.nRows; i++) {
      this.rowStart[i] = Math.round(this.cum(layout.plazaR + i * layout.rowH));
      if (i < this.nRows) {
        fMax = Math.max(fMax, this.f(layout.plazaR + (i + 0.5) * layout.rowH));
        const m = 2 * Math.PI * fMax / layout.aisleSpacing;
        this.rowAisles[i] = Math.max(4, Math.pow(2, Math.floor(Math.log2(Math.max(m, 1)))));
      }
    }
  }

  // ── radial functions ───────────────────────────────────────────────────────

  /** calendar year at distance ρ (outer region: linear) */
  time(rho: number): number {
    if (rho >= this.rho0) return T0 + (rho - this.rho0) / this.v;
    const { N, D0, r } = this.model.ancient;
    const graves = this.cum(rho);
    // ancient ramp: cum(t) = D0/r · (e^{r(t−T0)} − e^{−r·L}),  cum(T0) = N
    const eL = 1 - N * r / D0;
    return T0 + Math.log(Math.max(graves * r / D0 + eL, 1e-300)) / r;
  }

  rhoAtTime(t: number): number {
    if (t >= T0) return this.rho0 + (t - T0) * this.v;
    const { N, D0, r } = this.model.ancient;
    const eL = 1 - N * r / D0;
    const graves = Math.max(0, D0 / r * (Math.exp(r * (t - T0)) - eL));
    return Math.sqrt(graves / (Math.PI * this.sigma) + this.layout.plazaR ** 2);
  }

  /** circumference radius: ring at ρ has length 2π·f(ρ) */
  f(rho: number): number {
    if (rho <= this.rho0) return Math.max(rho, 1e-9);
    return deathsAt(this.model.D, this.time(rho)) / (2 * Math.PI * this.sigma * this.v);
  }

  /** graves closer to the centre than ρ */
  cum(rho: number): number {
    const pr = this.layout.plazaR;
    if (rho <= pr) return 0;
    if (rho <= this.rho0) return this.sigma * Math.PI * (rho * rho - pr * pr);
    return this.model.ancient.N + cumAt(this.model.D, this.model.cum, this.time(rho));
  }

  /** isothermal coordinate u(ρ) = ln ρ0 + ∫_{ρ0}^{ρ} dρ/f */
  u(rho: number): number {
    if (rho <= this.rho0) return Math.log(Math.max(rho, 1e-9));
    const D = this.model.D, K = D.length - 1;
    const x = (rho - this.rho0) / this.v;
    if (x >= K) return this.lnRho0 + this.U[K] + (x - K) * this.kU / D[K];
    const k = Math.floor(x), s = x - k;
    const dD = D[k + 1] - D[k];
    return this.lnRho0 + this.U[k] + this.kU * s * lnRatio(s * dD / D[k]) / D[k];
  }

  rhoAtU(u: number): number {
    if (u <= this.lnRho0) return Math.exp(u);
    const du = u - this.lnRho0;
    const U = this.U, K = U.length - 1;
    if (du >= U[K]) return this.rho0 + (K + (du - U[K]) * this.model.D[K] / this.kU) * this.v;
    let lo = 0, hi = K;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (U[m] <= du) lo = m; else hi = m; }
    // Newton inside the year (u is smooth and monotone there)
    let rho = this.rho0 + (lo + (du - U[lo]) / (U[hi] - U[lo])) * this.v;
    for (let i = 0; i < 3; i++) rho -= (this.u(rho) - u) * this.f(rho);
    return rho;
  }

  /** Gaussian curvature K = −f''/f (0 inside the flat circle) */
  gaussK(rho: number): number {
    if (rho <= this.rho0) return 0;
    const t = this.time(rho), h = 3;
    const D = this.model.D;
    const d2 = (deathsAt(D, t + h) - 2 * deathsAt(D, t) + deathsAt(D, t - h)) / (h * h);
    return -d2 / (deathsAt(D, t) * this.v * this.v);
  }

  // ── rows, aisles, graves ─────────────────────────────────────────────────────

  rowOf(rho: number): number {
    return Math.floor((rho - this.layout.plazaR) / this.layout.rowH);
  }

  rowInner(i: number): number {
    return this.layout.plazaR + i * this.layout.rowH;
  }

  /** first row index where aisles of this count exist */
  firstRowWithAisles(count: number): number {
    const a = this.rowAisles;
    let lo = 0, hi = a.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] >= count) hi = m; else lo = m + 1; }
    return lo;
  }

  /**
   * Row geometry for placing graves. Graves of row i are split evenly among
   * the M segments between aisles; grave j sits at angle
   *   seg·α + γ/2 + (l + ½)·(α − γ)/n_seg
   * with α = 2π/M and γ = aisle width / f.
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
export const surface = new Surface();
