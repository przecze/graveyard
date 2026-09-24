// Quintic B-splines and small dense linear algebra shared by the models.

export const DEG = 5;

export class BSpline {
  readonly t: Float64Array; // knot vector, clamped
  readonly n: number;       // number of coefficients
  constructor(breaks: number[]) {
    const k: number[] = [];
    for (let i = 0; i < DEG; i++) k.push(breaks[0]);
    k.push(...breaks);
    for (let i = 0; i < DEG; i++) k.push(breaks[breaks.length - 1]);
    this.t = Float64Array.from(k);
    this.n = k.length - DEG - 1;
  }
  span(x: number): number {
    const t = this.t, n = this.n;
    if (x >= t[n]) return n - 1;
    if (x <= t[DEG]) return DEG;
    let lo = DEG, hi = n;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (x < t[m]) hi = m; else lo = m; }
    return lo;
  }
  /** basis values and derivatives 0..nd at x: ders[d][j] for coefficient span−DEG+j (NURBS book A2.3) */
  basis(x: number, nd: number): { span: number; ders: number[][] } {
    const p = DEG, t = this.t, i = this.span(x);
    const ndu = Array.from({ length: p + 1 }, () => new Array(p + 1).fill(0));
    const left = new Array(p + 1).fill(0), right = new Array(p + 1).fill(0);
    ndu[0][0] = 1;
    for (let j = 1; j <= p; j++) {
      left[j] = x - t[i + 1 - j]; right[j] = t[i + j] - x;
      let saved = 0;
      for (let r = 0; r < j; r++) {
        ndu[j][r] = right[r + 1] + left[j - r];
        const tmp = ndu[r][j - 1] / ndu[j][r];
        ndu[r][j] = saved + right[r + 1] * tmp;
        saved = left[j - r] * tmp;
      }
      ndu[j][j] = saved;
    }
    const ders = Array.from({ length: nd + 1 }, () => new Array(p + 1).fill(0));
    for (let j = 0; j <= p; j++) ders[0][j] = ndu[j][p];
    const a = [new Array(p + 1).fill(0), new Array(p + 1).fill(0)];
    for (let r = 0; r <= p; r++) {
      let s1 = 0, s2 = 1;
      a[0][0] = 1;
      for (let k = 1; k <= nd; k++) {
        let d = 0;
        const rk = r - k, pk = p - k;
        if (r >= k) { a[s2][0] = a[s1][0] / ndu[pk + 1][rk]; d = a[s2][0] * ndu[rk][pk]; }
        const j1 = rk >= -1 ? 1 : -rk, j2 = r - 1 <= pk ? k - 1 : p - r;
        for (let j = j1; j <= j2; j++) { a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]; d += a[s2][j] * ndu[rk + j][pk]; }
        if (r <= pk) { a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]; d += a[s2][k] * ndu[r][pk]; }
        ders[k][r] = d;
        [s1, s2] = [s2, s1];
      }
    }
    let r = p;
    for (let k = 1; k <= nd; k++) { for (let j = 0; j <= p; j++) ders[k][j] *= r; r *= p - k; }
    return { span: i, ders };
  }
  private readonly nLeft = new Float64Array(DEG + 1);
  private readonly nRight = new Float64Array(DEG + 1);
  readonly N = new Float64Array(DEG + 1);
  /** basis values only (no allocation, NURBS book A2.2); fills this.N, returns span */
  basisValues(x: number): number {
    const t = this.t, i = this.span(x), N = this.N, left = this.nLeft, right = this.nRight;
    N[0] = 1;
    for (let j = 1; j <= DEG; j++) {
      left[j] = x - t[i + 1 - j]; right[j] = t[i + j] - x;
      let saved = 0;
      for (let r = 0; r < j; r++) {
        const tmp = N[r] / (right[r + 1] + left[j - r]);
        N[r] = saved + right[r + 1] * tmp;
        saved = left[j - r] * tmp;
      }
      N[j] = saved;
    }
    return i;
  }
  value(c: Float64Array, x: number): number {
    const i = this.basisValues(x);
    let acc = 0;
    for (let j = 0; j <= DEG; j++) acc += this.N[j] * c[i - DEG + j];
    return acc;
  }

  /** value and derivatives 0..nd of Σ c_i B_i at x */
  eval(c: Float64Array, x: number, nd: number): number[] {
    const { span, ders } = this.basis(x, nd);
    return ders.map(row => row.reduce((acc, b, j) => acc + b * c[span - DEG + j], 0));
  }
}

// Gauss–Legendre nodes/weights on [0,1]
export const GL8: [number, number][] = (() => {
  const x = [0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363];
  const w = [0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763];
  const out: [number, number][] = [];
  for (let i = 0; i < 4; i++) { out.push([0.5 - x[i] / 2, w[i] / 2], [0.5 + x[i] / 2, w[i] / 2]); }
  return out;
})();

export function solveLinear(A: number[][], y: number[]): number[] {
  const n = y.length;
  const M = A.map((row, i) => [...row, y[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const piv = M[c][c];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / piv;
      if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

export type Quad = { x: number; w: number; p: number; b: number[][]; span: number };

/** quadrature points (8 per knot interval) with basis values/derivatives 0..3 */
export function quadrature(sp: BSpline, breaks: number[], tag: (mid: number) => number = () => 0): Quad[] {
  const quad: Quad[] = [];
  for (let i = 0; i + 1 < breaks.length; i++) {
    const a = breaks[i], b = breaks[i + 1], p = tag((a + b) / 2);
    for (const [x, w] of GL8) {
      const xx = a + (b - a) * x;
      const { span, ders } = sp.basis(xx, 3);
      quad.push({ x: xx, w: w * (b - a), p, b: ders, span });
    }
  }
  return quad;
}
