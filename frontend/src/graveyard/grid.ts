// Grave grid on the surface of revolution.
//
// Radially (outward), square plots come in back-to-back pairs of rows, each pair
// followed by a narrow ring path, so every grave touches a path on one side:
//
//   | G G · G G · G G · G G · G G ═ G G · … ═ … ≡≡ …
//     └── block: pairsPerBlock pairs ─┘ road     avenue after blocksPerRing blocks
//
// Around each ring the graves sit in segments between radial roads. The roads
// branch in two wherever the ring has doubled (per block, so a split always
// starts at a ring road), and every mainEvery-th one is a wide avenue. Inside a
// segment, narrow radial cut-throughs split the graves into runs. Their angles
// depend only on the block, so they line up through all rows of the block.
//
// Grave counts: the ring holds σ·area graves, with σ uniform over the whole
// area (paths included). Row i "owns" the catchment [plazaR + i·rowH, +rowH)
// with rowH = cycle height / rows per cycle, so every row has the same pitch
// and grave ids stay in death order outward.

export type Layout = {
  plazaR: number;        // central plaza radius (m), no graves
  cell: number;          // square plot side (m)
  path: number;          // narrow paths: between row pairs and radial cut-throughs (m)
  pairsPerBlock: number;
  blockRoad: number;     // ring road between blocks, and ordinary radial roads (m)
  blocksPerRing: number; // a ring avenue after this many blocks
  avenue: number;        // ring and radial avenue width (m)
  run: number;           // graves between radial cut-throughs (nominal)
  aisleSpacing: number;  // radial roads split so neighbours are aisleSpacing..2× apart (m)
  mainEvery: number;     // every n-th radial road is an avenue
};

export const DEFAULT_LAYOUT: Layout = {
  plazaR: 14, cell: 1.2, path: 1.0, pairsPerBlock: 5, blockRoad: 1.8, blocksPerRing: 6, avenue: 3,
  run: 8, aisleSpacing: 40, mainEvery: 8,
};

export type GridMetrics = ReturnType<typeof gridMetrics>;

export function gridMetrics(L: Layout) {
  const pairH = 2 * L.cell + L.path;
  const blockH = L.pairsPerBlock * pairH - L.path + L.blockRoad;
  const cycleH = L.blocksPerRing * blockH - L.blockRoad + L.avenue;
  const rowsPerBlock = 2 * L.pairsPerBlock;
  const rowsPerCycle = rowsPerBlock * L.blocksPerRing;
  const rowH = cycleH / rowsPerCycle;
  // nominal ring length per grave: plot + a little gap, share of cut-throughs and radial roads
  const meanRoad = (L.blockRoad * (L.mainEvery - 1) + L.avenue) / L.mainEvery;
  const slot = L.cell + 0.12;
  const along = slot * (1 + meanRoad / (1.5 * L.aisleSpacing)) + L.path / L.run;
  return { pairH, blockH, cycleH, rowsPerBlock, rowsPerCycle, rowH, slot, sigma: 1 / (rowH * along) };
}

export interface RadialProfile {
  f(rho: number): number;
  cum(rho: number): number;
}

type SegGeom = { alpha: number; gL: number; gR: number; pA: number; R: number; runA: number };

export class Grid {
  readonly L: Layout;
  readonly m: GridMetrics;
  readonly nRows: number;
  readonly nBlocks: number;
  readonly rowStart: Float64Array;   // graves before row i
  readonly blockM: Float64Array;     // radial roads around block B (power of two ≥ 4)
  readonly blockF: Float64Array;     // f at the block's middle
  readonly blockRuns: Float64Array;  // runs per segment
  private readonly aisleRuns = new Map<number, [number, number][]>(); // level → row intervals

  constructor(L: Layout, P: RadialProfile, rhoEnd: number) {
    this.L = L;
    const m = this.m = gridMetrics(L);
    this.nRows = Math.ceil((rhoEnd - L.plazaR) / m.rowH);
    this.nBlocks = Math.ceil(this.nRows / m.rowsPerBlock);
    this.rowStart = new Float64Array(this.nRows + 1);
    for (let i = 0; i <= this.nRows; i++) this.rowStart[i] = Math.round(P.cum(L.plazaR + i * m.rowH));
    this.blockM = new Float64Array(this.nBlocks);
    this.blockF = new Float64Array(this.nBlocks);
    this.blockRuns = new Float64Array(this.nBlocks);
    for (let B = 0; B < this.nBlocks; B++) {
      const f = this.blockF[B] = P.f((this.blockInner(B) + this.blockOuter(B)) / 2);
      const M = this.blockM[B] = Math.max(4, 2 ** Math.floor(Math.log2(Math.max(1, 2 * Math.PI * f / L.aisleSpacing))));
      this.blockRuns[B] = Math.max(1, Math.round(2 * Math.PI * f / M / (L.run * m.slot)));
    }
    let maxL = 4;
    for (let B = 0; B < this.nBlocks; B++) maxL = Math.max(maxL, this.blockM[B]);
    for (let lvl = 4; lvl <= maxL; lvl *= 2) {
      const runs: [number, number][] = [];
      for (let B = 0; B < this.nBlocks; B++) {
        if (this.blockM[B] < lvl) continue;
        const a = B * m.rowsPerBlock, b = Math.min(this.nRows, a + m.rowsPerBlock);
        if (runs.length && runs[runs.length - 1][1] === a) runs[runs.length - 1][1] = b;
        else runs.push([a, b]);
      }
      this.aisleRuns.set(lvl, runs);
    }
  }

  // ── radial structure ──────────────────────────────────────────────────────

  blockOf(i: number): number { return Math.floor(i / this.m.rowsPerBlock); }
  blockInner(B: number): number {
    const c = Math.floor(B / this.L.blocksPerRing), b = B - c * this.L.blocksPerRing;
    return this.L.plazaR + c * this.m.cycleH + b * this.m.blockH;
  }
  /** outer edge of the block's last grave row */
  blockOuter(B: number): number { return this.blockInner(B) + this.L.pairsPerBlock * this.m.pairH - this.L.path; }
  /** width of the ring road just outside block B */
  roadAfter(B: number): number { return (B + 1) % this.L.blocksPerRing === 0 ? this.L.avenue : this.L.blockRoad; }

  /** inner edge of grave row i */
  rowInner(i: number): number {
    const rpb = this.m.rowsPerBlock, B = Math.floor(i / rpb), k = i - B * rpb;
    return this.blockInner(B) + (k >> 1) * this.m.pairH + (k & 1) * this.L.cell;
  }
  rowCenter(i: number): number { return this.rowInner(i) + this.L.cell / 2; }
  /** start of row i's grave-count catchment (uniform rowH pitch) */
  catchInner(i: number): number { return this.L.plazaR + i * this.m.rowH; }

  /** grave row containing ρ, or −1 on a path / road / plaza */
  rowAt(rho: number): number {
    const L = this.L, m = this.m;
    const x = rho - L.plazaR;
    if (x < 0) return -1;
    const c = Math.floor(x / m.cycleH), y = x - c * m.cycleH;
    const b = Math.floor(y / m.blockH);
    if (b >= L.blocksPerRing) return -1;
    const z = y - b * m.blockH;
    if (z >= L.pairsPerBlock * m.pairH - L.path) return -1;
    const pair = Math.floor(z / m.pairH), w = z - pair * m.pairH;
    if (w >= 2 * L.cell) return -1;
    return (c * L.blocksPerRing + b) * m.rowsPerBlock + pair * 2 + (w >= L.cell ? 1 : 0);
  }
  /** approximate row index for range queries (±3 rows of the true one) */
  rowNear(rho: number): number { return Math.floor((rho - this.L.plazaR) / this.m.rowH); }

  /** nearest radius to ρ at least `clear` metres from every row of plots (on a ring path or road) */
  walkableNear(rho: number, clear = 0.25): number {
    const L = this.L;
    if (rho < L.plazaR - clear) return rho;
    // gaps between consecutive grave rows around ρ, the plaza counting as one
    const i0 = Math.max(0, this.rowNear(rho) - 6), i1 = this.rowNear(rho) + 6;
    let best = rho, bestD = Infinity, prevOuter = -Infinity;
    for (let i = i0; i <= i1 + 1; i++) {
      const inner = this.rowInner(i);
      if (i > i0 || i === 0) {
        const lo = (i === 0 ? L.plazaR - 1e9 : prevOuter) + clear, hi = inner - clear;
        if (hi >= lo) {
          const x = Math.min(hi, Math.max(lo, rho)), d = Math.abs(x - rho);
          if (d < bestD) { bestD = d; best = x; }
        }
      }
      prevOuter = inner + L.cell;
    }
    return best;
  }

  /** row intervals [a, b) in which blocks have at least `count` radial roads */
  aisleRowRuns(count: number): [number, number][] { return this.aisleRuns.get(Math.max(4, count)) ?? []; }

  /** radial span [r0, r1] of a road running through rows [a, b), into the ring roads at both ends */
  aisleSpan(a: number, b: number): [number, number] {
    const B0 = this.blockOf(a), B1 = this.blockOf(b - 1);
    const r0 = B0 === 0 ? this.L.plazaR : this.blockInner(B0) - this.roadAfter(B0 - 1);
    return [r0, this.blockOuter(B1) + this.roadAfter(B1)];
  }

  // ── around the ring ───────────────────────────────────────────────────────

  /** is radial road k (of M) an avenue? roads branch, so level = first M at which k exists */
  isMain(k: number, M: number): boolean {
    let level = M, kk = ((k % M) + M) % M;
    while (level > 4 && kk % 2 === 0) { kk /= 2; level /= 2; }
    return level <= Math.max(4, M / this.L.mainEvery);
  }
  roadWidth(k: number, M: number): number { return this.isMain(k, M) ? this.L.avenue : this.L.blockRoad; }

  private seg(B: number, sm: number): SegGeom {
    const M = this.blockM[B], fB = this.blockF[B], alpha = 2 * Math.PI / M;
    const gL = this.roadWidth(sm, M) / 2 / fB, gR = this.roadWidth(sm + 1, M) / 2 / fB;
    let R = this.blockRuns[B], pA = this.L.path / fB;
    let runA = (alpha - gL - gR - (R - 1) * pA) / R;
    if (runA <= 0) { R = 1; pA = 0; runA = Math.max(1e-9, alpha - gL - gR); }
    return { alpha, gL, gR, pA, R, runA };
  }

  /** angular spans [φ0, φ1] of the runs (plot beds) of block B intersecting [a0, a1] */
  runs(B: number, a0: number, a1: number, cb: (p0: number, p1: number) => void) {
    const alpha = 2 * Math.PI / this.blockM[B], M = this.blockM[B];
    for (let seg = Math.floor(a0 / alpha); seg <= Math.floor(a1 / alpha); seg++) {
      const g = this.seg(B, ((seg % M) + M) % M), base = seg * alpha + g.gL;
      for (let r = 0; r < g.R; r++) {
        const rs = base + r * (g.runA + g.pA);
        if (rs + g.runA >= a0 && rs <= a1) cb(rs, rs + g.runA);
      }
    }
  }

  /** graves of row i with angle in [a0, a1] (unwrapped frame): cb(index in row, φ, slot angle) */
  graves(i: number, a0: number, a1: number, cb: (j: number, phi: number, slotA: number) => void) {
    const B = this.blockOf(i), M = this.blockM[B], alpha = 2 * Math.PI / M;
    const count = this.rowStart[i + 1] - this.rowStart[i];
    for (let seg = Math.floor(a0 / alpha); seg <= Math.floor(a1 / alpha); seg++) {
      const sm = ((seg % M) + M) % M;
      const jFirst = Math.floor(sm * count / M), n = Math.floor((sm + 1) * count / M) - jFirst;
      if (n <= 0) continue;
      const g = this.seg(B, sm), base = seg * alpha + g.gL;
      for (let r = 0; r < g.R; r++) {
        const rs = base + r * (g.runA + g.pA);
        if (rs + g.runA < a0 || rs > a1) continue;
        const l0 = Math.floor(r * n / g.R), mm = Math.floor((r + 1) * n / g.R) - l0;
        if (mm <= 0) continue;
        const slotA = g.runA / mm;
        const k0 = Math.max(0, Math.floor((a0 - rs) / slotA)), k1 = Math.min(mm - 1, Math.floor((a1 - rs) / slotA));
        for (let k = k0; k <= k1; k++) cb(jFirst + l0 + k, rs + (k + 0.5) * slotA, slotA);
      }
    }
  }

  /** the plot of row i whose slot contains φ, or null on a road / cut-through */
  graveAt(i: number, phi: number): { j: number; phi: number } | null {
    if (i < 0 || i >= this.nRows) return null;
    let hit: { j: number; phi: number } | null = null;
    this.graves(i, phi, phi, (j, p, slotA) => { if (Math.abs(phi - p) <= slotA / 2) hit = { j, phi: p }; });
    return hit;
  }
}
