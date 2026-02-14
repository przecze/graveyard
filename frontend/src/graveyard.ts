const TOTAL_GRAVES = 100e9; // 100 billion

// ----------------------------------------------------------------
// Exponential fit internals
// ----------------------------------------------------------------

function fitExponential(
  lastPeriodGraves: number,
  P: number,
  T: number,
): { alpha: number; A: number } {
  const ratio = lastPeriodGraves / TOTAL_GRAVES;

  const g = (a: number): number => {
    if (Math.abs(a) < 1e-15) return P / T - ratio;
    const aT = a * T;
    if (aT > 700) return 1 - ratio;
    if (aT < -700) return -ratio;
    const eaT = Math.exp(aT);
    return eaT * (1 - Math.exp(-a * P)) / (eaT - 1) - ratio;
  };

  const g0 = g(0);
  if (Math.abs(g0) < 1e-12) {
    return { alpha: 1e-15, A: TOTAL_GRAVES / T };
  }

  let lo: number, hi: number;
  if (g0 < 0) {
    lo = 0; hi = 0.01;
    while (g(hi) < 0 && hi < 1) hi *= 2;
  } else {
    hi = 0; lo = -0.01;
    while (g(lo) > 0 && lo > -1) lo *= 2;
  }

  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (g(mid) < 0) lo = mid; else hi = mid;
  }

  const alpha = (lo + hi) / 2;
  const aT = alpha * T;
  const eaT = Math.exp(Math.min(Math.max(aT, -700), 700));
  const A = Math.abs(aT) < 1e-10
    ? TOTAL_GRAVES / T
    : TOTAL_GRAVES * alpha / (eaT - 1);

  return { alpha, A };
}

function integralOfExp(A: number, alpha: number, t1: number, t2: number): number {
  if (Math.abs(alpha) < 1e-15) return A * (t2 - t1);
  const clamp = (v: number) => Math.min(Math.max(v, -700), 700);
  return (A / alpha) * (Math.exp(clamp(alpha * t2)) - Math.exp(clamp(alpha * t1)));
}

// ----------------------------------------------------------------
// Types
// ----------------------------------------------------------------

export interface ChunkAddr {
  period: number;
  chunk: number;
}

/** A neighbor that shares a radial boundary, with local edge coordinates. */
export interface BoundaryNeighbor {
  addr: ChunkAddr;
  /** Where the shared boundary starts on THIS chunk's edge (0 = clockwise edge, 1 = anticlockwise edge). */
  localStart: number;
  /** Where the shared boundary ends on THIS chunk's edge. */
  localEnd: number;
}

export interface ChunkInfo {
  period: number;
  chunkId: number;
  /** Clockwise edge angle (radians). */
  angleStart: number;
  /** Anticlockwise edge angle (radians). */
  angleEnd: number;
  /** Estimated graves in this chunk. */
  graves: number;
  /** Total graves in the whole period. */
  periodGraves: number;
  /** Total chunks in the whole period. */
  periodChunks: number;
  neighbors: {
    /** Chunks sharing the inner (radial) boundary, with local coordinates. */
    inner: BoundaryNeighbor[];
    /** Same-period neighbor in clockwise direction (null if period has 1 chunk). */
    clockwise: ChunkAddr | null;
    /** Same-period neighbor in anticlockwise direction. */
    anticlockwise: ChunkAddr | null;
    /** Chunks sharing the outer (radial) boundary, with local coordinates. */
    outer: BoundaryNeighbor[];
  };
}

export interface PeriodStats {
  yearsAgo: number;
  graves: number;
  fTimesP: number;
  chunks: number;
}

// ----------------------------------------------------------------
// Graveyard class
// ----------------------------------------------------------------

export class Graveyard {
  static readonly TOTAL_GRAVES = TOTAL_GRAVES;

  readonly periodLength: number;
  readonly lastPeriodGraves: number;
  readonly maxGraves: number;
  readonly totalSpan: number;
  readonly numPeriods: number;
  readonly alpha: number;
  readonly A: number;

  constructor(periodLength: number, lastPeriodGraves: number, maxGraves: number) {
    this.periodLength = periodLength;
    this.lastPeriodGraves = lastPeriodGraves;
    this.maxGraves = maxGraves;
    this.totalSpan = Math.round(22000 / periodLength) * periodLength;
    this.numPeriods = this.totalSpan / periodLength;
    const fit = fitExponential(lastPeriodGraves, periodLength, this.totalSpan);
    this.alpha = fit.alpha;
    this.A = fit.A;
  }

  gravesInPeriod(period: number): number {
    const t1 = period * this.periodLength;
    const t2 = (period + 1) * this.periodLength;
    return integralOfExp(this.A, this.alpha, t1, t2);
  }

  /** Number of chunks in a period. Period 0 (center) is always 1 chunk. */
  chunksInPeriod(period: number): number {
    if (period === 0) return 1;
    return Math.ceil(this.gravesInPeriod(period) / this.maxGraves);
  }

  periodStats(period: number): PeriodStats {
    const t1 = period * this.periodLength;
    const t2 = (period + 1) * this.periodLength;
    const tMid = (t1 + t2) / 2;
    return {
      yearsAgo: this.totalSpan - tMid,
      graves: integralOfExp(this.A, this.alpha, t1, t2),
      fTimesP: this.A * Math.exp(Math.min(this.alpha * tMid, 700)) * this.periodLength,
      chunks: this.chunksInPeriod(period),
    };
  }

  /** Angle range of a chunk in radians: [start, end). */
  chunkAngle(period: number, chunkId: number): [number, number] {
    const N = this.chunksInPeriod(period);
    return [(chunkId / N) * 2 * Math.PI, ((chunkId + 1) / N) * 2 * Math.PI];
  }

  /**
   * Stateless chunk info: provide (period, chunkId), get back everything
   * about that chunk including neighbor boundaries with local coordinates.
   */
  getChunkInfo(period: number, chunkId: number): ChunkInfo {
    const N = this.chunksInPeriod(period);
    const myStart = chunkId / N;       // normalized [0, 1)
    const myEnd = (chunkId + 1) / N;

    const periodGraves = this.gravesInPeriod(period);

    // Same-period angular neighbors
    const clockwise = N > 1 ? { period, chunk: (chunkId - 1 + N) % N } : null;
    const anticlockwise = N > 1 ? { period, chunk: (chunkId + 1) % N } : null;

    // Inner radial boundary neighbors
    const inner: BoundaryNeighbor[] = [];
    if (period > 0) {
      const M = this.chunksInPeriod(period - 1);
      for (const ov of this.findOverlapping(myStart, myEnd, M)) {
        inner.push({
          addr: { period: period - 1, chunk: ov.chunk },
          localStart: (ov.overlapStart - myStart) * N,
          localEnd: (ov.overlapEnd - myStart) * N,
        });
      }
    }

    // Outer radial boundary neighbors
    const outer: BoundaryNeighbor[] = [];
    if (period < this.numPeriods - 1) {
      const M = this.chunksInPeriod(period + 1);
      for (const ov of this.findOverlapping(myStart, myEnd, M)) {
        outer.push({
          addr: { period: period + 1, chunk: ov.chunk },
          localStart: (ov.overlapStart - myStart) * N,
          localEnd: (ov.overlapEnd - myStart) * N,
        });
      }
    }

    return {
      period,
      chunkId,
      angleStart: myStart * 2 * Math.PI,
      angleEnd: myEnd * 2 * Math.PI,
      graves: periodGraves / N,
      periodGraves,
      periodChunks: N,
      neighbors: { inner, clockwise, anticlockwise, outer },
    };
  }

  private findOverlapping(
    myStart: number, myEnd: number, M: number,
  ): { chunk: number; overlapStart: number; overlapEnd: number }[] {
    const result: { chunk: number; overlapStart: number; overlapEnd: number }[] = [];
    const jStart = Math.floor(myStart * M + 1e-9);
    const jEnd = Math.ceil(myEnd * M - 1e-9) - 1;
    for (let j = Math.max(0, jStart); j <= Math.min(jEnd, M - 1); j++) {
      result.push({
        chunk: j,
        overlapStart: Math.max(myStart, j / M),
        overlapEnd: Math.min(myEnd, (j + 1) / M),
      });
    }
    return result;
  }
}
