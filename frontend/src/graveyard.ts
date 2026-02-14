import rawPopData from './data.json';
import deathsOwid from './deathsOwid';

// ----------------------------------------------------------------
// Death-rate estimation from historical population data
// ----------------------------------------------------------------

const DATA_START = -10000;   // earliest data point (10 000 BC)
const PRESENT_YEAR = 2026;

/** Parse a population string like "2M", "5–10M", "<700M" → number | null */
function parsePop(s: string): number | null {
  s = s.trim().replace(/,/g, '');
  const range = s.match(/^(\d+(?:\.\d+)?)\s*[–\-]\s*(\d+(?:\.\d+)?)M?$/);
  if (range) return ((+range[1] + +range[2]) / 2) * 1e6;
  const lt = s.match(/^<\s*(\d+(?:\.\d+)?)M$/);
  if (lt) return +lt[1] * 1e6;
  const m = s.match(/^(\d+(?:\.\d+)?)M$/);
  if (m) return +m[1] * 1e6;
  const n = parseFloat(s);
  return isNaN(n) ? null : n * 1e6;
}

/** Crude death rate: 3 % before 1700, linearly falling to 2 % by 1940. */
function cdr(year: number): number {
  if (year <= 1700) return 0.03;
  if (year >= 1940) return 0.02;
  return 0.03 + ((year - 1700) / (1940 - 1700)) * (0.02 - 0.03);
}

/** Average population at each data-point year (sorted). */
function buildPopPoints(): { year: number; pop: number }[] {
  const data = rawPopData as Record<string, Record<string, string>>;
  const pts: { year: number; pop: number }[] = [];
  for (const [yearStr, estimates] of Object.entries(data)) {
    const year = Number(yearStr);
    const vals: number[] = [];
    for (const v of Object.values(estimates)) {
      const n = parsePop(v);
      if (n && n > 0) vals.push(n);
    }
    if (vals.length) {
      pts.push({ year, pop: vals.reduce((a, b) => a + b) / vals.length });
    }
  }
  pts.sort((a, b) => a.year - b.year);
  return pts;
}

/** Linearly interpolate population at an arbitrary year. */
function interpPop(pts: { year: number; pop: number }[], year: number): number {
  if (year <= pts[0].year) return pts[0].pop;
  if (year >= pts[pts.length - 1].year) return pts[pts.length - 1].pop;
  let lo = 0, hi = pts.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (pts[mid].year <= year) lo = mid; else hi = mid;
  }
  const t = (year - pts[lo].year) / (pts[hi].year - pts[lo].year);
  return pts[lo].pop + t * (pts[hi].pop - pts[lo].pop);
}

/** OWID actual deaths keyed by year for O(1) lookup. */
const owidMap = new Map<number, number>(deathsOwid);
const lastOwidYear = deathsOwid[deathsOwid.length - 1][0];
const lastOwidDeaths = deathsOwid[deathsOwid.length - 1][1];

/**
 * Build an array of annual deaths from `startYear` (inclusive)
 * to `endYear` (exclusive).  Index i → year startYear + i.
 *
 *  • Pre-1950: interpolated population × CDR
 *  • 1950–2023: OWID actuals
 *  • 2024+: extrapolate last OWID value
 */
function buildAnnualDeaths(startYear: number, endYear: number): Float64Array {
  const popPts = buildPopPoints();
  const len = endYear - startYear;
  const deaths = new Float64Array(len);
  for (let i = 0; i < len; i++) {
    const y = startYear + i;
    const owid = owidMap.get(y);
    if (owid !== undefined) {
      deaths[i] = owid;
    } else if (y > lastOwidYear) {
      deaths[i] = lastOwidDeaths;
    } else {
      deaths[i] = interpPop(popPts, y) * cdr(y);
    }
  }
  return deaths;
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
  deathsPerYear: number;
  chunks: number;
}

// ----------------------------------------------------------------
// Graveyard class — data-driven (no curve fitting)
// ----------------------------------------------------------------

export class Graveyard {
  readonly periodLength: number;
  readonly maxGraves: number;
  /** Calendar year the graveyard begins at. */
  readonly startYear: number;
  readonly totalSpan: number;
  readonly numPeriods: number;
  readonly totalGraves: number;

  /** Pre-computed graves per period and mid-period deaths/yr. */
  private readonly _gravesPerPeriod: Float64Array;
  private readonly _deathsPerYearMid: Float64Array;

  constructor(periodLength: number, maxGraves: number) {
    this.periodLength = periodLength;
    this.maxGraves = maxGraves;

    // Span rounded up so the last period includes the present
    this.totalSpan = Math.ceil((PRESENT_YEAR - DATA_START) / periodLength) * periodLength;
    this.startYear = PRESENT_YEAR - this.totalSpan;
    this.numPeriods = this.totalSpan / periodLength;

    const annualDeaths = buildAnnualDeaths(this.startYear, PRESENT_YEAR);

    this._gravesPerPeriod = new Float64Array(this.numPeriods);
    this._deathsPerYearMid = new Float64Array(this.numPeriods);
    let total = 0;

    for (let p = 0; p < this.numPeriods; p++) {
      const yOff0 = p * periodLength;               // offset into annualDeaths
      const yOff1 = Math.min((p + 1) * periodLength, annualDeaths.length);
      let sum = 0;
      for (let y = yOff0; y < yOff1; y++) sum += annualDeaths[y];
      this._gravesPerPeriod[p] = sum;
      total += sum;

      // Deaths/year at period midpoint (for display)
      const midIdx = Math.min(Math.floor(yOff0 + periodLength / 2), annualDeaths.length - 1);
      this._deathsPerYearMid[p] = annualDeaths[midIdx];
    }
    this.totalGraves = total;
  }

  gravesInPeriod(period: number): number {
    return this._gravesPerPeriod[period] ?? 0;
  }

  /** Number of chunks in a period. Period 0 (center) is always 1 chunk. */
  chunksInPeriod(period: number): number {
    if (period === 0) return 1;
    return Math.max(1, Math.ceil(this.gravesInPeriod(period) / this.maxGraves));
  }

  /**
   * Geographical area of a period's ring.
   * Period 0 (center): π·P². Period i: π·P²·(2i+1).
   */
  areaOfPeriod(period: number): number {
    const P = this.periodLength;
    return Math.PI * P * P * (2 * period + 1);
  }

  /** Grave density = graves / geographical area. */
  densityOfPeriod(period: number): number {
    return this.gravesInPeriod(period) / this.areaOfPeriod(period);
  }

  /**
   * Scaling factor for a period: period density / central density.
   * Can be < 1 for periods less dense than the center.
   */
  scalingFactor(period: number): number {
    const centralDensity = this.densityOfPeriod(0);
    if (centralDensity === 0) return 0;
    return this.densityOfPeriod(period) / centralDensity;
  }

  periodStats(period: number): PeriodStats {
    const tMid = (period + 0.5) * this.periodLength;
    return {
      yearsAgo: this.totalSpan - tMid,
      graves: this.gravesInPeriod(period),
      deathsPerYear: this._deathsPerYearMid[period],
      chunks: this.chunksInPeriod(period),
    };
  }

  /** Calendar year at the start of a period. */
  periodStartYear(period: number): number {
    return this.startYear + period * this.periodLength;
  }

  /** Angular offset for a period: odd periods shift by half a chunk for brick-like tiling. */
  chunkOffset(period: number): number {
    return period % 2 === 1 ? 0.5 : 0;
  }

  /** Angle range of a chunk in radians: [start, end). Accounts for odd-period offset. */
  chunkAngle(period: number, chunkId: number): [number, number] {
    const N = this.chunksInPeriod(period);
    const off = this.chunkOffset(period);
    return [
      ((chunkId + off) / N) * 2 * Math.PI,
      ((chunkId + 1 + off) / N) * 2 * Math.PI,
    ];
  }

  /**
   * Stateless chunk info: provide (period, chunkId), get back everything
   * about that chunk including neighbor boundaries with local coordinates.
   */
  getChunkInfo(period: number, chunkId: number): ChunkInfo {
    const N = this.chunksInPeriod(period);
    const off = this.chunkOffset(period);
    const myStart = (chunkId + off) / N;       // normalized, may exceed 1
    const myEnd = (chunkId + 1 + off) / N;

    const periodGraves = this.gravesInPeriod(period);

    // Same-period angular neighbors (offset doesn't change these)
    const clockwise = N > 1 ? { period, chunk: (chunkId - 1 + N) % N } : null;
    const anticlockwise = N > 1 ? { period, chunk: (chunkId + 1) % N } : null;

    // Inner radial boundary neighbors
    const inner: BoundaryNeighbor[] = [];
    if (period > 0) {
      const M = this.chunksInPeriod(period - 1);
      const nOff = this.chunkOffset(period - 1);
      for (const ov of this.findOverlapping(myStart, myEnd, M, nOff)) {
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
      const nOff = this.chunkOffset(period + 1);
      for (const ov of this.findOverlapping(myStart, myEnd, M, nOff)) {
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

  /**
   * Find chunks in a ring with M sectors and angular offset `nOff`
   * that overlap the normalized range [myStart, myEnd].
   * Chunk j spans [(j + nOff)/M, (j + 1 + nOff)/M].
   * Handles wrap-around correctly via unwrapped indices.
   */
  private findOverlapping(
    myStart: number, myEnd: number, M: number, nOff: number,
  ): { chunk: number; overlapStart: number; overlapEnd: number }[] {
    const result: { chunk: number; overlapStart: number; overlapEnd: number }[] = [];
    const jStart = Math.floor(myStart * M - nOff + 1e-9);
    const jEnd = Math.ceil(myEnd * M - nOff - 1e-9) - 1;
    // Cap iterations — period 0 borders every chunk in period 1 which can be 100K+.
    const MAX_NEIGHBORS = 500;
    for (let j = jStart; j <= jEnd && result.length < MAX_NEIGHBORS; j++) {
      const actualJ = ((j % M) + M) % M;
      const nStart = (j + nOff) / M;
      const nEnd = (j + 1 + nOff) / M;
      const overlapStart = Math.max(myStart, nStart);
      const overlapEnd = Math.min(myEnd, nEnd);
      if (overlapEnd > overlapStart + 1e-12) {
        result.push({ chunk: actualJ, overlapStart, overlapEnd });
      }
    }
    return result;
  }
}
