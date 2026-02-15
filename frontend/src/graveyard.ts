import rawPrbData from './data.json';
import deathsOwid from './deathsOwid';

// ----------------------------------------------------------------
// Death-rate estimation from PRB grave-count data + OWID
// ----------------------------------------------------------------

const DATA_START = -8000;    // 8000 BCE = start of ring model
const PRESENT_YEAR = 2026;

/**
 * Right-anchored fit: exponential population × linear CDR per interval.
 *   pop(t) = P0·e^(g·t),  g = ln(P1/P0)/T
 *   CDR(t) = c0 + (c1−c0)·t/T
 *   d(t)   = CDR(t)·pop(t)
 * Integral: P0·[c0·A + c1·B] = G where A,B from ∫e^(gt) and ∫t·e^(gt).
 * Anchor: 1950 OWID death rate → chain right-to-left.
 */
interface FitIv {
  startYr: number; endYr: number;
  startPop: number; endPop: number;
  cdrLeft: number; cdrRight: number;
  g: number;
}

function computeIvIntegrals(g: number, T: number): { I1: number; I2: number } {
  if (Math.abs(g * T) < 1e-10) return { I1: T, I2: T * T / 2 };
  const egT = Math.exp(g * T);
  return { I1: (egT - 1) / g, I2: T * egT / g - (egT - 1) / (g * g) };
}

function buildFitIntervals(): FitIv[] {
  const entries = Object.entries(rawPrbData as Record<string, { graves: number; pop: number }>)
    .map(([k, v]) => ({ year: Number(k), graves: v.graves, pop: v.pop }))
    .sort((a, b) => a.year - b.year);

  const d1950 = deathsOwid[0][1];
  const p1950 = entries[entries.length - 1].pop;
  let cdrRight = d1950 / p1950;

  const result: FitIv[] = [];
  for (let i = entries.length - 1; i >= 1; i--) {
    const left = entries[i - 1];
    const right = entries[i];
    const T = right.year - left.year;
    const P0 = left.pop;
    const P1 = right.pop;
    const G = right.graves;
    const g = Math.log(P1 / P0) / T;
    const { I1, I2 } = computeIvIntegrals(g, T);
    const A = I1 - I2 / T;
    const B = I2 / T;
    const c0 = (G / P0 - cdrRight * B) / A;
    result.push({ startYr: left.year, endYr: right.year, startPop: P0, endPop: P1, cdrLeft: c0, cdrRight, g });
    cdrRight = c0;
  }
  return result.reverse();
}

const fitIvs = buildFitIntervals();

/** Interpolate death rate at a given year. */
function interpFitRate(year: number): number {
  for (const iv of fitIvs) {
    if (year >= iv.startYr && year < iv.endYr) {
      const t = year - iv.startYr;
      const T = iv.endYr - iv.startYr;
      const cdr = iv.cdrLeft + (iv.cdrRight - iv.cdrLeft) * t / T;
      const pop = iv.startPop * Math.exp(iv.g * t);
      return cdr * pop;
    }
  }
  const last = fitIvs[fitIvs.length - 1];
  return last.cdrRight * last.endPop;
}

/** OWID actual deaths keyed by year for O(1) lookup. */
const owidMap = new Map<number, number>(deathsOwid);
const lastOwidYear = deathsOwid[deathsOwid.length - 1][0];
const lastOwidDeaths = deathsOwid[deathsOwid.length - 1][1];

/**
 * Build an array of annual deaths from `startYear` (inclusive)
 * to `endYear` (exclusive).  Index i → year startYear + i.
 *
 *  • Pre-1950: PRB interval constant rate
 *  • 1950–2023: OWID actuals
 *  • 2024+: extrapolate last OWID value
 */
function buildAnnualDeaths(startYear: number, endYear: number): Float64Array {
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
      deaths[i] = interpFitRate(y);
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
// Ancient graves (first entry in data.json = all graves before 8000 BCE)
// ----------------------------------------------------------------
const prbSorted = Object.entries(rawPrbData as Record<string, { graves: number; pop: number }>)
  .map(([k, v]) => ({ year: Number(k), graves: v.graves, pop: v.pop }))
  .sort((a, b) => a.year - b.year);
const ANCIENT_GRAVES = prbSorted[0].graves;   // ~9 billion

// ----------------------------------------------------------------
// Graveyard class — data-driven, with ancient circle
// ----------------------------------------------------------------

export class Graveyard {
  readonly periodLength: number;
  readonly maxGraves: number;
  /** Ancient circle radius in years. */
  readonly ancientCircleRadius: number;
  /** Ancient graves count (A). */
  readonly ancientGraves: number;
  /** Ancient circle density: A / yc² (π cancels). */
  readonly ancientDensity: number;
  /** Calendar year the graveyard begins at. */
  readonly startYear: number;
  readonly totalSpan: number;
  readonly numPeriods: number;
  readonly totalGraves: number;

  /** Pre-computed graves per period and mid-period deaths/yr. */
  private readonly _gravesPerPeriod: Float64Array;
  private readonly _deathsPerYearMid: Float64Array;

  constructor(periodLength: number, maxGraves: number, ancientCircleRadius: number = 20000) {
    this.periodLength = periodLength;
    this.maxGraves = maxGraves;
    this.ancientCircleRadius = ancientCircleRadius;
    this.ancientGraves = ANCIENT_GRAVES;
    this.ancientDensity = ANCIENT_GRAVES / (ancientCircleRadius * ancientCircleRadius);

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

  /** Number of chunks in a period (all periods are rings around the ancient circle). */
  chunksInPeriod(period: number): number {
    return Math.max(1, Math.ceil(this.gravesInPeriod(period) / this.maxGraves));
  }

  /**
   * Geographical area of a period's ring, accounting for the ancient circle.
   * Period i: inner radius = yc + i·P, outer radius = yc + (i+1)·P
   * Area = π·((yc+(i+1)P)² - (yc+iP)²) = π·P²·(2·(yc/P + i) + 1)
   */
  areaOfPeriod(period: number): number {
    const P = this.periodLength;
    const ycP = this.ancientCircleRadius / P;  // yc in units of P
    return Math.PI * P * P * (2 * (ycP + period) + 1);
  }

  /** Grave density = graves / geographical area. */
  densityOfPeriod(period: number): number {
    return this.gravesInPeriod(period) / this.areaOfPeriod(period);
  }

  /**
   * Scaling factor for a period: period density / ancient circle density.
   * π cancels: graves_i·yc² / (A·P²·(2·(yc/P+i)+1))
   */
  scalingFactor(period: number): number {
    const P = this.periodLength;
    const yc = this.ancientCircleRadius;
    const ringFactor = 2 * (yc / P + period) + 1;
    const denom = this.ancientGraves * P * P * ringFactor;
    if (denom === 0) return 0;
    return (this.gravesInPeriod(period) * yc * yc) / denom;
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
