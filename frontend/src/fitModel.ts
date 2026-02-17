/**
 * Shared death-rate fitting model.
 *
 * Right-anchored fit: exponential population × linear CDR per interval.
 *   pop(t) = P0·e^(g·t),  g = ln(P1/P0)/T,  t ∈ [0,T]
 *   CDR(t) = c0 + (c1−c0)·t/T
 *   d(t)   = CDR(t)·pop(t)
 *
 * Integral constraint:  ∫₀ᵀ d(t) dt = G
 *   P0·[c0·A + c1·B] = G
 *   where A = I1 − I2/T,  B = I2/T
 *     I1 = ∫₀ᵀ e^(gt) dt
 *     I2 = ∫₀ᵀ t·e^(gt) dt
 *
 * Anchor: 1950 OWID death rate → chain right-to-left.
 */

import rawPrbData from './data.json';
import deathsOwid from './deathsOwid';

// ----------------------------------------------------------------
// Constants
// ----------------------------------------------------------------

export const DATA_START = -8000;   // 8 000 BCE = start of ring model
export const PRESENT_YEAR = 2026;

// ----------------------------------------------------------------
// PRB data (parsed once)
// ----------------------------------------------------------------

export interface PrbEntry {
  year: number;
  graves: number;
  pop: number;
}

export const prbEntries: PrbEntry[] = Object.entries(
  rawPrbData as Record<string, { graves: number; pop: number }>,
)
  .map(([k, v]) => ({ year: Number(k), graves: v.graves, pop: v.pop }))
  .sort((a, b) => a.year - b.year);

/** Ancient graves — all deaths before 8 000 BCE (first entry in data.json). */
export const ANCIENT_GRAVES = prbEntries[0].graves;

// ----------------------------------------------------------------
// Fit interval types & helpers
// ----------------------------------------------------------------

export interface FitInterval {
  startYear: number;
  endYear: number;
  startPop: number;
  endPop: number;
  graves: number;
  cdrLeft: number;
  cdrRight: number;
  /** Population growth rate ln(P1/P0)/T. */
  g: number;
}

export function computeIntegrals(g: number, T: number): { I1: number; I2: number } {
  if (Math.abs(g * T) < 1e-10) {
    return { I1: T, I2: T * T / 2 };
  }
  const egT = Math.exp(g * T);
  return {
    I1: (egT - 1) / g,
    I2: T * egT / g - (egT - 1) / (g * g),
  };
}

// ----------------------------------------------------------------
// Build fit intervals (computed once at module load)
// ----------------------------------------------------------------

function buildFitIntervals(): FitInterval[] {
  const d1950 = deathsOwid[0][1];
  const p1950 = prbEntries[prbEntries.length - 1].pop;
  let cdrRight = d1950 / p1950;

  const result: FitInterval[] = [];

  for (let i = prbEntries.length - 1; i >= 1; i--) {
    const left = prbEntries[i - 1];
    const right = prbEntries[i];
    const T = right.year - left.year;
    const P0 = left.pop;
    const P1 = right.pop;
    const G = right.graves;
    const g = Math.log(P1 / P0) / T;

    const { I1, I2 } = computeIntegrals(g, T);
    const A = I1 - I2 / T;
    const B = I2 / T;
    const c0 = (G / P0 - cdrRight * B) / A;

    result.push({
      startYear: left.year, endYear: right.year,
      startPop: P0, endPop: P1,
      graves: G,
      cdrLeft: c0, cdrRight,
      g,
    });

    cdrRight = c0;
  }

  return result.reverse();
}

export const fitIntervals = buildFitIntervals();
