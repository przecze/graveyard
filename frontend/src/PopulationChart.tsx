import { useMemo } from 'react';
import {
  ComposedChart, Line, XAxis, YAxis,
  Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts';
import rawData from './data.json';
import deathsOwid from './deathsOwid';

// ----------------------------------------------------------------
// PRB table (data.json) format:
//   "-8000": { graves: A, pop: 5_000_000 }   ← ancient graves
//   "1":     { graves: 46B, pop: 300_000_000 } ← graves from -8000→1
//   ...
//   "1950":  { graves: 3.4B, pop: 2_499_000_000 }
// Then OWID takes over from 1950.
// ----------------------------------------------------------------

const START_YEAR = -8000;
const END_YEAR = 2026;

interface PrbEntry { year: number; graves: number; pop: number }

const prbEntries: PrbEntry[] = Object.entries(
  rawData as Record<string, { graves: number; pop: number }>,
)
  .map(([k, v]) => ({ year: Number(k), graves: v.graves, pop: v.pop }))
  .sort((a, b) => a.year - b.year);

/** A = ancient graves (all before 8000 BCE). */
export const ANCIENT_GRAVES = prbEntries[0].graves;

// ----------------------------------------------------------------
// Right-anchored fit: exponential population × linear CDR.
//
// For each interval [y0, y1], T = y1 − y0:
//   pop(t) = P0 · e^(g·t),  g = ln(P1/P0) / T,  t ∈ [0,T]
//   CDR(t) = c0 + (c1−c0)·t/T                    (linear)
//   d(t)   = CDR(t) · pop(t)                      (deaths/yr)
//
// Integral constraint:  ∫₀ᵀ d(t) dt = G
//   P0·[c0·A + c1·B] = G
//   where A = I1 − I2/T,  B = I2/T
//     I1 = ∫₀ᵀ e^(gt) dt
//     I2 = ∫₀ᵀ t·e^(gt) dt
//
// Given c1 (CDR at right edge), solve:
//   c0 = (G/P0 − c1·B) / A
//
// Anchor: at 1950 the OWID death rate gives c1 of the last interval.
// Chain right→left: c0 of interval i becomes c1 of interval i−1.
// ----------------------------------------------------------------

interface FitInterval {
  startYear: number; endYear: number;
  startPop: number;  endPop: number;
  graves: number;
  cdrLeft: number;   cdrRight: number;
  g: number; // population growth rate
}

function computeIntegrals(g: number, T: number): { I1: number; I2: number } {
  if (Math.abs(g * T) < 1e-10) {
    return { I1: T, I2: T * T / 2 };
  }
  const egT = Math.exp(g * T);
  const I1 = (egT - 1) / g;
  const I2 = T * egT / g - (egT - 1) / (g * g);
  return { I1, I2 };
}

function buildFitIntervals(): FitInterval[] {
  // Anchor at 1950: OWID death rate
  const d1950 = deathsOwid[0][1];
  const p1950 = prbEntries[prbEntries.length - 1].pop;
  let cdrRight = d1950 / p1950;

  const result: FitInterval[] = [];

  // Work backwards through PRB intervals
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
    const c1 = cdrRight;
    const c0 = (G / P0 - c1 * B) / A;

    result.push({
      startYear: left.year, endYear: right.year,
      startPop: P0, endPop: P1,
      graves: G,
      cdrLeft: c0, cdrRight: c1,
      g,
    });

    cdrRight = c0; // chain: left CDR becomes right CDR of next interval
  }

  return result.reverse(); // chronological order
}

const fitIntervals = buildFitIntervals();

// ----------------------------------------------------------------
// Deaths-per-year series from fitted intervals + OWID
// ----------------------------------------------------------------

interface DeathsPt { year: number; deathsPerYear: number }

function buildDeathsPerYear(): DeathsPt[] {
  const pts: DeathsPt[] = [];

  for (const iv of fitIntervals) {
    const T = iv.endYear - iv.startYear;
    const step = T > 500 ? 50 : T > 100 ? 10 : 1;
    for (let y = iv.startYear; y < iv.endYear; y += step) {
      const t = y - iv.startYear;
      const cdr = iv.cdrLeft + (iv.cdrRight - iv.cdrLeft) * t / T;
      const pop = iv.startPop * Math.exp(iv.g * t);
      pts.push({ year: y, deathsPerYear: cdr * pop });
    }
  }
  // Last PRB point
  const last = fitIntervals[fitIntervals.length - 1];
  pts.push({ year: last.endYear, deathsPerYear: last.cdrRight * last.endPop });

  // OWID from 1950 onward
  for (const [year, d] of deathsOwid) {
    pts.push({ year, deathsPerYear: d });
  }
  return pts;
}

// ----------------------------------------------------------------
// Available graves from ancient circle density
// ----------------------------------------------------------------

interface AncientPt { year: number; availAncient: number }

function buildAncientDensitySeries(yc: number): AncientPt[] {
  const A = ANCIENT_GRAVES;
  const yc2 = yc * yc;
  const pts: AncientPt[] = [];
  for (let calYear = START_YEAR; calYear <= END_YEAR; calYear += 10) {
    const y = calYear - START_YEAR;
    pts.push({ year: calYear, availAncient: A * (2 * (yc + y) + 1) / yc2 });
  }
  const yLast = END_YEAR - START_YEAR;
  pts.push({ year: END_YEAR, availAncient: A * (2 * (yc + yLast) + 1) / yc2 });
  return pts;
}

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------

const fmtPop = (v: number) => {
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return v.toFixed(0);
};

const fmtYear = (y: number) => (y <= 0 ? `${Math.abs(y)} BC` : `${y}`);

// ----------------------------------------------------------------
// Component
// ----------------------------------------------------------------

interface Props {
  yc: number;
  onYcChange: (yc: number) => void;
}

export default function PopulationChart({ yc, onYcChange }: Props) {
  const deathsData = useMemo(buildDeathsPerYear, []);
  const ancientData = useMemo(() => buildAncientDensitySeries(yc), [yc]);

  return (
    <>
      {/* Ancient circle slider */}
      <label className="ctrl" style={{ marginBottom: 8 }}>
        <span>Ancient circle radius: <b>{yc} yr</b></span>
        <input type="range" min={100} max={50000} step={100}
          value={yc} onChange={e => onYcChange(+e.target.value)}
          style={{ width: '100%' }} />
      </label>
      <div style={{ fontSize: 13, color: '#999', marginBottom: 8 }}>
        A = {ANCIENT_GRAVES.toLocaleString()} ancient graves
        &nbsp;&middot;&nbsp; density = A/yc&sup2; = {(ANCIENT_GRAVES / (yc * yc)).toFixed(2)}/yr&sup2;
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
          <XAxis
            dataKey="year" type="number" domain={[START_YEAR, END_YEAR]}
            tick={{ fill: '#666', fontSize: 11 }} tickFormatter={fmtYear}
          />
          <YAxis
            scale="log" domain={[1e5, 1e9]}
            tick={{ fill: '#666', fontSize: 11 }} tickFormatter={fmtPop}
            allowDataOverflow
          />
          <Tooltip
            formatter={(v: number, name: string) => [fmtPop(v), name]}
            labelFormatter={fmtYear}
            contentStyle={{ background: '#1a1a1a', border: '1px solid #333', borderRadius: 4 }}
            itemStyle={{ color: '#ccc' }} labelStyle={{ color: '#999' }}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: '#999' }} />
          <Line
            name="Graves/yr" data={deathsData} dataKey="deathsPerYear"
            stroke="#e06c75" strokeWidth={2} dot={false}
          />
          <Line
            name="Available (ancient circle density)"
            data={ancientData} dataKey="availAncient"
            stroke="#61afef" strokeWidth={2} dot={false}
            strokeDasharray="6 3"
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Interval table */}
      <details style={{ marginTop: 12 }} open>
        <summary style={{ color: '#999', fontSize: 13, cursor: 'pointer' }}>
          Exp-pop &times; linear-CDR fit (anchored at 1950 OWID)
        </summary>
        <table style={{ fontSize: 11, color: '#ccc', borderCollapse: 'collapse', marginTop: 8, width: '100%' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #444' }}>
              <th style={{ textAlign: 'left', padding: '4px 6px' }}>Interval</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>Pop start</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>Pop end</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>CDR start</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>CDR end</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>d(start)</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>d(end)</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>Source graves</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>Recon. graves</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>Diff</th>
            </tr>
          </thead>
          <tbody>
            {fitIntervals.map((iv, i) => {
              // Reconstructed integral: P0 * [c0*A + c1*B]
              const T = iv.endYear - iv.startYear;
              const { I1, I2 } = computeIntegrals(iv.g, T);
              const A = I1 - I2 / T;
              const B = I2 / T;
              const recon = iv.startPop * (iv.cdrLeft * A + iv.cdrRight * B);
              const diff = recon - iv.graves;
              const dStart = iv.cdrLeft * iv.startPop;
              const dEnd = iv.cdrRight * iv.endPop;
              return (
                <tr key={i} style={{ borderBottom: '1px solid #333' }}>
                  <td style={{ padding: '3px 6px' }}>
                    {fmtYear(iv.startYear)} &rarr; {fmtYear(iv.endYear)}
                  </td>
                  <td style={{ textAlign: 'right', padding: '3px 6px' }}>{fmtPop(iv.startPop)}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px' }}>{fmtPop(iv.endPop)}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px',
                    color: iv.cdrLeft < 0 ? '#e06c75' : undefined }}>
                    {(iv.cdrLeft * 100).toFixed(2)}%
                  </td>
                  <td style={{ textAlign: 'right', padding: '3px 6px' }}>
                    {(iv.cdrRight * 100).toFixed(2)}%
                  </td>
                  <td style={{ textAlign: 'right', padding: '3px 6px',
                    color: dStart < 0 ? '#e06c75' : undefined }}>
                    {fmtPop(dStart)}
                  </td>
                  <td style={{ textAlign: 'right', padding: '3px 6px' }}>{fmtPop(dEnd)}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px' }}>{fmtPop(iv.graves)}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px' }}>{fmtPop(recon)}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px',
                    color: Math.abs(diff) / iv.graves < 1e-6 ? '#666' : '#e06c75' }}>
                    {Math.abs(diff) < 1 ? '0' : fmtPop(diff)}
                  </td>
                </tr>
              );
            })}
            <tr style={{ borderBottom: '1px solid #333', color: '#888' }}>
              <td style={{ padding: '3px 6px' }}>1950 &rarr; (OWID anchor)</td>
              <td style={{ textAlign: 'right', padding: '3px 6px' }}>{fmtPop(2499000000)}</td>
              <td style={{ textAlign: 'right', padding: '3px 6px' }}>&mdash;</td>
              <td style={{ textAlign: 'right', padding: '3px 6px' }}>
                {(deathsOwid[0][1] / 2499000000 * 100).toFixed(2)}%
              </td>
              <td colSpan={6} style={{ textAlign: 'right', padding: '3px 6px' }}>annual OWID data</td>
            </tr>
          </tbody>
        </table>
      </details>
    </>
  );
}
