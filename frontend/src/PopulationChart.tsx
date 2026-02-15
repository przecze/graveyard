import { useMemo, useState } from 'react';
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
// Exponential fit per interval, chained left→right.
//
// Within each interval:  d(t) = dLeft · e^(k · (t − t_start))
// Integral constraint:   dLeft · (e^(kT) − 1) / k = G
//   → solve for k (Newton on (e^u−1)/u = R, where u = kT, R = G/(dLeft·T))
//   → dRight = dLeft · e^(kT)
//   → chain: dLeft of next = dRight of current
// ----------------------------------------------------------------

/** Solve (e^u − 1)/u = R for u using Newton's method. */
function solveExpU(R: number): number {
  if (Math.abs(R - 1) < 1e-10) return 0;
  // Initial guess
  let u = R > 10 ? Math.log(R) + Math.log(Math.log(R) + 1) : 2 * (R - 1);
  if (R < 0.1) u = -2; // shrinking case
  for (let iter = 0; iter < 100; iter++) {
    const eu = Math.exp(u);
    // Avoid division by very small u
    if (Math.abs(u) < 1e-14) { u = 2 * (R - 1); break; }
    const f = (eu - 1) / u - R;
    const fp = (u * eu - eu + 1) / (u * u);
    if (Math.abs(fp) < 1e-30) break;
    const du = f / fp;
    u -= du;
    if (Math.abs(du) < 1e-12 * Math.max(1, Math.abs(u))) break;
  }
  return u;
}

interface ExpInterval {
  startYear: number; endYear: number;
  startPop: number;  endPop: number;
  graves: number;
  dLeft: number;  dRight: number;
  k: number;
}

function buildExpIntervals(cdr0: number): ExpInterval[] {
  const result: ExpInterval[] = [];
  let dLeft = cdr0 * prbEntries[0].pop;

  for (let i = 1; i < prbEntries.length; i++) {
    const prev = prbEntries[i - 1];
    const cur = prbEntries[i];
    const T = cur.year - prev.year;
    const G = cur.graves;
    const R = G / (dLeft * T);
    const u = solveExpU(R);
    const k = u / T;
    const dRight = dLeft * Math.exp(u);

    result.push({
      startYear: prev.year, endYear: cur.year,
      startPop: prev.pop, endPop: cur.pop,
      graves: G, dLeft, dRight, k,
    });

    dLeft = dRight;
  }
  return result;
}

// ----------------------------------------------------------------
// Deaths-per-year series from exponential intervals + OWID
// ----------------------------------------------------------------

interface DeathsPt { year: number; deathsPerYear: number }

function buildDeathsPerYear(intervals: ExpInterval[]): DeathsPt[] {
  const pts: DeathsPt[] = [];

  for (const iv of intervals) {
    const T = iv.endYear - iv.startYear;
    const step = T > 500 ? 50 : T > 100 ? 10 : 1;
    for (let y = iv.startYear; y < iv.endYear; y += step) {
      pts.push({ year: y, deathsPerYear: iv.dLeft * Math.exp(iv.k * (y - iv.startYear)) });
    }
  }
  // Last PRB point
  const last = intervals[intervals.length - 1];
  pts.push({ year: last.endYear, deathsPerYear: last.dRight });

  // OWID from 1950 onward
  for (const [year, d] of deathsOwid) {
    pts.push({ year, deathsPerYear: d });
  }
  return pts;
}

// ----------------------------------------------------------------
// Available graves from ancient circle density (commented-out per request,
// but kept for later use)
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
  const [cdr0, setCdr0] = useState(0.035); // 3.5%

  const expIntervals = useMemo(() => buildExpIntervals(cdr0), [cdr0]);
  const deathsData = useMemo(() => buildDeathsPerYear(expIntervals), [expIntervals]);
  const ancientData = useMemo(() => buildAncientDensitySeries(yc), [yc]);

  return (
    <>
      {/* Initial CDR slider */}
      <label className="ctrl" style={{ marginBottom: 4 }}>
        <span>Initial CDR at 8000 BCE: <b>{(cdr0 * 100).toFixed(1)}%</b>
          &nbsp;&rarr; d₀ = {fmtPop(cdr0 * prbEntries[0].pop)}/yr</span>
        <input type="range" min={0.001} max={0.10} step={0.001}
          value={cdr0} onChange={e => setCdr0(+e.target.value)}
          style={{ width: '100%' }} />
      </label>

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
      <details style={{ marginTop: 12 }}>
        <summary style={{ color: '#999', fontSize: 13, cursor: 'pointer' }}>
          Exponential fit per interval (chained)
        </summary>
        <table style={{ fontSize: 11, color: '#ccc', borderCollapse: 'collapse', marginTop: 8, width: '100%' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #444' }}>
              <th style={{ textAlign: 'left', padding: '4px 6px' }}>Interval</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>d(start)</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>CDR start</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>d(end)</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>CDR end</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>Source graves</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>Recon. graves</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>Diff</th>
            </tr>
          </thead>
          <tbody>
            {expIntervals.map((iv, i) => {
              const T = iv.endYear - iv.startYear;
              // Reconstructed integral: dLeft * (e^(kT)-1)/k, or dLeft*T when k≈0
              const recon = Math.abs(iv.k) < 1e-14
                ? iv.dLeft * T
                : iv.dLeft * (Math.exp(iv.k * T) - 1) / iv.k;
              const diff = recon - iv.graves;
              const cdrStart = iv.dLeft / iv.startPop;
              const cdrEnd = iv.dRight / iv.endPop;
              return (
                <tr key={i} style={{ borderBottom: '1px solid #333' }}>
                  <td style={{ padding: '3px 6px' }}>
                    {fmtYear(iv.startYear)} &rarr; {fmtYear(iv.endYear)}
                  </td>
                  <td style={{ textAlign: 'right', padding: '3px 6px' }}>{fmtPop(iv.dLeft)}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px' }}>{(cdrStart * 100).toFixed(2)}%</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px' }}>{fmtPop(iv.dRight)}</td>
                  <td style={{ textAlign: 'right', padding: '3px 6px' }}>{(cdrEnd * 100).toFixed(2)}%</td>
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
              <td style={{ padding: '3px 6px' }}>1950 &rarr; (OWID)</td>
              <td style={{ textAlign: 'right', padding: '3px 6px' }}>{fmtPop(deathsOwid[0][1])}</td>
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
