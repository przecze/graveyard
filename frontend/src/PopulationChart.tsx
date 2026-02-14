import { useMemo, useState } from 'react';
import {
  ComposedChart, Line, Scatter, XAxis, YAxis,
  Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';
import rawData from './data.json';
import deathsOwid from './deathsOwid';

/** Parse a population string like "2M", "1,265M", "5–10M", "<700M", "226" → number */
function parsePop(s: string): number | null {
  s = s.trim().replace(/,/g, '');
  // Range: "5–10M", "270–330M", "1650–1710M"
  const range = s.match(/^(\d+(?:\.\d+)?)\s*[–\-]\s*(\d+(?:\.\d+)?)M?$/);
  if (range) return ((+range[1] + +range[2]) / 2) * 1e6;
  // "<700M"
  const lt = s.match(/^<\s*(\d+(?:\.\d+)?)M$/);
  if (lt) return +lt[1] * 1e6;
  // "2M", "1265M"
  const m = s.match(/^(\d+(?:\.\d+)?)M$/);
  if (m) return +m[1] * 1e6;
  // Plain number → assume millions
  const n = parseFloat(s);
  return isNaN(n) ? null : n * 1e6;
}

const CDR_ANCIENT = 0.03;
const CDR_MODERN = 0.02;
const CDR_TRANSITION_START = 1700;
const CDR_TRANSITION_END = 1940;

/** Crude death rate: 3% before 1700, linearly falling to 1.2% by 1940 */
function cdr(year: number): number {
  if (year <= CDR_TRANSITION_START) return CDR_ANCIENT;
  if (year >= CDR_TRANSITION_END) return CDR_MODERN;
  const t = (year - CDR_TRANSITION_START) / (CDR_TRANSITION_END - CDR_TRANSITION_START);
  return CDR_ANCIENT + t * (CDR_MODERN - CDR_ANCIENT);
}

interface ScatterPt { year: number; pop: number }
interface AvgPt { year: number; avg: number }
interface DeathsPt { year: number; deaths: number }

/** Linearly interpolate population between data-point years, return annual deaths. */
function interpolateDeaths(avgPts: AvgPt[]): DeathsPt[] {
  if (avgPts.length < 2) return avgPts.map(p => ({ year: p.year, deaths: p.avg * CRUDE_DEATH_RATE }));
  const deaths: DeathsPt[] = [];
  for (let i = 0; i < avgPts.length - 1; i++) {
    const { year: y0, avg: p0 } = avgPts[i];
    const { year: y1, avg: p1 } = avgPts[i + 1];
    const span = y1 - y0;
    // For very large spans (>500 yr), sample every 100 years; otherwise every year
    const step = span > 500 ? 100 : span > 50 ? 10 : 1;
    for (let y = y0; y < y1; y += step) {
      const t = (y - y0) / span;
      const pop = p0 + t * (p1 - p0);
      deaths.push({ year: y, deaths: pop * cdr(y) });
    }
  }
  // Last point
  const last = avgPts[avgPts.length - 1];
  deaths.push({ year: last.year, deaths: last.avg * cdr(last.year) });
  return deaths;
}

function buildData() {
  const data = rawData as Record<string, Record<string, string>>;
  const scatter: ScatterPt[] = [];
  const avg: AvgPt[] = [];

  for (const [yearStr, estimates] of Object.entries(data)) {
    const year = Number(yearStr);
    const vals: number[] = [];
    for (const v of Object.values(estimates)) {
      const n = parsePop(v);
      if (n && n > 0) { scatter.push({ year, pop: n }); vals.push(n); }
    }
    if (vals.length) {
      avg.push({ year, avg: vals.reduce((a, b) => a + b) / vals.length });
    }
  }

  avg.sort((a, b) => a.year - b.year);
  scatter.sort((a, b) => a.year - b.year);

  // Deaths: interpolated 3% CDR up to 1949, then OWID actuals 1950–2023
  const cdrDeaths = interpolateDeaths(avg);
  const deaths: DeathsPt[] = cdrDeaths.filter(d => d.year < 1950);
  for (const [year, d] of deathsOwid) {
    deaths.push({ year, deaths: d });
  }

  return { scatter, avg, deaths };
}

const fmtPop = (v: number) =>
  v >= 1e9 ? `${(v / 1e9).toFixed(1)}B` : `${(v / 1e6).toFixed(0)}M`;

const fmtYear = (y: number) => (y <= 0 ? `${Math.abs(y)} BC` : `${y}`);

const MIN_YEAR = -10000;
const MAX_YEAR = 2023;

export default function PopulationChart() {
  const { scatter, avg, deaths } = useMemo(buildData, []);
  const [startYear, setStartYear] = useState(MIN_YEAR);

  const filteredAvg = useMemo(() => avg.filter(d => d.year >= startYear), [avg, startYear]);
  const filteredScatter = useMemo(() => scatter.filter(d => d.year >= startYear), [scatter, startYear]);
  const filteredDeaths = useMemo(() => deaths.filter(d => d.year >= startYear), [deaths, startYear]);

  return (
    <>
      <label className="ctrl" style={{ marginBottom: 8 }}>
        <span>Start year: <b>{fmtYear(startYear)}</b></span>
        <input type="range" min={MIN_YEAR} max={MAX_YEAR - 100} step={100}
          value={startYear} onChange={e => setStartYear(+e.target.value)}
          style={{ width: '100%' }} />
      </label>
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={filteredAvg} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
          <XAxis
            dataKey="year" type="number" domain={['dataMin', 'dataMax']}
            tick={{ fill: '#666', fontSize: 11 }} tickFormatter={fmtYear}
          />
          <YAxis
            scale="log" domain={[5e4, 3e9]}
            tick={{ fill: '#666', fontSize: 11 }} tickFormatter={fmtPop}
            allowDataOverflow
          />
          <Tooltip
            formatter={(v: number, name: string) => [fmtPop(v), name]}
            labelFormatter={fmtYear}
            contentStyle={{ background: '#1a1a1a', border: '1px solid #333', borderRadius: 4 }}
            itemStyle={{ color: '#ccc' }} labelStyle={{ color: '#999' }}
          />
          <Scatter name="Estimates" data={filteredScatter} dataKey="pop" fill="#666" fillOpacity={0.5} />
          <Line
            name="Avg population" dataKey="avg" stroke="#4ecdc4" strokeWidth={3}
            dot={{ r: 4, fill: '#4ecdc4', stroke: 'none' }}
          />
          <Line
            name="Deaths/yr" data={filteredDeaths} dataKey="deaths"
            stroke="#e06c75" strokeWidth={2} dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </>
  );
}
