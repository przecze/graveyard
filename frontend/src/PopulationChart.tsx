import { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import deathsOwid from './deathsOwid';
import {
  ANCIENT_GRAVES, DATA_START, PRESENT_YEAR,
  fitIntervals, computeIntegrals, prbEntries,
} from './fitModel';

// ----------------------------------------------------------------
// Deaths-per-year series from fitted intervals + OWID
// ----------------------------------------------------------------

const START_YEAR = DATA_START;
const END_YEAR = PRESENT_YEAR;

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
  const last = fitIntervals[fitIntervals.length - 1];
  pts.push({ year: last.endYear, deathsPerYear: last.cdrRight * last.endPop });

  for (const [year, d] of deathsOwid) {
    pts.push({ year, deathsPerYear: d });
  }
  return pts;
}

// ----------------------------------------------------------------
// Available graves from ancient circle density
// ----------------------------------------------------------------

function buildAncientDensitySeries(yc: number): { years: number[]; vals: number[] } {
  const A = ANCIENT_GRAVES;
  const yc2 = yc * yc;
  const years: number[] = [];
  const vals: number[] = [];
  for (let calYear = START_YEAR; calYear <= END_YEAR; calYear += 10) {
    const y = calYear - START_YEAR;
    years.push(calYear);
    vals.push(A * (2 * (yc + y) + 1) / yc2);
  }
  const yLast = END_YEAR - START_YEAR;
  years.push(END_YEAR);
  vals.push(A * (2 * (yc + yLast) + 1) / yc2);
  return { years, vals };
}

// ----------------------------------------------------------------
// Density series: graves/yr²
// X axis = radius coordinate = calYear - DATA_START + yc (true log scale)
// ----------------------------------------------------------------

interface DensityPt { yearsAgo: number; density: number }

function buildDensityData(yc: number): DensityPt[] {
  const ancientDensity = ANCIENT_GRAVES / (Math.PI * yc * yc);

  const pts: DensityPt[] = [];

  // Ancient circle: yearsAgo spans from (PRESENT_YEAR - DATA_START) down to
  // (PRESENT_YEAR - DATA_START) - yc, sampled on a log grid
  const maxYearsAgo = PRESENT_YEAR - DATA_START;
  const ancientEdgeYearsAgo = maxYearsAgo; // yearsAgo at r=yc boundary = yc mapped back
  const steps = 80;
  for (let i = 0; i <= steps; i++) {
    // log-spaced from ancientEdgeYearsAgo down toward large yearsAgo
    const yearsAgo = ancientEdgeYearsAgo * Math.exp(Math.log(3) * i / steps);
    pts.push({ yearsAgo, density: ancientDensity });
  }

  // Ring model: yearsAgo = PRESENT_YEAR - calYear
  for (const pt of buildDeathsPerYear()) {
    const yearsAgo = PRESENT_YEAR - pt.year;
    if (yearsAgo <= 0) continue;
    const r = pt.year - DATA_START + yc;
    if (r > 0) {
      pts.push({ yearsAgo, density: pt.deathsPerYear / (2 * Math.PI * r) });
    }
  }

  return pts.sort((a, b) => a.yearsAgo - b.yearsAgo);
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
// Shared Plotly dark layout
// ----------------------------------------------------------------

const DARK_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { color: '#999', size: 11 },
  margin: { t: 10, r: 30, b: 50, l: 70 },
  xaxis: {
    gridcolor: '#222',
    zerolinecolor: '#333',
    tickfont: { color: '#666', size: 11 },
  },
  yaxis: {
    gridcolor: '#222',
    zerolinecolor: '#333',
    tickfont: { color: '#666', size: 11 },
  },
  legend: {
    font: { color: '#999', size: 12 },
    bgcolor: 'transparent',
  },
};

const PLOTLY_CONFIG: Partial<Plotly.Config> = {
  displayModeBar: false,
  responsive: true,
};

// ----------------------------------------------------------------
// Component
// ----------------------------------------------------------------

interface Props {
  yc: number;
  onYcChange: (yc: number) => void;
}

export default function PopulationChart({ yc, onYcChange }: Props) {
  const deathsData = useMemo(buildDeathsPerYear, []);
  const ancientSeries = useMemo(() => buildAncientDensitySeries(yc), [yc]);
  const [logY, setLogY] = useState(false);
  const densityData = useMemo(() => buildDensityData(yc), [yc]);
  const ancientCircleDensity = ANCIENT_GRAVES / (Math.PI * yc * yc);

  // ---- Chart 1: Deaths per year ----
  const deathsYears = useMemo(() => deathsData.map(d => d.year), [deathsData]);
  const deathsVals = useMemo(() => deathsData.map(d => d.deathsPerYear), [deathsData]);

  // ---- Chart 2: Density ----
  const densityX = useMemo(() => densityData.map(d => d.yearsAgo), [densityData]);
  const densityY = useMemo(() => densityData.map(d => d.density), [densityData]);

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

      {/* Chart 1: Deaths per year */}
      <Plot
        data={[
          {
            x: deathsYears,
            y: deathsVals,
            type: 'scatter',
            mode: 'lines',
            name: 'Graves/yr',
            line: { color: '#e06c75', width: 2 },
          },
          {
            x: ancientSeries.years,
            y: ancientSeries.vals,
            type: 'scatter',
            mode: 'lines',
            name: 'Available (ancient circle density)',
            line: { color: '#61afef', width: 2, dash: 'dash' },
          },
        ]}
        layout={{
          ...DARK_LAYOUT,
          height: 400,
          xaxis: {
            ...DARK_LAYOUT.xaxis,
            range: [START_YEAR, END_YEAR],
          },
          yaxis: {
            ...DARK_LAYOUT.yaxis,
            type: 'log',
            range: [5, 9],
          },
        }}
        config={PLOTLY_CONFIG}
        useResizeHandler
        style={{ width: '100%' }}
      />

      {/* Interval table */}
      <details style={{ marginTop: 12 }}>
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
              <td style={{ textAlign: 'right', padding: '3px 6px' }}>{fmtPop(prbEntries[prbEntries.length - 1].pop)}</td>
              <td style={{ textAlign: 'right', padding: '3px 6px' }}>&mdash;</td>
              <td style={{ textAlign: 'right', padding: '3px 6px' }}>
                {(deathsOwid[0][1] / prbEntries[prbEntries.length - 1].pop * 100).toFixed(2)}%
              </td>
              <td colSpan={6} style={{ textAlign: 'right', padding: '3px 6px' }}>annual OWID data</td>
            </tr>
          </tbody>
        </table>
      </details>

      {/* Chart 2: Density — true log X scale */}
      <div style={{ marginTop: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 6 }}>
          <div style={{ fontSize: 13, color: '#999' }}>
            Grave density &mdash; graves/yr&sup2; = deaths/yr &divide; (2&pi;&thinsp;&times;&thinsp;X)
            &nbsp;&middot;&nbsp;
            ancient circle: {ancientCircleDensity.toExponential(3)}/yr&sup2;
          </div>
          <button
            onClick={() => setLogY(v => !v)}
            style={{
              fontSize: 11, padding: '2px 10px', borderRadius: 4, cursor: 'pointer',
              background: logY ? '#4ecdc4' : '#222',
              color: logY ? '#111' : '#888',
              border: '1px solid ' + (logY ? '#4ecdc4' : '#333'),
              flexShrink: 0,
            }}
          >
            Y: {logY ? 'log' : 'linear'}
          </button>
        </div>
        <Plot
          data={[
            {
              x: densityX,
              y: densityY,
              type: 'scatter',
              mode: 'lines',
              name: 'Density (graves/yr²)',
              line: { color: '#e06c75', width: 2 },
            },
          ]}
          layout={{
            ...DARK_LAYOUT,
            height: 320,
            xaxis: {
              ...DARK_LAYOUT.xaxis,
              type: 'log',
              autorange: 'reversed',
              title: { text: 'Years ago', font: { color: '#555', size: 11 }, standoff: 8 },
              tickformat: ',d',
            },
            yaxis: {
              ...DARK_LAYOUT.yaxis,
              type: logY ? 'log' : 'linear',
              title: { text: 'graves / yr²', font: { color: '#555', size: 11 }, standoff: 8 },
              rangemode: logY ? undefined : 'tozero',
            },
            shapes: [
              {
                type: 'line',
                x0: PRESENT_YEAR - DATA_START, x1: PRESENT_YEAR - DATA_START,
                y0: 0, y1: 1,
                xref: 'x', yref: 'paper',
                line: { color: '#61afef', width: 1.5, dash: 'dash' },
              },
            ],
            annotations: [
              {
                x: Math.log10(PRESENT_YEAR - DATA_START),
                y: 1, xref: 'x', yref: 'paper',
                text: `ancient edge (${PRESENT_YEAR - DATA_START} yr ago)`,
                showarrow: false,
                font: { color: '#61afef', size: 10 },
                yanchor: 'bottom',
              },
            ],
          }}
          config={PLOTLY_CONFIG}
          useResizeHandler
          style={{ width: '100%' }}
        />
      </div>
    </>
  );
}
