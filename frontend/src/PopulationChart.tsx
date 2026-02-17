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
// Circumference series: required C(r) to maintain ancient density
//
// Ancient circle (r ≤ yc):  C(r) = 2π·r  (flat, by definition)
// Post-ancient  (r > yc):   C(r) = deaths_per_year / ρ
//                                 = deaths_per_year · π·yc² / A
// where ρ = A / (π·yc²) is the ancient circle density.
// X axis: r = (calYear − DATA_START) + yc  (years from centre)
// ----------------------------------------------------------------

interface CircumferencePt { yearsAgo: number; C: number }

function buildCircumferenceSeries(yc: number): CircumferencePt[] {
  const A = ANCIENT_GRAVES;
  const density = A / (Math.PI * yc * yc); // graves / yr²
  const pts: CircumferencePt[] = [];

  // Ancient circle: flat C = 2π·r, parameterised as yearsAgo = (yc−r) + (PRESENT−DATA_START)
  const anchorSteps = 200;
  const postSpan = PRESENT_YEAR - DATA_START; // 10026
  for (let i = 0; i <= anchorSteps; i++) {
    const r = (i / anchorSteps) * yc;
    const yearsAgo = (yc - r) + postSpan;
    pts.push({ yearsAgo, C: 2 * Math.PI * r });
  }

  // Post-ancient: one point per deaths-series sample
  for (const { year, deathsPerYear } of buildDeathsPerYear()) {
    if (year < DATA_START) continue;
    const yearsAgo = PRESENT_YEAR - year;
    const C = deathsPerYear / density;
    pts.push({ yearsAgo, C });
  }

  return pts.sort((a, b) => b.yearsAgo - a.yearsAgo);
}

// ----------------------------------------------------------------
// Uniform-grid C(r): dr = 1 yr, r from 0 to yc + postSpan
// Ancient (r ≤ yc): C = 2πr
// Post-ancient (r > yc): C = deaths[calYear] / density
// ----------------------------------------------------------------

function buildUniformCircumference(yc: number): { r: Float64Array; C: Float64Array } {
  const A = ANCIENT_GRAVES;
  const density = A / (Math.PI * yc * yc);
  const postSpan = PRESENT_YEAR - DATA_START; // 10026
  const annualDeaths = (() => {
    const map = new Map<number, number>(deathsOwid as [number, number][]);
    const last = (deathsOwid as [number, number][])[deathsOwid.length - 1];
    const arr = new Float64Array(postSpan);
    for (let i = 0; i < postSpan; i++) {
      const calYear = DATA_START + i;
      const owid = map.get(calYear);
      arr[i] = owid !== undefined ? owid : last[1];
    }
    // fill pre-OWID from fit intervals
    for (const iv of fitIntervals) {
      const T = iv.endYear - iv.startYear;
      for (let y = iv.startYear; y < iv.endYear && y < PRESENT_YEAR; y++) {
        if (y < DATA_START) continue;
        if (map.has(y)) continue;
        const t = y - iv.startYear;
        const cdr = iv.cdrLeft + (iv.cdrRight - iv.cdrLeft) * t / T;
        const pop = iv.startPop * Math.exp(iv.g * t);
        arr[y - DATA_START] = cdr * pop;
      }
    }
    return arr;
  })();

  const len = Math.round(yc) + postSpan + 1;
  const r = new Float64Array(len);
  const C = new Float64Array(len);
  for (let i = 0; i < len; i++) {
    r[i] = i;
    if (i <= yc) {
      C[i] = 2 * Math.PI * i;
    } else {
      const idx = Math.min(i - Math.round(yc), postSpan - 1);
      C[i] = annualDeaths[idx] / density;
    }
  }
  return { r, C };
}

// ----------------------------------------------------------------
// Curvature reconstruction: K(r) = -f''(r)/f(r), f = C/2π
// ----------------------------------------------------------------

function gaussianSmooth(arr: Float64Array, sigma: number): Float64Array {
  const out = new Float64Array(arr.length);
  const radius = Math.ceil(3 * sigma);
  const kernel: number[] = [];
  let ksum = 0;
  for (let k = -radius; k <= radius; k++) {
    const v = Math.exp(-(k * k) / (2 * sigma * sigma));
    kernel.push(v);
    ksum += v;
  }
  for (let i = 0; i < arr.length; i++) {
    let s = 0;
    for (let k = -radius; k <= radius; k++) {
      const j = Math.max(0, Math.min(arr.length - 1, i + k));
      s += kernel[k + radius] * arr[j];
    }
    out[i] = s / ksum;
  }
  return out;
}

interface CurvatureData {
  r: Float64Array;
  C: Float64Array;
  flatC: Float64Array;
  K: Float64Array;
  R_kappa: Float64Array;   // 1/κ = 1/√|K|·sign(K)
  calYearLabel: string[];  // hover label per point
}

function computeCurvature(yc: number): CurvatureData {
  const { r, C } = buildUniformCircumference(yc);
  const dr = 1;
  const n = r.length;

  const f = new Float64Array(n);
  for (let i = 0; i < n; i++) f[i] = C[i] / (2 * Math.PI);

  const fSmooth = gaussianSmooth(f, 20);

  // second derivative via central differences
  const f2 = new Float64Array(n);
  for (let i = 1; i < n - 1; i++) {
    f2[i] = (fSmooth[i + 1] - 2 * fSmooth[i] + fSmooth[i - 1]) / (dr * dr);
  }
  f2[0] = f2[1];
  f2[n - 1] = f2[n - 2];

  const R_MAX = 5 * r[n - 1];
  const K = new Float64Array(n);
  const R_kappa = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const fi = fSmooth[i];
    if (Math.abs(fi) < 1e-6) { K[i] = NaN; R_kappa[i] = NaN; continue; }
    const k = -f2[i] / fi;
    K[i] = k;
    const sqrtAbsK = Math.sqrt(Math.abs(k));
    if (sqrtAbsK < 1 / R_MAX) { R_kappa[i] = NaN; continue; }
    R_kappa[i] = Math.sign(k) / sqrtAbsK;
  }

  const flatC = new Float64Array(n);
  for (let i = 0; i < n; i++) flatC[i] = 2 * Math.PI * r[i];

  // calendar year labels: calYear = DATA_START + (r - yc)
  const calYearLabel: string[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const cy = Math.round(DATA_START + (r[i] - yc));
    calYearLabel[i] = cy <= 0 ? `${Math.abs(cy)} BC` : `${cy} AD`;
  }

  // trim boundary artefacts (1%)
  const trim = Math.max(1, Math.floor(n / 100));
  return {
    r: r.slice(trim, n - trim),
    C: C.slice(trim, n - trim),
    flatC: flatC.slice(trim, n - trim),
    K: K.slice(trim, n - trim),
    R_kappa: R_kappa.slice(trim, n - trim),
    calYearLabel: calYearLabel.slice(trim, n - trim),
  };
}

// ----------------------------------------------------------------
// Curvature chart component
// ----------------------------------------------------------------

function CurvatureChart({ yc }: { yc: number }) {
  const data = useMemo(() => computeCurvature(yc), [yc]);
  const rArr = Array.from(data.r);
  const labels = data.calYearLabel;
  const hoverX = '%{customdata}<br>r = %{x:.0f} yr<extra></extra>';

  return (
    <div style={{ marginTop: 32 }}>
      <div style={{ fontSize: 13, color: '#999', marginBottom: 6 }}>
        Curvature reconstruction &mdash; K(r) = &minus;f&Prime;(r)/f(r), &nbsp;
        f = C/2&pi;, &nbsp; radius = 1/&radic;|K|
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {/* K(r) */}
        <Plot
          data={[
            { x: rArr, y: Array.from(data.K), type: 'scatter', mode: 'lines',
              name: 'K(r)', line: { color: '#98c379', width: 2 },
              customdata: labels,
              hovertemplate: hoverX + '<br>K = %{y:.3e} yr⁻²',
            },
          ]}
          layout={{ ...DARK_LAYOUT, height: 280,
            xaxis: { ...DARK_LAYOUT.xaxis, title: { text: 'r (yrs)', font: { color: '#555', size: 11 }, standoff: 6 } },
            yaxis: { ...DARK_LAYOUT.yaxis, title: { text: 'K (1/yr²)', font: { color: '#555', size: 11 }, standoff: 6 } },
          }}
          config={PLOTLY_CONFIG} useResizeHandler style={{ width: '100%' }}
        />

        {/* 1/√|K| radius */}
        <Plot
          data={[
            { x: rArr, y: Array.from(data.R_kappa), type: 'scatter', mode: 'lines',
              name: '1/√|K| (signed)', line: { color: '#c678dd', width: 2 },
              customdata: labels,
              hovertemplate: hoverX + '<br>radius = %{y:.0f} yr',
            },
          ]}
          layout={{ ...DARK_LAYOUT, height: 280,
            xaxis: { ...DARK_LAYOUT.xaxis, title: { text: 'r (yrs)', font: { color: '#555', size: 11 }, standoff: 6 } },
            yaxis: { ...DARK_LAYOUT.yaxis, title: { text: 'radius (yrs)', font: { color: '#555', size: 11 }, standoff: 6 } },
          }}
          config={PLOTLY_CONFIG} useResizeHandler style={{ width: '100%' }}
        />
      </div>
    </div>
  );
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
  const circData = useMemo(() => buildCircumferenceSeries(yc), [yc]);

  // ---- Chart 1: Deaths per year ----
  const deathsYears = useMemo(() => deathsData.map(d => d.year), [deathsData]);
  const deathsVals = useMemo(() => deathsData.map(d => d.deathsPerYear), [deathsData]);

  // ---- Chart 2: Circumference ----
  const circX = useMemo(() => circData.map(d => d.yearsAgo), [circData]);
  const circC = useMemo(() => circData.map(d => d.C), [circData]);
  // flat reference: r = yc + postSpan − yearsAgo, so 2πr = 2π(yc + postSpan − yearsAgo)
  const postSpan = PRESENT_YEAR - DATA_START;
  const circFlat = useMemo(() => circX.map(ya => 2 * Math.PI * Math.max(0, yc + postSpan - ya)), [circX, yc]);
  const circLabels = useMemo(() => circX.map(ya => {
    const cy = Math.round(PRESENT_YEAR - ya);
    return cy <= 0 ? `${Math.abs(cy)} BC` : `${cy} AD`;
  }), [circX]);

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

      {/* Chart 2: Required circumference C(r) */}
      <div style={{ marginTop: 24 }}>
        <div style={{ fontSize: 13, color: '#999', marginBottom: 6 }}>
          Required circumference C(r) to maintain ancient density &mdash; both axes in years
          &nbsp;&middot;&nbsp; r = (year &minus; {DATA_START}) + yc
        </div>
        <Plot
          data={[
            {
              x: circX,
              y: circFlat,
              type: 'scatter',
              mode: 'lines',
              name: '2πr (flat)',
              line: { color: '#61afef', width: 1.5, dash: 'dash' },
              customdata: circLabels,
              hovertemplate: '%{customdata}<br>C = %{y:.0f} yr<extra></extra>',
            },
            {
              x: circX,
              y: circC,
              type: 'scatter',
              mode: 'lines',
              name: 'C(r) required',
              line: { color: '#e06c75', width: 2 },
              customdata: circLabels,
              hovertemplate: '%{customdata}<br>C = %{y:.0f} yr<extra></extra>',
            },
          ]}
          layout={{
            ...DARK_LAYOUT,
            height: 360,
            xaxis: {
              ...DARK_LAYOUT.xaxis,
              type: 'log',
              autorange: 'reversed',
              title: { text: 'Years ago', font: { color: '#555', size: 11 }, standoff: 8 },
              tickformat: ',d',
            },
            yaxis: {
              ...DARK_LAYOUT.yaxis,
              title: { text: 'C (years)', font: { color: '#555', size: 11 }, standoff: 8 },
              rangemode: 'tozero',
            },
            shapes: [
              {
                type: 'line',
                x0: postSpan, x1: postSpan,
                y0: 0, y1: 1,
                xref: 'x', yref: 'paper',
                line: { color: '#98c379', width: 1, dash: 'dot' },
              },
            ],
            annotations: [
              {
                x: Math.log10(postSpan), y: 1, xref: 'x', yref: 'paper',
                text: `ancient edge (${postSpan} yr ago)`, showarrow: false,
                font: { color: '#98c379', size: 10 }, yanchor: 'bottom', xanchor: 'left',
              },
            ],
          }}
          config={PLOTLY_CONFIG}
          useResizeHandler
          style={{ width: '100%' }}
        />
      </div>

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

      <CurvatureChart yc={yc} />

    </>
  );
}
