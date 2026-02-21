import { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import deathsOwid from './deathsOwid';
import {
  ANCIENT_GRAVES, DATA_START, PRESENT_YEAR,
  PRB_TABLE, prbCBR, cbrDeathsPerYear, prbIntervals,
} from './fitModel';

// ----------------------------------------------------------------
// PRB era summary table — for each benchmark interval shows:
//   • CBR used by PRB for that era
//   • OWID population at start/end
//   • Model deaths/year at start/end (= CBR × OWID pop)
//   • PRB's published births-between-benchmarks (≈ deaths for that period)
// ----------------------------------------------------------------

function PrbEraTable() {
  const th: React.CSSProperties = { textAlign: 'right', padding: '4px 6px', fontWeight: 600 };
  const thL: React.CSSProperties = { ...th, textAlign: 'left' };
  const td: React.CSSProperties = { textAlign: 'right', padding: '3px 6px' };
  const tdL: React.CSSProperties = { ...td, textAlign: 'left' };

  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ fontSize: 11, color: '#666', margin: '6px 0 4px' }}>
        Keyfitz (1966) / Haub formula: deaths/yr(t) = births · g · P₀ · e^(g·t) / (P₁−P₀),
        &nbsp; g = ln(P₁/P₀) / T. &nbsp;
        Period totals match PRB exactly (analytically); &lt;0.5% discretisation error when summed year-by-year.
        <br />
        Sources: Kaneda, Greenbaum &amp; Haub, PRB 2022 · UN WPP 2022 · Poston (Texas A&amp;M).
      </div>
      <table style={{ fontSize: 11, color: '#ccc', borderCollapse: 'collapse', width: '100%' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #444' }}>
            <th style={thL}>Era</th>
            <th style={th}>PRB pop (start)</th>
            <th style={th}>PRB pop (end)</th>
            <th style={th}>PRB births</th>
            <th style={th}>Deaths (births−ΔPop)</th>
            <th style={th}>Deaths/yr (start)</th>
            <th style={th}>Deaths/yr (end)</th>
          </tr>
        </thead>
        <tbody>
          {prbIntervals.map((iv, i) => {
            const prbBirths = PRB_TABLE.find(e => e.year === iv.endYear)?.births ?? 0;
            const dStart = cbrDeathsPerYear(iv.startYear);
            const dEnd   = cbrDeathsPerYear(iv.endYear - 1);
            return (
              <tr key={i} style={{ borderBottom: '1px solid #2a2a2a' }}>
                <td style={tdL}>{fmtYear(iv.startYear)} → {fmtYear(iv.endYear)}</td>
                <td style={td}>{fmtPop(iv.startPop)}</td>
                <td style={td}>{fmtPop(iv.endPop)}</td>
                <td style={{ ...td, color: '#888' }}>{fmtPop(prbBirths)}</td>
                <td style={{ ...td, color: '#98c379' }}>{fmtPop(iv.births)}</td>
                <td style={td}>{fmtPop(dStart)}</td>
                <td style={td}>{fmtPop(dEnd)}</td>
              </tr>
            );
          })}
          <tr style={{ borderBottom: '1px solid #2a2a2a', color: '#888' }}>
            <td style={tdL}>1950 → present</td>
            <td colSpan={5} style={td}>actual UN WPP annual deaths (deathsOwid.ts)</td>
            <td style={td}>—</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

// ----------------------------------------------------------------
// Deaths-per-year series: CBR × OWID population + OWID actuals post-1950
// ----------------------------------------------------------------

const START_YEAR = DATA_START;
const END_YEAR = PRESENT_YEAR;

interface DeathsPt { year: number; deathsPerYear: number }

function buildDeathsPerYear(): DeathsPt[] {
  const pts: DeathsPt[] = [];

  // Pre-1950: sample CBR × OWID population at regular intervals
  // Use finer steps near the transition to 1950 for a smooth join
  for (let y = START_YEAR; y < 1950; y++) {
    pts.push({ year: y, deathsPerYear: cbrDeathsPerYear(y) });
  }

  // 1950+: actual OWID annual deaths (UN WPP 2024)
  for (const [year, d] of deathsOwid) {
    pts.push({ year, deathsPerYear: d });
  }
  return pts;
}

// ----------------------------------------------------------------
// Hyperbolic geometry helpers
// k = √|K|, K < 0 is Gaussian curvature of the ancient circle.
// All functions degrade smoothly to flat formulas as k → 0.
// ----------------------------------------------------------------

function hypCircumference(r: number, k: number): number {
  if (k < 1e-10) return 2 * Math.PI * r;
  return 2 * Math.PI * Math.sinh(k * r) / k;
}

function hypDiskArea(r: number, k: number): number {
  if (k < 1e-10) return Math.PI * r * r;
  return 2 * Math.PI * (Math.cosh(k * r) - 1) / (k * k);
}

// ----------------------------------------------------------------
// Available graves from ancient circle density
// ----------------------------------------------------------------

function buildAncientDensitySeries(yc: number, k: number): { years: number[]; vals: number[] } {
  const A = ANCIENT_GRAVES;
  const density = A / hypDiskArea(yc, k);
  const years: number[] = [];
  const vals: number[] = [];
  for (let calYear = START_YEAR; calYear <= END_YEAR; calYear += 10) {
    const r = (calYear - START_YEAR) + yc;
    years.push(calYear);
    // Available graves/yr at this radius = density × circumference × 1 yr radial width
    vals.push(density * 2 * Math.PI * r);
  }
  const rLast = (END_YEAR - START_YEAR) + yc;
  years.push(END_YEAR);
  vals.push(density * 2 * Math.PI * rLast);
  return { years, vals };
}

// ----------------------------------------------------------------
// Circumference series: required C(r) to maintain ancient density
//
// Ancient circle (r ≤ yc):  C(r) = hypCircumference(r, k)
//   (= 2π·r when k = 0, larger for k > 0)
// Post-ancient  (r > yc):   C(r) = deaths_per_year / ρ
//   where ρ = A / hypDiskArea(yc, k)
// X axis: r = (calYear − DATA_START) + yc  (years from centre)
// ----------------------------------------------------------------

interface CircumferencePt { yearsAgo: number; C: number }

function buildCircumferenceSeries(yc: number, k: number): CircumferencePt[] {
  const A = ANCIENT_GRAVES;
  const density = A / hypDiskArea(yc, k);
  const pts: CircumferencePt[] = [];

  const anchorSteps = 200;
  const postSpan = PRESENT_YEAR - DATA_START;
  for (let i = 0; i <= anchorSteps; i++) {
    const r = (i / anchorSteps) * yc;
    const yearsAgo = (yc - r) + postSpan;
    pts.push({ yearsAgo, C: hypCircumference(r, k) });
  }

  for (const { year, deathsPerYear } of buildDeathsPerYear()) {
    if (year < DATA_START) continue;
    const yearsAgo = PRESENT_YEAR - year;
    pts.push({ yearsAgo, C: deathsPerYear / density });
  }

  return pts.sort((a, b) => b.yearsAgo - a.yearsAgo);
}

// ----------------------------------------------------------------
// Uniform-grid C(r): dr = 1 yr, r from 0 to yc + postSpan
// Ancient (r ≤ yc): C = hypCircumference(r, k)
// Post-ancient (r > yc): C = deaths[calYear] / density
// ----------------------------------------------------------------

function buildUniformCircumference(yc: number, k: number): { r: Float64Array; C: Float64Array } {
  const A = ANCIENT_GRAVES;
  const density = A / hypDiskArea(yc, k);
  const postSpan = PRESENT_YEAR - DATA_START;
  const annualDeaths = (() => {
    const map = new Map<number, number>(deathsOwid as [number, number][]);
    const last = (deathsOwid as [number, number][])[deathsOwid.length - 1];
    const arr = new Float64Array(postSpan);
    for (let i = 0; i < postSpan; i++) {
      const calYear = DATA_START + i;
      const owid = map.get(calYear);
      if (owid !== undefined) {
        arr[i] = owid;
      } else if (calYear > last[0]) {
        arr[i] = last[1];
      } else {
        arr[i] = cbrDeathsPerYear(calYear);
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
      C[i] = hypCircumference(i, k);
    } else {
      const idx = Math.min(i - Math.round(yc), postSpan - 1);
      C[i] = annualDeaths[idx] / density;
    }
  }
  return { r, C };
}

// ----------------------------------------------------------------
// metric(ρ): C(r) / (2π·r)
// = 1 inside the ancient circle (by construction)
// = deaths_per_year / (density · 2π · r) in the post-ancient rings
// Encodes how much the required circumference deviates from flat.
// ----------------------------------------------------------------

function buildMetricRho(yc: number, k: number): { yearsAgo: number[]; rho: number[]; labels: string[] } {
  const { r, C } = buildUniformCircumference(yc, k);
  const postSpan = PRESENT_YEAR - DATA_START;
  const yearsAgo: number[] = [];
  const rho: number[] = [];
  const labels: string[] = [];

  for (let i = 1; i < r.length; i++) {
    const ri = r[i];
    const Ci = C[i];
    const metric = Ci / (2 * Math.PI * ri);
    const ya = postSpan - (ri - yc);
    const calYear = Math.round(DATA_START + (ri - yc));
    const label = calYear <= 0 ? `${Math.abs(calYear)} BC` : `${calYear} AD`;
    yearsAgo.push(ya);
    rho.push(metric);
    labels.push(label);
  }

  return { yearsAgo, rho, labels };
}

function MetricRhoChart({ yc, k }: { yc: number; k: number }) {
  const { yearsAgo, rho, labels } = useMemo(() => buildMetricRho(yc, k), [yc, k]);
  const postSpan = PRESENT_YEAR - DATA_START;

  return (
    <div style={{ marginTop: 32 }}>
      <div style={{ fontSize: 13, color: '#999', marginBottom: 6 }}>
        &rho;(r) = C(r) / (2&pi;r) &mdash; ratio of actual to flat circumference &nbsp;&middot;&nbsp;
        &rho; = 1 inside the ancient circle, &rho; &gt; 1 means &ldquo;more space&rdquo; than flat
      </div>
      <Plot
        data={[
          {
            x: yearsAgo,
            y: Array(yearsAgo.length).fill(1),
            type: 'scatter',
            mode: 'lines',
            name: 'flat (ρ = 1)',
            line: { color: '#61afef', width: 1.5, dash: 'dash' },
            hoverinfo: 'skip',
          },
          {
            x: yearsAgo,
            y: rho,
            type: 'scatter',
            mode: 'lines',
            name: 'ρ(r) = C / 2πr',
            line: { color: '#ff8c42', width: 2 },
            customdata: labels,
            hovertemplate: '%{customdata}<br>ρ = %{y:.4f}<extra></extra>',
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
            title: { text: 'ρ = C / 2πr', font: { color: '#555', size: 11 }, standoff: 8 },
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
  );
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

function computeCurvature(yc: number, k: number): CurvatureData {
  const { r, C } = buildUniformCircumference(yc, k);
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

function CurvatureChart({ yc, k }: { yc: number; k: number }) {
  const data = useMemo(() => computeCurvature(yc, k), [yc, k]);
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

function fmtTime(seconds: number): string {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return `${m}m ${s.toString().padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `${h}h ${rm.toString().padStart(2, '0')}m`;
}

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
  const [showCurvature, setShowCurvature] = useState(true);
  // kappaYc = k·yc (dimensionless curvature scale), k = √|K| where K < 0
  const [kappaYc, setKappaYc] = useState(3.95);
  const clampedKappaYc = Math.min(kappaYc, yc / 500);
  const k = clampedKappaYc / Math.max(1, yc);
  const [walkMinutes, setWalkMinutes] = useState(30);

  const deathsData = useMemo(buildDeathsPerYear, []);
  const ancientSeries = useMemo(() => buildAncientDensitySeries(yc, k), [yc, k]);
  const circData = useMemo(() => buildCircumferenceSeries(yc, k), [yc, k]);

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
      {/* Ancient circle sliders */}
      <label className="ctrl" style={{ marginBottom: 8 }}>
        <span>Ancient circle radius: <b>{yc} yr</b></span>
        <input type="range" min={100} max={50000} step={100}
          value={yc} onChange={e => onYcChange(+e.target.value)}
          style={{ width: '100%' }} />
      </label>
      <label className="ctrl" style={{ marginBottom: 8 }}>
        <span>
          Ancient circle curvature: <b>κ̃ = k·yc = {clampedKappaYc.toFixed(2)}</b>
          {clampedKappaYc > 0
            ? ` (K = −${(k * k).toExponential(2)} yr⁻², R = ${Math.round(1 / k).toLocaleString()} yr)`
            : ' (flat, R = ∞)'}
        </span>
        <input type="range" min={0} max={yc / 500} step={Math.max(0.001, yc / 500 / 400)}
          value={Math.min(kappaYc, yc / 500)}
          onChange={e => setKappaYc(+e.target.value)}
          style={{ width: '100%' }} />
      </label>
      <div style={{ fontSize: 13, color: '#999', marginBottom: 8 }}>
        A = {ANCIENT_GRAVES.toLocaleString()} ancient graves
        &nbsp;&middot;&nbsp;
        Area(yc, κ) = {hypDiskArea(yc, k).toExponential(3)} yr²
        &nbsp;&middot;&nbsp;
        density = {(ANCIENT_GRAVES / hypDiskArea(yc, k)).toFixed(4)}/yr²
        &nbsp;&middot;&nbsp;
        C(yc) = {Math.round(hypCircumference(yc, k)).toLocaleString()} yr
        {clampedKappaYc > 0 && (
          <span style={{ color: '#ff8c42' }}>
            &nbsp;&middot;&nbsp;
            C(yc)/2πyc = {(hypCircumference(yc, k) / (2 * Math.PI * yc)).toFixed(3)}×
          </span>
        )}
      </div>

      {/* Walk calculator */}
      {(() => {
        const walkSec = walkMinutes * 60;
        const rTotal = yc + postSpan;
        const speedYrSec = rTotal / walkSec;
        const density0 = ANCIENT_GRAVES / hypDiskArea(yc, k);
        const gravesPerSec = Math.sqrt(density0) * speedYrSec;

        const rAncient = yc;
        const rOldHist = 1900 - DATA_START;   // DATA_START → 1900 AD
        const rModern  = PRESENT_YEAR - 1900; // 1900 → present

        const tAncient = (rAncient / rTotal) * walkSec;
        const tOldHist = (rOldHist / rTotal) * walkSec;
        const tModern  = (rModern  / rTotal) * walkSec;

        const cell: React.CSSProperties = {
          padding: '6px 10px', borderRadius: 6,
          background: '#1a1a1a', textAlign: 'center' as const,
        };
        const label: React.CSSProperties = { fontSize: 11, color: '#666', marginBottom: 2 };
        const val: React.CSSProperties = { fontSize: 15, color: '#ccc', fontWeight: 600 };

        return (
          <div style={{ marginTop: 20, marginBottom: 8 }}>
            <label className="ctrl" style={{ marginBottom: 10 }}>
              <span>Walk from centre to edge: <b>{fmtTime(walkSec)}</b></span>
              <input type="range" min={10} max={120} step={1}
                value={walkMinutes} onChange={e => setWalkMinutes(+e.target.value)}
                style={{ width: '100%' }} />
            </label>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6, fontSize: 13 }}>
              <div style={cell}>
                <div style={label}>Speed</div>
                <div style={val}>{speedYrSec.toFixed(1)}</div>
                <div style={{ ...label, marginTop: 2 }}>yr / sec</div>
              </div>
              <div style={cell}>
                <div style={label}>Speed</div>
                <div style={val}>{gravesPerSec.toFixed(1)}</div>
                <div style={{ ...label, marginTop: 2 }}>graves / sec</div>
              </div>
              <div style={{ ...cell, borderLeft: '2px solid #333' }}>
                <div style={label}>Ancient circle</div>
                <div style={{ ...val, color: '#c678dd' }}>{fmtTime(tAncient)}</div>
                <div style={{ ...label, marginTop: 2 }}>&gt; {Math.round(PRESENT_YEAR - DATA_START)} yr ago</div>
              </div>
              <div style={cell}>
                <div style={label}>History</div>
                <div style={{ ...val, color: '#61afef' }}>{fmtTime(tOldHist)}</div>
                <div style={{ ...label, marginTop: 2 }}>{DATA_START < 0 ? `${Math.abs(DATA_START)} BC` : DATA_START} → 1900</div>
              </div>
              <div style={cell}>
                <div style={label}>20th & 21st c.</div>
                <div style={{ ...val, color: '#98c379' }}>{fmtTime(tModern)}</div>
                <div style={{ ...label, marginTop: 2 }}>1900 → present</div>
              </div>
            </div>
          </div>
        );
      })()}

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

      {/* Anchor values: ancient edge + OWID 1950 */}
      {(() => {
        const density    = ANCIENT_GRAVES / hypDiskArea(yc, k);
        const edgeDeaths = density * hypCircumference(yc, k);
        const owid1950   = deathsOwid[0][1];   // first OWID entry is 1950
        const stat: React.CSSProperties = {
          display: 'flex', flexDirection: 'column', gap: 2,
          padding: '8px 14px', borderRadius: 6, background: '#161616',
        };
        const label: React.CSSProperties = { fontSize: 10, color: '#555', textTransform: 'uppercase', letterSpacing: '0.05em' };
        const value: React.CSSProperties = { fontSize: 15, fontWeight: 700 };
        return (
          <div style={{ display: 'flex', gap: 10, marginTop: 8, marginBottom: 4 }}>
            <div style={stat}>
              <span style={label}>deaths / yr — ancient edge (ρ · C(y&#x2C)))</span>
              <span style={{ ...value, color: '#e06c75' }}>{fmtPop(edgeDeaths)}</span>
            </div>
            <div style={stat}>
              <span style={label}>deaths / yr — 1950 (OWID)</span>
              <span style={{ ...value, color: '#98c379' }}>{fmtPop(owid1950)}</span>
            </div>
          </div>
        );
      })()}

      {/* PRB 2022 era table */}
      <details style={{ marginTop: 12 }}>
        <summary style={{ color: '#999', fontSize: 13, cursor: 'pointer' }}>
          PRB 2022 Table 1 — "How Many People Have Ever Lived on Earth?"
        </summary>
        <PrbEraTable />
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

      <div style={{ marginTop: 32, display: 'flex', alignItems: 'center', gap: 10 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', userSelect: 'none' }}>
          <span style={{ position: 'relative', display: 'inline-block', width: 36, height: 20 }}>
            <input
              type="checkbox"
              checked={showCurvature}
              onChange={e => setShowCurvature(e.target.checked)}
              style={{ opacity: 0, width: 0, height: 0, position: 'absolute' }}
            />
            <span style={{
              position: 'absolute', inset: 0, borderRadius: 20,
              background: showCurvature ? '#61afef' : '#333',
              transition: 'background 0.2s',
            }} />
            <span style={{
              position: 'absolute', top: 3, left: showCurvature ? 19 : 3,
              width: 14, height: 14, borderRadius: '50%',
              background: '#fff', transition: 'left 0.2s',
            }} />
          </span>
          <span style={{ fontSize: 13, color: '#999' }}>Curvature reconstruction</span>
        </label>
      </div>

      {showCurvature && <CurvatureChart yc={yc} k={k} />}

      <MetricRhoChart yc={yc} k={k} />

  </>
);
}
