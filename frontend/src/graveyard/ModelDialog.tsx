// "Model & charts" dialog: the death model against its sources, the surface
// profile along the walk, and an independent check of every period total.
// Lazy-loaded from Walker (plotly is large).
import { useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
import type { Data, Shape as PlotShape } from 'plotly.js';
import { band, baseLayout, C, vline, WalkCharts } from './charts';
import { ANCIENT_START, deathModel, priorPoints, T_END, type DeathModel } from './deathModel';
import { OWID_DEATHS, OWID_FIRST_YEAR, PRB } from './data';
import { GL8 } from './spline';
import { formatYear } from './lore';
import type { Surface } from './surface';

const PRB_URL = 'https://www.prb.org/articles/how-many-people-have-ever-lived-on-earth/';
const OWID_URL = 'https://ourworldindata.org/grapher/number-of-deaths-per-year';

const fmtB = (n: number) => `${(n / 1e9).toFixed(3)} B`;

/** year grid: coarse where nothing happens, yearly where data is dense */
function yearGrid(): number[] {
  const ts: number[] = [];
  for (let t = ANCIENT_START; t < -8000; t += 50) ts.push(t);
  for (let t = -8000; t < 1500; t += 5) ts.push(t);
  for (let t = 1500; t <= T_END; t += 0.5) ts.push(t);
  return ts;
}

/** ∫ D dt over [a, b] by 8-point Gauss–Legendre on every whole year: independent of the model's cumulative table */
function integrate(m: DeathModel, a: number, b: number): number {
  let acc = 0;
  for (let y = a; y < b; y++) for (const [x, w] of GL8) acc += w * m.D(y + x);
  return acc;
}

export default function ModelDialog({ S, onClose }: { S: Surface; onClose: () => void }) {
  const [yLog, setYLog] = useState(true);
  const m = S.model;
  const base = deathModel(undefined, m.params);
  const yr = (metres: number) => `${Math.round(S.yr(metres)).toLocaleString('en-US')} yr`;

  const time = useMemo(() => {
    const ts = yearGrid();
    // period means as steps: area under a step = the period total
    const sx: (number | null)[] = [], sy: (number | null)[] = [];
    for (const p of m.periods) { sx.push(p.a, p.b, null); const r = p.target / (p.b - p.a); sy.push(r, r, null); }
    const est = priorPoints();
    // cumulative benchmarks: running sum of period totals
    const cx = [ANCIENT_START], cy = [0];
    for (const p of m.periods) { cx.push(p.b); cy.push(cy[cy.length - 1] + p.target); }
    return { ts, D: ts.map(t => m.D(t)), Dbase: ts.map(t => base.D(t)), N: ts.map(t => m.cum(t)), sx, sy, est, cx, cy };
  }, [m, base]);

  const check = useMemo(() => m.periods.map(p => {
    const direct = integrate(m, p.a, p.b);
    return { ...p, direct, abs: direct - p.target, rel: (direct - p.target) / p.target };
  }), [m]);
  const worst = Math.max(...check.map(c => Math.abs(c.rel)));
  const sumData = check.reduce((a, c) => a + c.target, 0), sumModel = check.reduce((a, c) => a + c.direct, 0);

  const r = S.report;
  const fw = m.flattenWindow;
  const timeShapes: Partial<PlotShape>[] = [
    band(ANCIENT_START, S.t0, 'rgba(214,180,100,0.06)'),
    ...(fw ? [band(fw[0], fw[1], 'rgba(143,209,143,0.08)')] : []),
    vline(S.t0, C.model),
  ];
  const deathsData: Data[] = [
    { x: time.ts, y: time.Dbase, name: 'model without flatten', type: 'scatter', mode: 'lines', line: { color: C.base, width: 1, dash: 'dash' }, hoverinfo: 'skip', visible: fw ? true : 'legendonly' },
    { x: time.sx, y: time.sy, name: 'period mean (total ÷ length)', type: 'scatter', mode: 'lines', line: { color: C.steps, width: 1.5 }, connectgaps: false, hovertemplate: '%{y:.4s}/yr' },
    { x: time.ts, y: time.D, name: 'model D(t)', type: 'scatter', mode: 'lines', line: { color: C.model, width: 2 }, hovertemplate: '%{y:.4s}/yr' },
    {
      x: time.est.map(p => p[0]), y: time.est.map(p => p[1]), name: 'PRB-style estimate pop·(CBR − growth)', type: 'scatter', mode: 'markers',
      marker: { color: C.prb, size: 8, symbol: 'diamond' }, hovertemplate: 'estimate %{y:.4s}/yr',
    },
    {
      x: OWID_DEATHS.map((_, i) => OWID_FIRST_YEAR + i + 0.5), y: OWID_DEATHS, name: 'OWID deaths per year', type: 'scatter', mode: 'markers',
      marker: { color: C.owid, size: 4 }, hovertemplate: 'OWID %{y:.4s}',
    },
  ];

  const cumData: Data[] = [
    { x: time.ts, y: time.N.map(n => n / 1e9), name: 'model ∫D dt', type: 'scatter', mode: 'lines', line: { color: C.model, width: 2 }, hovertemplate: '%{y:.3f} B' },
    {
      x: time.cx, y: time.cy.map(n => n / 1e9), name: 'benchmarks: Σ period totals (PRB + OWID)', type: 'scatter', mode: 'markers',
      marker: { color: C.prb, size: 7, symbol: 'circle-open', line: { width: 2 } }, hovertemplate: 'data %{y:.3f} B',
    },
  ];

  return (
    <div className="gy-about-backdrop" onClick={onClose}>
      <div className="gy-about gy-model gy-panel" onClick={e => e.stopPropagation()}>
        <button className="gy-close" onClick={onClose}>×</button>

        <h2>Ancient vs history</h2>
        <table>
          <tbody>
            <tr><td>ancient era (centre → {formatYear(S.t0)})</td><td><b>{yr(r.ancientLen)}</b></td><td className="dim">{fmtB(S.ancientGraves)} graves</td></tr>
            <tr><td>history ({formatYear(S.t0)} → {T_END})</td><td><b>{yr(r.historyLen)}</b></td><td className="dim">{S.rowsPerYear.toFixed(2)} rows per year, {fmtB(S.rowStart[S.nRows] - S.ancientGraves)} graves</td></tr>
            <tr><td>ancient : history</td><td><b>{(r.ancientLen / r.historyLen).toFixed(3)}</b></td><td className="dim">ancient is {(100 * r.ancientShare).toFixed(1)}% of the walk (cylinder floor {(100 * r.floorShare).toFixed(1)}%)</td></tr>
            <tr><td>D at the switch</td><td><b>{(m.D(S.t0) / 1e6).toFixed(2)} M/yr</b></td><td className="dim">{fw ? `flattened ${formatYear(fw[0])} → ${formatYear(fw[1])}; unflattened ${(base.D(S.t0) / 1e6).toFixed(2)} M/yr` : 'no flattening'}</td></tr>
          </tbody>
        </table>
        <p className="dim">
          Only ≈ N_before · v / D(switch) is forced (rings never shrink), so a higher death rate right after the switch shrinks the ancient zone.
          PRB gives only one total for 8000 BCE → 1 CE, so how deaths are spread inside that period is a free choice. Flattening
          {fw ? ` ${formatYear(fw[0])} → ${formatYear(fw[1])}` : ''} starts the rate at the window mean and lets it rise slowly. The period total stays exact (table below).
        </p>

        <h2>Deaths per year D(t)</h2>
        <div className="gy-model-bar">
          <button onClick={() => setYLog(l => !l)}>{yLog ? 'linear y' : 'log y'}</button>
          <span className="dim">drag to zoom · double-click resets to the full range (50 000 BCE → {T_END})</span>
        </div>
        <Plot
          data={deathsData}
          layout={baseLayout({
            shapes: timeShapes,
            xaxis: { range: [-9000, T_END], title: { text: 'year' } },
            yaxis: { type: yLog ? 'log' : 'linear', title: { text: 'deaths / year' } },
          })}
          config={{ displaylogo: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
        <p className="dim small">
          Shaded: ancient era (undated){fw ? ', flatten window (green)' : ''}. Green steps are each period's total divided by its length.
          Diamonds are PRB's own assumption, population × (birth rate − growth), used only to shape the prior. The 50 000 BCE diamond
          is an assumed exponential ramp that carries PRB's pre-8000 BCE total.
        </p>

        <h2>Cumulative graves N(t)</h2>
        <Plot
          data={cumData}
          layout={baseLayout({
            shapes: timeShapes,
            xaxis: { range: [-9000, T_END], title: { text: 'year' } },
            yaxis: { title: { text: 'graves (billions)' }, rangemode: 'tozero' },
          })}
          config={{ displaylogo: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />

        <h2>Along the walk: ring length and curvature</h2>
        <WalkCharts S={S} />
        <p className="dim small">
          Ring length 2πf in yr (log). In history it is D(t) / (σv²): the death rate read on another scale, so it dips where deaths fell
          (e.g. 1350–1650 and after 1900). In the ancient zone it is one smooth fit from the centre to the rim that never shrinks (red shading: positive curvature).
          Curvature K = −f″/f, in history −D″/D per yr², set by the data alone. It is continuous everywhere (f is C⁴, K is C²).
          Positive is sphere-like, negative saddle-like. The tightest ancient curvature radius is {yr(r.minKRadius)}.
        </p>

        <h2>Proof: every period total is kept</h2>
        <p className="dim">
          "Model" here integrates D(t) directly (8-point Gauss–Legendre on every whole year), independent of the lookup table the walker uses.
          Largest relative error: <b>{worst.toExponential(1)}</b>. Sum of all periods: data {fmtB(sumData)}, model {fmtB(sumModel)}.
        </p>
        <table className="gy-num">
          <thead><tr><th>period</th><th>data (deaths)</th><th>model ∫D dt</th><th>model − data</th><th>rel. error</th></tr></thead>
          <tbody>
            {check.map(c => (
              <tr key={c.source}>
                <td>{c.source}</td>
                <td>{Math.round(c.target).toLocaleString('en-US')}</td>
                <td>{Math.round(c.direct).toLocaleString('en-US')}</td>
                <td>{c.abs.toExponential(1)}</td>
                <td>{c.rel.toExponential(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h2>Sources</h2>
        <p>
          <a href={PRB_URL} target="_blank" rel="noreferrer">PRB, "How Many People Have Ever Lived on Earth?"</a>. Benchmarks give
          births between benchmarks, population and birth rate. Deaths in a period = births − Δpopulation.
        </p>
        <table className="gy-num">
          <thead><tr><th>year</th><th>births since prev.</th><th>population</th><th>CBR ‰</th><th>deaths in period</th></tr></thead>
          <tbody>
            {PRB.map((row, i) => (
              <tr key={row.year}>
                <td>{formatYear(row.year)}</td>
                <td>{row.birthsSincePrev.toLocaleString('en-US')}</td>
                <td>{row.pop.toLocaleString('en-US')}</td>
                <td>{row.cbr}</td>
                <td>{i ? (row.birthsSincePrev - row.pop + PRB[i - 1].pop).toLocaleString('en-US') : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p>
          <a href={OWID_URL} target="_blank" rel="noreferrer">Our World in Data, "Number of deaths per year"</a> (UN WPP), {OWID_FIRST_YEAR}–{OWID_FIRST_YEAR + OWID_DEATHS.length - 1}.
          The model is fitted to decade totals. After {OWID_FIRST_YEAR + OWID_DEATHS.length - 1} the data is extrapolated at +0.8 %/yr to {T_END}.
        </p>
      </div>
    </div>
  );
}
