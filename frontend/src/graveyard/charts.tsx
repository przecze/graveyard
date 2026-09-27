// Shared plotly helpers and the "along the walk" charts (ring length and
// curvature against distance from the centre, in history-years).
import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import type { Data, Layout, Shape as PlotShape } from 'plotly.js';
import { formatYear } from './lore';
import type { Surface } from './surface';

export const C = {
  model: '#e0b860', base: '#8a8472', steps: '#8fd18f', prb: '#ff8a5c', owid: '#7ad9ff', curv: '#d68cff',
  grid: 'rgba(230,220,195,0.08)', line: 'rgba(230,220,195,0.2)', bg: 'rgba(0,0,0,0)',
};

export function baseLayout(extra: Partial<Layout>): Partial<Layout> {
  const axis = { gridcolor: C.grid, linecolor: C.line, zerolinecolor: C.line, tickcolor: C.line };
  return {
    paper_bgcolor: C.bg, plot_bgcolor: C.bg, height: 340,
    margin: { t: 10, r: 60, b: 44, l: 70 },
    font: { color: '#c9bfa6', family: 'ui-monospace, Menlo, monospace', size: 11 },
    legend: { orientation: 'h', y: -0.2, font: { size: 10 } },
    hovermode: 'x unified',
    ...extra,
    xaxis: { ...axis, ...extra.xaxis },
    yaxis: { ...axis, ...extra.yaxis },
  };
}

export const band = (x0: number, x1: number, color: string): Partial<PlotShape> =>
  ({ type: 'rect', xref: 'x', yref: 'paper', x0, x1, y0: 0, y1: 1, fillcolor: color, line: { width: 0 }, layer: 'below' });
export const vline = (x: number, color = C.line): Partial<PlotShape> =>
  ({ type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color, width: 1, dash: 'dot' } });

/** ring length (yr) and curvature (1/yr²) sampled along the walk */
function profile(S: Surface) {
  const rs: number[] = [];
  const a = S.grid.L.plazaR, end = S.rhoEndTable, n = 3000;
  for (let i = 0; i <= n; i++) rs.push(a + (end - a) * i / n);
  for (const t of [1900, 1950, 2000, 2020]) rs.push(S.rhoAtTime(t)); // keep the edge detail
  rs.sort((p, q) => p - q);
  return {
    x: rs.map(r => S.yr(r)),
    ring: rs.map(r => S.yr(2 * Math.PI * S.f(r))),
    K: rs.map(r => S.gaussK(r) * S.v * S.v),
    year: rs.map(r => (r >= S.rho0 ? formatYear(Math.round(S.time(r))) : 'ancient (undated)')),
  };
}

export function WalkCharts({ S, compare, height = 300 }: { S: Surface; compare?: Surface; height?: number }) {
  const p = useMemo(() => profile(S), [S]);
  const q = useMemo(() => (compare && compare !== S ? profile(compare) : null), [compare, S]);
  const zone = (x0: number, x1: number, c: string) => band(S.yr(x0), S.yr(x1), c);
  const r = S.report;
  const shapes: Partial<PlotShape>[] = [
    zone(0, S.rho0, 'rgba(214,180,100,0.08)'),
    ...(r.bulge ? [zone(r.bulge[0], r.bulge[1], 'rgba(255,120,80,0.10)')] : []),
    vline(S.yr(S.rho0), C.model),
  ];
  const notes = ([['ancient', 0, S.rho0], ['history →', S.rho0 + 0.1 * (S.rhoEndTable - S.rho0), S.rho0 + 0.3 * (S.rhoEndTable - S.rho0)]] as [string, number, number][]).map(([text, a, b]) => ({
    x: S.yr((a + b) / 2), y: 1, xref: 'x' as const, yref: 'paper' as const, text,
    showarrow: false, yanchor: 'bottom' as const, font: { size: 10, color: '#9d957f' },
  }));
  const trace = (ys: number[], color: string, name: string, fmt: string): Data =>
    ({ x: p.x, y: ys, customdata: p.year, name, type: 'scatter', mode: 'lines', line: { color, width: 2 }, hovertemplate: `%{x:.0f} yr · %{customdata}<br>${fmt}<extra></extra>` });
  const ghost = (xs: number[], ys: number[]): Data =>
    ({ x: xs, y: ys, name: 'applied now', type: 'scatter', mode: 'lines', line: { color: C.base, width: 1, dash: 'dot' }, hoverinfo: 'skip' });
  const common = { shapes, annotations: notes, margin: { t: 16, r: 20, b: 40, l: 64 }, height };
  const cfg = { displaylogo: false, responsive: true };
  return (
    <>
      <Plot
        data={[...(q ? [ghost(q.x, q.ring)] : []), trace(p.ring, C.model, 'ring length', 'ring %{y:.3s} yr')]}
        layout={baseLayout({ ...common, showlegend: !!q, xaxis: { title: { text: 'distance from the centre (yr)' } },
          yaxis: { type: 'log', title: { text: 'ring length (yr)' } } })}
        config={cfg} style={{ width: '100%' }} useResizeHandler
      />
      <Plot
        data={[...(q ? [ghost(q.x, q.K)] : []), trace(p.K, C.curv, 'curvature', 'K = %{y:.3g} /yr²')]}
        layout={baseLayout({ ...common, showlegend: false, xaxis: { title: { text: 'distance from the centre (yr)' } },
          yaxis: { title: { text: 'curvature K (1/yr²)' } } })}
        config={cfg} style={{ width: '100%' }} useResizeHandler
      />
    </>
  );
}
