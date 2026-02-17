import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Graveyard } from './graveyard';
import { DATA_START } from './fitModel';

const GRID = 500;

function fmtCalYear(year: number): string {
  if (year <= 0) return `${Math.abs(year)} BCE`;
  return `${year} CE`;
}

interface Props {
  gy: Graveyard;
}

export default function DensityHeatmap({ gy }: Props) {
  const { data, customdata, x, y, R, yc, dMin, dMax } = useMemo(() => {
    const yc = gy.ancientCircleRadius;
    const P = gy.periodLength;
    const R = yc + gy.numPeriods * P;
    const ancientDensity = gy.ancientGraves / (Math.PI * yc * yc);

    const x = new Array<number>(GRID);
    const y = new Array<number>(GRID);
    for (let i = 0; i < GRID; i++) {
      x[i] = -R + (2 * R * (i + 0.5)) / GRID;
      y[i] = -R + (2 * R * (i + 0.5)) / GRID;
    }

    let dMin = Infinity;
    let dMax = -Infinity;

    const z: (number | null)[][] = new Array(GRID);
    const cd: (string | null)[][] = new Array(GRID);
    for (let row = 0; row < GRID; row++) {
      z[row] = new Array<number | null>(GRID);
      cd[row] = new Array<string | null>(GRID);
      const py = y[row];
      for (let col = 0; col < GRID; col++) {
        const px = x[col];
        const r = Math.sqrt(px * px + py * py);

        if (r > R) {
          z[row][col] = null;
          cd[row][col] = null;
          continue;
        }

        let d: number;
        if (r <= yc) {
          d = ancientDensity;
        } else {
          const period = Math.min(
            Math.floor((r - yc) / P),
            gy.numPeriods - 1,
          );
          d = gy.densityOfPeriod(period);
        }

        z[row][col] = d;
        if (d > 0 && d < dMin) dMin = d;
        if (d > dMax) dMax = d;

        const rRound = Math.round(r);
        const calYear = r <= yc
          ? `< ${fmtCalYear(DATA_START)} (ancient)`
          : fmtCalYear(Math.round(DATA_START + (r - yc)));
        cd[row][col] = `r = ${rRound} yr | ${calYear}`;
      }
    }

    return { data: z, customdata: cd, x, y, R, yc, dMin, dMax };
  }, [gy]);

  const shapes: Partial<Plotly.Shape>[] = [
    {
      type: 'circle',
      x0: -yc, y0: -yc, x1: yc, y1: yc,
      xref: 'x', yref: 'y',
      line: { color: '#61afef', width: 1.5, dash: 'dash' },
    },
    {
      type: 'circle',
      x0: -R, y0: -R, x1: R, y1: R,
      xref: 'x', yref: 'y',
      line: { color: '#fff', width: 1, dash: 'dot' },
    },
  ];

  const annotations: Partial<Plotly.Annotations>[] = [
    {
      x: 0, y: yc + R * 0.02,
      xref: 'x', yref: 'y',
      text: `ancient circle (yc=${yc})`,
      showarrow: false,
      font: { color: '#61afef', size: 10 },
    },
  ];

  return (
    <Plot
      data={[
        {
          z: data,
          x,
          y,
          customdata,
          type: 'heatmap',
          colorscale: 'Inferno',
          zmin: dMin,
          zmax: dMax,
          colorbar: {
            title: { text: 'graves/yr²', side: 'right', font: { color: '#999', size: 11 } },
            tickfont: { color: '#999', size: 10 },
            bgcolor: 'transparent',
            outlinecolor: '#333',
            outlinewidth: 1,
          },
          hovertemplate:
            '%{customdata}<br>density: %{z:.4e} graves/yr²<extra></extra>',
          showscale: true,
        },
      ]}
      layout={{
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#999', size: 11 },
        height: 600,
        margin: { t: 10, r: 80, b: 50, l: 60 },
        xaxis: {
          scaleanchor: 'y',
          scaleratio: 1,
          title: { text: 'x (yr)', font: { color: '#555', size: 11 } },
          gridcolor: '#222',
          zerolinecolor: '#333',
          tickfont: { color: '#666', size: 10 },
        },
        yaxis: {
          title: { text: 'y (yr)', font: { color: '#555', size: 11 } },
          gridcolor: '#222',
          zerolinecolor: '#333',
          tickfont: { color: '#666', size: 10 },
        },
        shapes,
        annotations,
      }}
      config={{
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
        responsive: true,
        scrollZoom: true,
      }}
      useResizeHandler
      style={{ width: '100%' }}
    />
  );
}
