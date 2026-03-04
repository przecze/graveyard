import Plot from 'react-plotly.js';

// PRB data rows: each cumulative_births = births in the period ending at this year.
// Cumulative deaths at year Y = sum(cumulative_births up to and including Y) - pop at Y.
const PRB = [
  { year: -8000, cumBirths:  8_993_889_771, pop:    5_000_000 },
  { year:     1, cumBirths: 46_025_332_354, pop:  300_000_000 },
  { year:  1200, cumBirths: 26_591_343_000, pop:  450_000_000 },
  { year:  1650, cumBirths: 12_782_002_453, pop:  500_000_000 },
  { year:  1750, cumBirths:  3_171_931_513, pop:  795_000_000 },
  { year:  1850, cumBirths:  4_046_240_009, pop: 1_265_000_000 },
  { year:  1900, cumBirths:  2_900_237_856, pop: 1_656_000_000 },
  { year:  1950, cumBirths:  3_390_198_215, pop: 2_499_000_000 },
  { year:  2024, cumBirths:  8_200_000_000, pop: 8_100_000_000 }, // estimated
];

// Anchor points: year → cumulative deaths
const ANCHORS: { year: number; cumDeaths: number }[] = [];
let runningBirths = 0;
for (const row of PRB) {
  runningBirths += row.cumBirths;
  ANCHORS.push({ year: row.year, cumDeaths: runningBirths - row.pop });
}

function fmtYear(y: number): string {
  if (y < 0) return `${-y} BCE`;
  if (y === 0) return '1 CE';
  return `${y} CE`;
}

function computePoints() {
  const xYears: number[] = [];
  const yCumB: number[] = [];
  const hover: string[] = [];

  for (let i = 0; i < ANCHORS.length - 1; i++) {
    const a = ANCHORS[i];
    const b = ANCHORS[i + 1];
    const span = b.year - a.year;
    for (let dy = 0; dy < span; dy++) {
      const yr = a.year + dy;
      xYears.push(yr);
      yCumB.push((a.cumDeaths + (b.cumDeaths - a.cumDeaths) * (dy / span)) / 1e9);
      hover.push(fmtYear(yr));
    }
  }
  const last = ANCHORS[ANCHORS.length - 1];
  xYears.push(last.year);
  yCumB.push(last.cumDeaths / 1e9);
  hover.push(fmtYear(last.year));

  return { xYears, yCumB, hover };
}

const TICK_VALS = [-8000, -6000, -4000, -2000, -1000, 1, 500, 1000, 1500, 2000];
const TICK_TEXT = TICK_VALS.map(fmtYear);

const { xYears, yCumB, hover } = computePoints();

export default function AboutDialog({ onClose }: { onClose: () => void }) {
  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0, 0, 0, 0.7)',
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: '#161616',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          borderRadius: 8,
          padding: '20px 20px 12px',
          width: 780,
          maxWidth: '92vw',
          fontFamily: 'monospace',
          color: '#ccc',
        }}
      >
        <div style={{ marginBottom: 4, fontSize: 14, color: '#eee' }}>
          Cumulative human deaths since 8000 BCE
        </div>
        <div style={{ marginBottom: 12, fontSize: 11, color: 'rgba(200,200,200,0.5)' }}>
          PRB methodology · post-1950 estimated · total ~{(yCumB[yCumB.length - 1]).toFixed(1)}B
        </div>

        <Plot
          data={[
            {
              x: xYears,
              y: yCumB,
              type: 'scatter',
              mode: 'lines',
              fill: 'tozeroy',
              line: { color: '#7ad9ff', width: 1.5 },
              fillcolor: 'rgba(122, 217, 255, 0.12)',
              customdata: hover,
              hovertemplate: '<b>%{customdata}</b><br>%{y:.2f}B deaths<extra></extra>',
            },
          ]}
          layout={{
            paper_bgcolor: '#161616',
            plot_bgcolor: '#161616',
            height: 360,
            margin: { t: 8, r: 20, b: 52, l: 72 },
            font: { color: '#aaa', family: 'monospace', size: 11 },
            xaxis: {
              tickvals: TICK_VALS,
              ticktext: TICK_TEXT,
              tickangle: -35,
              gridcolor: 'rgba(255,255,255,0.07)',
              linecolor: 'rgba(255,255,255,0.15)',
              tickcolor: 'rgba(255,255,255,0.15)',
              tickfont: { size: 10 },
            },
            yaxis: {
              title: { text: 'Cumulative deaths (billions)', font: { size: 11 } },
              gridcolor: 'rgba(255,255,255,0.07)',
              linecolor: 'rgba(255,255,255,0.15)',
              tickcolor: 'rgba(255,255,255,0.15)',
              ticksuffix: 'B',
              rangemode: 'tozero',
            },
            hoverlabel: {
              bgcolor: '#222',
              bordercolor: 'rgba(122,217,255,0.4)',
              font: { family: 'monospace', size: 12, color: '#eee' },
            },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  );
}
