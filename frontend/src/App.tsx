import { useState, useMemo } from 'react';
import {
  ComposedChart, Bar, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { Graveyard, type ChunkInfo, type ChunkAddr, type BoundaryNeighbor } from './graveyard';
import './App.css';

// ----------------------------------------------------------------
// Formatting helpers
// ----------------------------------------------------------------
function fmtCount(n: number): string {
  const a = Math.abs(n);
  if (a >= 1e9) return (n / 1e9).toFixed(a >= 10e9 ? 0 : 1) + 'B';
  if (a >= 1e6) return (n / 1e6).toFixed(a >= 10e6 ? 0 : 1) + 'M';
  if (a >= 1e3) return (n / 1e3).toFixed(a >= 10e3 ? 0 : 1) + 'K';
  return n.toFixed(0);
}

function fmtYearsAgo(y: number): string {
  if (Math.abs(y) < 1) return 'now';
  if (y >= 1000) {
    const k = y / 1000;
    return (k % 1 === 0 ? k.toFixed(0) : k.toFixed(1)) + 'k ya';
  }
  return y.toFixed(0) + ' ya';
}

// ----------------------------------------------------------------
// SVG sector path
// ----------------------------------------------------------------
function sectorPath(
  cx: number, cy: number,
  rInner: number, rOuter: number,
  startAngle: number, endAngle: number,
): string {
  const span = endAngle - startAngle;

  if (span >= 2 * Math.PI - 0.001) {
    if (rInner < 0.5) {
      return [
        `M ${cx - rOuter} ${cy}`,
        `A ${rOuter} ${rOuter} 0 1 1 ${cx + rOuter} ${cy}`,
        `A ${rOuter} ${rOuter} 0 1 1 ${cx - rOuter} ${cy}`,
        'Z',
      ].join(' ');
    }
    return [
      `M ${cx - rOuter} ${cy}`,
      `A ${rOuter} ${rOuter} 0 1 1 ${cx + rOuter} ${cy}`,
      `A ${rOuter} ${rOuter} 0 1 1 ${cx - rOuter} ${cy}`,
      `M ${cx - rInner} ${cy}`,
      `A ${rInner} ${rInner} 0 1 0 ${cx + rInner} ${cy}`,
      `A ${rInner} ${rInner} 0 1 0 ${cx - rInner} ${cy}`,
      'Z',
    ].join(' ');
  }

  const x1o = cx + rOuter * Math.cos(startAngle);
  const y1o = cy + rOuter * Math.sin(startAngle);
  const x2o = cx + rOuter * Math.cos(endAngle);
  const y2o = cy + rOuter * Math.sin(endAngle);
  const largeArc = span > Math.PI ? 1 : 0;

  if (rInner < 0.5) {
    return `M ${cx} ${cy} L ${x1o} ${y1o} A ${rOuter} ${rOuter} 0 ${largeArc} 1 ${x2o} ${y2o} Z`;
  }

  const x1i = cx + rInner * Math.cos(startAngle);
  const y1i = cy + rInner * Math.sin(startAngle);
  const x2i = cx + rInner * Math.cos(endAngle);
  const y2i = cy + rInner * Math.sin(endAngle);

  return [
    `M ${x1o} ${y1o}`,
    `A ${rOuter} ${rOuter} 0 ${largeArc} 1 ${x2o} ${y2o}`,
    `L ${x2i} ${y2i}`,
    `A ${rInner} ${rInner} 0 ${largeArc} 0 ${x1i} ${y1i}`,
    'Z',
  ].join(' ');
}

// ----------------------------------------------------------------
// Bounding box of a circular sector
// ----------------------------------------------------------------
function sectorBBox(
  cx: number, cy: number,
  rMin: number, rMax: number,
  a1: number, a2: number,
): { x: number; y: number; w: number; h: number } {
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  const upd = (x: number, y: number) => {
    if (x < xMin) xMin = x;
    if (x > xMax) xMax = x;
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  };

  for (const r of [rMin, rMax]) {
    upd(cx + r * Math.cos(a1), cy + r * Math.sin(a1));
    upd(cx + r * Math.cos(a2), cy + r * Math.sin(a2));
  }

  // Check cardinal angles (multiples of π/2) that fall within the arc
  const step = Math.PI / 2;
  const mStart = Math.ceil(a1 / step);
  const mEnd = Math.floor(a2 / step);
  for (let m = mStart; m <= mEnd; m++) {
    const a = m * step;
    if (a > a1 && a < a2) {
      for (const r of [rMin, rMax]) {
        upd(cx + r * Math.cos(a), cy + r * Math.sin(a));
      }
    }
  }

  return { x: xMin, y: yMin, w: xMax - xMin, h: yMax - yMin };
}

// ----------------------------------------------------------------
// Focused chunk view
// ----------------------------------------------------------------
const RING_COLORS = ['#4ecdc4', '#ff8c42', '#c678dd', '#e06c75', '#61afef', '#98c379'];

function FocusedView({
  gy,
  chunkInfo,
}: {
  gy: Graveyard;
  chunkInfo: ChunkInfo;
}) {
  const { period: selP, chunkId: selC, neighbors } = chunkInfo;

  // Rings: inner, selected, outer (when they exist)
  const ringPeriods: number[] = [];
  if (selP > 0) ringPeriods.push(selP - 1);
  ringPeriods.push(selP);
  if (selP < gy.numPeriods - 1) ringPeriods.push(selP + 1);
  const nRings = ringPeriods.length;

  // SVG layout
  const R = 500;
  const rMin = 150;
  const rw = (R - rMin) / nRings;
  const cx = R + 20;
  const cy = R + 20;

  // Collect all neighbor + selected addresses
  const allAddrs: ChunkAddr[] = [
    { period: selP, chunk: selC },
    ...(neighbors.clockwise ? [neighbors.clockwise] : []),
    ...(neighbors.anticlockwise ? [neighbors.anticlockwise] : []),
    ...neighbors.inner.map(b => b.addr),
    ...neighbors.outer.map(b => b.addr),
  ];

  // Build set of relevant chunks per ring: neighbors + 2 padding on each side
  const relevantByPeriod = new Map<number, Set<number>>();
  for (const pi of ringPeriods) relevantByPeriod.set(pi, new Set());
  for (const addr of allAddrs) {
    relevantByPeriod.get(addr.period)?.add(addr.chunk);
  }
  for (const [pi, chunks] of relevantByPeriod) {
    const N = gy.chunksInPeriod(pi);
    const expanded = new Set(chunks);
    for (const c of chunks) {
      for (let d = 1; d <= 2; d++) {
        expanded.add((c - d + N) % N);
        expanded.add((c + d) % N);
      }
    }
    relevantByPeriod.set(pi, expanded);
  }

  // Compute visible angular extent from all relevant chunks
  const selCenter = (chunkInfo.angleStart + chunkInfo.angleEnd) / 2;
  let minAngle = selCenter, maxAngle = selCenter;
  for (const [pi, chunks] of relevantByPeriod) {
    const N = gy.chunksInPeriod(pi);
    for (const ci of chunks) {
      let a1 = (ci / N) * 2 * Math.PI;
      let a2 = ((ci + 1) / N) * 2 * Math.PI;
      // Handle wrap-around: keep angles near selCenter
      while (a1 > selCenter + Math.PI) { a1 -= 2 * Math.PI; a2 -= 2 * Math.PI; }
      while (a1 < selCenter - Math.PI) { a1 += 2 * Math.PI; a2 += 2 * Math.PI; }
      minAngle = Math.min(minAngle, a1);
      maxAngle = Math.max(maxAngle, a2);
    }
  }

  const span = maxAngle - minAngle;
  const fullCircle = span >= 2 * Math.PI * 0.85;

  // Compute viewBox
  let vb: string;
  if (fullCircle) {
    const s = 2 * (R + 20);
    vb = `0 0 ${s} ${s}`;
  } else {
    const bb = sectorBBox(cx, cy, rMin, R, minAngle, maxAngle);
    const pad = 25;
    let vbW = bb.w + 2 * pad;
    let vbH = bb.h + 2 * pad;
    let vbX = bb.x - pad;
    let vbY = bb.y - pad;
    // Enforce min aspect ratio
    const minW = vbH * 1.3;
    if (vbW < minW) { vbX -= (minW - vbW) / 2; vbW = minW; }
    const minH = vbW / 2.5;
    if (vbH < minH) { vbY -= (minH - vbH) / 2; vbH = minH; }
    vb = `${vbX} ${vbY} ${vbW} ${vbH}`;
  }

  // Highlight sets
  const selectedKey = `${selP}:${selC}`;
  const neighborSet = new Set(allAddrs.slice(1).map(a => `${a.period}:${a.chunk}`));

  return (
    <svg viewBox={vb} className="tess-svg" preserveAspectRatio="xMidYMid meet">
      {ringPeriods.map((pi, ri) => {
        const rI = rMin + ri * rw;
        const rO = rMin + (ri + 1) * rw;
        const N = gy.chunksInPeriod(pi);
        const astep = (2 * Math.PI) / N;
        const color = RING_COLORS[pi % RING_COLORS.length];
        const relevant = relevantByPeriod.get(pi) || new Set();

        const els: JSX.Element[] = [];

        // Background arc
        if (fullCircle) {
          els.push(
            <path key="bg" d={sectorPath(cx, cy, rI, rO, 0, 2 * Math.PI)}
              fill={color} fillOpacity={0.04} stroke={color} strokeOpacity={0.12} strokeWidth={0.5} />
          );
        } else {
          els.push(
            <path key="bg" d={sectorPath(cx, cy, rI, rO, minAngle, maxAngle)}
              fill={color} fillOpacity={0.04} stroke={color} strokeOpacity={0.12} strokeWidth={0.5} />
          );
        }

        // Individual chunks at their true angles
        for (const ci of relevant) {
          let a1 = ci * astep;
          let a2 = (ci + 1) * astep;
          // Handle wrap-around
          while (a1 > selCenter + Math.PI) { a1 -= 2 * Math.PI; a2 -= 2 * Math.PI; }
          while (a1 < selCenter - Math.PI) { a1 += 2 * Math.PI; a2 += 2 * Math.PI; }

          const key = `${pi}:${ci}`;
          const isSel = key === selectedKey;
          const isNb = neighborSet.has(key);

          els.push(
            <path key={`c${ci}`}
              d={sectorPath(cx, cy, rI, rO, a1, a2)}
              fill={isSel ? '#fff' : color}
              fillOpacity={isSel ? 0.5 : isNb ? 0.4 : 0.1}
              stroke={isSel ? '#fff' : isNb ? '#fff' : color}
              strokeOpacity={isSel ? 1 : isNb ? 0.7 : 0.25}
              strokeWidth={isSel ? 3 : isNb ? 2 : 0.5}
            />
          );
        }

        // Ring label
        const labelAngle = fullCircle ? 0 : maxAngle + 0.04;
        const labelR = (rI + rO) / 2;
        els.push(
          <text key="label"
            x={cx + labelR * Math.cos(labelAngle)}
            y={cy + labelR * Math.sin(labelAngle)}
            textAnchor="start" dominantBaseline="central"
            fill={color} fontSize={14} fontFamily="monospace" opacity={0.8}
          >
            P{pi} ({N > 999 ? fmtCount(N) : N})
          </text>
        );

        return <g key={ri}>{els}</g>;
      })}
    </svg>
  );
}

// ----------------------------------------------------------------
// Boundary neighbor formatter
// ----------------------------------------------------------------
function BoundaryList({ label, items }: { label: string; items: BoundaryNeighbor[] }) {
  if (items.length === 0) return (
    <div className="tess-neighbor-row">
      <span className="tess-n-label">{label}:</span>
      <span className="tess-n-value">none</span>
    </div>
  );

  const shown = items.length > 12 ? items.slice(0, 10) : items;
  return (
    <>
      <div className="tess-neighbor-row">
        <span className="tess-n-label">{label}:</span>
        <span className="tess-n-value">{items.length} chunk{items.length > 1 ? 's' : ''}</span>
      </div>
      {shown.map((b, i) => (
        <div key={i} className="tess-neighbor-detail">
          ({b.addr.period}, {b.addr.chunk}): {b.localStart.toFixed(3)} &rarr; {b.localEnd.toFixed(3)}
        </div>
      ))}
      {items.length > 12 && (
        <div className="tess-neighbor-detail">... +{items.length - 10} more</div>
      )}
    </>
  );
}

// ================================================================
// Main component
// ================================================================
export default function App() {
  const [periodLength, setPeriodLength] = useState(75);
  const [lastPeriodGraves, setLastPeriodGraves] = useState(5e9);
  const [maxGraves, setMaxGraves] = useState(1000);
  const [selectedPeriod, setSelectedPeriod] = useState(0);
  const [selectedChunk, setSelectedChunk] = useState(0);

  const gy = useMemo(
    () => new Graveyard(periodLength, lastPeriodGraves, maxGraves),
    [periodLength, lastPeriodGraves, maxGraves],
  );

  // Chart data
  const chartData = useMemo(() => {
    const data: { yearsAgo: number; bar: number; line: number; chunks: number }[] = [];
    for (let i = 0; i < gy.numPeriods; i++) {
      const s = gy.periodStats(i);
      data.push({ yearsAgo: s.yearsAgo, bar: s.graves, line: s.fTimesP, chunks: s.chunks });
    }
    return data;
  }, [gy]);

  const totalChunks = useMemo(
    () => { let s = 0; for (let i = 0; i < gy.numPeriods; i++) s += gy.chunksInPeriod(i); return s; },
    [gy],
  );

  // Clamp selection
  const safePeriod = Math.min(selectedPeriod, gy.numPeriods - 1);
  const safePeriodChunks = gy.chunksInPeriod(safePeriod);
  const safeChunk = Math.min(selectedChunk, Math.max(0, safePeriodChunks - 1));

  // Get chunk info from the Graveyard class
  const chunkInfo: ChunkInfo = useMemo(
    () => gy.getChunkInfo(safePeriod, safeChunk),
    [gy, safePeriod, safeChunk],
  );

  const xTicks = useMemo(() => {
    const n = chartData.length;
    if (n <= 10) return chartData.map(d => d.yearsAgo);
    const step = Math.max(1, Math.floor(n / 7));
    const ticks: number[] = [];
    for (let i = n - 1; i >= 0; i -= step) ticks.push(chartData[i].yearsAgo);
    if (!ticks.includes(chartData[0].yearsAgo)) ticks.push(chartData[0].yearsAgo);
    return ticks.sort((a, b) => b - a);
  }, [chartData]);

  const selStats = gy.periodStats(safePeriod);

  const totalNeighbors =
    chunkInfo.neighbors.inner.length +
    (chunkInfo.neighbors.clockwise ? 1 : 0) +
    (chunkInfo.neighbors.anticlockwise ? 1 : 0) +
    chunkInfo.neighbors.outer.length;

  return (
    <div className="app">
      <h1 className="title">Graveyard</h1>
      <p className="subtitle">
        Exponential model of {fmtCount(Graveyard.TOTAL_GRAVES)} human graves
        &nbsp;&middot;&nbsp; span: {gy.totalSpan.toLocaleString()} years ({gy.numPeriods} periods)
        &nbsp;&middot;&nbsp; &alpha; = {gy.alpha.toExponential(4)}
        &nbsp;&middot;&nbsp; total chunks: {fmtCount(totalChunks)}
      </p>

      <div className="controls">
        <label className="ctrl">
          <span>Period length: <b>{periodLength} years</b></span>
          <input type="range" min={25} max={500} step={5}
            value={periodLength} onChange={e => setPeriodLength(+e.target.value)} />
        </label>
        <label className="ctrl">
          <span>Graves in last period: <b>{fmtCount(lastPeriodGraves)}</b></span>
          <input type="range" min={1e9} max={10e9} step={0.5e9}
            value={lastPeriodGraves} onChange={e => setLastPeriodGraves(+e.target.value)} />
        </label>
        <label className="ctrl">
          <span>Max graves per chunk: <b>{maxGraves.toLocaleString()}</b></span>
          <input type="range" min={100} max={100000} step={100}
            value={maxGraves} onChange={e => setMaxGraves(+e.target.value)} />
        </label>
      </div>

      <h2 className="chart-title">Graves per period</h2>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={chartData} margin={{ top: 16, right: 24, bottom: 20, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
            <XAxis dataKey="yearsAgo" type="number" domain={['dataMin', 'dataMax']}
              reversed ticks={xTicks} tickFormatter={fmtYearsAgo}
              tick={{ fill: '#666', fontSize: 11 }} stroke="#333"
              label={{ value: 'years ago', position: 'insideBottom', offset: -10, fill: '#555', fontSize: 12 }} />
            <YAxis scale="log" domain={['auto', 'auto']} allowDataOverflow
              tickFormatter={fmtCount} tick={{ fill: '#666', fontSize: 11 }} stroke="#333" width={60} />
            <Tooltip
              contentStyle={{ background: '#1a1a1a', border: '1px solid #333', borderRadius: 6, fontSize: 12 }}
              labelFormatter={(v: number) => fmtYearsAgo(v)}
              formatter={(value: number, name: string) => [fmtCount(value), name === 'bar' ? '∫ per period' : 'f(t) × period']} />
            <Legend formatter={(v: string) => v === 'bar' ? '∫ per period' : 'f(t) × period'}
              wrapperStyle={{ fontSize: 12, color: '#888' }} />
            <Bar dataKey="bar" fill="rgba(78, 205, 196, 0.5)" stroke="rgba(78, 205, 196, 0.3)" isAnimationActive={false} />
            <Line dataKey="line" stroke="#ff8c42" strokeWidth={2} dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <h2 className="chart-title">Chunk inspector</h2>
      <div className="tess-section">
        <div className="tess-svg-wrap">
          <FocusedView gy={gy} chunkInfo={chunkInfo} />
        </div>

        <div className="tess-panel">
          <label className="ctrl">
            <span>Period: <b>{safePeriod}</b> / {gy.numPeriods - 1}</span>
            <input type="range" min={0} max={gy.numPeriods - 1} step={1}
              value={safePeriod}
              onChange={e => { setSelectedPeriod(+e.target.value); setSelectedChunk(0); }} />
          </label>

          <div className="tess-info-grid">
            <div className="tess-info-item">
              <span className="tess-label">Time window</span>
              <span className="tess-value">{fmtYearsAgo(selStats.yearsAgo + periodLength / 2)} &rarr; {fmtYearsAgo(selStats.yearsAgo - periodLength / 2)}</span>
            </div>
            <div className="tess-info-item">
              <span className="tess-label">Graves</span>
              <span className="tess-value">{fmtCount(chunkInfo.periodGraves)}</span>
            </div>
            <div className="tess-info-item">
              <span className="tess-label">Chunks in period</span>
              <span className="tess-value">{chunkInfo.periodChunks.toLocaleString()}</span>
            </div>
            <div className="tess-info-item">
              <span className="tess-label">Graves / chunk</span>
              <span className="tess-value">{fmtCount(chunkInfo.graves)}</span>
            </div>
          </div>

          <label className="ctrl">
            <span>Chunk angular ID: <b>{safeChunk}</b> / {Math.max(0, safePeriodChunks - 1)}</span>
            <input type="range" min={0} max={Math.max(0, safePeriodChunks - 1)} step={1}
              value={safeChunk}
              onChange={e => setSelectedChunk(+e.target.value)} />
          </label>

          <div className="tess-neighbors">
            <h3>Neighbors of ({safePeriod}, {safeChunk}) &mdash; {totalNeighbors} total</h3>

            <BoundaryList label="Inner boundary" items={chunkInfo.neighbors.inner} />

            <div className="tess-neighbor-row">
              <span className="tess-n-label">Clockwise:</span>
              <span className="tess-n-value">
                {chunkInfo.neighbors.clockwise
                  ? `(${chunkInfo.neighbors.clockwise.period}, ${chunkInfo.neighbors.clockwise.chunk})`
                  : 'none'}
              </span>
            </div>
            <div className="tess-neighbor-row">
              <span className="tess-n-label">Anticlockwise:</span>
              <span className="tess-n-value">
                {chunkInfo.neighbors.anticlockwise
                  ? `(${chunkInfo.neighbors.anticlockwise.period}, ${chunkInfo.neighbors.anticlockwise.chunk})`
                  : 'none'}
              </span>
            </div>

            <BoundaryList label="Outer boundary" items={chunkInfo.neighbors.outer} />
          </div>
        </div>
      </div>
    </div>
  );
}
