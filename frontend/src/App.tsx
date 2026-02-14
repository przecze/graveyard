import { useState, useMemo } from 'react';
import { Graveyard, type ChunkInfo, type ChunkAddr, type BoundaryNeighbor } from './graveyard';
import PopulationChart from './PopulationChart';
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

const CURRENT_YEAR = 2026;

function fmtCalendarYear(yearsAgo: number): string {
  const year = Math.round(CURRENT_YEAR - yearsAgo);
  if (year <= 0) return `${Math.abs(year - 1)} BC`;
  return `${year} AD`;
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
// Focused chunk view — fixed 200×200 year canvas, no rotation/rescaling
// ----------------------------------------------------------------
const RING_COLORS = ['#4ecdc4', '#ff8c42', '#c678dd', '#e06c75', '#61afef', '#98c379'];
const CANVAS_YEARS = 400;
const HALF = CANVAS_YEARS / 2;
const DIAG = HALF * Math.SQRT2; // ~141 years — max distance from view center to corner

function FocusedView({
  gy,
  chunkInfo,
}: {
  gy: Graveyard;
  chunkInfo: ChunkInfo;
}) {
  const { period: selP, chunkId: selC, neighbors } = chunkInfo;
  const P = gy.periodLength;

  // Center of selected chunk in year-coordinate space (tessellation origin = 0,0)
  const rMid = (selP + 0.5) * P;
  const thetaMid = (chunkInfo.angleStart + chunkInfo.angleEnd) / 2;
  const viewCx = rMid * Math.cos(thetaMid);
  const viewCy = rMid * Math.sin(thetaMid);

  // Fixed viewBox — always 200×200 years centered on selected chunk
  const vb = `${viewCx - HALF} ${viewCy - HALF} ${CANVAS_YEARS} ${CANVAS_YEARS}`;

  // Visible radial range
  const minR = Math.max(0, rMid - DIAG);
  const maxR = rMid + DIAG;
  const minPeriod = Math.max(0, Math.floor(minR / P));
  const maxPeriod = Math.min(gy.numPeriods - 1, Math.ceil(maxR / P));

  // Highlight sets
  const allNeighborAddrs: ChunkAddr[] = [
    ...(neighbors.clockwise ? [neighbors.clockwise] : []),
    ...(neighbors.anticlockwise ? [neighbors.anticlockwise] : []),
    ...neighbors.inner.map(b => b.addr),
    ...neighbors.outer.map(b => b.addr),
  ];
  const selectedKey = `${selP}:${selC}`;
  const neighborSet = new Set(allNeighborAddrs.map(a => `${a.period}:${a.chunk}`));

  return (
    <svg viewBox={vb} className="tess-svg" preserveAspectRatio="xMidYMid meet">
      {Array.from({ length: maxPeriod - minPeriod + 1 }, (_, i) => {
        const pi = minPeriod + i;
        const rI = pi * P;
        const rO = (pi + 1) * P;
        const Npi = gy.chunksInPeriod(pi);
        const astep = (2 * Math.PI) / Npi;
        const piOff = gy.chunkOffset(pi); // 0.5 for odd periods
        const color = RING_COLORS[pi % RING_COLORS.length];
        const rCenter = (rI + rO) / 2;

        const els: JSX.Element[] = [];

        // Visible angular margin at this ring's radius
        const angMargin = rCenter > 1 ? DIAG / rCenter + astep * 2 : Math.PI;
        const clampedMargin = Math.min(angMargin, Math.PI);

        // Ring background band
        els.push(
          <path key="bg"
            d={sectorPath(0, 0, rI, rO,
              thetaMid - clampedMargin, thetaMid + clampedMargin)}
            fill={color} fillOpacity={0.04}
            stroke={color} strokeOpacity={0.12} strokeWidth={0.3}
          />
        );

        // Arc length per chunk (years) at ring center
        const arcLen = (2 * Math.PI * rCenter) / Npi;

        // Only draw individual chunk outlines if they're large enough to see
        const drawAll = arcLen > 0.4;

        if (drawAll) {
          const minIdx = Math.floor((thetaMid - clampedMargin) / astep - piOff) - 1;
          const maxIdx = Math.ceil((thetaMid + clampedMargin) / astep - piOff) + 1;
          for (let ci = minIdx; ci <= maxIdx; ci++) {
            const wci = ((ci % Npi) + Npi) % Npi; // wrapped chunk id
            const a1 = (ci + piOff) * astep;
            const a2 = (ci + 1 + piOff) * astep;
            const key = `${pi}:${wci}`;
            const isSel = key === selectedKey;
            const isNb = neighborSet.has(key);

            els.push(
              <path key={`c${ci}`}
                d={sectorPath(0, 0, rI, rO, a1, a2)}
                fill={isSel ? '#fff' : color}
                fillOpacity={isSel ? 0.5 : isNb ? 0.4 : 0.08}
                stroke={isSel ? '#fff' : isNb ? '#fff' : color}
                strokeOpacity={isSel ? 1 : isNb ? 0.7 : 0.2}
                strokeWidth={isSel ? 1.5 : isNb ? 1 : 0.2}
              />
            );
          }
        } else {
          // Chunks too small to draw individually — only highlight selected + neighbors
          const toHighlight: { c: number; isSel: boolean }[] = [];
          if (selP === pi) toHighlight.push({ c: selC, isSel: true });
          for (const addr of allNeighborAddrs) {
            if (addr.period === pi) toHighlight.push({ c: addr.chunk, isSel: false });
          }
          for (let hi = 0; hi < toHighlight.length; hi++) {
            const { c, isSel } = toHighlight[hi];
            let a1 = (c + piOff) * astep;
            // Wrap near thetaMid
            while (a1 - thetaMid > Math.PI) a1 -= 2 * Math.PI;
            while (thetaMid - a1 > Math.PI) a1 += 2 * Math.PI;
            const a2 = a1 + astep;
            els.push(
              <path key={`h${hi}`}
                d={sectorPath(0, 0, rI, rO, a1, a2)}
                fill={isSel ? '#fff' : color}
                fillOpacity={isSel ? 0.5 : 0.4}
                stroke={isSel ? '#fff' : '#fff'}
                strokeOpacity={isSel ? 1 : 0.7}
                strokeWidth={isSel ? 1.5 : 1}
              />
            );
          }
        }

        // Period label at the right edge of the visible band
        const labelA = thetaMid + clampedMargin * 0.85;
        const labelR = rCenter;
        const fontSize = Math.min(4, P * 0.3);
        els.push(
          <text key="label"
            x={labelR * Math.cos(labelA)}
            y={labelR * Math.sin(labelA)}
            textAnchor="start" dominantBaseline="central"
            fill={color} fontSize={fontSize} fontFamily="monospace" opacity={0.7}
          >
            P{pi}
          </text>
        );

        return <g key={pi}>{els}</g>;
      })}
    </svg>
  );
}

// ----------------------------------------------------------------
// Solo chunk view — 4× zoom, auto-height to fit the chunk
// ----------------------------------------------------------------
const SOLO_SCALE = 4; // 4× zoom compared to the neighbourhood canvas
const SOLO_W = 1000 / SOLO_SCALE;  // years wide (fixed horizontal span)

/** Bounding box of a sector arc centered at origin. */
function sectorBBox(
  rI: number, rO: number, a1: number, a2: number,
): { x: number; y: number; w: number; h: number } {
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  const upd = (x: number, y: number) => {
    if (x < xMin) xMin = x; if (x > xMax) xMax = x;
    if (y < yMin) yMin = y; if (y > yMax) yMax = y;
  };
  for (const r of [rI, rO]) {
    upd(r * Math.cos(a1), r * Math.sin(a1));
    upd(r * Math.cos(a2), r * Math.sin(a2));
  }
  const step = Math.PI / 2;
  for (let m = Math.ceil(a1 / step); m <= Math.floor(a2 / step); m++) {
    const a = m * step;
    if (a > a1 && a < a2) {
      for (const r of [rI, rO]) upd(r * Math.cos(a), r * Math.sin(a));
    }
  }
  // For a full circle or pie slice that includes the origin
  if (rI < 0.5) upd(0, 0);
  return { x: xMin, y: yMin, w: xMax - xMin, h: yMax - yMin };
}

function ChunkSoloView({ gy, chunkInfo }: { gy: Graveyard; chunkInfo: ChunkInfo }) {
  const { period, angleStart, angleEnd, periodChunks } = chunkInfo;
  const PL = gy.periodLength;
  const N = periodChunks;

  const rI = period * PL;
  const rO = (period + 1) * PL;
  let a1 = angleStart;
  let a2 = angleEnd;
  if (N === 1) { a1 = 0; a2 = 2 * Math.PI; }

  // Cartesian stretch matrix in the tangential direction
  const sf = gy.scalingFactor(period);
  const theta = (a1 + a2) / 2;
  const sn = Math.sin(theta);
  const cs = Math.cos(theta);
  const ma = sf * sn * sn + cs * cs;
  const mb = (1 - sf) * sn * cs;
  const md = sf * cs * cs + sn * sn;

  // Transform helper
  const tfx = (x: number, y: number) => ma * x + mb * y;
  const tfy = (x: number, y: number) => mb * x + md * y;

  // Sample boundary of the sector and compute rescaled bbox
  const nSamp = 48;
  let sxMin = Infinity, sxMax = -Infinity, syMin = Infinity, syMax = -Infinity;
  for (let i = 0; i <= nSamp; i++) {
    const a = a1 + (a2 - a1) * i / nSamp;
    for (const r of [rI, rO]) {
      const px = r * Math.cos(a), py = r * Math.sin(a);
      const x = tfx(px, py), y = tfy(px, py);
      if (x < sxMin) sxMin = x; if (x > sxMax) sxMax = x;
      if (y < syMin) syMin = y; if (y > syMax) syMax = y;
    }
  }

  // ViewBox sized to rescaled chunk only — original may overflow
  const pad = PL * 0.1;
  const scaledCx = (sxMin + sxMax) / 2;
  const vbW = Math.max(SOLO_W, sxMax - sxMin + 2 * pad);
  const vbH = (syMax - syMin) + pad; // pad only at bottom, top flush
  const vb = `${scaledCx - vbW / 2} ${syMin} ${vbW} ${vbH}`;

  const sw = Math.max(0.3, vbW * 0.002);
  const color = RING_COLORS[period % RING_COLORS.length];
  const sectorD = sectorPath(0, 0, rI, rO, a1, a2);

  return (
    <svg viewBox={vb} className="chunk-solo-svg"
      style={{ aspectRatio: `${vbW} / ${vbH}` }}
      overflow="visible"
      preserveAspectRatio="xMidYMin slice">
      {/* Original chunk — semi-transparent, can overflow */}
      <path d={sectorD} fill="none"
        stroke={color} strokeOpacity={0.2} strokeWidth={sw} />
      {/* Rescaled chunk — main element */}
      <g transform={`matrix(${ma}, ${mb}, ${mb}, ${md}, 0, 0)`}>
        <path d={sectorD} fill="none"
          stroke="#fff" strokeOpacity={0.8}
          strokeWidth={sw / Math.max(sf, 0.01)} />
      </g>
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
  const [maxGraves, setMaxGraves] = useState(200);
  const [selectedPeriod, setSelectedPeriod] = useState(50);
  const [selectedChunk, setSelectedChunk] = useState(0);

  const gy = useMemo(
    () => new Graveyard(periodLength, maxGraves),
    [periodLength, maxGraves],
  );

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
        {fmtCount(gy.totalGraves)} human graves
        &nbsp;&middot;&nbsp; {fmtCalendarYear(gy.totalSpan)} &rarr; present ({gy.numPeriods} periods of {periodLength} yr)
        &nbsp;&middot;&nbsp; total chunks: {fmtCount(totalChunks)}
      </p>

      <h2 className="chart-title">World population estimates</h2>
      <div className="chart-wrap">
        <PopulationChart />
      </div>

      <div className="controls">
        <label className="ctrl">
          <span>Period length: <b>{periodLength} years</b></span>
          <input type="range" min={25} max={500} step={5}
            value={periodLength} onChange={e => setPeriodLength(+e.target.value)} />
        </label>
        <label className="ctrl">
          <span>Max graves per chunk: <b>{maxGraves.toLocaleString()}</b></span>
          <input type="range" min={100} max={100000} step={100}
            value={maxGraves} onChange={e => setMaxGraves(+e.target.value)} />
        </label>
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

          <div className="tess-scaling">
            Scaling factor: <b>{gy.scalingFactor(safePeriod).toFixed(4)}</b>
            <span className="tess-scaling-detail">
              &nbsp;= d / d₀ &nbsp;&middot;&nbsp;
              d = {gy.densityOfPeriod(safePeriod).toExponential(4)},
              d₀ = {gy.densityOfPeriod(0).toExponential(4)}
            </span>
            <br />
            <span className="tess-scaling-detail">
              center: {fmtCount(gy.gravesInPeriod(0))} graves / {gy.areaOfPeriod(0).toFixed(0)} area
              &nbsp;&middot;&nbsp;
              ring {safePeriod}: {fmtCount(gy.gravesInPeriod(safePeriod))} graves / {gy.areaOfPeriod(safePeriod).toFixed(0)} area
            </span>
          </div>

          <div className="tess-info-grid">
            <div className="tess-info-item">
              <span className="tess-label">Time window</span>
              <span className="tess-value">{fmtCalendarYear(selStats.yearsAgo + periodLength / 2)} &rarr; {fmtCalendarYear(selStats.yearsAgo - periodLength / 2)}</span>
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

      <h2 className="chart-title">Chunk ({safePeriod}, {safeChunk})</h2>
      <div className="chunk-solo-wrap">
        <ChunkSoloView gy={gy} chunkInfo={chunkInfo} />
      </div>
    </div>
  );
}
