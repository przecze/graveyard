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
// Focused chunk view — only selected chunk + direct neighbors
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
  const P = gy.periodLength;
  const yc = gy.ancientCircleRadius;

  // Build list of chunks to render: selected + all direct neighbors
  const allNeighborAddrs: ChunkAddr[] = [
    ...(neighbors.clockwise ? [neighbors.clockwise] : []),
    ...(neighbors.anticlockwise ? [neighbors.anticlockwise] : []),
    ...neighbors.inner.map(b => b.addr),
    ...neighbors.outer.map(b => b.addr),
  ];
  const toDraw: { addr: ChunkAddr; isSel: boolean }[] = [
    { addr: { period: selP, chunk: selC }, isSel: true },
    ...allNeighborAddrs.map(a => ({ addr: a, isSel: false })),
  ];

  // Center viewBox on the selected chunk
  const rMid = yc + (selP + 0.5) * P;
  const thetaMid = (chunkInfo.angleStart + chunkInfo.angleEnd) / 2;
  const viewCx = rMid * Math.cos(thetaMid);
  const viewCy = rMid * Math.sin(thetaMid);
  const span = P * 4;
  const half = span / 2;
  const vb = `${viewCx - half} ${viewCy - half} ${span} ${span}`;
  const sw = span * 0.004;

  return (
    <svg viewBox={vb} className="tess-svg" preserveAspectRatio="xMidYMid meet">
      {toDraw.map(({ addr, isSel }, i) => {
        const pi = addr.period;
        const ci = addr.chunk;
        const rI = yc + pi * P;
        const rO = yc + (pi + 1) * P;
        const Npi = gy.chunksInPeriod(pi);
        const piOff = gy.chunkOffset(pi);
        const astep = (2 * Math.PI) / Npi;
        let a1 = (ci + piOff) * astep;
        // Wrap angle near thetaMid to avoid drawing on the far side
        while (a1 - thetaMid > Math.PI) a1 -= 2 * Math.PI;
        while (thetaMid - a1 > Math.PI) a1 += 2 * Math.PI;
        const a2 = a1 + astep;
        const color = RING_COLORS[pi % RING_COLORS.length];

        return (
          <path key={i}
            d={sectorPath(0, 0, rI, rO, a1, a2)}
            fill={isSel ? '#fff' : color}
            fillOpacity={isSel ? 0.45 : 0.3}
            stroke={isSel ? '#fff' : '#fff'}
            strokeOpacity={isSel ? 1 : 0.5}
            strokeWidth={isSel ? sw * 2 : sw}
          />
        );
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
  const [maxGraves, setMaxGraves] = useState(100000);
  const [ancientCircleRadius, setAncientCircleRadius] = useState(20000);
  const [selectedPeriod, setSelectedPeriod] = useState(50);
  const [selectedChunk, setSelectedChunk] = useState(0);

  const gy = useMemo(
    () => new Graveyard(periodLength, maxGraves, ancientCircleRadius),
    [periodLength, maxGraves, ancientCircleRadius],
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

      <div className="chart-wrap">
        <PopulationChart yc={ancientCircleRadius} onYcChange={setAncientCircleRadius} />
      </div>

      <div className="controls">
        <label className="ctrl">
          <span>Period length: <b>{periodLength} years</b></span>
          <input type="range" min={25} max={500} step={5}
            value={periodLength} onChange={e => setPeriodLength(+e.target.value)} />
        </label>
        <label className="ctrl">
          <span>Max graves per chunk: <b>{maxGraves.toLocaleString()}</b></span>
          <input type="range" min={0} max={1000} step={1}
            value={Math.round(Math.log(maxGraves / 1000) / Math.log(5000) * 1000)}
            onChange={e => setMaxGraves(Math.round(1000 * Math.pow(5000, +e.target.value / 1000)))} />
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
