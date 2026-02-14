import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import './App.css';

// ----------------------------------------------------------------
// Types
// ----------------------------------------------------------------
interface Point { x: number; y: number; r: number }
interface BinData { counts: number[]; areas: number[]; densities: number[]; dr: number }
interface TessRing {
  rInner: number;
  rOuter: number;
  numSlices: number;
  sliceCounts: number[];
}

// ----------------------------------------------------------------
// Constants
// ----------------------------------------------------------------
const S = 480;
const HALF = S / 2;
const MARGIN = 12;
const SCALE = HALF - MARGIN;
const VP_W = 480;
const VP_H = 360;
const END_YEAR = 2026;

// ----------------------------------------------------------------
// Seeded PRNG (mulberry32)
// ----------------------------------------------------------------
function makePRNG(seed: number) {
  let s = seed | 0;
  return () => {
    s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ----------------------------------------------------------------
// Model:  f(t) = A · (e^(αt) − 1),   t ∈ [0, T]
//
//   Constraints (3 equations, 2 unknowns → unique):
//     f(0) = 0                        (by construction)
//     ∫₀ᵀ f(t) dt  = N_tot           (total graves)
//     f(T) · δ      = N_curr          (deaths in last δ years)
//
//   Eliminating A:
//     N_curr · [1/α − T/(e^(αT)−1)] = N_tot · δ
//   → bisect for α, then A = N_curr / (δ·(e^(αT)−1))
// ----------------------------------------------------------------
function solveModel(
  nTot: number, nCurr: number, T: number, delta: number,
): { alpha: number; A: number; beta: number } {
  const target = nTot * delta;

  const h = (a: number): number => {
    if (a < 1e-12) return nCurr * T / 2 - target;
    const aT = a * T;
    if (aT > 600) return nCurr / a - target;
    const eaT = Math.exp(aT);
    return nCurr * (1 / a - T / (eaT - 1)) - target;
  };

  const h0 = h(0);
  if (h0 <= 1e-6) {
    const alpha = 1e-10;
    return { alpha, A: nCurr / (delta * alpha * T), beta: alpha * T };
  }

  let lo = 1e-12, hi = 0.01;
  while (h(hi) > 0 && hi < 10) hi *= 2;

  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (h(mid) > 0) lo = mid; else hi = mid;
  }

  const alpha = (lo + hi) / 2;
  const beta = alpha * T;
  const eaT = Math.exp(Math.min(beta, 600));
  const A = nCurr / (delta * Math.max(eaT - 1, 1e-20));

  return { alpha, A, beta };
}

// ----------------------------------------------------------------
// Expected counts per bin from the model
// ----------------------------------------------------------------
function modelExpectedCounts(
  A: number, alpha: number, T: number, numBins: number,
): number[] {
  const dt = T / numBins;
  if (alpha < 1e-12 || A < 1e-20) return new Array(numBins).fill(0);
  return Array.from({ length: numBins }, (_, i) => {
    const t1 = i * dt;
    const t2 = (i + 1) * dt;
    return Math.max(0,
      A * ((Math.exp(alpha * t2) - Math.exp(alpha * t1)) / alpha - dt),
    );
  });
}

// ----------------------------------------------------------------
// Point generation via CDF table for g(r) ∝ e^(βr) − 1
// ----------------------------------------------------------------
function generatePoints(n: number, beta: number, seed: number): Point[] {
  const rand = makePRNG(seed);
  const pts: Point[] = new Array(n);

  if (beta < 0.1) {
    // g(r) ≈ βr → CDF ∝ r² → r = √u
    for (let i = 0; i < n; i++) {
      const r = Math.sqrt(rand());
      const θ = rand() * 2 * Math.PI;
      pts[i] = { x: r * Math.cos(θ), y: r * Math.sin(θ), r };
    }
    return pts;
  }

  // Build CDF lookup table
  const TBL = 4000;
  const cdf = new Float64Array(TBL + 1);
  const step = 1 / TBL;
  for (let i = 0; i <= TBL; i++) {
    const r = i * step;
    cdf[i] = (Math.exp(beta * r) - 1) / beta - r;
  }
  const total = cdf[TBL];
  for (let i = 0; i <= TBL; i++) cdf[i] /= total;

  for (let i = 0; i < n; i++) {
    const u = rand();
    let lo = 0, hi = TBL;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid] < u) lo = mid + 1; else hi = mid;
    }
    let r: number;
    if (lo === 0) {
      r = 0;
    } else {
      const c0 = cdf[lo - 1];
      const c1 = cdf[lo];
      const frac = c1 > c0 ? (u - c0) / (c1 - c0) : 0;
      r = Math.max(0, Math.min(1, ((lo - 1) + frac) * step));
    }
    const θ = rand() * 2 * Math.PI;
    pts[i] = { x: r * Math.cos(θ), y: r * Math.sin(θ), r };
  }
  return pts;
}

// ----------------------------------------------------------------
// Bin points into rings
// ----------------------------------------------------------------
function computeBins(points: Point[], numBins: number): BinData {
  const dr = 1 / numBins;
  const counts = new Array(numBins).fill(0);
  for (const p of points) {
    const bin = Math.min(Math.floor(p.r / dr), numBins - 1);
    counts[bin]++;
  }
  const areas = Array.from({ length: numBins }, (_, i) => {
    const ri = i * dr;
    const ro = (i + 1) * dr;
    return Math.PI * (ro * ro - ri * ri);
  });
  const densities = counts.map((c, i) => (areas[i] > 0 ? c / areas[i] : 0));
  return { counts, areas, densities, dr };
}

// ----------------------------------------------------------------
// Tessellation:
//   - ring width = ringYears mapped to r-space (dr = ringYears / T)
//   - first cell is a circle of radius dr/2
//   - subsequent rings of width dr
//   - only slice a ring if its count > maxPts
// ----------------------------------------------------------------
function computeTessellation(
  points: Point[], ringYears: number, maxPts: number, T: number,
): TessRing[] {
  const dr = ringYears / T;          // ring width in r-space [0,1]
  const r0 = dr / 2;                 // first circle radius = half a ring width

  // Build ring boundaries: [0, r0], [r0, r0+dr], [r0+2dr, ...], ... up to 1
  const edges: number[] = [0, Math.min(r0, 1)];
  let r = r0;
  while (r < 1) {
    r = Math.min(r + dr, 1);
    edges.push(r);
  }
  const numRings = edges.length - 1;

  // Count per ring
  const ringCounts = new Array(numRings).fill(0);
  for (const p of points) {
    // First ring [0, r0), then uniform dr after that
    let bin: number;
    if (p.r < edges[1]) {
      bin = 0;
    } else {
      bin = Math.min(
        1 + Math.floor((p.r - edges[1]) / dr),
        numRings - 1,
      );
    }
    ringCounts[bin]++;
  }

  // Build tessellation rings
  const rings: TessRing[] = [];
  for (let i = 0; i < numRings; i++) {
    const count = ringCounts[i];
    const numSlices = count > maxPts
      ? Math.max(2, Math.round(count / maxPts))
      : 1;
    rings.push({
      rInner: edges[i],
      rOuter: edges[i + 1],
      numSlices,
      sliceCounts: new Array(numSlices).fill(0),
    });
  }

  // Assign each point to its cell
  for (const p of points) {
    let bin: number;
    if (p.r < edges[1]) {
      bin = 0;
    } else {
      bin = Math.min(
        1 + Math.floor((p.r - edges[1]) / dr),
        numRings - 1,
      );
    }
    const ring = rings[bin];
    if (ring.numSlices === 1) {
      ring.sliceCounts[0]++;
    } else {
      let θ = Math.atan2(p.y, p.x);
      if (θ < 0) θ += 2 * Math.PI;
      const slice = Math.min(
        Math.floor(θ / (2 * Math.PI) * ring.numSlices),
        ring.numSlices - 1,
      );
      ring.sliceCounts[slice]++;
    }
  }

  return rings;
}

// ----------------------------------------------------------------
// Heatmap color: count → HSL  (blue=low → green → yellow → red=high)
// ----------------------------------------------------------------
function heatColor(count: number, maxCount: number, alpha: number): string {
  if (count === 0) return `rgba(0,0,0,${alpha})`;
  const t = Math.min(count / maxCount, 1);      // 0..1
  // hue: 240 (blue) → 0 (red)
  const h = (1 - t) * 240;
  const s = 80 + t * 20;
  const l = 25 + t * 30;
  return `hsla(${h},${s}%,${l}%,${alpha})`;
}

// ----------------------------------------------------------------
// Draw tessellation: heatmap fills + grid lines
// ----------------------------------------------------------------
function drawTessellation(canvas: HTMLCanvasElement, tess: TessRing[]) {
  const ctx = canvas.getContext('2d')!;

  // Find max count across all cells for color scale
  let maxCount = 1;
  for (const ring of tess) {
    for (const c of ring.sliceCounts) {
      if (c > maxCount) maxCount = c;
    }
  }

  // --- Fill cells with heatmap ---
  for (const ring of tess) {
    const r1 = ring.rInner * SCALE;
    const r2 = ring.rOuter * SCALE;
    const sliceAngle = (2 * Math.PI) / ring.numSlices;

    for (let s = 0; s < ring.numSlices; s++) {
      const θ0 = s * sliceAngle;
      const θ1 = θ0 + sliceAngle;

      ctx.fillStyle = heatColor(ring.sliceCounts[s], maxCount, 0.45);
      ctx.beginPath();
      if (ring.rInner === 0) {
        // Central circle — just an arc from 0
        ctx.moveTo(HALF, HALF);
        ctx.arc(HALF, HALF, r2, -θ1, -θ0);
        ctx.closePath();
      } else {
        // Annular wedge
        ctx.arc(HALF, HALF, r2, -θ1, -θ0);
        ctx.arc(HALF, HALF, r1, -θ0, -θ1, true);
        ctx.closePath();
      }
      ctx.fill();
    }
  }

  // --- Grid lines on top ---
  // Ring boundaries
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
  ctx.lineWidth = 0.5;
  for (const ring of tess) {
    const rPx = ring.rOuter * SCALE;
    ctx.beginPath();
    ctx.arc(HALF, HALF, rPx, 0, 2 * Math.PI);
    ctx.stroke();
  }

  // Angular divisions
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  for (const ring of tess) {
    if (ring.numSlices <= 1) continue;
    const r1 = ring.rInner * SCALE;
    const r2 = ring.rOuter * SCALE;
    for (let s = 0; s < ring.numSlices; s++) {
      const θ = (s / ring.numSlices) * 2 * Math.PI;
      const c = Math.cos(θ);
      const sn = Math.sin(θ);
      ctx.moveTo(HALF + r1 * c, HALF - r1 * sn);
      ctx.lineTo(HALF + r2 * c, HALF - r2 * sn);
    }
  }
  ctx.stroke();
}

// ----------------------------------------------------------------
// Hit-test: pixel → cell
// ----------------------------------------------------------------
function findCell(
  tess: TessRing[], wx: number, wy: number,
): { ringIdx: number; sliceIdx: number } | null {
  const r = Math.sqrt(wx * wx + wy * wy);
  if (r > 1 || tess.length === 0) return null;

  let ringIdx = -1;
  for (let i = 0; i < tess.length; i++) {
    if (r < tess[i].rOuter || i === tess.length - 1) { ringIdx = i; break; }
  }
  if (ringIdx < 0) return null;

  const ring = tess[ringIdx];
  let θ = Math.atan2(wy, wx);
  if (θ < 0) θ += 2 * Math.PI;
  const sliceIdx = Math.min(
    Math.floor(θ / (2 * Math.PI) * ring.numSlices),
    ring.numSlices - 1,
  );
  return { ringIdx, sliceIdx };
}

// ----------------------------------------------------------------
// Draw ring histogram: bar chart of sliceCounts for one ring
// ----------------------------------------------------------------
function drawRingHistogram(
  canvas: HTMLCanvasElement,
  tess: TessRing[],
  ringIdx: number,
  maxPts: number,
  startYear: number,
  T: number,
  selectedSlice: number | null,
) {
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width;
  const H = canvas.height;
  const pad = { t: 28, r: 12, b: 22, l: 44 };
  const pW = W - pad.l - pad.r;
  const pH = H - pad.t - pad.b;

  ctx.fillStyle = '#141414';
  ctx.fillRect(0, 0, W, H);

  if (ringIdx < 0 || ringIdx >= tess.length) return;

  const ring = tess[ringIdx];
  const data = ring.sliceCounts;
  const n = data.length;
  const maxVal = Math.max(...data, maxPts, 1);
  const barW = pW / n;

  // Find global max for heatmap scale
  let globalMax = 1;
  for (const r of tess) for (const c of r.sliceCounts) if (c > globalMax) globalMax = c;

  // Grid
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 0.5;
  for (let i = 1; i <= 4; i++) {
    const y = pad.t + pH * (1 - i / 4);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pW, y); ctx.stroke();
  }

  // maxPts threshold line
  const threshY = pad.t + pH * (1 - maxPts / maxVal);
  ctx.strokeStyle = '#ff6b6b55';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(pad.l, threshY); ctx.lineTo(pad.l + pW, threshY); ctx.stroke();
  ctx.setLineDash([]);

  // Bars
  for (let i = 0; i < n; i++) {
    const barH = (data[i] / maxVal) * pH;
    const x = pad.l + i * barW;
    const y = pad.t + pH - barH;

    ctx.fillStyle = heatColor(data[i], globalMax, 0.7);
    ctx.fillRect(x + 0.5, y, barW - 1, barH);

    // Highlight selected slice
    if (i === selectedSlice) {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x + 0.5, y, barW - 1, barH);
    }
  }

  // Axes
  ctx.strokeStyle = '#444';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + pH); ctx.lineTo(pad.l + pW, pad.t + pH);
  ctx.stroke();

  // Title
  const y0 = Math.round(startYear + ring.rInner * T);
  const y1 = Math.round(startYear + ring.rOuter * T);
  ctx.fillStyle = '#ff8c42';
  ctx.font = 'bold 11px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(
    `Ring ${ringIdx}: ${fmtYear(y0)}→${fmtYear(y1)}  (${n} slices, ${ring.sliceCounts.reduce((a, b) => a + b, 0)} pts)`,
    pad.l, 16,
  );

  // Y labels
  ctx.fillStyle = '#666';
  ctx.font = '10px monospace';
  ctx.textAlign = 'right';
  ctx.fillText(fmt(maxVal), pad.l - 4, pad.t + 10);
  ctx.fillText('0', pad.l - 4, pad.t + pH + 3);

  // X labels
  ctx.textAlign = 'center';
  ctx.fillText('slice 0', pad.l + barW / 2, H - 5);
  ctx.fillText(String(n - 1), pad.l + pW - barW / 2, H - 5);

  // maxPts label
  ctx.fillStyle = '#ff6b6b88';
  ctx.textAlign = 'left';
  ctx.fillText(`max=${maxPts}`, pad.l + 4, threshY - 4);
}

// ----------------------------------------------------------------
// Formatting
// ----------------------------------------------------------------
function fmt(n: number): string {
  if (n === 0) return '0';
  const a = Math.abs(n);
  if (a >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (a >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  if (a >= 100) return n.toFixed(0);
  if (a >= 1) return n.toFixed(1);
  if (a >= 0.01) return n.toFixed(2);
  return n.toExponential(1);
}

function fmtYear(y: number): string {
  const abs = Math.abs(y);
  const s = abs >= 1000 ? (abs / 1000).toFixed(0) + 'k' : String(abs);
  return y <= 0 ? s + ' BC' : String(y);
}

// ----------------------------------------------------------------
// Combined chart: deaths per period (left axis) + density (right axis)
// with model fit overlay on deaths
// ----------------------------------------------------------------
interface Series {
  data: number[];
  color: string;
  label: string;
  fill?: boolean;
}

function drawCombinedChart(
  canvas: HTMLCanvasElement,
  left: Series,                    // deaths per period (sampled)
  right: Series,                   // grave density
  modelOverlay: { data: number[]; label: string; color: string },
  xLabels: { start: string; end: string },
) {
  const ctx = canvas.getContext('2d')!;
  const W = canvas.width;
  const H = canvas.height;
  const pad = { t: 28, r: 52, b: 22, l: 52 };  // room for right axis
  const pW = W - pad.l - pad.r;
  const pH = H - pad.t - pad.b;
  const n = left.data.length;
  const barW = pW / n;

  // background
  ctx.fillStyle = '#141414';
  ctx.fillRect(0, 0, W, H);

  const maxL = Math.max(...left.data, ...modelOverlay.data, 1e-10);
  const maxR = Math.max(...right.data, 1e-10);

  // grid
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 0.5;
  for (let i = 1; i <= 4; i++) {
    const y = pad.t + pH * (1 - i / 4);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pW, y); ctx.stroke();
  }

  // --- Left series: deaths (filled area + line) ---
  ctx.fillStyle = left.color + '22';
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t + pH);
  for (let i = 0; i < n; i++) {
    ctx.lineTo(pad.l + (i + 0.5) * barW, pad.t + pH * (1 - left.data[i] / maxL));
  }
  ctx.lineTo(pad.l + pW, pad.t + pH);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = left.color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = pad.l + (i + 0.5) * barW;
    const y = pad.t + pH * (1 - left.data[i] / maxL);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // --- Model overlay (dashed, same scale as left) ---
  ctx.strokeStyle = modelOverlay.color;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  for (let i = 0; i < modelOverlay.data.length; i++) {
    const x = pad.l + (i + 0.5) * barW;
    const y = pad.t + pH * (1 - modelOverlay.data[i] / maxL);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // --- Right series: density (line only, own scale) ---
  ctx.strokeStyle = right.color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = pad.l + (i + 0.5) * barW;
    const y = pad.t + pH * (1 - right.data[i] / maxR);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // --- Axes ---
  ctx.strokeStyle = '#444';
  ctx.lineWidth = 1;
  ctx.beginPath();
  // left axis
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + pH);
  // bottom
  ctx.lineTo(pad.l + pW, pad.t + pH);
  // right axis
  ctx.lineTo(pad.l + pW, pad.t);
  ctx.stroke();

  // --- Labels ---
  // left axis labels (deaths)
  ctx.fillStyle = left.color;
  ctx.font = '10px monospace';
  ctx.textAlign = 'right';
  ctx.fillText(fmt(maxL), pad.l - 4, pad.t + 10);
  ctx.fillText('0', pad.l - 4, pad.t + pH + 3);

  // right axis labels (density)
  ctx.fillStyle = right.color;
  ctx.textAlign = 'left';
  ctx.fillText(fmt(maxR), pad.l + pW + 4, pad.t + 10);
  ctx.fillText('0', pad.l + pW + 4, pad.t + pH + 3);

  // title / legend
  ctx.font = 'bold 11px monospace';
  ctx.textAlign = 'left';
  ctx.fillStyle = left.color;
  ctx.fillText(left.label, pad.l, 16);
  const leftW = ctx.measureText(left.label).width;

  ctx.fillStyle = '#444';
  ctx.fillText(' | ', pad.l + leftW, 16);
  const sepW = ctx.measureText(' | ').width;

  ctx.fillStyle = right.color;
  ctx.fillText(right.label, pad.l + leftW + sepW, 16);

  // model legend (right-aligned)
  ctx.fillStyle = modelOverlay.color;
  ctx.font = '10px monospace';
  ctx.textAlign = 'right';
  ctx.fillText(modelOverlay.label, pad.l + pW, 16);

  // x-axis
  ctx.fillStyle = '#666';
  ctx.font = '10px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(xLabels.start, pad.l, H - 5);
  ctx.textAlign = 'right';
  ctx.fillText(xLabels.end, pad.l + pW, H - 5);
}

// ----------------------------------------------------------------
// Cloud drawing
// ----------------------------------------------------------------
function drawCloud(canvas: HTMLCanvasElement, points: Point[]) {
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#0a0a0a';
  ctx.fillRect(0, 0, S, S);

  ctx.strokeStyle = '#1c1c1c';
  ctx.lineWidth = 0.5;
  ctx.fillStyle = '#2a2a2a';
  ctx.font = '10px monospace';
  ctx.textAlign = 'left';
  for (const frac of [0.25, 0.5, 0.75, 1.0]) {
    const rad = frac * SCALE;
    ctx.beginPath(); ctx.arc(HALF, HALF, rad, 0, 2 * Math.PI); ctx.stroke();
    ctx.fillText(frac.toFixed(2), HALF + 3, HALF - rad + 12);
  }

  ctx.fillStyle = 'rgba(220, 230, 240, 0.65)';
  for (const p of points) {
    ctx.fillRect(HALF + p.x * SCALE - 0.5, HALF - p.y * SCALE - 0.5, 1.5, 1.5);
  }
}

// ================================================================
// Main component
// ================================================================
export default function App() {
  // --- State ---
  // nTotLog: log-scale slider value (exponent), range log10(100k)=5 to log10(1M)=6
  const [nTotLog, setNTotLog] = useState(Math.log10(200000));
  const nTot = Math.round(Math.pow(10, nTotLog));
  const [alivePct, setAlivePct] = useState(5);     // % of nTot currently alive
  const [startYear, setStartYear] = useState(-20000);
  const [delta, setDelta] = useState(80);
  const [numBins, setNumBins] = useState(100);
  const [ringYears, setRingYears] = useState(500);
  const [maxPts, setMaxPts] = useState(200);
  const [seed, setSeed] = useState(42);
  const [vpSize, setVpSize] = useState(0.20);
  const [inspectRing, setInspectRing] = useState(0);
  const [selectedCell, setSelectedCell] = useState<{ ringIdx: number; sliceIdx: number } | null>(null);

  // nCurr = people alive now = alivePct% of nTot, dying over δ years
  const nCurr = Math.round(nTot * alivePct / 100);

  // --- Refs ---
  const cloudRef = useRef<HTMLCanvasElement>(null);
  const vpRef = useRef<HTMLCanvasElement>(null);
  const offRef = useRef<HTMLCanvasElement | null>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);
  const histRef = useRef<HTMLCanvasElement>(null);
  const vpSizeRef = useRef(vpSize);
  vpSizeRef.current = vpSize;
  const lastMouseRef = useRef<{ wx: number; wy: number } | null>(null);
  const startYearRef = useRef(startYear);
  startYearRef.current = startYear;

  // --- Derived model ---
  const T = END_YEAR - startYear;
  const model = useMemo(
    () => solveModel(nTot, nCurr, T, delta),
    [nTot, nCurr, T, delta],
  );

  const points = useMemo(
    () => generatePoints(nTot, model.beta, seed),
    [nTot, model.beta, seed],
  );
  const pointsRef = useRef(points);
  pointsRef.current = points;

  const bins = useMemo(() => computeBins(points, numBins), [points, numBins]);

  const tess = useMemo(
    () => computeTessellation(points, ringYears, maxPts, T),
    [points, ringYears, maxPts, T],
  );

  const fitCounts = useMemo(
    () => modelExpectedCounts(model.A, model.alpha, T, numBins),
    [model.A, model.alpha, T, numBins],
  );

  // --- Draw cloud + tessellation to offscreen ---
  useEffect(() => {
    if (!offRef.current) {
      offRef.current = document.createElement('canvas');
      offRef.current.width = S;
      offRef.current.height = S;
    }
    drawCloud(offRef.current, points);
    drawTessellation(offRef.current, tess);
    const ctx = cloudRef.current?.getContext('2d');
    if (ctx) ctx.drawImage(offRef.current, 0, 0);
  }, [points, tess]);

  // --- Draw chart ---
  useEffect(() => {
    const c = chartRef.current;
    if (!c) return;
    drawCombinedChart(
      c,
      { data: bins.counts, color: '#4ecdc4', label: 'deaths/period' },
      { data: bins.densities, color: '#c678dd', label: 'density' },
      { data: fitCounts, color: '#ffe066', label: `model α=${model.alpha.toExponential(2)}` },
      { start: fmtYear(startYear), end: String(END_YEAR) },
    );
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bins, fitCounts, model.alpha, startYear]);

  // --- Draw ring histogram ---
  useEffect(() => {
    const c = histRef.current;
    if (!c || tess.length === 0) return;
    const ri = Math.min(inspectRing, tess.length - 1);
    drawRingHistogram(c, tess, ri, maxPts, startYear, T, selectedCell?.ringIdx === ri ? selectedCell.sliceIdx : null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tess, inspectRing, maxPts, startYear, T, selectedCell]);

  // Clamp inspectRing when tess changes
  useEffect(() => {
    if (inspectRing >= tess.length) setInspectRing(Math.max(0, tess.length - 1));
  }, [tess.length, inspectRing]);

  // --- Cloud click → select cell ---
  const handleCloudClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = cloudRef.current!;
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (S / rect.width);
    const py = (e.clientY - rect.top) * (S / rect.height);
    const wx = (px - HALF) / SCALE;
    const wy = (HALF - py) / SCALE;
    const cell = findCell(tess, wx, wy);
    if (cell) {
      setSelectedCell(cell);
      setInspectRing(cell.ringIdx);
    } else {
      setSelectedCell(null);
    }
  }, [tess]);

  // --- Imperative overlay + viewport ---
  const renderOverlay = useCallback((wx: number, wy: number) => {
    const ctx = cloudRef.current?.getContext('2d');
    if (!ctx || !offRef.current) return;
    ctx.drawImage(offRef.current, 0, 0);

    const vw = vpSizeRef.current;
    const vh = vw * 3 / 4;
    const rx = HALF + (wx - vw / 2) * SCALE;
    const ry = HALF - (wy + vh / 2) * SCALE;

    ctx.strokeStyle = '#4ecdc4';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(rx, ry, vw * SCALE, vh * SCALE);
    ctx.setLineDash([]);

    const cpx = HALF + wx * SCALE;
    const cpy = HALF - wy * SCALE;
    ctx.strokeStyle = 'rgba(78, 205, 196, 0.35)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(cpx - 8, cpy); ctx.lineTo(cpx + 8, cpy);
    ctx.moveTo(cpx, cpy - 8); ctx.lineTo(cpx, cpy + 8);
    ctx.stroke();
  }, []);

  const renderViewport = useCallback((wx: number, wy: number) => {
    const vpCtx = vpRef.current?.getContext('2d');
    if (!vpCtx) return;

    const vw = vpSizeRef.current;
    const vh = vw * 3 / 4;
    const x1 = wx - vw / 2, x2 = wx + vw / 2;
    const y1 = wy - vh / 2, y2 = wy + vh / 2;

    vpCtx.fillStyle = '#0a0a0a';
    vpCtx.fillRect(0, 0, VP_W, VP_H);

    const pts = pointsRef.current;
    let count = 0;
    const dotR = Math.max(1.5, Math.min(6, 0.006 / vw * VP_W));

    vpCtx.fillStyle = 'rgba(220, 230, 240, 0.8)';
    for (const p of pts) {
      if (p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2) {
        const vx = ((p.x - x1) / vw) * VP_W;
        const vy = ((y2 - p.y) / vh) * VP_H;
        vpCtx.beginPath();
        vpCtx.arc(vx, vy, dotR, 0, 2 * Math.PI);
        vpCtx.fill();
        count++;
      }
    }

    // Info bar
    const r = Math.sqrt(wx * wx + wy * wy);
    const sy = startYearRef.current;
    const span = END_YEAR - sy;
    const year = Math.round(sy + r * span);

    vpCtx.fillStyle = '#0a0a0aCC';
    vpCtx.fillRect(0, VP_H - 24, VP_W, 24);
    vpCtx.fillStyle = '#4ecdc4';
    vpCtx.font = '11px monospace';
    vpCtx.textAlign = 'left';
    vpCtx.fillText(
      `${count} graves  ·  ~${fmtYear(year)} (r=${r.toFixed(2)})  ·  view ${vw.toFixed(2)}`,
      8, VP_H - 8,
    );
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = cloudRef.current!;
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (S / rect.width);
    const py = (e.clientY - rect.top) * (S / rect.height);
    const wx = (px - HALF) / SCALE;
    const wy = (HALF - py) / SCALE;
    lastMouseRef.current = { wx, wy };
    renderOverlay(wx, wy);
    renderViewport(wx, wy);
  }, [renderOverlay, renderViewport]);

  const clearViewport = useCallback(() => {
    const vpCtx = vpRef.current?.getContext('2d');
    if (vpCtx) {
      vpCtx.fillStyle = '#0a0a0a';
      vpCtx.fillRect(0, 0, VP_W, VP_H);
      vpCtx.fillStyle = '#333';
      vpCtx.font = '12px monospace';
      vpCtx.textAlign = 'center';
      vpCtx.fillText('hover over the cloud above', VP_W / 2, VP_H / 2);
    }
  }, []);

  const handleMouseLeave = useCallback(() => {
    lastMouseRef.current = null;
    const ctx = cloudRef.current?.getContext('2d');
    if (ctx && offRef.current) ctx.drawImage(offRef.current, 0, 0);
    clearViewport();
  }, [clearViewport]);

  // Refresh overlay when vpSize slider moves
  useEffect(() => {
    const m = lastMouseRef.current;
    if (m) { renderOverlay(m.wx, m.wy); renderViewport(m.wx, m.wy); }
  }, [vpSize, renderOverlay, renderViewport]);

  // Initial viewport prompt
  useEffect(() => { clearViewport(); }, [clearViewport]);

  // --- Render ---
  return (
    <div className="app">
      <h1 className="title">Graveyard — Density Explorer</h1>
      <p className="subtitle">
        f(t) = A·(e<sup>αt</sup> − 1) &nbsp;|&nbsp;
        α = {model.alpha.toExponential(3)} &nbsp;|&nbsp;
        N<sub>curr</sub> = {nCurr.toLocaleString()} &nbsp;|&nbsp;
        {fmtYear(startYear)} → {END_YEAR} &nbsp;({T.toLocaleString()} yrs)
      </p>

      <div className="controls">
        <label className="ctrl">
          <span>N<sub>tot</sub>: <b>{nTot.toLocaleString()}</b></span>
          <input type="range" min={5} max={6} step={0.01}
            value={nTotLog} onChange={e => setNTotLog(+e.target.value)} />
        </label>
        <label className="ctrl">
          <span>Alive: <b>{alivePct}%</b> ({nCurr.toLocaleString()})</span>
          <input type="range" min={1} max={30} step={0.5}
            value={alivePct} onChange={e => setAlivePct(+e.target.value)} />
        </label>
        <label className="ctrl">
          <span>δ (lifespan): <b>{delta}</b> yrs</span>
          <input type="range" min={20} max={200} step={5}
            value={delta} onChange={e => setDelta(+e.target.value)} />
        </label>
        <label className="ctrl">
          <span>Start: <b>{fmtYear(startYear)}</b></span>
          <input type="range" min={-200000} max={-1000} step={1000}
            value={startYear} onChange={e => setStartYear(+e.target.value)} />
        </label>
        <label className="ctrl">
          <span>Bins: <b>{numBins}</b></span>
          <input type="range" min={5} max={500} step={1}
            value={numBins} onChange={e => setNumBins(+e.target.value)} />
        </label>
        <label className="ctrl">
          <span>Ring: <b>{ringYears}</b> yrs ({tess.length} rings)</span>
          <input type="range" min={50} max={2000} step={10}
            value={ringYears} onChange={e => setRingYears(+e.target.value)} />
        </label>
        <label className="ctrl">
          <span>Max/cell: <b>{maxPts}</b> ({tess.reduce((s, r) => s + r.numSlices, 0)} cells)</span>
          <input type="range" min={5} max={2000} step={5}
            value={maxPts} onChange={e => setMaxPts(+e.target.value)} />
        </label>
        <label className="ctrl">
          <span>Viewport: <b>{vpSize.toFixed(2)}</b></span>
          <input type="range" min={0.03} max={0.5} step={0.01}
            value={vpSize} onChange={e => setVpSize(+e.target.value)} />
        </label>
        <button className="btn" onClick={() => setSeed(s => s + 1)}>
          Regenerate
        </button>
      </div>

      <div className="main">
        <div className="left-col">
          <canvas
            ref={cloudRef} className="cloud" width={S} height={S}
            onMouseMove={handleMouseMove} onMouseLeave={handleMouseLeave}
            onClick={handleCloudClick}
          />
          <canvas ref={vpRef} className="viewport" width={VP_W} height={VP_H} />
        </div>
        <div className="charts">
          <canvas ref={chartRef} width={420} height={280} />

          <div className="ring-inspect">
            <label className="ctrl">
              <span>Inspect ring: <b>{Math.min(inspectRing, tess.length - 1)}</b> / {tess.length - 1}</span>
              <input type="range" min={0} max={Math.max(0, tess.length - 1)} step={1}
                value={Math.min(inspectRing, tess.length - 1)}
                onChange={e => { setInspectRing(+e.target.value); setSelectedCell(null); }} />
            </label>
          </div>
          <canvas ref={histRef} width={420} height={180} />

          {selectedCell && tess[selectedCell.ringIdx] && (() => {
            const ring = tess[selectedCell.ringIdx];
            const count = ring.sliceCounts[selectedCell.sliceIdx];
            const y0 = Math.round(startYear + ring.rInner * T);
            const y1 = Math.round(startYear + ring.rOuter * T);
            return (
              <div className="cell-info">
                <b>Selected cell</b>
                &nbsp;Ring {selectedCell.ringIdx} ({fmtYear(y0)} → {fmtYear(y1)})
                &nbsp;·&nbsp;Slice {selectedCell.sliceIdx}/{ring.numSlices}
                &nbsp;·&nbsp;<b>{count.toLocaleString()}</b> graves
              </div>
            );
          })()}
        </div>
      </div>
    </div>
  );
}
