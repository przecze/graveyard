import { useEffect, useRef, useState } from 'react';
import AboutDialog from './AboutDialog';

type PolarPos = { r: number; phi: number };
type PolarVel = { vr: number; vphi: number };

// Fixed world-space scale: 1 year = PIX_PER_YEAR world pixels
const PIX_PER_YEAR = 100;
const INNER_RADIUS = 1000; // years — empty central zone

const DEFAULT_GRAVE_DENSITY = 5; // graves / yr²
const MIN_GRAVE_DENSITY = 1;
const MAX_GRAVE_DENSITY = 200;

const DEFAULT_CHUNK_RADIAL_LEN = 10; // years
const MIN_CHUNK_RADIAL_LEN = 1;
const MAX_CHUNK_RADIAL_LEN = 100;

const DEFAULT_MAX_GRAVES = 1000;
const MIN_MAX_GRAVES = 100;
const MAX_MAX_GRAVES = 10000;

const TRAVEL_DISTANCE_YEARS = 16000;
const DEFAULT_TRAVEL_TIME = 3.5 * 60 * 60;
const MIN_TRAVEL_TIME = 30;
const MAX_TRAVEL_TIME = 5 * 60 * 60;

const DEFAULT_VIEWPORT_YEARS = 50;
const MIN_VIEWPORT_YEARS = 1;
const MAX_VIEWPORT_YEARS = 10000;

const ACCELERATION = 7;
const DAMPING = 9;

// 4×2 block layout: gap between blocks = 0.5 * s
// effective area per grave = (4.5s/4) * (2.5s/2) = 1.40625 * s²
const BLOCK_GAP_FACTOR = 0.5;
const BLOCK_COLS = 4;
const BLOCK_ROWS = 2;
const BLOCK_OVERHEAD = ((BLOCK_COLS + BLOCK_GAP_FACTOR) / BLOCK_COLS) *
                       ((BLOCK_ROWS + BLOCK_GAP_FACTOR) / BLOCK_ROWS);

const VIEWPORT_LOG_MIN = Math.log10(MIN_VIEWPORT_YEARS);
const VIEWPORT_LOG_MAX = Math.log10(MAX_VIEWPORT_YEARS);
const viewportToSlider = (v: number) =>
  ((Math.log10(v) - VIEWPORT_LOG_MIN) / (VIEWPORT_LOG_MAX - VIEWPORT_LOG_MIN)) * 1000;
const sliderToViewport = (v: number) =>
  Math.pow(10, VIEWPORT_LOG_MIN + (v / 1000) * (VIEWPORT_LOG_MAX - VIEWPORT_LOG_MIN));

function fmtYears(y: number): string {
  if (y >= 1000) return `${Math.round(y).toLocaleString()} yr`;
  if (y >= 10) return `${Math.round(y)} yr`;
  return `${y.toFixed(1)} yr`;
}

function formatDuration(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) return 'n/a';
  if (totalSeconds < 60) return `${totalSeconds.toFixed(1)}s`;
  if (totalSeconds < 3600) return `${(totalSeconds / 60).toFixed(1)}m`;
  if (totalSeconds < 86400) return `${(totalSeconds / 3600).toFixed(1)}h`;
  if (totalSeconds < 31557600) return `${(totalSeconds / 86400).toFixed(1)}d`;
  return `${(totalSeconds / 31557600).toFixed(2)}y`;
}

function calcNumChunks(density: number, chunkL: number, maxGraves: number): number {
  const ringArea = Math.PI * ((INNER_RADIUS + chunkL) ** 2 - INNER_RADIUS ** 2);
  return Math.max(1, Math.ceil((density * ringArea) / maxGraves));
}

export default function VisitorView() {
  const [travelTimeSeconds, setTravelTimeSeconds] = useState(DEFAULT_TRAVEL_TIME);
  const [graveDensity, setGraveDensity] = useState(DEFAULT_GRAVE_DENSITY);
  const [viewportWidthYears, setViewportWidthYears] = useState(DEFAULT_VIEWPORT_YEARS);
  const [chunkRadialLen, setChunkRadialLen] = useState(DEFAULT_CHUNK_RADIAL_LEN);
  const [maxGraves, setMaxGraves] = useState(DEFAULT_MAX_GRAVES);
  const [showAbout, setShowAbout] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const speedRef = useRef(TRAVEL_DISTANCE_YEARS / DEFAULT_TRAVEL_TIME);
  const densityRef = useRef(graveDensity);
  const viewportRef = useRef(viewportWidthYears);
  const chunkLRef = useRef(chunkRadialLen);
  const maxGravesRef = useRef(maxGraves);

  useEffect(() => { speedRef.current = TRAVEL_DISTANCE_YEARS / Math.max(travelTimeSeconds, 0.0001); }, [travelTimeSeconds]);
  useEffect(() => { densityRef.current = graveDensity; }, [graveDensity]);
  useEffect(() => { viewportRef.current = viewportWidthYears; }, [viewportWidthYears]);
  useEffect(() => { chunkLRef.current = chunkRadialLen; }, [chunkRadialLen]);
  useEffect(() => { maxGravesRef.current = maxGraves; }, [maxGraves]);


  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let raf = 0;
    let lastTs = performance.now();
    const keys = new Set<string>();
    const position: PolarPos = { r: INNER_RADIUS * PIX_PER_YEAR, phi: -Math.PI / 2 };
    const velocity: PolarVel = { vr: 0, vphi: 0 };

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(window.innerWidth * dpr);
      canvas.height = Math.round(window.innerHeight * dpr);
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','w','a','s','d','h','j','k','l'].includes(e.key)) {
        keys.add(e.key);
        e.preventDefault();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => keys.delete(e.key);
    const clearKeys = () => keys.clear();

    const update = (dt: number) => {
      let ix = 0, iy = 0;
      if (keys.has('ArrowLeft') || keys.has('a') || keys.has('h')) ix -= 1;
      if (keys.has('ArrowRight') || keys.has('d') || keys.has('l')) ix += 1;
      if (keys.has('ArrowUp') || keys.has('w') || keys.has('k')) iy -= 1;
      if (keys.has('ArrowDown') || keys.has('s') || keys.has('j')) iy += 1;

      const spd = speedRef.current * PIX_PER_YEAR;
      const accel = 1 - Math.exp(-ACCELERATION * dt);
      const damp = Math.exp(-DAMPING * dt);

      // left/right → tangential (angular), target angular speed = linear speed / r
      const targetVphi = ix * spd / Math.max(position.r, 1);
      velocity.vphi += (targetVphi - velocity.vphi) * accel;
      if (ix === 0) velocity.vphi *= damp;

      // up/down → radial (up = inward = decreasing r)
      const targetVr = -iy * spd;
      velocity.vr += (targetVr - velocity.vr) * accel;
      if (iy === 0) velocity.vr *= damp;

      position.phi += velocity.vphi * dt;
      position.r = Math.max(INNER_RADIUS * PIX_PER_YEAR, position.r + velocity.vr * dt);
    };

    const draw = () => {
      const W = window.innerWidth;
      const H = window.innerHeight;

      ctx.fillStyle = '#101010';
      ctx.fillRect(0, 0, W, H);

      const vpYears = viewportRef.current;
      const scale = (vpYears * PIX_PER_YEAR) / Math.max(W, 1); // world px per screen px
      const camX = position.r * Math.cos(position.phi);
      const camY = position.r * Math.sin(position.phi);
      const wl = camX - (W / 2) * scale;
      const wt = camY - (H / 2) * scale;
      const sx = (wx: number) => (wx - wl) / scale;
      const sy = (wy: number) => (wy - wt) / scale;

      const density = densityRef.current;
      const chunkL = chunkLRef.current;
      const mg = maxGravesRef.current;
      const numChunks = calcNumChunks(density, chunkL, mg);
      const dTheta = (2 * Math.PI) / numChunks;

      // Grave spacing sized so max_graves fit in chunk with 4×2 block layout
      const rMid = INNER_RADIUS + chunkL / 2;
      const chunkTang = rMid * dTheta; // years at mid-radius
      const chunkArea = chunkTang * chunkL;
      const s = Math.sqrt(chunkArea / (BLOCK_OVERHEAD * mg)); // years
      const periodT = (BLOCK_COLS + BLOCK_GAP_FACTOR) * s; // tangential period
      const periodR = (BLOCK_ROWS + BLOCK_GAP_FACTOR) * s; // radial period
      const dotR = Math.max(1.5, (s * 0.3 * PIX_PER_YEAR) / scale);

      // Debug radial grid
      {
        const rCamYr = position.r / PIX_PER_YEAR;
        const rMinYr = Math.max(0, rCamYr - vpYears * 0.75);
        const rMaxYr = rCamYr + vpYears * 0.75;

        // Pick step giving ~6 concentric circles in view
        const rawRStep = (rMaxYr - rMinYr) / 6;
        const mag = Math.pow(10, Math.floor(Math.log10(rawRStep)));
        const rStep = Math.ceil(rawRStep / mag) * mag;
        const rStartYr = Math.ceil(rMinYr / rStep) * rStep;

        ctx.strokeStyle = 'rgba(255,220,80,0.18)';
        ctx.lineWidth = 0.5;
        ctx.fillStyle = 'rgba(255,220,80,0.55)';
        ctx.font = '10px monospace';
        for (let ry = rStartYr; ry <= rMaxYr; ry += rStep) {
          const screenR = (ry * PIX_PER_YEAR) / scale;
          ctx.beginPath();
          ctx.arc(sx(0), sy(0), screenR, 0, 2 * Math.PI);
          ctx.stroke();
          // label at current phi
          ctx.fillText(
            `${Math.round(ry)} yr`,
            sx(ry * PIX_PER_YEAR * Math.cos(position.phi)) + 3,
            sy(ry * PIX_PER_YEAR * Math.sin(position.phi)) - 3,
          );
        }

        // Visible arc half-angle; pick degree step giving ~6 angular lines
        const visHalfAngle = Math.atan2(vpYears / 2, rCamYr);
        const rawDegStep = (visHalfAngle * 2 * 180 / Math.PI) / 6;
        const degSteps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 45, 90];
        const degStep = degSteps.find(d => d >= rawDegStep) ?? 90;
        const phiStepRad = degStep * Math.PI / 180;
        const phiMin = position.phi - visHalfAngle - phiStepRad;
        const phiMax = position.phi + visHalfAngle + phiStepRad;
        const phiStart = Math.ceil(phiMin / phiStepRad) * phiStepRad;

        ctx.strokeStyle = 'rgba(255,220,80,0.18)';
        ctx.lineWidth = 0.5;
        for (let phi = phiStart; phi <= phiMax; phi += phiStepRad) {
          const cosP = Math.cos(phi), sinP = Math.sin(phi);
          ctx.beginPath();
          ctx.moveTo(sx(rMinYr * PIX_PER_YEAR * cosP), sy(rMinYr * PIX_PER_YEAR * sinP));
          ctx.lineTo(sx(rMaxYr * PIX_PER_YEAR * cosP), sy(rMaxYr * PIX_PER_YEAR * sinP));
          ctx.stroke();
          const deg = ((phi * 180 / Math.PI) % 360 + 360) % 360;
          const labelFrac = degStep < 1 ? 1 : 0;
          ctx.fillStyle = 'rgba(255,220,80,0.55)';
          ctx.fillText(
            `${deg.toFixed(labelFrac)}°`,
            sx(rCamYr * PIX_PER_YEAR * cosP) + 3,
            sy(rCamYr * PIX_PER_YEAR * sinP) - 3,
          );
        }
      }

      // Draw inner / outer ring arcs
      const ocx = sx(0), ocy = sy(0);
      const innerR = (INNER_RADIUS * PIX_PER_YEAR) / scale;
      const outerR = ((INNER_RADIUS + chunkL) * PIX_PER_YEAR) / scale;
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(ocx, ocy, innerR, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(ocx, ocy, outerR, 0, 2 * Math.PI);
      ctx.stroke();

      // Determine visible chunk range from camera angle
      const camAngle = position.phi;
      const angHalf = Math.min(Math.PI, Math.asin(Math.min(1, (vpYears * 2) / INNER_RADIUS)));
      const iCenter = Math.round(camAngle / dTheta);
      const iHalf = Math.ceil(angHalf / dTheta) + 1;

      for (let i = iCenter - iHalf; i <= iCenter + iHalf; i++) {
        const theta = i * dTheta;
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);
        const iPx = INNER_RADIUS * PIX_PER_YEAR;
        const oPx = (INNER_RADIUS + chunkL) * PIX_PER_YEAR;

        // Chunk border (radial line at chunk start angle)
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(sx(iPx * cosT), sy(iPx * sinT));
        ctx.lineTo(sx(oPx * cosT), sy(oPx * sinT));
        ctx.stroke();

        // Graves in 4×2 blocks within this chunk
        ctx.fillStyle = '#7ad9ff';
        for (let bt = 0; bt * periodT < chunkTang; bt++) {
          for (let br = 0; br * periodR < chunkL; br++) {
            for (let gc = 0; gc < BLOCK_COLS; gc++) {
              for (let gr = 0; gr < BLOCK_ROWS; gr++) {
                const tang = bt * periodT + gc * s + s / 2;
                const r = INNER_RADIUS + br * periodR + gr * s + s / 2;
                if (tang > chunkTang || r > INNER_RADIUS + chunkL) continue;

                // Map (tang, r) to world coords: theta offset = tang / r
                const graveTheta = theta + tang / r;
                const wx = r * PIX_PER_YEAR * Math.cos(graveTheta);
                const wy = r * PIX_PER_YEAR * Math.sin(graveTheta);
                const gsx = sx(wx), gsy = sy(wy);
                if (gsx < -dotR || gsx > W + dotR || gsy < -dotR || gsy > H + dotR) continue;

                ctx.beginPath();
                ctx.arc(gsx, gsy, dotR, 0, Math.PI * 2);
                ctx.fill();
              }
            }
          }
        }
      }

      // HUD overlay
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.font = '14px monospace';
      ctx.fillText('Circular plan  —  move with arrows / wasd / hjkl', 16, 28);
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      const rYears = position.r / PIX_PER_YEAR;
      const phiDeg = (((position.phi * 180 / Math.PI) % 360) + 360) % 360;
      ctx.fillText(`r = ${rYears.toFixed(1)} yr  |  φ = ${phiDeg.toFixed(1)}°  |  chunks: ${numChunks.toLocaleString()}`, 16, 50);

      // Ruler
      const yearPx = W / vpYears;
      const rl = 16, ry = H - 24, rr = rl + yearPx;
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(rl, ry); ctx.lineTo(rr, ry);
      ctx.moveTo(rl, ry - 6); ctx.lineTo(rl, ry + 6);
      ctx.moveTo(rr, ry - 6); ctx.lineTo(rr, ry + 6);
      ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fillText('1 year', rl, ry - 10);
    };

    const tick = (ts: number) => {
      const dt = Math.min(0.05, (ts - lastTs) / 1000);
      lastTs = ts;
      update(dt);
      draw();
      raf = window.requestAnimationFrame(tick);
    };

    resize();
    window.addEventListener('resize', resize);
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', clearKeys);
    document.addEventListener('visibilitychange', clearKeys);
    raf = window.requestAnimationFrame(tick);

    return () => {
      window.cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', clearKeys);
      document.removeEventListener('visibilitychange', clearKeys);
    };
  }, []);

  // Computed for display panel
  const ringArea = Math.PI * ((INNER_RADIUS + chunkRadialLen) ** 2 - INNER_RADIUS ** 2);
  const totalGravesInRing = graveDensity * ringArea;
  const numChunks = calcNumChunks(graveDensity, chunkRadialLen, maxGraves);
  const dTheta = (2 * Math.PI) / numChunks;
  const chunkWidthMid = (INNER_RADIUS + chunkRadialLen / 2) * dTheta;
  const speedYearsPerSecond = TRAVEL_DISTANCE_YEARS / Math.max(travelTimeSeconds, 0.0001);

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', overflow: 'hidden' }}>
      {showAbout && <AboutDialog onClose={() => setShowAbout(false)} />}
      <canvas ref={canvasRef} />
      <div
        style={{
          position: 'absolute',
          top: 12,
          right: 12,
          background: 'rgba(0,0,0,0.5)',
          border: '1px solid rgba(255,255,255,0.15)',
          borderRadius: 6,
          padding: '10px 12px',
          color: '#ddd',
          fontSize: 12,
          fontFamily: 'monospace',
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
          minWidth: 240,
        }}
      >
        <button
          onClick={() => setShowAbout(true)}
          style={{
            background: 'rgba(255,255,255,0.06)',
            border: '1px solid rgba(255,255,255,0.2)',
            borderRadius: 4,
            color: '#bbb',
            cursor: 'pointer',
            fontFamily: 'monospace',
            fontSize: 12,
            padding: '4px 8px',
            textAlign: 'left',
          }}
        >
          About
        </button>

        <div style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: 6 }}>
          <div style={{ color: '#7ad9ff', fontSize: 13 }}>
            Segments in ring:{' '}
            <strong style={{ color: '#fff' }}>{numChunks.toLocaleString()}</strong>
          </div>
          <div style={{ color: 'rgba(180,200,220,0.7)', fontSize: 11, marginTop: 3 }}>
            Ring area: {Math.round(ringArea).toLocaleString()} yr²
          </div>
          <div style={{ color: 'rgba(180,200,220,0.7)', fontSize: 11 }}>
            Total graves in ring: {Math.round(totalGravesInRing).toLocaleString()}
          </div>
          <div style={{ color: 'rgba(180,200,220,0.7)', fontSize: 11 }}>
            Chunk width at mid: {chunkWidthMid.toFixed(1)} yr
          </div>
        </div>

        <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span>Chunk radial length: {chunkRadialLen} yr</span>
          <input
            type="range"
            min={MIN_CHUNK_RADIAL_LEN}
            max={MAX_CHUNK_RADIAL_LEN}
            step={1}
            value={chunkRadialLen}
            onChange={e => setChunkRadialLen(Number(e.target.value))}
          />
        </label>

        <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span>Max graves / chunk: {maxGraves.toLocaleString()}</span>
          <input
            type="range"
            min={MIN_MAX_GRAVES}
            max={MAX_MAX_GRAVES}
            step={100}
            value={maxGraves}
            onChange={e => setMaxGraves(Number(e.target.value))}
          />
        </label>

        <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span>Grave density: {graveDensity.toFixed(1)} graves/yr²</span>
          <input
            type="range"
            min={MIN_GRAVE_DENSITY}
            max={MAX_GRAVE_DENSITY}
            step={1}
            value={graveDensity}
            onChange={e => setGraveDensity(Number(e.target.value))}
          />
        </label>

        <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span>16k yr travel time: {formatDuration(travelTimeSeconds)}</span>
          <input
            type="range"
            min={MIN_TRAVEL_TIME}
            max={MAX_TRAVEL_TIME}
            step={30}
            value={travelTimeSeconds}
            onChange={e => setTravelTimeSeconds(Number(e.target.value))}
          />
        </label>

        <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span>Viewport: {fmtYears(viewportWidthYears)}</span>
          <input
            type="range"
            min={0}
            max={1000}
            step={1}
            value={viewportToSlider(viewportWidthYears)}
            onChange={e => setViewportWidthYears(sliderToViewport(Number(e.target.value)))}
          />
        </label>

        <div style={{ color: 'rgba(180,200,200,0.6)', fontSize: 11 }}>
          Speed: {speedYearsPerSecond.toFixed(1)} yr/s
        </div>
      </div>
    </div>
  );
}
