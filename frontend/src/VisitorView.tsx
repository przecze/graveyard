import { useEffect, useRef, useState } from 'react';

type Vec2 = { x: number; y: number };

const GRID_SPACING = 36;
const GRAVE_RADIUS = 8;
const FILLED_GRID_FRACTION = (2 / 3) * (4 / 5);
const GRAVES_PER_PIXEL_SQ = FILLED_GRID_FRACTION / (GRID_SPACING * GRID_SPACING);
const DEFAULT_GRAVE_DENSITY = 50; // graves / year^2
const MIN_GRAVE_DENSITY = 5;
const MAX_GRAVE_DENSITY = 200;
const DEFAULT_SPEED = 0.9; // years per second
const TRAVEL_DISTANCE_YEARS = 16000;
const DEFAULT_TRAVEL_TIME_SECONDS = TRAVEL_DISTANCE_YEARS / DEFAULT_SPEED;
const MIN_TRAVEL_TIME_SECONDS = 30;
const MAX_TRAVEL_TIME_SECONDS = 5 * 60 * 60;
const DEFAULT_VIEWPORT_WIDTH_YEARS = 3;
const MIN_VIEWPORT_WIDTH_YEARS = 1;
const MAX_VIEWPORT_WIDTH_YEARS = 20;
const ACCELERATION = 7;
const DAMPING = 9;

function shouldDrawGrave(gridX: number, gridY: number): boolean {
  return gridX % 3 !== 0 && gridY % 5 !== 0;
}

function pixelsPerYearForDensity(graveDensity: number): number {
  return Math.sqrt(graveDensity / GRAVES_PER_PIXEL_SQ);
}

function formatDuration(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) return 'n/a';
  if (totalSeconds < 60) return `${totalSeconds.toFixed(1)}s`;
  if (totalSeconds < 3600) return `${(totalSeconds / 60).toFixed(1)}m`;
  if (totalSeconds < 86400) return `${(totalSeconds / 3600).toFixed(1)}h`;
  if (totalSeconds < 31557600) return `${(totalSeconds / 86400).toFixed(1)}d`;
  return `${(totalSeconds / 31557600).toFixed(2)}y`;
}

export default function VisitorView() {
  const [travelTimeSeconds, setTravelTimeSeconds] = useState(DEFAULT_TRAVEL_TIME_SECONDS);
  const [graveDensity, setGraveDensity] = useState(DEFAULT_GRAVE_DENSITY);
  const [viewportWidthYears, setViewportWidthYears] = useState(DEFAULT_VIEWPORT_WIDTH_YEARS);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const speedYearsPerSecondRef = useRef(TRAVEL_DISTANCE_YEARS / DEFAULT_TRAVEL_TIME_SECONDS);
  const graveDensityRef = useRef(graveDensity);
  const pixelsPerYearRef = useRef(pixelsPerYearForDensity(graveDensity));
  const viewportWidthYearsRef = useRef(viewportWidthYears);

  useEffect(() => {
    speedYearsPerSecondRef.current = TRAVEL_DISTANCE_YEARS / Math.max(travelTimeSeconds, 0.0001);
  }, [travelTimeSeconds]);

  useEffect(() => {
    graveDensityRef.current = graveDensity;
    pixelsPerYearRef.current = pixelsPerYearForDensity(graveDensity);
  }, [graveDensity]);

  useEffect(() => {
    viewportWidthYearsRef.current = viewportWidthYears;
  }, [viewportWidthYears]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let raf = 0;
    let lastTs = performance.now();
    const keys = new Set<string>();
    const position: Vec2 = { x: 0, y: 0 };
    const velocity: Vec2 = { x: 0, y: 0 };

    const resize = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.key === 'ArrowUp' ||
        event.key === 'ArrowDown' ||
        event.key === 'ArrowLeft' ||
        event.key === 'ArrowRight'
      ) {
        keys.add(event.key);
        event.preventDefault();
      }
    };

    const onKeyUp = (event: KeyboardEvent) => {
      keys.delete(event.key);
    };

    const update = (dt: number) => {
      let inputX = 0;
      let inputY = 0;
      if (keys.has('ArrowLeft')) inputX -= 1;
      if (keys.has('ArrowRight')) inputX += 1;
      if (keys.has('ArrowUp')) inputY -= 1;
      if (keys.has('ArrowDown')) inputY += 1;

      const speedPxPerSec = speedYearsPerSecondRef.current * pixelsPerYearRef.current;
      const targetVX = inputX * speedPxPerSec;
      const targetVY = inputY * speedPxPerSec;
      const accelT = 1 - Math.exp(-ACCELERATION * dt);
      velocity.x += (targetVX - velocity.x) * accelT;
      velocity.y += (targetVY - velocity.y) * accelT;

      const dampingT = Math.exp(-DAMPING * dt);
      if (inputX === 0) velocity.x *= dampingT;
      if (inputY === 0) velocity.y *= dampingT;

      position.x += velocity.x * dt;
      position.y += velocity.y * dt;
    };

    const draw = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;

      ctx.fillStyle = '#101010';
      ctx.fillRect(0, 0, width, height);

      const worldPxPerScreenPx =
        (viewportWidthYearsRef.current * pixelsPerYearRef.current) / Math.max(width, 1);
      const worldHalfW = (width / 2) * worldPxPerScreenPx;
      const worldHalfH = (height / 2) * worldPxPerScreenPx;
      const worldLeftScaled = position.x - worldHalfW;
      const worldTop = position.y - worldHalfH;
      const worldRight = position.x + worldHalfW;
      const worldBottom = position.y + worldHalfH;

      const startX = Math.floor(worldLeftScaled / GRID_SPACING) - 1;
      const endX = Math.ceil(worldRight / GRID_SPACING) + 1;
      const startY = Math.floor(worldTop / GRID_SPACING) - 1;
      const endY = Math.ceil(worldBottom / GRID_SPACING) + 1;

      ctx.fillStyle = '#7ad9ff';
      for (let gy = startY; gy <= endY; gy++) {
        for (let gx = startX; gx <= endX; gx++) {
          if (!shouldDrawGrave(gx, gy)) continue;
          const sx = (gx * GRID_SPACING - worldLeftScaled) / worldPxPerScreenPx;
          const sy = (gy * GRID_SPACING - worldTop) / worldPxPerScreenPx;
          ctx.beginPath();
          ctx.arc(sx, sy, GRAVE_RADIUS / worldPxPerScreenPx, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.font = '14px monospace';
      ctx.fillText('Visitor view  -  move with arrow keys', 16, 28);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.55)';
      const positionInYearsX = position.x / pixelsPerYearRef.current;
      const positionInYearsY = position.y / pixelsPerYearRef.current;
      ctx.fillText(
        `position: (${positionInYearsX.toFixed(2)}, ${positionInYearsY.toFixed(2)}) years`,
        16,
        50,
      );
      ctx.fillText(`speed: ${speedYearsPerSecondRef.current.toFixed(2)} years/s`, 16, 72);
      ctx.fillText(`grave density: ${graveDensityRef.current.toFixed(1)} graves/yr^2`, 16, 94);

      const yearPixels = width / viewportWidthYearsRef.current;
      const rulerLeft = 16;
      const rulerY = height - 24;
      const rulerRight = rulerLeft + yearPixels;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(rulerLeft, rulerY);
      ctx.lineTo(rulerRight, rulerY);
      ctx.moveTo(rulerLeft, rulerY - 6);
      ctx.lineTo(rulerLeft, rulerY + 6);
      ctx.moveTo(rulerRight, rulerY - 6);
      ctx.lineTo(rulerRight, rulerY + 6);
      ctx.stroke();
      ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
      ctx.fillText('1 year', rulerLeft, rulerY - 10);
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
    raf = window.requestAnimationFrame(tick);

    return () => {
      window.cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', overflow: 'hidden' }}>
      <canvas ref={canvasRef} />
      <div
        style={{
          position: 'absolute',
          top: 12,
          right: 12,
          background: 'rgba(0, 0, 0, 0.45)',
          border: '1px solid rgba(255, 255, 255, 0.15)',
          borderRadius: 6,
          padding: '10px 12px',
          color: '#ddd',
          fontSize: 12,
          fontFamily: 'monospace',
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
          minWidth: 220,
        }}
      >
        <div style={{ color: '#bfe7ff' }}>
          Calculated speed: {(TRAVEL_DISTANCE_YEARS / travelTimeSeconds).toFixed(2)} years/s
        </div>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span>16k years travel time: {formatDuration(travelTimeSeconds)}</span>
          <input
            type="range"
            min={MIN_TRAVEL_TIME_SECONDS}
            max={MAX_TRAVEL_TIME_SECONDS}
            step={30}
            value={travelTimeSeconds}
            onChange={event => setTravelTimeSeconds(Number(event.target.value))}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span>Grave density: {graveDensity.toFixed(1)} graves/yr^2</span>
          <input
            type="range"
            min={MIN_GRAVE_DENSITY}
            max={MAX_GRAVE_DENSITY}
            step={1}
            value={graveDensity}
            onChange={event => setGraveDensity(Number(event.target.value))}
          />
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span>Viewport width: {viewportWidthYears.toFixed(1)} years</span>
          <input
            type="range"
            min={MIN_VIEWPORT_WIDTH_YEARS}
            max={MAX_VIEWPORT_WIDTH_YEARS}
            step={0.1}
            value={viewportWidthYears}
            onChange={event => setViewportWidthYears(Number(event.target.value))}
          />
        </label>
      </div>
    </div>
  );
}
