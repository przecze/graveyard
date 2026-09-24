import { useEffect, useRef, useState } from 'react';
import { surface as S, T0, T_END } from './surface';
import { buildMarker, buildSprites, SPRITE_H, SPRITE_W, STYLES_BY_ERA, type Style } from './sprites';
import { formatYear, graveInfo, hash01, LANDMARKS, styleFor, type GraveInfo } from './lore';
import './walker.css';

const TAU = 2 * Math.PI;
const MIN_ZOOM = 0.002;          // px per metre (whole graveyard)
const MAX_ZOOM = 90;
const DEFAULT_ZOOM = 22;
const MAX_GRAVES_DRAWN = 7000;
const MIN_GRAVE_PX = 3;          // below this plot width rows are drawn as bands
const MAX_AISLES_DRAWN = 240;
const WALK_SPEED = 1.4;          // m/s
const SCREENS_PER_SECOND = 1 / 3; // auto speed: cross the screen in 3 s
const LIVING_BAND = 45;          // metres of candles beyond the edge

const wrapPi = (a: number) => a - TAU * Math.floor((a + Math.PI) / TAU);

const BAND_COLOR: Record<Style, string> = {
  mound: '#3b4a2e', cairn: '#6a675c', menhir: '#57553f', stele: '#86795e',
  cross: '#4f4838', headstone: '#9f9c93', granite: '#47474a',
};
function eraBandColor(t: number): string {
  const era = STYLES_BY_ERA.find(e => t < e.until) ?? STYLES_BY_ERA[STYLES_BY_ERA.length - 1];
  return BAND_COLOR[era.styles[0][0]];
}

// "Now" as a fractional calendar year, drives the edge in real time
function nowYear(): number {
  const d = new Date();
  const y = d.getUTCFullYear();
  const start = Date.UTC(y, 0, 1), end = Date.UTC(y + 1, 0, 1);
  return Math.min(T_END - 0.01, y + (d.getTime() - start) / (end - start));
}

function markStep(t: number): number {
  if (t < T0) return 5000;
  if (t < 0) return 500;
  if (t < 1500) return 100;
  if (t < 1800) return 50;
  return 10;
}

const fmtBig = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(n >= 1e11 ? 1 : 2)} B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} M` : Math.round(n).toLocaleString('en-US');
const fmtLen = (m: number) =>
  m >= 1e7 ? `${Math.round(m / 1000).toLocaleString('en-US')} km` : m >= 1000 ? `${(m / 1000).toFixed(m >= 1e5 ? 0 : 1)} km` : `${m.toFixed(m >= 10 ? 0 : 1)} m`;
const fmtDur = (s: number) =>
  !isFinite(s) ? '—' : s < 90 ? `${s.toFixed(0)} s` : s < 5400 ? `${(s / 60).toFixed(0)} min` : s < 172800 ? `${(s / 3600).toFixed(1)} h` : `${(s / 86400).toFixed(1)} days`;

const WELCOME = {
  year: -50000, title: 'The dawn of humanity',
  text: `Every person who ever died has a grave here, about ${Math.round(S.rowStart[S.nRows] / 1e9)} billion of them. `
    + `Walk outward and time moves forward: the oldest graves are here at the centre, the newest at the edge, ${Math.round(S.rhoEndTable / 1000)} km away.`,
};

type Hit = { info: GraveInfo; row: number; rho: number; phi: number };

const JUMPS: [string, number][] = [
  ['Centre', -50000], ['20,000 BCE', -20000], ['Rim 8000 BCE', -8000], ['1 CE', 1],
  ['1500', 1500], ['1900', 1900], ['2000', 2000], ['Edge (now)', NaN],
];

export default function Walker() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mapRef = useRef<HTMLCanvasElement>(null);
  const hudRef = useRef<HTMLDivElement>(null);
  const signRef = useRef<HTMLDivElement>(null);
  const [selected, setSelected] = useState<Hit | null>(null);
  const [showAbout, setShowAbout] = useState(false);
  const [mapMode, setMapMode] = useState<'distance' | 'graves'>('graves');
  const [bigMap, setBigMap] = useState(false);
  const [walkMode, setWalkMode] = useState(false);

  const state = useRef({
    rho: 6,
    phi: 0,
    zoom: DEFAULT_ZOOM,
    speed: 0,
    hover: null as Hit | null,
    mouse: null as [number, number] | null,
    patternOffset: [0, 0] as [number, number],
  });
  const opts = useRef({ mapMode, walkMode, bigMap });
  opts.current = { mapMode, walkMode, bigMap };
  const setSelectedRef = useRef(setSelected);
  setSelectedRef.current = setSelected;

  const jump = (t: number) => {
    const st = state.current;
    st.rho = isNaN(t) ? S.rhoAtTime(nowYear()) - 12 : t <= -50000 ? 4 : S.rhoAtTime(t) + 1.3;
  };

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    const mapCanvas = mapRef.current!;
    const mctx = mapCanvas.getContext('2d')!;
    const sprites = buildSprites();
    const marker = buildMarker();
    const landmarkMarker = buildMarker('#e8c872');
    const st = state.current;

    // grass noise pattern
    const grass = document.createElement('canvas');
    grass.width = grass.height = 256;
    {
      const g = grass.getContext('2d')!;
      g.fillStyle = '#1d2a1b';
      g.fillRect(0, 0, 256, 256);
      for (let i = 0; i < 5000; i++) {
        const l = 10 + Math.random() * 16;
        g.fillStyle = `hsla(${85 + Math.random() * 30},${25 + Math.random() * 20}%,${l}%,0.5)`;
        g.fillRect(Math.random() * 256, Math.random() * 256, 1 + Math.random() * 2, 1 + Math.random() * 3);
      }
    }
    const grassPattern = ctx.createPattern(grass, 'repeat')!;
    const glow = document.createElement('canvas');
    glow.width = glow.height = 64;
    {
      const g = glow.getContext('2d')!;
      const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
      gr.addColorStop(0, 'rgba(255,220,150,0.9)'); gr.addColorStop(0.3, 'rgba(255,170,70,0.35)'); gr.addColorStop(1, 'rgba(255,140,40,0)');
      g.fillStyle = gr;
      g.fillRect(0, 0, 64, 64);
    }

    let W = 0, H = 0, dpr = 1;
    const resize = () => {
      dpr = window.devicePixelRatio || 1;
      W = window.innerWidth; H = window.innerHeight;
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      canvas.style.width = `${W}px`; canvas.style.height = `${H}px`;
    };
    resize();
    window.addEventListener('resize', resize);

    const keys = new Set<string>();
    const MOVE_KEYS = ['arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'w', 'a', 's', 'd', 'shift'];
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
      const k = e.key.toLowerCase();
      if (MOVE_KEYS.includes(k)) { keys.add(k); e.preventDefault(); }
      if (k === '+' || k === '=') st.zoom = Math.min(MAX_ZOOM, st.zoom * 1.25);
      if (k === '-' || k === '_') st.zoom = Math.max(MIN_ZOOM, st.zoom / 1.25);
      if (k === 'escape') setSelectedRef.current(null);
    };
    const onKeyUp = (e: KeyboardEvent) => keys.delete(e.key.toLowerCase());
    const onBlur = () => keys.clear();
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      st.zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, st.zoom * Math.exp(-e.deltaY * 0.0015)));
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    const onMouseMove = (e: MouseEvent) => { st.mouse = [e.clientX, e.clientY]; };
    const onMouseLeave = () => { st.mouse = null; };
    const onClick = () => setSelectedRef.current(st.hover);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseleave', onMouseLeave);
    canvas.addEventListener('click', onClick);

    // ── movement ─────────────────────────────────────────────────────────────
    function update(dt: number) {
      let ix = 0, iy = 0;
      if (keys.has('arrowleft') || keys.has('a')) ix -= 1;
      if (keys.has('arrowright') || keys.has('d')) ix += 1;
      if (keys.has('arrowup') || keys.has('w')) iy += 1;
      if (keys.has('arrowdown') || keys.has('s')) iy -= 1;
      const base = opts.current.walkMode ? WALK_SPEED : (W / st.zoom) * SCREENS_PER_SECOND;
      const speed = base * (keys.has('shift') ? 5 : 1);
      st.speed = ix || iy ? speed : 0;
      if (!ix && !iy) return;
      const n = Math.hypot(ix, iy);
      const dx = ix / n * speed * dt, dy = iy / n * speed * dt;
      st.patternOffset[0] -= dx * st.zoom;
      st.patternOffset[1] += dy * st.zoom;
      if (st.rho + dy <= S.rho0) {
        // exact Euclidean step in the flat circle (lets you walk through the centre)
        const x = st.rho * Math.cos(st.phi) - dx * Math.sin(st.phi) + dy * Math.cos(st.phi);
        const y = st.rho * Math.sin(st.phi) + dx * Math.cos(st.phi) + dy * Math.sin(st.phi);
        st.rho = Math.max(0.3, Math.hypot(x, y));
        st.phi = Math.atan2(y, x);
      } else {
        st.rho += dy;
        st.phi += dx / S.f(st.rho);
      }
      st.rho = Math.min(st.rho, S.rhoAtTime(nowYear()) + LIVING_BAND + 60);
      st.phi = ((st.phi % TAU) + TAU) % TAU;
    }

    // ── drawing ──────────────────────────────────────────────────────────────
    function draw(time: number) {
      const { rho: rp, phi: pp, zoom: z } = st;
      const fp = S.f(rp), up = S.u(rp);
      const cx = W / 2, cy = H / 2;
      const Oy = cy + fp * z;              // screen position of the graveyard centre
      const tNow = nowYear();
      const rhoEdge = S.rhoAtTime(tNow);
      const cumNow = S.cum(rhoEdge);

      // projection (ρ, φ) → screen; dU/e computed per ring
      const ringOf = (rho: number) => {
        const em1 = Math.expm1(S.u(rho) - up);
        return { em1, e: 1 + em1, scale: z * fp * (1 + em1) / S.f(rho) };
      };
      const projRing = (em1: number, dphi: number): [number, number] => {
        const e = 1 + em1, s2 = Math.sin(dphi / 2);
        const tx = fp * e * Math.sin(dphi);
        const ty = fp * (em1 * Math.cos(dphi) - 2 * s2 * s2);
        return [cx + tx * z, cy - ty * z];
      };
      const proj = (rho: number, phi: number) => projRing(ringOf(rho).em1, wrapPi(phi - pp));
      const unproj = (sx: number, sy: number): [number, number] => {
        const a = -(sy - cy) / z / fp, b = (sx - cx) / z / fp;
        const dU = 0.5 * Math.log1p(2 * a + a * a + b * b);
        return [S.rhoAtU(up + dU), pp + Math.atan2(b, 1 + a)];
      };

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.globalCompositeOperation = 'source-over';
      ctx.save();
      ctx.translate(st.patternOffset[0] % 256, st.patternOffset[1] % 256);
      ctx.fillStyle = grassPattern;
      ctx.fillRect(-256, -256, W + 512, H + 512);
      ctx.restore();

      // visible radial range and angular sector (the view is a sector around O)
      const ddx = Math.max(0 - cx, 0, cx - W), ddy = Math.max(0 - Oy, 0, Oy - H);
      const dmin = Math.hypot(ddx, ddy);
      const dmax = Math.max(Math.hypot(cx, Oy), Math.hypot(W - cx, Oy), Math.hypot(cx, Oy - H), Math.hypot(W - cx, Oy - H));
      const R0 = fp * z;
      const rhoMin = dmin <= 0 ? 0 : S.rhoAtU(up + Math.log(dmin / R0)) - 3;
      const rhoMax = S.rhoAtU(up + Math.log(dmax / R0)) + 3;
      let thMin = -Math.PI, thMax = Math.PI;
      if (dmin > 0) {
        const ths = [[0, 0], [W, 0], [0, H], [W, H]].map(([x, y]) => Math.atan2(x - cx, Oy - y));
        thMin = Math.min(...ths); thMax = Math.max(...ths);
      }
      const fullCircle = thMax - thMin > Math.PI;

      const arcPath = (rho: number, a0: number, a1: number) => {
        const { em1, e } = ringOf(rho);
        const rs = R0 * e;
        const n = Math.max(2, Math.min(400, Math.ceil((a1 - a0) * rs / 12)));
        ctx.beginPath();
        for (let i = 0; i <= n; i++) {
          const [x, y] = projRing(em1, a0 + (a1 - a0) * i / n);
          if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        }
      };
      const ring = (rho: number, color: string, widthM: number, minPx = 1) => {
        if (rho < rhoMin || rho > rhoMax) return;
        const { scale } = ringOf(rho);
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(minPx, widthM * scale);
        arcPath(rho, thMin - 0.01, thMax + 0.01);
        ctx.stroke();
      };

      // central plaza
      if (rhoMin < S.layout.plazaR) {
        ctx.fillStyle = '#5a554a';
        arcPath(S.layout.plazaR, -Math.PI, Math.PI);
        ctx.fill();
        ring(S.layout.plazaR * 0.55, '#6a6456', 0.6);
        const [mx, my] = proj(0.01, pp);
        const g = ctx.createRadialGradient(mx, my, 0, mx, my, 6 * z);
        g.addColorStop(0, 'rgba(255,190,90,0.9)'); g.addColorStop(1, 'rgba(255,160,60,0)');
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(mx, my, 6 * z, 0, TAU); ctx.fill();
        ctx.fillStyle = '#d8d0bc';
        ctx.beginPath(); ctx.arc(mx, my, 0.8 * z, 0, TAU); ctx.fill();
      }

      // ── aisles (radial gravel paths, branching outward) ────────────────────
      const lastRow = Math.min(S.nRows - 1, S.rowOf(rhoMax));
      const firstRow = Math.max(0, S.rowOf(rhoMin));
      let aislesDrawn: number[] = [];
      let aisleLevel = 4;
      if (lastRow >= 0 && firstRow <= lastRow) {
        let M = S.rowAisles[lastRow];
        const span = fullCircle ? TAU : thMax - thMin;
        while (M > 4 && span / (TAU / M) > MAX_AISLES_DRAWN) M /= 2;
        aisleLevel = M;
        const alpha = TAU / M;
        const k0 = fullCircle ? 0 : Math.floor((pp + thMin) / alpha);
        const k1 = fullCircle ? M - 1 : Math.ceil((pp + thMax) / alpha);
        const rhoTop = Math.min(rhoMax, rhoEdge + LIVING_BAND);
        ctx.fillStyle = '#4f4a3e';
        for (let k = k0; k <= k1; k++) {
          const km = ((k % M) + M) % M;
          let level = M, kk = km;
          while (level > 4 && kk % 2 === 0) { kk /= 2; level /= 2; }
          const startRho = Math.max(S.rowInner(S.firstRowWithAisles(level)), S.layout.plazaR, rhoMin);
          if (startRho >= rhoTop) continue;
          const dphi = wrapPi(km * alpha - pp);
          const a = ringOf(startRho), b = ringOf(rhoTop);
          const [x1, y1] = projRing(a.em1, dphi), [x2, y2] = projRing(b.em1, dphi);
          const w1 = S.layout.aisleW * a.scale / 2, w2 = S.layout.aisleW * b.scale / 2;
          const nx = Math.cos(dphi), ny = Math.sin(dphi);
          ctx.beginPath();
          ctx.moveTo(x1 - nx * w1, y1 - ny * w1); ctx.lineTo(x2 - nx * w2, y2 - ny * w2);
          ctx.lineTo(x2 + nx * w2, y2 + ny * w2); ctx.lineTo(x1 + nx * w1, y1 + ny * w1);
          ctx.fill();
          aislesDrawn.push(km * alpha);
        }
      }
      if (aislesDrawn.length > 60) {
        const near = aislesDrawn.map(a => [Math.abs(wrapPi(a - pp)), a]).sort((p, q) => p[0] - q[0]);
        aislesDrawn = near.slice(0, 60).map(p => p[1]);
      }

      // rim of the ancient circle and the frontier line
      ring(S.rho0, 'rgba(214,180,100,0.55)', 0.5);

      // ── graves ─────────────────────────────────────────────────────────────
      let drawn = 0;
      let hover: Hit | null = null;
      const mouseW = st.mouse ? unproj(st.mouse[0], st.mouse[1]) : null;
      const candles: [number, number, number][] = [];
      const lh = S.layout.rowH;

      for (let i = firstRow; i <= lastRow; i++) {
        const { count, M, fMid, alpha, gamma } = S.rowGeometry(i);
        const rhoIn = S.rowInner(i), rhoC = rhoIn + lh / 2;
        if (rhoC > rhoEdge + LIVING_BAND) break;
        const rg = ringOf(rhoC);
        const plotPx = SPRITE_W * rg.scale;
        const angMargin = 1.5 / fMid;
        const a0 = fullCircle ? pp - Math.PI : pp + thMin - angMargin;
        const a1 = fullCircle ? pp + Math.PI : pp + thMax + angMargin;
        const expected = count * (a1 - a0) / TAU;
        const rowStart = S.rowStart[i];
        const fill = Math.min(1, Math.max(0, (cumNow - rowStart) / count));

        if (plotPx < MIN_GRAVE_PX || drawn + expected > MAX_GRAVES_DRAWN) {
          // band mode: one stroke per ≥2 px of rows
          const merged = Math.max(1, Math.ceil(2 / (lh * rg.scale)));
          const mid = rhoIn + merged * lh / 2;
          i += merged - 1;
          if (fill <= 0) continue;
          ctx.strokeStyle = eraBandColor(S.time(mid));
          ctx.globalAlpha = 0.75 * fill;
          ctx.lineWidth = Math.max(1, merged * lh * rg.scale * 0.8);
          arcPath(mid, a0 - pp, a1 - pp);
          ctx.stroke();
          ctx.globalAlpha = 1;
          continue;
        }

        const s = rg.scale;
        const usable = alpha - gamma;
        const segA = Math.floor(a0 / alpha), segB = Math.floor(a1 / alpha);
        const rowT0 = S.time(rhoIn), rowT1 = S.time(rhoIn + lh);
        for (let seg = segA; seg <= segB; seg++) {
          const sm = ((seg % M) + M) % M;
          const jFirst = Math.floor(sm * count / M), jEnd = Math.floor((sm + 1) * count / M);
          const n = jEnd - jFirst;
          if (n <= 0) continue;
          const segBase = seg * alpha + gamma / 2;
          const step = usable / n;
          const l0 = Math.max(0, Math.floor((a0 - segBase) / step - 0.5));
          const l1 = Math.min(n - 1, Math.ceil((a1 - segBase) / step - 0.5));
          for (let l = l0; l <= l1; l++) {
            const phi = segBase + (l + 0.5) * step;
            const dphi = phi - pp;
            const id = rowStart + jFirst + l;
            const [sx, sy] = projRing(rg.em1, dphi);
            if (sx < -60 || sx > W + 60 || sy < -60 || sy > H + 60) continue;
            if (hash01(id, 9) >= fill) {
              // an empty plot at the frontier: sometimes a candle for the living
              if (hash01(id, 10) < 0.5 * Math.exp(-3 * Math.max(0, rhoC - rhoEdge) / LIVING_BAND))
                candles.push([sx, sy, s]);
              continue;
            }
            const year = rowT0 + (rowT1 - rowT0) * ((jFirst + l + 0.5) / count);
            const style = styleFor(id, year);
            const variants = sprites[style];
            const img = variants[hash01(id, 11) * variants.length | 0];
            const c = Math.cos(dphi), sn = Math.sin(dphi);
            ctx.setTransform(dpr * s * c, dpr * s * sn, -dpr * s * sn, dpr * s * c, dpr * sx, dpr * sy);
            ctx.drawImage(img, -SPRITE_W / 2, -SPRITE_H / 2, SPRITE_W, SPRITE_H);
            if (year > 1880 && hash01(id, 12) < 0.04 + 0.08 * Math.min(1, (year - 1880) / 140))
              candles.push([sx + (-0.35 * c) * s, sy + (-0.35 * sn) * s - 0.1 * s, s]);
            drawn++;
            if (mouseW) {
              const dr = mouseW[0] - rhoC, dt = wrapPi(mouseW[1] - phi) * fMid;
              if (Math.abs(dr) < SPRITE_H / 2 && Math.abs(dt) < SPRITE_W / 2)
                hover = { info: graveInfo(id, year), row: i, rho: rhoC, phi };
            }
          }
        }
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // ── frontier glow ──────────────────────────────────────────────────────
      if (rhoEdge > rhoMin && rhoEdge < rhoMax + 40) {
        ring(rhoEdge + 1, 'rgba(255,214,140,0.20)', 6, 2);
        ring(rhoEdge + 1, 'rgba(255,230,170,0.55)', 0.4, 1);
      }
      // living band: candles in plots beyond the last filled row
      if (rhoEdge + LIVING_BAND > rhoMin && z > 3) {
        const rowA = Math.max(S.rowOf(rhoEdge) + 1, firstRow);
        const rowB = Math.min(S.rowOf(Math.min(rhoMax, rhoEdge + LIVING_BAND)), rowA + 40);
        const f = S.f(rhoEdge), pitch = S.layout.pitch;
        for (let i = rowA; i <= rowB; i++) {
          if (i < S.nRows) continue; // those rows are handled with the graves above
          const rhoC = S.rowInner(i) + S.layout.rowH / 2;
          const rg = ringOf(rhoC);
          const d = (rhoC - rhoEdge) / LIVING_BAND;
          const j0 = Math.floor((pp + thMin) * f / pitch), j1 = Math.ceil((pp + thMax) * f / pitch);
          if (j1 - j0 > 3000) continue;
          for (let j = j0; j <= j1; j++) {
            const id = i * 1e10 + j;
            if (hash01(id, 13) > 0.5 * Math.exp(-3 * d)) continue;
            candles.push([...projRing(rg.em1, (j + 0.5) * pitch / f - pp), rg.scale]);
          }
        }
      }
      // candles (additive glow)
      if (candles.length) {
        ctx.globalCompositeOperation = 'lighter';
        const flick = time * 0.008;
        for (const [x, y, s] of candles) {
          const r = Math.max(2, 0.45 * s) * (0.85 + 0.15 * Math.sin(flick + x * 0.37 + y * 0.11));
          ctx.drawImage(glow, x - r, y - r, 2 * r, 2 * r);
        }
        ctx.globalCompositeOperation = 'source-over';
      }

      // ── year markers at aisle crossings ────────────────────────────────────
      const tA = S.time(Math.max(rhoMin, S.layout.plazaR)), tB = S.time(Math.min(rhoMax, rhoEdge));
      const marks: { t: number; landmark: boolean }[] = [];
      for (let t = Math.ceil(tA / markStep(tA)) * markStep(tA); t <= tB && marks.length < 80; ) {
        marks.push({ t, landmark: false });
        const s = markStep(t + 0.5);
        const next = Math.floor(t / s) * s + s;
        t = t < T0 && next > T0 ? T0 : next;
      }
      for (const lm of LANDMARKS) if (lm.year >= tA && lm.year <= tB) marks.push({ t: lm.year, landmark: true });
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (const { t, landmark } of marks) {
        const rho = S.rhoAtTime(t);
        const rg = ringOf(rho);
        const size = 2 * rg.scale * (landmark ? 1.6 : 1);
        if (size < 4) continue;
        const row = S.rowOf(rho);
        for (const a of aislesDrawn) {
          const lvl = Math.round(a / (TAU / aisleLevel));
          let level = aisleLevel, kk = lvl % aisleLevel;
          while (level > 4 && kk % 2 === 0) { kk /= 2; level /= 2; }
          if (row < S.firstRowWithAisles(level)) continue;
          const dphi = wrapPi(a - pp);
          const [x, y] = projRing(rg.em1, dphi);
          if (x < -40 || x > W + 40 || y < -40 || y > H + 40) continue;
          ctx.save();
          ctx.translate(x, y);
          ctx.rotate(dphi);
          ctx.drawImage(landmark ? landmarkMarker : marker, -size / 2, -size / 2, size, size);
          ctx.restore();
          if (size > 22) {
            ctx.font = `${Math.min(16, Math.max(9, size * 0.16))}px Georgia, serif`;
            ctx.fillStyle = landmark ? '#2a1f08' : '#2b271d';
            ctx.fillText(formatYear(t), x, y - size * 0.07);
          }
        }
      }

      // hover highlight
      st.hover = hover;
      if (hover) {
        const rg = ringOf(hover.rho);
        const dphi = wrapPi(hover.phi - pp);
        const [x, y] = projRing(rg.em1, dphi);
        ctx.save();
        ctx.translate(x, y); ctx.rotate(dphi);
        ctx.strokeStyle = 'rgba(255,235,180,0.9)';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(-SPRITE_W / 2 * rg.scale, -SPRITE_H / 2 * rg.scale, SPRITE_W * rg.scale, SPRITE_H * rg.scale);
        ctx.restore();
      }

      // vignette + lantern
      const vg = ctx.createRadialGradient(cx, cy, Math.min(W, H) * 0.25, cx, cy, Math.hypot(W, H) * 0.6);
      vg.addColorStop(0, 'rgba(0,0,0,0)'); vg.addColorStop(1, 'rgba(4,6,10,0.65)');
      ctx.fillStyle = vg;
      ctx.fillRect(0, 0, W, H);
      // player
      ctx.fillStyle = 'rgba(0,0,0,0.4)';
      ctx.beginPath(); ctx.ellipse(cx + 2, cy + 3, 7, 4, 0, 0, TAU); ctx.fill();
      ctx.fillStyle = '#e9dfc4';
      ctx.beginPath(); ctx.arc(cx, cy, 5, 0, TAU); ctx.fill();
      ctx.strokeStyle = '#3a3428'; ctx.lineWidth = 1.5; ctx.stroke();

      return { fp, rhoEdge, cumNow, drawn };
    }

    // ── minimap ──────────────────────────────────────────────────────────────
    const mapRadiusOf = (rho: number, rhoEdge: number, cumNow: number) =>
      opts.current.mapMode === 'distance' ? rho / rhoEdge : Math.sqrt(Math.max(0, S.cum(rho)) / cumNow);
    function drawMap(rhoEdge: number, cumNow: number) {
      const size = opts.current.bigMap ? Math.min(window.innerWidth, window.innerHeight) * 0.8 : 200;
      const px = Math.round(size * dpr);
      if (mapCanvas.width !== px) {
        mapCanvas.width = mapCanvas.height = px;
        mapCanvas.style.width = mapCanvas.style.height = `${size}px`;
      }
      mctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      mctx.clearRect(0, 0, size, size);
      const c = size / 2, R = size / 2 - 14;
      const at = (rho: number, phi: number): [number, number] => {
        const r = mapRadiusOf(rho, rhoEdge, cumNow) * R;
        return [c + r * Math.sin(phi), c - r * Math.cos(phi)];
      };
      mctx.fillStyle = 'rgba(10,14,10,0.85)';
      mctx.beginPath(); mctx.arc(c, c, R + 12, 0, TAU); mctx.fill();
      // era bands
      const eras: [number, number, string][] = [
        [-50000, -8000, '#2f3a27'], [-8000, -3000, '#3b4630'], [-3000, 400, '#4a4a36'],
        [400, 1700, '#4b4639'], [1700, 1920, '#5b5a55'], [1920, nowYear(), '#404046'],
      ];
      for (let i = eras.length - 1; i >= 0; i--) {
        const r = mapRadiusOf(S.rhoAtTime(eras[i][1]), rhoEdge, cumNow) * R;
        mctx.fillStyle = eras[i][2];
        mctx.beginPath(); mctx.arc(c, c, r, 0, TAU); mctx.fill();
      }
      mctx.strokeStyle = 'rgba(214,180,100,0.8)';
      mctx.lineWidth = 1;
      mctx.beginPath(); mctx.arc(c, c, mapRadiusOf(S.rho0, rhoEdge, cumNow) * R, 0, TAU); mctx.stroke();
      mctx.strokeStyle = 'rgba(255,225,160,0.9)';
      mctx.beginPath(); mctx.arc(c, c, R, 0, TAU); mctx.stroke();
      mctx.font = `${opts.current.bigMap ? 12 : 9}px monospace`;
      mctx.fillStyle = 'rgba(230,220,190,0.75)';
      mctx.textAlign = 'left';
      const labelled = opts.current.bigMap ? [-40000, -20000, -8000, 1, 1000, 1500, 1800, 1900, 1950, 2000] : [-8000, 1];
      for (const t of labelled) {
        const r = mapRadiusOf(S.rhoAtTime(t), rhoEdge, cumNow) * R;
        if (R - r < 4 && t !== labelled[labelled.length - 1]) continue;
        mctx.strokeStyle = 'rgba(230,220,190,0.18)';
        mctx.beginPath(); mctx.arc(c, c, r, 0, TAU); mctx.stroke();
        mctx.fillText(t === 1 ? '1 CE' : t < 0 ? `${-t / 1000}k BCE` : `${t}`, c + r * 0.72 + 2, c - r * 0.7);
      }
      const [x, y] = at(state.current.rho, state.current.phi);
      mctx.fillStyle = '#ffeab0';
      mctx.beginPath(); mctx.arc(x, y, 3.5, 0, TAU); mctx.fill();
      mctx.strokeStyle = '#ffeab0';
      mctx.beginPath();
      mctx.moveTo(x, y);
      mctx.lineTo(x + 9 * Math.sin(state.current.phi), y - 9 * Math.cos(state.current.phi));
      mctx.stroke();
    }
    const onMapClick = (e: MouseEvent) => {
      const rect = mapCanvas.getBoundingClientRect();
      const size = rect.width, c = size / 2, R = size / 2 - 14;
      const dx = e.clientX - rect.left - c, dy = e.clientY - rect.top - c;
      const frac = Math.min(1, Math.hypot(dx, dy) / R);
      const rhoEdge = S.rhoAtTime(nowYear());
      let rho: number;
      if (opts.current.mapMode === 'distance') rho = frac * rhoEdge;
      else {
        // invert sqrt(cum/cumNow) by bisection
        const target = frac * frac * S.cum(rhoEdge);
        let lo = 0, hi = rhoEdge;
        for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (S.cum(m) < target) lo = m; else hi = m; }
        rho = lo;
      }
      st.rho = Math.max(0.5, rho);
      st.phi = ((Math.atan2(dx, -dy) % TAU) + TAU) % TAU;
    };
    mapCanvas.addEventListener('click', onMapClick);

    // ── HUD ──────────────────────────────────────────────────────────────────
    let lastHud = 0;
    function drawHud(fp: number, rhoEdge: number, cumNow: number, drawn: number) {
      const rho = st.rho;
      const t = S.time(rho);
      const inAncient = rho < S.rho0;
      const K = S.gaussK(rho);
      const kText = inAncient ? 'flat (K = 0)' :
        Math.abs(K) < 1e-12 ? '≈ flat' : `K = ${K < 0 ? '−' : '+'}1/(${fmtLen(1 / Math.sqrt(Math.abs(K)))})²`;
      const D = S.model.D;
      const deaths = inAncient ? S.model.ancient.D0 * Math.exp(S.model.ancient.r * (t - T0))
        : D[Math.max(0, Math.min(D.length - 1, Math.round(t - T0)))];
      const toEdge = Math.max(0, rhoEdge - rho);
      const speed = st.speed || (opts.current.walkMode ? WALK_SPEED : (W / st.zoom) * SCREENS_PER_SECOND);
      const lines = [
        `<b>${rho < S.layout.plazaR ? 'Central plaza' : formatYear(t, inAncient)}</b>${inAncient ? ' <span class="dim">· ancient circle</span>' : ''}`,
        `graves nearer the centre <b>${fmtBig(S.cum(rho))}</b> <span class="dim">of ${fmtBig(cumNow)}</span>`,
        `deaths that year <b>${fmtBig(deaths)}</b>`,
        `from centre <b>${fmtLen(rho)}</b> · to the edge <b>${fmtLen(toEdge)}</b>`,
        `ring around here <b>${fmtLen(TAU * fp)}</b> <span class="dim">(flat plane: ${fmtLen(TAU * rho)})</span>`,
        `curvature <b>${kText}</b>`,
        inAncient ? `time here is not linear` : `1 year = <b>${S.v.toFixed(2)} m</b> outward`,
        `speed ${fmtLen(speed)}/s · edge in ${fmtDur(toEdge / speed)} · view ${fmtLen(W / st.zoom)} · ${drawn} graves drawn`,
      ];
      hudRef.current!.innerHTML = lines.join('<br>');
      const near = rho < S.layout.plazaR + 40 ? WELCOME : LANDMARKS.find(l => Math.abs(S.rhoAtTime(l.year) - rho) < 40);
      signRef.current!.innerHTML = near ? `<b>${near.title}</b>${near === WELCOME ? '' : ` · ${formatYear(near.year)}`}<br>${near.text}` : '';
      signRef.current!.style.display = near ? 'block' : 'none';
    }

    let raf = 0, last = performance.now();
    const loop = (ts: number) => {
      const dt = Math.min(0.1, (ts - last) / 1000);
      last = ts;
      update(dt);
      const { fp, rhoEdge, cumNow, drawn } = draw(ts);
      if (ts - lastHud > 100) {
        lastHud = ts;
        drawHud(fp, rhoEdge, cumNow, drawn);
        drawMap(rhoEdge, cumNow);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('mousemove', onMouseMove);
      canvas.removeEventListener('mouseleave', onMouseLeave);
      canvas.removeEventListener('click', onClick);
      mapCanvas.removeEventListener('click', onMapClick);
    };
  }, []);

  return (
    <div className="gy-root">
      <canvas ref={canvasRef} className="gy-canvas" />
      <div ref={hudRef} className="gy-hud gy-panel" />
      <div ref={signRef} className="gy-sign gy-panel" />
      <div className={`gy-map ${bigMap ? 'big' : ''}`}>
        <canvas ref={mapRef} title="click to travel" />
        <div className="gy-map-buttons">
          <button onClick={() => setMapMode(m => (m === 'graves' ? 'distance' : 'graves'))}>
            {mapMode === 'graves' ? 'area ∝ graves' : 'radius ∝ distance'}
          </button>
          <button onClick={() => setBigMap(b => !b)}>{bigMap ? 'smaller' : 'bigger'}</button>
        </div>
      </div>
      <div className="gy-controls gy-panel">
        <div className="gy-jumps">
          {JUMPS.map(([label, t]) => <button key={label} onClick={() => jump(t)}>{label}</button>)}
        </div>
        <label>
          <input type="checkbox" checked={walkMode} onChange={e => setWalkMode(e.target.checked)} /> real walking speed (1.4 m/s)
        </label>
        <div className="dim">WASD / arrows move · Shift ×5 · wheel or +/− zoom · click a grave · click the map to travel</div>
        <button className="gy-about-btn" onClick={() => setShowAbout(true)}>how is this built?</button>
      </div>
      {selected && (
        <div className="gy-card gy-panel">
          <button className="gy-close" onClick={() => setSelected(null)}>×</button>
          <div className="gy-card-title">Grave #{selected.info.id.toLocaleString('en-US')}</div>
          <div>died <b>{formatYear(selected.info.year, selected.info.year < T0)}</b></div>
          <div>{selected.info.sex}, aged {selected.info.ageText}</div>
          <div className="dim">row {selected.row.toLocaleString('en-US')} · {fmtLen(selected.rho)} from the centre</div>
          <div className="dim small">Details are procedurally generated from the grave id — plausible, not real.</div>
        </div>
      )}
      {showAbout && <About onClose={() => setShowAbout(false)} />}
    </div>
  );
}

function About({ onClose }: { onClose: () => void }) {
  const m = S.model;
  return (
    <div className="gy-about-backdrop" onClick={onClose}>
      <div className="gy-about gy-panel" onClick={e => e.stopPropagation()}>
        <button className="gy-close" onClick={onClose}>×</button>
        <h2>The geometry</h2>
        <p>
          One grave per person who ever died, at uniform density σ = 1 / ({S.layout.rowH} m × {S.layout.pitch} m).
          The surface is rotationally symmetric, ds² = dρ² + f(ρ)² dφ², so a ring at distance ρ is 2π·f(ρ) long.
        </p>
        <ul>
          <li><b>Ancient circle</b> (before 8000 BCE, {fmtBig(m.ancient.N)} graves): flat disc, f = ρ, radius ρ₀ = <b>{fmtLen(S.rho0)}</b>.</li>
          <li><b>Outer region</b>: time is linear, ρ = ρ₀ + v·(t + 8000). Uniform density forces f = D(t) / (2πσv), where D is deaths per year.</li>
          <li>f(ρ₀) = ρ₀ (no tear at the rim) fixes v = D(−8000) / (2πσρ₀) = <b>{S.v.toFixed(3)} m / year</b>. Nothing else is tunable; σ only scales everything.</li>
          <li>Gaussian curvature K = −f″/f = −D″ / (D v²). Where deaths grow faster than exponentially the ground is saddle-shaped.</li>
          <li>The view is a conformal map: isothermal coordinate u = ∫dρ/f, screen = f(ρ_you)·(e^(Δu + iΔφ) − 1). It is exact at your feet; zoom out to see the rest of the world shrink or grow.</li>
        </ul>
        <h2>The death model</h2>
        <p>
          PRB benchmark periods plus OWID yearly deaths (5-year bins). Each period gets a fixed shape, the whole curve is smoothed
          (Gaussian, 200 yr → 2.5 yr wide), and the period weights are solved as a linear system so every total matches exactly.
        </p>
        <table>
          <thead><tr><th>period</th><th>deaths (data)</th><th>model</th></tr></thead>
          <tbody>
            {m.periods.filter(p => p.b - p.a > 5 || p.a % 25 === 0).map(p => (
              <tr key={p.source}><td>{p.source}</td><td>{fmtBig(p.target)}</td><td>{((p.fitted / p.target - 1) * 100).toExponential(1)} %</td></tr>
            ))}
          </tbody>
        </table>
        <p className="dim">Total by {Math.floor(nowYear())}: {fmtBig(S.cum(S.rhoAtTime(nowYear())))} graves · {S.nRows.toLocaleString('en-US')} rows · edge {fmtLen(S.rhoAtTime(nowYear()))} from the centre.</p>
      </div>
    </div>
  );
}
