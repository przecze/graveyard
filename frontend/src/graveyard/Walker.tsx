import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { Surface, T_END, type Shape } from './surface';
import { fmtDur, type Motion } from './motion';
import { DEFAULT_PRESET } from './presets';
import { buildMarker, buildSprites, CELL, SPRITE_H, SPRITE_W, AGES, type Age } from './sprites';
import { ageOf, formatYear, hash01, styleFor, type GraveInfo, type Landmark } from './lore';
import { OWID_DEATHS, OWID_FIRST_YEAR, PRB } from './data';
import { Soundscape } from './soundscape';
import { buildProfile, drawProfile, type Profile } from './profile';
import './walker.css';

const ModelDialog = lazy(() => import('./ModelDialog'));
const Settings = lazy(() => import('./Settings'));

const TAU = 2 * Math.PI;
// zoom is px per metre internally (a grave is a fixed number of metres); the interface uses years
const MAX_ZOOM = 90;
const DEFAULT_VIEW_YR = DEFAULT_PRESET.viewYr; // view width at start
const MIN_GRAVE_PX = 3;          // below this plot width rows are drawn as bands
const SIMPLE_GRAVE_PX = 10;      // below this graves are simple stone blocks (one path per style)
const ROTATE_GROUND_PX = 24;     // below this grave sprites are drawn unrotated (cheaper, invisible at that size)
const MAX_AISLES_DRAWN = 240;
const MAX_BEDS_PER_RING = 400;
const LIVING_BAND = 45;          // walkable metres (empty plots) beyond the edge
const PLAYER_R = 0.22;           // walker radius for collisions (m)

const wrapPi = (a: number) => a - TAU * Math.floor((a + Math.PI) / TAU);

// simple stand-in for small graves: the typical stone colour for its age (ancient, older, recent)
const BLOCK_COLOR: Record<Age, string> = { 0: '#8f8c80', 1: '#a8a192', 2: '#6f7075' };
// far-away / zoomed-out ground: that colour mixed into the grass beds
const AGE_BORDER = 1775; // older graves look weathered (switch year is the other border)
function mixHex(a: string, b: string, k: number): string {
  const p = (h: string, i: number) => parseInt(h.slice(1 + 2 * i, 3 + 2 * i), 16);
  return `rgb(${[0, 1, 2].map(i => Math.round(p(a, i) * (1 - k) + p(b, i) * k)).join(',')})`;
}
const eraTint = (t: number) => mixHex(BLOCK_COLOR[ageOf(t, S.t0)], '#243020', 0.6);

// "Now" as a fractional calendar year, drives the edge in real time
function nowYear(): number {
  const d = new Date();
  const y = d.getUTCFullYear();
  const start = Date.UTC(y, 0, 1), end = Date.UTC(y + 1, 0, 1);
  return Math.min(T_END - 0.01, y + (d.getTime() - start) / (end - start));
}

function markStep(t: number): number {
  if (t < 0) return 500;
  if (t < 1500) return 100;
  if (t < 1800) return 50;
  return 10;
}

/** a fraction as a percentage with enough digits to be non-zero */
const fmtPct = (x: number) => { const p = 100 * x; return p <= 0 ? '0%' : p >= 10 ? `${p.toFixed(0)}%` : `${p.toPrecision(2)}%`; };
const fmtBig = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(n >= 1e11 ? 1 : 2)} B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} M` : Math.round(n).toLocaleString('en-US');
/** a distance in metres, shown in history-years */
const fmtLen = (m: number) => {
  const y = S.yr(m), a = Math.abs(y);
  return a >= 1e6 ? `${(y / 1e6).toFixed(1)} M yr` : a >= 1e5 ? `${Math.round(y / 1e3)} k yr`
    : a >= 100 ? `${Math.round(y).toLocaleString('en-US')} yr` : `${y.toFixed(a >= 10 ? 0 : a >= 1 ? 1 : 2)} yr`;
};

// settings survive reloads (tuning sessions); world changes rebuild S and restart at the centre
const STORE_KEY = 'gy-settings-v9';
function loadSettings(): { shape: Shape; motion: Motion } {
  try {
    const raw = JSON.parse(localStorage.getItem(STORE_KEY) ?? '{}');
    return { shape: { ...DEFAULT_PRESET.shape, ...raw.shape }, motion: { ...DEFAULT_PRESET.motion, ...raw.motion } };
  } catch { return { shape: DEFAULT_PRESET.shape, motion: DEFAULT_PRESET.motion }; }
}
function saveSettings(shape: Shape, motion: Motion) {
  try { localStorage.setItem(STORE_KEY, JSON.stringify({ shape, motion })); } catch { /* private mode */ }
}
const initial = loadSettings();
function initialSurface(): Surface {
  try { const s = new Surface(initial.shape); if (s.report.ok) return s; } catch { /* fall back */ }
  return new Surface(DEFAULT_PRESET.shape);
}

// the render loop reads S every frame
let S = initialSurface();
const sound = new Soundscape();

/** fraction of row i already filled at cumNow graves (the newest row fills in real time) */
function rowFill(i: number, cumNow: number): number {
  const G = S.grid;
  if (i < 0 || i >= G.nRows) return 0;
  const c = G.rowStart[i + 1] - G.rowStart[i];
  return c > 0 ? Math.min(1, Math.max(0, (cumNow - G.rowStart[i]) / c)) : 0;
}
const minZoom = () => Math.min(window.innerWidth, window.innerHeight) / (2.4 * S.rhoEndTable);

const welcome = () => ({
  year: -50000, title: 'The dawn of humanity',
  text: `Every person who ever died has a grave here, about ${Math.round(S.rowStart[S.nRows] / 1e9)} billion of them. `
    + `First you cross the ancient era: everyone who died before ${formatYear(S.t0)}, undated. Then history begins and time moves forward with every step, to the newest graves at the edge, ${fmtLen(S.rhoEndTable)} away (distances here are in years of history).`,
});

type Hit = { info: GraveInfo; row: number; rho: number; phi: number };

// NaN = the edge; strings are special places
const JUMPS: [string, number | 'centre' | 'ancient' | 'history'][] = [
  ['Centre', 'centre'], ['Ancient era', 'ancient'], ['History begins', 'history'], ['1 CE', 1],
  ['1500', 1500], ['1900', 1900], ['2000', 2000], ['Edge (now)', NaN],
];

const ancientSign = () => ({
  year: -50000, title: 'The ancient era',
  text: `${fmtBig(S.ancientGraves)} people died before ${formatYear(S.t0)}. They rest here together, undated, in no particular order of years. `
    + `The ground curves so all of them fit; ${fmtLen(S.rho0)} from the centre, history begins.`,
});
const historySign = () => ({
  year: S.t0, title: 'History begins',
  text: `From this ring outward every grave is dated and time is linear: every ${S.rowsPerYear.toFixed(2)} rows of graves is one year. `
    + `Behind you lie the ${fmtBig(S.ancientGraves)} of the ancient era.`,
});
// landmark signs: numbers come from the data (PRB, OWID) and the model, never typed in
const owid = (y: number) => OWID_DEATHS[y - OWID_FIRST_YEAR];
const prbPop = (y: number) => PRB.find(r => r.year === y)?.pop ?? NaN;
const landmarks = (): Landmark[] => [historySign(), ...([
  { year: 1, title: 'Year 1', text: `${fmtBig(S.model.cum(1))} people had died before this ring: ${Math.round(100 * S.model.cum(1) / S.model.cum(T_END))}% of all graves here. PRB: world population ${fmtBig(prbPop(1))}.` },
  { year: 1200, title: 'Coarse data', text: 'From 1 CE to 1650 PRB gives only totals for 1–1200 and 1200–1650, so the Black Death (1347–1351) is inside a smoothed period: the rings do not show its spike.' },
  { year: 1650, title: '500 million alive', text: `PRB: world population ${fmtBig(prbPop(1650))}. The model has ${fmtBig(S.model.D(1650))} deaths a year here, ${fmtBig(S.model.D(1850))} by 1850.` },
  { year: 1850, title: 'A billion alive', text: `PRB: world population ${fmtBig(prbPop(1850))}. The ring here is ${fmtLen(2 * Math.PI * S.f(S.rhoAtTime(1850)))} around; a flat plane would give ${fmtLen(2 * Math.PI * S.rhoAtTime(1850))}.` },
  { year: 1918, title: 'Pandemic & wars', text: `The PRB period 1900–1950 (${fmtBig(S.model.cum(1950) - S.model.cum(1900))} deaths) includes the 1918 flu and both world wars; only its total is known, so the model spreads it smoothly.` },
  { year: 1950, title: 'Yearly data', text: `From here the model is fitted to Our World in Data decade totals: ${fmtBig(owid(1950))} deaths in 1950.` },
  { year: 2020, title: 'COVID-19', text: `OWID: ${fmtBig(owid(2019))} deaths in 2019, ${fmtBig(owid(2021))} in 2021. After ${OWID_FIRST_YEAR + OWID_DEATHS.length - 1} the model extrapolates.` },
] as Landmark[]).filter(l => l.year > S.t0 + 100)];

export default function Walker() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mapRef = useRef<HTMLCanvasElement>(null);
  const hudRef = useRef<HTMLDivElement>(null);
  const signRef = useRef<HTMLDivElement>(null);
  const profRef = useRef<HTMLCanvasElement>(null);
  const [selected, setSelected] = useState<Hit | null>(null);
  const [showAbout, setShowAbout] = useState(false);
  const [showModel, setShowModel] = useState(false);
  const [mapMode, setMapMode] = useState<'distance' | 'graves'>('graves');
  const [bigMap, setBigMap] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [soundOn, setSoundOn] = useState(false);
  const [motion, setMotionState] = useState<Motion>(initial.motion);
  // settings are saved only on an explicit change (never from an effect, so a
  // hot reload keeping old in-memory state cannot overwrite them)
  const setMotion = (m: Motion) => { setMotionState(m); saveSettings(S.shape, m); };
  const [surface, setSurface] = useState(S);

  const state = useRef({
    rho: 6,
    phi: 0,
    zoom: window.innerWidth / (DEFAULT_VIEW_YR * S.v),
    speed: 0,
    hover: null as Hit | null,
    mouse: null as [number, number] | null,
    patternOffset: [0, 0] as [number, number],
  });
  // dev only: lets browser tests place the walker (window.__gy.state.rho = …)
  if (import.meta.env.DEV) (window as unknown as { __gy: unknown }).__gy = { state: state.current, S: () => S };
  const opts = useRef({ mapMode, motion, bigMap });
  opts.current = { mapMode, motion, bigMap };
  const setSelectedRef = useRef(setSelected);
  setSelectedRef.current = setSelected;


  // world settings: rebuild and restart at the centre, keeping the view width in years
  const applyWorld = (next: Surface) => {
    if (!next.report.ok || next === S) return;
    const st = state.current;
    st.zoom *= S.v / next.v;
    S = next;
    st.rho = 4; st.phi = 0;
    setSelected(null);
    setSurface(next);
    saveSettings(next.shape, motion);
  };
  const viewYr = () => window.innerWidth / (state.current.zoom * S.v);
  const setViewYr = (y: number) => {
    state.current.zoom = Math.min(MAX_ZOOM, Math.max(minZoom(), window.innerWidth / (y * S.v)));
  };

  // jumps land on the nearest path, so collisions never trap you
  const jump = (t: number | 'centre' | 'ancient' | 'history') => {
    const st = state.current;
    st.rho = S.grid.walkableNear(t === 'centre' ? 4 : t === 'ancient' ? S.rho0 / 2 : t === 'history' ? S.rho0 + 1.3
      : isNaN(t) ? S.rhoAtTime(nowYear()) - 12 : S.rhoAtTime(t) + 1.3);
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
      if (k === '-' || k === '_') st.zoom = Math.max(minZoom(), st.zoom / 1.25);
      if (k === 'escape') setSelectedRef.current(null);
    };
    const onKeyUp = (e: KeyboardEvent) => keys.delete(e.key.toLowerCase());
    const onBlur = () => keys.clear();
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      st.zoom = Math.min(MAX_ZOOM, Math.max(minZoom(), st.zoom * Math.exp(-e.deltaY * 0.0015)));
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    const onMouseMove = (e: MouseEvent) => { st.mouse = [e.clientX, e.clientY]; };
    const onMouseLeave = () => { st.mouse = null; };
    const onClick = () => setSelectedRef.current(st.hover);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseleave', onMouseLeave);
    canvas.addEventListener('click', onClick);

    // ── movement ─────────────────────────────────────────────────────────────
    /** base speed in m/s */
    const baseSpeed = () => {
      const m = opts.current.motion;
      return m.mode === 'fixed' ? m.yrPerSec * S.v : (W / st.zoom) * m.screensPerSec;
    };
    /** does the walker at (ρ, φ) overlap a filled plot? (plots are solid, paths are free) */
    function blocked(rho: number, phi: number, cumNow: number): boolean {
      const G = S.grid, half = CELL / 2 + PLAYER_R, f = S.f(rho);
      for (const r of [rho - half, rho, rho + half]) {
        const i = G.rowAt(r);
        if (i < 0 || i >= G.nRows || Math.abs(rho - G.rowCenter(i)) >= half) continue;
        for (const d of [-half, 0, half]) {
          const g = G.graveAt(i, phi + d / f);
          if (g && Math.abs(wrapPi(phi - g.phi)) * f < half && hash01(G.rowStart[i] + g.j, 9) < rowFill(i, cumNow)) return true;
        }
      }
      return false;
    }
    function update(dt: number) {
      let ix = 0, iy = 0;
      if (keys.has('arrowleft') || keys.has('a')) ix -= 1;
      if (keys.has('arrowright') || keys.has('d')) ix += 1;
      if (keys.has('arrowup') || keys.has('w')) iy += 1;
      if (keys.has('arrowdown') || keys.has('s')) iy -= 1;
      const speed = baseSpeed() * (keys.has('shift') ? opts.current.motion.boost : 1);
      st.speed = ix || iy ? speed : 0;
      if (!ix && !iy) return;
      const n = Math.hypot(ix, iy);
      const dx = ix / n * speed * dt, dy = iy / n * speed * dt;
      const step = (dx: number, dy: number): [number, number] => {
        if (st.rho + dy <= S.coreR) {
          // exact Euclidean step in the flat circle (lets you walk through the centre)
          const x = st.rho * Math.cos(st.phi) - dx * Math.sin(st.phi) + dy * Math.cos(st.phi);
          const y = st.rho * Math.sin(st.phi) + dx * Math.cos(st.phi) + dy * Math.sin(st.phi);
          return [Math.max(0.3, Math.hypot(x, y)), Math.atan2(y, x)];
        }
        return [st.rho + dy, st.phi + dx / S.f(st.rho + dy)];
      };
      // collisions: try the move, else slide along one axis (substeps so fast walking can't tunnel)
      // if a plot ever ends up under the walker (a jump, the frontier filling in, a
      // toggle), step out onto the nearest path first
      if (opts.current.motion.collide && blocked(st.rho, st.phi, S.cum(S.rhoAtTime(nowYear())))) st.rho = S.grid.walkableNear(st.rho);
      const nSub = opts.current.motion.collide ? Math.min(40, Math.ceil(Math.hypot(dx, dy) / 0.2)) : 1;
      const cumNow = S.cum(S.rhoAtTime(nowYear()));
      let moved = false;
      for (let k = 0; k < nSub; k++) {
        const sx = dx / nSub, sy = dy / nSub;
        let next: [number, number] | null = null;
        for (const [ax, ay] of [[sx, sy], [0, sy], [sx, 0]] as [number, number][]) {
          if (!ax && !ay) continue;
          const c = step(ax, ay);
          if (!opts.current.motion.collide || !blocked(c[0], c[1], cumNow)) { next = c; break; }
        }
        if (!next) break;
        [st.rho, st.phi] = next;
        moved = true;
      }
      if (moved) { st.patternOffset[0] -= dx * st.zoom; st.patternOffset[1] += dy * st.zoom; }
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

      const G = S.grid, L = G.L;
      const rhoPlots = Math.min(rhoMax, rhoEdge + LIVING_BAND); // plots (filled or not) end here

      // ── ground: gravel everywhere inside the graveyard, grass beds under the plots ──
      // the gaps between beds are the narrow paths and cut-throughs
      const annulus = (r0: number, r1: number, color: string) => {
        if (r1 <= r0) return;
        const a0 = thMin - 0.01, a1 = thMax + 0.01;
        const pts = (rho: number, from: number, to: number) => {
          const { em1 } = ringOf(rho), rs = R0 * (1 + em1);
          const n = Math.max(2, Math.min(600, Math.ceil(Math.abs(to - from) * rs / 12)));
          const out: [number, number][] = [];
          for (let k = 0; k <= n; k++) out.push(projRing(em1, from + (to - from) * k / n));
          return out;
        };
        ctx.fillStyle = color;
        ctx.beginPath();
        const outer = pts(r1, a0, a1), inner = r0 > 0.5 ? pts(r0, a1, a0) : [[cx, Oy] as [number, number]];
        outer.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        inner.forEach(([x, y]) => ctx.lineTo(x, y));
        ctx.closePath();
        ctx.fill();
      };
      const skip = (import.meta.env.DEV && (window as unknown as { __gySkip?: Record<string, boolean> }).__gySkip) || {};
      if (!skip.annulus) annulus(Math.max(rhoMin, L.plazaR), Math.min(rhoMax + 3, rhoEdge + LIVING_BAND + L.avenue), '#4d4739');

      // detail range [dIn, dOut]: where plots are at least MIN_GRAVE_PX on screen. Beyond it
      // (far out in a bulb, or zoomed out) the ground is one era-tinted fill: no per-row
      // stripes, no sub-pixel roads
      const pxAt = (rho: number) => L.cell * ringOf(rho).scale;
      const lo = Math.max(rhoMin, L.plazaR), hi = Math.max(lo, Math.min(rhoMax, rhoEdge));
      const here = Math.min(Math.max(rp, lo), hi);
      let dIn = lo, dOut = hi;
      if (pxAt(here) < MIN_GRAVE_PX) dIn = dOut = here;
      else {
        if (pxAt(hi) < MIN_GRAVE_PX) { let a = here, b = hi; for (let k = 0; k < 40; k++) { const m = (a + b) / 2; if (pxAt(m) >= MIN_GRAVE_PX) a = m; else b = m; } dOut = a; }
        if (pxAt(lo) < MIN_GRAVE_PX) { let a = lo, b = here; for (let k = 0; k < 40; k++) { const m = (a + b) / 2; if (pxAt(m) >= MIN_GRAVE_PX) b = m; else a = m; } dIn = b; }
      }
      const tinted = (r0: number, r1: number) => { // split at era borders
        if (r1 <= r0) return;
        const cuts = [r0, ...[S.t0, AGE_BORDER].map(t => S.rhoAtTime(t)).filter(r => r > r0 && r < r1), r1];
        for (let k = 0; k + 1 < cuts.length; k++) annulus(cuts[k], cuts[k + 1], eraTint(S.time((cuts[k] + cuts[k + 1]) / 2)));
      };
      if (!skip.annulus) { tinted(lo, dIn); tinted(dOut, hi); }
      const roadsTo = dOut >= hi ? rhoPlots : dOut;

      const lastRow = Math.min(G.nRows - 1, G.rowNear(rhoMax) + 3);
      const firstRow = Math.max(0, G.rowNear(rhoMin) - 3);
      const cellPx = L.cell * z * fp / S.f(rp); // plot size near the player
      const B0 = G.blockOf(firstRow), B1 = G.blockOf(Math.max(firstRow, G.rowNear(rhoPlots) + 3));

      if (cellPx >= 2 && !skip.beds) {
        // beds: one grass strip per pair of rows per run
        ctx.strokeStyle = '#243020';
        for (let B = B0; B <= B1; B++) {
          for (let p = 0; p < L.pairsPerBlock; p++) {
            const r0 = G.blockInner(B) + p * G.m.pairH, rc = r0 + L.cell;
            if (rc + L.cell < dIn || rc - L.cell > dOut) continue;
            // judge each ring on its own: in a bulb, rings a little further out can be
            // hundreds of times longer, with far too many (sub-pixel) beds to draw
            const { scale } = ringOf(rc);
            if (2 * L.cell * scale < 1.5) continue;
            const fr = S.f(rc), margin = 2 / fr;
            const nRuns = (thMax - thMin + 2 * margin) * fr / (L.run * G.m.slot);
            if (nRuns > MAX_BEDS_PER_RING) continue;
            ctx.lineWidth = 2 * L.cell * scale;
            G.runs(B, pp + thMin - margin, pp + thMax + margin, (p0, p1) => { arcPath(rc, p0 - pp, p1 - pp); ctx.stroke(); });
          }
        }
      }

      // ── ring roads and avenues ───────────────────────────────────────────────
      for (let B = Math.max(0, B0 - 1); B <= B1; B++) {
        if (skip.roads) break;
        const w = G.roadAfter(B), rc = G.blockOuter(B) + w / 2;
        if (rc > roadsTo) break;
        if (rc < dIn) continue;
        const main = w > L.blockRoad;
        if (w * ringOf(rc).scale < (main ? 0.6 : 1.2)) continue;
        ring(rc, main ? '#7a7261' : '#5d5748', w, 0.5);
      }

      // ── radial roads (branching outward), avenues lighter ─────────────────────
      let aislesDrawn: number[] = [];
      let aisleLevel = 4;
      if (lastRow >= 0 && firstRow <= lastRow && !skip.aisles) {
        let M = G.blockM[G.blockOf(lastRow)];
        const span = fullCircle ? TAU : thMax - thMin;
        while (M > 4 && span / (TAU / M) > MAX_AISLES_DRAWN) M /= 2;
        aisleLevel = M;
        const alpha = TAU / M;
        const k0 = fullCircle ? 0 : Math.floor((pp + thMin) / alpha);
        const k1 = fullCircle ? M - 1 : Math.ceil((pp + thMax) / alpha);
        // blocks tall enough on screen to get their own road wedge (computed once per frame)
        const big = new Uint8Array(Math.max(0, B1 - B0 + 1));
        for (let B = B0; B <= B1; B++) big[B - B0] = G.m.blockH * ringOf((G.blockInner(B) + G.blockOuter(B)) / 2).scale >= 3 ? 1 : 0;
        const bigBlock = (B: number) => big[B - B0] === 1;
        for (let k = k0; k <= k1; k++) {
          const km = ((k % M) + M) % M;
          let level = M, kk = km;
          while (level > 4 && kk % 2 === 0) { kk /= 2; level /= 2; }
          const dphi = wrapPi(km * alpha - pp);
          const nx = Math.cos(dphi), ny = Math.sin(dphi);
          let any = false;
          const quad = (r0: number, r1: number, B: number) => {
            r0 = Math.max(r0, rhoMin, dIn); r1 = Math.min(r1, roadsTo);
            if (r0 >= r1) return;
            any = true;
            const Mb = G.blockM[B], wM = G.roadWidth(km * Mb / M, Mb);
            // the road fills exactly the angular gap the graves leave for it (wM at the block's
            // middle ring): a wedge, which the conformal view draws as straight edges through O
            const half = wM / 2 / G.blockF[B];
            const a = ringOf(r0), b = ringOf(r1);
            const [x1, y1] = projRing(a.em1, dphi), [x2, y2] = projRing(b.em1, dphi);
            const w1 = half * R0 * a.e, w2 = half * R0 * b.e;
            ctx.fillStyle = wM > L.blockRoad ? '#7a7261' : '#5d5748';
            ctx.beginPath();
            ctx.moveTo(x1 - nx * w1, y1 - ny * w1); ctx.lineTo(x2 - nx * w2, y2 - ny * w2);
            ctx.lineTo(x2 + nx * w2, y2 + ny * w2); ctx.lineTo(x1 + nx * w1, y1 + ny * w1);
            ctx.fill();
          };
          for (const [ra, rb] of G.aisleRowRuns(level)) {
            const Ba = Math.max(G.blockOf(ra), B0), Bb = Math.min(G.blockOf(rb - 1), B1);
            if (Ba > Bb) continue;
            // one wedge per block where blocks are visible; tiny far-away blocks of equal width
            // are merged (a road only widens where it becomes an avenue)
            const widthAt = (B: number) => G.roadWidth(km * G.blockM[B] / M, G.blockM[B]);
            let start = Ba;
            for (let B = Ba; B <= Bb; B++) {
              if (B < Bb && widthAt(B + 1) === widthAt(start) && !bigBlock(B) && !bigBlock(B + 1)) continue;
              const r0 = start === 0 ? L.plazaR : G.blockInner(start) - G.roadAfter(start - 1);
              quad(r0, B === G.nBlocks - 1 ? Infinity : G.blockOuter(B) + G.roadAfter(B), start);
              start = B + 1;
            }
          }
          if (any) aislesDrawn.push(km * alpha);
        }
      }
      if (aislesDrawn.length > 60) {
        const near = aislesDrawn.map(a => [Math.abs(wrapPi(a - pp)), a]).sort((p, q) => p[0] - q[0]);
        aislesDrawn = near.slice(0, 60).map(p => p[1]);
      }

      // central plaza
      if (rhoMin < L.plazaR) {
        ctx.fillStyle = '#5a554a';
        arcPath(L.plazaR, -Math.PI, Math.PI);
        ctx.fill();
        ring(L.plazaR * 0.55, '#6a6456', 0.6);
        const [mx, my] = proj(0.01, pp);
        ctx.fillStyle = '#d8d0bc';
        ctx.beginPath(); ctx.arc(mx, my, 0.8 * z, 0, TAU); ctx.fill();
      }

      // rim of the ancient zone
      ring(S.rho0, 'rgba(214,180,100,0.55)', 0.5);

      // ── graves ─────────────────────────────────────────────────────────────
      let drawn = 0;
      let rotated = true; // a rotated transform is set (reset before screen-space draws)
      const blocks: Record<Age, number[]> = { 0: [], 1: [], 2: [] };

      for (let i = firstRow; i <= lastRow; i++) {
        const rhoC = G.rowCenter(i);
        if (rhoC < rhoMin - L.cell || rhoC > rhoMax + L.cell) continue;
        if (rhoC > rhoEdge + LIVING_BAND) break;
        const count = G.rowStart[i + 1] - G.rowStart[i];
        const fRow = S.f(rhoC);
        const rg = ringOf(rhoC);
        const plotPx = L.cell * rg.scale;
        const angMargin = 1.5 / fRow;
        const a0 = fullCircle ? pp - Math.PI : pp + thMin - angMargin;
        const a1 = fullCircle ? pp + Math.PI : pp + thMax + angMargin;
        const rowStart = G.rowStart[i];
        const fill = rowFill(i, cumNow);

        if (skip.graves || plotPx < MIN_GRAVE_PX || rhoC < dIn - L.cell || rhoC > dOut + L.cell) continue; // tinted fill there

        const s = rg.scale;
        const rowT0 = S.time(G.catchInner(i)), rowT1 = S.time(G.catchInner(i + 1));
        const cullM = s + 20;
        G.graves(i, a0, a1, (j, phi, slotA) => {
          const dphi = phi - pp;
          const id = rowStart + j;
          const [sx, sy] = projRing(rg.em1, dphi);
          if (sx < -cullM || sx > W + cullM || sy < -cullM || sy > H + cullM) return;
          if (hash01(id, 9) >= fill) return; // an empty plot at the frontier
          const year = rowT0 + (rowT1 - rowT0) * ((j + 0.5) / count);
          const age = ageOf(year, S.t0);
          const k = Math.min(1, 0.97 * slotA * fRow / L.cell); // narrow inner slots: shrink to fit the bed
          if (plotPx < SIMPLE_GRAVE_PX) { // small: just the stone's footprint, batched by age
            blocks[age].push(sx - 0.28 * s * k, sy - 0.32 * s * k, 0.56 * s * k);
            drawn++;
            return;
          }
          const variants = sprites[styleFor(id)][age];
          const img = variants[hash01(id, 11) * variants.length | 0];
          // one sprite per grave: the monument inside its square, rotated with the grid
          // (small plots unrotated: cheaper, and invisible at that size)
          if (plotPx >= ROTATE_GROUND_PX) {
            const c = Math.cos(dphi), sn = Math.sin(dphi);
            ctx.setTransform(dpr * s * k * c, dpr * s * k * sn, -dpr * s * k * sn, dpr * s * k * c, dpr * sx, dpr * sy);
            ctx.drawImage(img, -SPRITE_W / 2, -SPRITE_H / 2, SPRITE_W, SPRITE_H);
            rotated = true;
          } else {
            if (rotated) { ctx.setTransform(dpr, 0, 0, dpr, 0, 0); rotated = false; }
            ctx.drawImage(img, sx - SPRITE_W / 2 * s * k, sy - SPRITE_H / 2 * s * k, SPRITE_W * s * k, SPRITE_H * s * k);
          }
          drawn++;
        });
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      // small graves: one filled path per age (size varies per row, so each block carries its row scale)
      for (const age of AGES) {
        const b = blocks[age];
        if (!b.length) continue;
        ctx.fillStyle = BLOCK_COLOR[age];
        ctx.beginPath();
        for (let k = 0; k < b.length; k += 3) ctx.rect(b[k], b[k + 1], b[k + 2], b[k + 2] * 1.1);
        ctx.fill();
      }

      // hover: the plot under the mouse
      let hover: Hit | null = null;
      if (st.mouse && cellPx >= MIN_GRAVE_PX) {
        const [mr, mphi] = unproj(st.mouse[0], st.mouse[1]);
        const i = G.rowAt(mr), g = G.graveAt(i, mphi);
        if (g && Math.abs(wrapPi(mphi - g.phi)) * S.f(mr) < L.cell / 2) {
          const id = G.rowStart[i] + g.j, count = G.rowStart[i + 1] - G.rowStart[i];
          if (hash01(id, 9) < rowFill(i, cumNow)) {
            const year = S.time(G.catchInner(i)) + (S.time(G.catchInner(i + 1)) - S.time(G.catchInner(i))) * ((g.j + 0.5) / count);
            hover = { info: { id, year }, row: i, rho: G.rowCenter(i), phi: g.phi };
          }
        }
      }

      // ── year markers at aisle crossings ────────────────────────────────────
      // no dates in the ancient zone: markers start at the switch
      const tA = Math.max(S.t0, S.time(Math.max(rhoMin, S.rho0))), tB = S.time(Math.max(S.rho0, Math.min(rhoMax, rhoEdge)));
      const marks: { t: number; landmark: boolean }[] = [];
      for (let t = Math.ceil(tA / markStep(tA)) * markStep(tA); t <= tB && marks.length < 80; ) {
        marks.push({ t, landmark: false });
        const s = markStep(t + 0.5);
        t = Math.floor(t / s) * s + s;
      }
      if (rhoMax < S.rho0) marks.length = 0;
      for (const lm of landmarks()) if (lm.year >= tA && lm.year <= tB) marks.push({ t: lm.year, landmark: true });
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (const { t, landmark } of marks) {
        const rho = S.rhoAtTime(t);
        const rg = ringOf(rho);
        const size = 2 * rg.scale * (landmark ? 1.6 : 1);
        if (size < 4) continue;
        const row = G.rowNear(rho);
        for (const a of aislesDrawn) {
          const lvl = Math.round(a / (TAU / aisleLevel));
          let level = aisleLevel, kk = lvl % aisleLevel;
          while (level > 4 && kk % 2 === 0) { kk /= 2; level /= 2; }
          if (!G.aisleRowRuns(level).some(([ra, rb]) => row >= ra && row < rb)) continue;
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

      // vignette
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
        [-50000, S.t0, '#2f3a27'], [S.t0, 400, '#4a4a36'],
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
      const labelled = opts.current.bigMap ? [S.t0, 1, 1000, 1500, 1800, 1900, 1950, 2000] : [S.t0, 1];
      for (const t of labelled) {
        const r = mapRadiusOf(S.rhoAtTime(t), rhoEdge, cumNow) * R;
        if (R - r < 4 && t !== labelled[labelled.length - 1]) continue;
        mctx.strokeStyle = 'rgba(230,220,190,0.18)';
        mctx.beginPath(); mctx.arc(c, c, r, 0, TAU); mctx.stroke();
        mctx.fillText(t === S.t0 ? 'history' : t === 1 ? '1 CE' : `${t}`, c + r * 0.72 + 2, c - r * 0.7);
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
      st.rho = S.grid.walkableNear(Math.max(0.5, rho));
      st.phi = ((Math.atan2(dx, -dy) % TAU) + TAU) % TAU;
    };
    mapCanvas.addEventListener('click', onMapClick);

    // ── HUD ──────────────────────────────────────────────────────────────────
    let lastHud = 0;
    let prof: Profile | null = null;
    function drawHud(fp: number, rhoEdge: number, cumNow: number, drawn: number) {
      const rho = st.rho;
      const t = S.time(rho);
      const inAncient = rho < S.rho0;
      const K = S.gaussK(rho);
      const kText = rho < S.grid.L.plazaR ? 'flat (plaza)' :
        Math.abs(K) < 1e-12 ? '≈ flat' : `K = ${K < 0 ? '−' : '+'}1/(${fmtLen(1 / Math.sqrt(Math.abs(K)))})²`;
      const deaths = S.model.D(t);
      const toEdge = Math.max(0, rhoEdge - rho);
      const speed = st.speed || baseSpeed();
      const lines = [
        `<b>${rho < S.grid.L.plazaR ? 'Central plaza' : inAncient ? `Ancient era` : formatYear(t)}</b>`
          + (inAncient ? ` <span class="dim">· before ${formatYear(S.t0)}, undated</span>` : ''),
        `graves nearer the centre <b>${fmtBig(S.cum(rho))}</b> <span class="dim">of ${fmtBig(cumNow)}</span>`,
        inAncient ? `ancient era <b>${(100 * rho / S.rho0).toFixed(0)}%</b> walked <span class="dim">(${fmtPct(S.cum(rho) / S.ancientGraves)} of its graves behind you)</span> · history in <b>${fmtLen(S.rho0 - rho)}</b>`
          : `deaths that year <b>${fmtBig(deaths)}</b>`,
        `from centre <b>${fmtLen(rho)}</b> · to the edge <b>${fmtLen(toEdge)}</b>`,
        `ring around here <b>${fmtLen(TAU * fp)}</b> <span class="dim">(flat plane: ${fmtLen(TAU * rho)})</span>`,
        `curvature <b>${kText}</b>`,
        inAncient ? `<span class="dim">no dates here</span>` : `1 year = <b>${S.rowsPerYear.toFixed(2)}</b> rows outward`,
        `speed ${fmtLen(speed)}/s = ${(S.yr(speed) * S.rowsPerYear).toPrecision(2)} graves/s${opts.current.motion.mode === 'explore' ? ' (explore)' : ''} · edge in ${fmtDur(toEdge / speed)} · view ${fmtLen(W / st.zoom)} · ${drawn} graves drawn`,
      ];
      hudRef.current!.innerHTML = lines.join('<br>');
      const near = rho < S.grid.L.plazaR + 40 ? welcome()
        : rho > S.grid.L.plazaR + 40 && rho < S.grid.L.plazaR + 110 ? ancientSign()
        : landmarks().map(l => ({ l, d: Math.abs(S.yr(S.rhoAtTime(l.year) - rho)) })).filter(x => x.d < 15).sort((a, b) => a.d - b.d)[0]?.l;
      signRef.current!.innerHTML = near ? `<b>${near.title}</b>${near.year === -50000 ? '' : ` · ${formatYear(near.year)}`}<br>${near.text}` : '';
      signRef.current!.style.display = near ? 'block' : 'none';
    }

    let raf = 0, last = performance.now();
    const loop = (ts: number) => {
      const dt = Math.min(0.1, (ts - last) / 1000);
      last = ts;
      update(dt);
      sound.update(dt, {
        moving: st.speed > 0,
        stepRate: Math.min(4, Math.max(1.3, st.speed / 0.75)),
        ancient: Math.min(1, Math.max(0, (S.rho0 - st.rho) / (0.3 * S.rho0))),
      });
      const { fp, rhoEdge, cumNow, drawn } = draw(ts);
      if (ts - lastHud > 100) {
        lastHud = ts;
        drawHud(fp, rhoEdge, cumNow, drawn);
        if (profRef.current) {
          if (!prof || prof.S !== S) prof = buildProfile(S);
          drawProfile(profRef.current, prof, st.rho);
        }
        drawMap(rhoEdge, cumNow);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      sound.stop();
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
      <div className={`gy-profile gy-panel ${bigMap ? 'hidden' : ''}`}><canvas ref={profRef} /></div>
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
          <input type="checkbox" checked={motion.collide} onChange={e => setMotion({ ...motion, collide: e.target.checked })} />
          collisions (walk only on paths)
        </label>
        <label>
          <input type="checkbox" checked={motion.mode === 'explore'}
            onChange={e => setMotion({ ...motion, mode: e.target.checked ? 'explore' : 'fixed' })} />
          explore speed ({motion.mode === 'explore' ? `${motion.screensPerSec} view/s` : `off: ${motion.yrPerSec} yr/s`})
        </label>
        <div className="dim">WASD / arrows move · Shift ×{motion.boost} · wheel or +/− zoom · click a grave · click the map to travel</div>
        <div className="gy-jumps">
          <button onClick={() => setShowSettings(true)}>settings</button>
          <button onClick={() => { if (soundOn) sound.stop(); else sound.start(); setSoundOn(!soundOn); }}>
            sound {soundOn ? 'on' : 'off'}
          </button>
          <button onClick={() => setShowAbout(true)}>how is this built?</button>
          <button onClick={() => setShowModel(true)}>model &amp; charts</button>
        </div>
      </div>
      {selected && (
        <div className="gy-card gy-panel">
          <button className="gy-close" onClick={() => setSelected(null)}>×</button>
          <div className="gy-card-title">Grave #{selected.info.id.toLocaleString('en-US')}</div>
          <div>died <b>{selected.info.year < S.t0 ? `before ${formatYear(S.t0)} (ancient era)` : formatYear(selected.info.year, true)}</b></div>
          <div className="dim">row {selected.row.toLocaleString('en-US')} · {fmtLen(selected.rho)} from the centre</div>
          <div className="dim small">One grave for each person who died, in order of death; the year is where this grave falls in that order (a row spans about {Math.max(1, Math.round(1 / S.rowsPerYear))} yr). Nothing else is known about them.</div>
        </div>
      )}
      {showAbout && <About onClose={() => setShowAbout(false)} />}
      {showSettings && (
        <Suspense fallback={<div className="gy-about-backdrop"><div className="gy-panel">loading…</div></div>}>
          <Settings surface={surface} motion={motion} setMotion={setMotion} viewYr={viewYr()} setViewYr={setViewYr}
            onApply={applyWorld} onClose={() => setShowSettings(false)} />
        </Suspense>
      )}
      {showModel && (
        <Suspense fallback={<div className="gy-about-backdrop"><div className="gy-panel">loading charts…</div></div>}>
          <ModelDialog S={S} onClose={() => setShowModel(false)} />
        </Suspense>
      )}
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
          One grave per person who ever died, on square {S.grid.L.cell} m plots at uniform average density (paths included).
          The surface is rotationally symmetric, ds² = dρ² + f(ρ)² dφ², so a ring at distance ρ is 2π·f(ρ) long.
        </p>
        <ul>
          <li><b>History</b> (from {formatYear(S.t0)}): time is linear and distance is measured in years of it. Uniform density σ forces <b>f = D(t) / (2πσv²)</b> (in yr), where σv² = {(S.sigma * S.v * S.v).toPrecision(3)} graves per yr² ({S.rowsPerYear.toFixed(2)} rows per year) is the only world-shape knob. Curvature K = −D″/D per yr²: set by the data alone.</li>
          <li><b>Ancient era</b> ({fmtBig(S.ancientGraves)} graves before {formatYear(S.t0)}, undated): one smooth curve from the centre (where it starts flat) to the rim at {fmtLen(S.rho0)}, matching the history ring and its first three derivatives there and holding exactly these graves. Rings never shrink; it curves as much as the radius forces it to. Graves are placed by area.</li>
          <li>Every blend is C∞ and D is C⁴, so f is C⁴ and the curvature is C²: no creases or jumps anywhere. Rings never shrink outward.</li>
          <li>Hard limit: with rings that never shrink the ancient zone needs ≳ N_before / D(switch) years (a cylinder), here {fmtLen(S.ancientGraves * S.v / m.D(S.t0))}: {(100 * S.report.floorShare).toFixed(0)}% of the walk. Best switch ≈ 3000 BCE. "Flatten after" starts the rate after the switch higher and lets it rise slower (same totals), which lowers the floor.</li>
          <li>The view is a conformal map: isothermal coordinate u = ∫dρ/f, screen = f(ρ_you)·(e^(Δu + iΔφ) − 1). It is exact at your feet; zoom out to see the rest of the world shrink or grow.</li>
        </ul>
        <h2>The death model</h2>
        <p>
          D(t) = P(t)·m(t), both quintic B-splines. P: smooth log-space fit of PRB's own exponential-per-period estimates and OWID
          decade means. m ≈ 1: smoothest multiplier (min ∫m′² + α m‴²) making every period total exact. Both are single linear
          solves: no iterative fitting.
        </p>
        <table>
          <thead><tr><th>period</th><th>deaths (data)</th><th>model</th></tr></thead>
          <tbody>
            {m.periods.map(p => (
              <tr key={p.source}><td>{p.source}</td><td>{fmtBig(p.target)}</td><td>{((p.fitted / p.target - 1) * 100).toExponential(1)} %</td></tr>
            ))}
          </tbody>
        </table>
        <p className="dim">Total by {Math.floor(nowYear())}: {fmtBig(S.cum(S.rhoAtTime(nowYear())))} graves · {S.nRows.toLocaleString('en-US')} rows · edge {fmtLen(S.rhoAtTime(nowYear()))} from the centre.</p>
      </div>
    </div>
  );
}
