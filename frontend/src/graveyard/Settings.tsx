// Settings dialog. Distances are in history-years (1 yr = the radial depth of
// one year of history). Main knobs: a preset, the ancient era's share of the
// walk, the pace (graves passed per second, which sets the speed) and the view
// width. Everything else is under "advanced".
// World changes are fitted live and applied straight away (restart at the
// centre); motion changes apply immediately.
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Surface, T_END, type Shape } from './surface';
import { maxFlatten } from './deathModel';
import { formatYear } from './lore';
import { CELL } from './sprites';
import { WalkCharts } from './charts';
import { fmtDur, speedYr, type Motion } from './motion';
import { PRESETS, type Preset } from './presets';

const sig = (x: number, n = 3) => Number(x.toPrecision(n));
const fmtY = (y: number) => {
  const a = Math.abs(y);
  return a >= 1e6 ? `${sig(y / 1e6)} M yr` : a >= 1e5 ? `${Math.round(y / 1e3)} k yr` : a >= 100 ? `${Math.round(y).toLocaleString('en-US')} yr` : `${sig(y)} yr`;
};
const fmtN = (n: number) => (n >= 1e6 ? `${sig(n / 1e6)} M` : n >= 1e4 ? `${Math.round(n / 1e3)} k` : Math.round(n).toLocaleString('en-US'));

function Slider({ label, value, min, max, step, log, onChange, fmt, hint }: {
  label: string; value: number; min: number; max: number; step?: number; log?: boolean;
  onChange: (v: number) => void; fmt: (v: number) => string; hint?: ReactNode;
}) {
  const to = (v: number) => (log ? Math.log10(v) : v);
  const from = (x: number) => (log ? sig(10 ** x) : x);
  return (
    <>
      <label className="gy-slider">
        <span>{label}</span>
        <input type="range" min={to(min)} max={to(max)} step={log ? 0.005 : step} value={to(Math.min(max, Math.max(min, value)))}
          onChange={e => onChange(from(Number(e.target.value)))} />
        <b>{fmt(value)}</b>
      </label>
      {hint && <div className="gy-hint dim">{hint}</div>}
    </>
  );
}

export default function Settings({ surface, motion, setMotion, viewYr, setViewYr, onApply, onClose }: {
  surface: Surface; motion: Motion; setMotion: (m: Motion) => void;
  viewYr: number; setViewYr: (y: number) => void;
  onApply: (S: Surface) => void; onClose: () => void;
}) {
  const [draft, setDraft] = useState<Shape>(surface.shape);
  const [view, setView] = useState(viewYr);
  const [error, setError] = useState<string | null>(null);
  const set = (patch: Partial<Shape>) => setDraft(d => ({ ...d, ...patch }));
  const applyRef = useRef(onApply);
  applyRef.current = onApply;

  // fit the draft world and apply it as soon as it is valid
  useEffect(() => {
    if (JSON.stringify(draft) === JSON.stringify(surface.shape)) { setError(null); return; }
    const id = setTimeout(() => {
      try {
        const S = new Surface(draft);
        if (S.report.ok) { setError(null); applyRef.current(S); } else setError(S.report.problem ?? 'invalid shape');
      } catch (e) {
        setError((e as Error).message);
      }
    }, 150);
    return () => clearTimeout(id);
  }, [draft, surface]);

  const applyPreset = (p: Preset) => {
    setDraft(p.shape);
    setMotion(p.motion);
    setView(p.viewYr); setViewYr(p.viewYr);
  };

  const P = surface, r = P.report;
  const flatMax = Math.floor(maxFlatten(draft.switchYear) / 100) * 100;

  // ── experience numbers ────────────────────────────────────────────────────
  const W = window.innerWidth, H = window.innerHeight;
  const spd = speedYr(motion, view);
  const anc = P.yr(r.ancientLen), hist = P.yr(r.historyLen);
  const rows = P.rowsPerYear; // rows crossed per yr walking straight out (≈ 0.9·√density)
  const pace = rows * spd;
  const plotPx = CELL * W / (view * P.v);
  const onScreen = draft.density * view * view * H / W;
  const tightest = P.yr(r.minKRadius);

  return (
    <div className="gy-about-backdrop" onClick={onClose}>
      <div className="gy-about gy-settings gy-panel" onClick={e => e.stopPropagation()}>
        <button className="gy-close" onClick={onClose}>×</button>

        <h2>Presets</h2>
        <div className="gy-presets">
          {PRESETS.map(p => (
            <button key={p.name} onClick={() => applyPreset(p)}><b>{p.name}</b><span>{p.blurb}</span></button>
          ))}
        </div>

        <h2>Experience</h2>
        <div className="gy-exp">
          <div><b>{fmtDur((anc + hist) / spd)}</b><span>whole walk</span></div>
          <div><b>{fmtDur(anc / spd)}</b><span>ancient era ({fmtY(anc)}, {(100 * r.ancientShare).toFixed(0)}%)</span></div>
          <div><b>{fmtDur(hist / spd)}</b><span>history ({fmtY(hist)})</span></div>
          <div><b>{sig(pace)} /s</b><span>graves you pass walking out</span></div>
          <div><b>{fmtN(rows * (anc + hist))}</b><span>graves along one radius</span></div>
          <div><b>{fmtDur(view / spd)}</b><span>to cross the screen</span></div>
          <div><b>{fmtN(onScreen)}</b><span>graves on screen</span></div>
          <div><b>{plotPx < 10 ? plotPx.toFixed(1) : Math.round(plotPx)} px</b><span>a plot on screen</span></div>
        </div>

        <Slider label="ancient era" value={Math.round(draft.ancientShare * 100)} min={2} max={50} step={1}
          onChange={v => set({ ancientShare: v / 100 })} fmt={v => `${v}% of the walk`}
          hint={<>{fmtY(anc)} · restarts at the centre</>} />
        {motion.mode === 'fixed'
          ? <Slider label="pace" value={sig(pace)} min={0.05} max={300} log
              onChange={v => setMotion({ ...motion, yrPerSec: sig(v / rows) })} fmt={v => `${v} graves/s`}
              hint={`graves passed per second walking straight out · ${sig(spd)} yr/s`} />
          : <Slider label="explore speed" value={motion.screensPerSec} min={0.02} max={3} log
              onChange={v => setMotion({ ...motion, screensPerSec: v })} fmt={v => `${v} view/s`} hint={`= ${sig(spd)} yr/s at this view width`} />}
        <Slider label="view width" value={view} min={0.5} max={50000} log onChange={v => { setView(v); setViewYr(v); }} fmt={fmtY}
          hint="also the mouse wheel and +/−" />
        <div className="gy-jumps">
          <label><input type="checkbox" checked={motion.mode === 'explore'} onChange={e => setMotion({ ...motion, mode: e.target.checked ? 'explore' : 'fixed' })} /> explore speed (debug)</label>
          <label><input type="checkbox" checked={motion.collide} onChange={e => setMotion({ ...motion, collide: e.target.checked })} /> collisions</label>
        </div>

        <p className="small">
          {error ? <span className="gy-warn">{error}</span> : <>
            <span className="dim">The ancient era is one smooth curve from the centre to its rim, holding all {sig(P.ancientGraves / 1e9)} B graves before {formatYear(P.t0)}
              {r.ancientLen < r.floorLen ? ` as a bulb: its rings grow up to ${fmtY(P.yr(r.maxRing))} around (the history ring is ${fmtY(P.yr(r.rimRing))}) and narrow back at the rim` : ''}.
              {' '}Tightest curvature radius </span><b>{fmtY(tightest)}</b>
            {tightest < view
              ? <span className="gy-warn"> — smaller than the view ({fmtY(view)}), so the ground visibly warps there. A longer ancient era is gentler.</span>
              : <span className="dim"> (view {fmtY(view)}).</span>}
          </>}
        </p>
        <WalkCharts S={P} height={200} />

        <details className="gy-advanced">
          <summary>advanced</summary>
          <Slider label="grave density" value={draft.density} min={0.1} max={100} log onChange={v => set({ density: v })} fmt={v => `${v} /yr²`}
            hint={<>graves per yr²: {sig(rows)} rows per year of history, 1 yr = {sig(P.v)} m. Ancient/history lengths and curvature do not depend on it.</>} />
          <Slider label="history begins" value={draft.switchYear} min={-8000} max={-1000} step={100}
            onChange={v => set({ switchYear: v, flatten: Math.min(draft.flatten, Math.floor(maxFlatten(v) / 100) * 100) })} fmt={formatYear} />
          <Slider label="flatten window" value={Math.min(draft.flatten, flatMax)} min={0} max={flatMax} step={100}
            onChange={v => set({ flatten: v })} fmt={v => (v ? `${formatYear(draft.switchYear)} → ${formatYear(draft.switchYear + v)}` : 'off')}
            hint={`deaths/yr at the switch ${sig(P.model.D(P.t0) / 1e6)} M`} />
          <Slider label="prior smoothing" value={draft.priorScale} min={0.5} max={200} log onChange={v => set({ priorScale: v })} fmt={v => `${v} yr`} />
          <Slider label="multiplier bend" value={draft.multScale} min={1} max={500} log onChange={v => set({ multScale: v })} fmt={v => `${v} yr`} />
          <Slider label="growth after 2023" value={draft.extrapGrowth * 100} min={-2} max={3} step={0.1}
            onChange={v => set({ extrapGrowth: sig(v / 100) })} fmt={v => `${v.toFixed(1)} %/yr`} />
          <Slider label="Shift boost" value={motion.boost} min={1} max={50} log onChange={v => setMotion({ ...motion, boost: v })} fmt={v => `×${v}`} />
          <Slider label="explore speed" value={motion.screensPerSec} min={0.02} max={3} log onChange={v => setMotion({ ...motion, screensPerSec: v })} fmt={v => `${v} view/s`} />
          <p className="dim small">History runs {formatYear(P.t0)} → {T_END}; the ancient era holds everyone before, undated.</p>
        </details>
      </div>
    </div>
  );
}
