// Live HUD profile: ring length and curvature against distance from the
// centre, with a marker at the walker. Plain canvas so it can redraw often.
// The x axis gives the ancient era ANCIENT_FRAC of the width (it is only ~10%
// of the distance but holds the interesting shape).
import type { Surface } from './surface';

const ANCIENT_FRAC = 0.35;
const N = 700;
const K0 = 1e-5; // 1/yr², curvature axis is asinh(K/K0): linear near 0, log-like beyond

export type Profile = { S: Surface; rho: Float64Array; ring: Float64Array; K: Float64Array; ringMin: number; ringMax: number; kMax: number };

export function buildProfile(S: Surface): Profile {
  const end = S.rhoEndTable, r0 = S.rho0;
  const rho = new Float64Array(N + 1), ring = new Float64Array(N + 1), K = new Float64Array(N + 1);
  let ringMin = Infinity, ringMax = 0, kMax = 0;
  for (let i = 0; i <= N; i++) {
    const s = i / N;
    const r = s <= ANCIENT_FRAC ? (s / ANCIENT_FRAC) * r0 : r0 + (s - ANCIENT_FRAC) / (1 - ANCIENT_FRAC) * (end - r0);
    rho[i] = Math.max(r, S.grid.L.plazaR);
    ring[i] = Math.log10(S.yr(2 * Math.PI * S.f(rho[i])));
    K[i] = Math.asinh(S.gaussK(rho[i]) * S.v * S.v / K0);
    ringMin = Math.min(ringMin, ring[i]); ringMax = Math.max(ringMax, ring[i]);
    kMax = Math.max(kMax, Math.abs(K[i]));
  }
  return { S, rho, ring, K, ringMin, ringMax, kMax: Math.max(kMax, 1) };
}

const xOf = (p: Profile, rho: number) => {
  const r0 = p.S.rho0, end = p.S.rhoEndTable;
  return rho <= r0 ? ANCIENT_FRAC * rho / r0 : ANCIENT_FRAC + (1 - ANCIENT_FRAC) * Math.min(1, (rho - r0) / (end - r0));
};

const fmtYr = (y: number) => (y >= 1e6 ? `${(y / 1e6).toFixed(1)} M yr` : y >= 1e3 ? `${(y / 1e3).toFixed(y >= 1e4 ? 0 : 1)} k yr` : `${y.toFixed(0)} yr`);

export function drawProfile(canvas: HTMLCanvasElement, p: Profile, rhoNow: number) {
  const dpr = window.devicePixelRatio || 1, W = 300, H = 150;
  if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; canvas.style.width = `${W}px`; canvas.style.height = `${H}px`; }
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const S = p.S, padL = 6, padR = 6, pw = W - padL - padR;
  const top = { y: 14, h: 52 }, bot = { y: 86, h: 52 };
  const X = (s: number) => padL + s * pw;

  // ancient band + labels
  ctx.fillStyle = 'rgba(214,180,100,0.08)';
  ctx.fillRect(X(0), top.y, X(ANCIENT_FRAC) - X(0), top.h);
  ctx.fillRect(X(0), bot.y, X(ANCIENT_FRAC) - X(0), bot.h);
  ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.fillStyle = '#9d957f';
  ctx.textBaseline = 'alphabetic';

  const line = (ys: Float64Array, map: (v: number) => number, color: string) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i <= N; i++) { const x = X(i / N), y = map(ys[i]); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
    ctx.stroke();
  };
  const ringY = (v: number) => top.y + top.h - (v - p.ringMin) / (p.ringMax - p.ringMin || 1) * top.h;
  const kY = (v: number) => bot.y + bot.h / 2 - v / p.kMax * bot.h / 2;
  ctx.strokeStyle = 'rgba(230,220,195,0.2)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(0), kY(0)); ctx.lineTo(X(1), kY(0)); ctx.stroke();
  line(p.ring, ringY, '#e0b860');
  line(p.K, kY, '#d68cff');

  // walker
  const s = xOf(p, Math.min(rhoNow, S.rhoEndTable));
  const ringNow = S.yr(2 * Math.PI * S.f(Math.max(rhoNow, S.grid.L.plazaR)));
  const Know = S.gaussK(rhoNow) * S.v * S.v;
  ctx.strokeStyle = 'rgba(255,234,176,0.8)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(s), top.y); ctx.lineTo(X(s), top.y + top.h); ctx.moveTo(X(s), bot.y); ctx.lineTo(X(s), bot.y + bot.h); ctx.stroke();
  ctx.fillStyle = '#ffeab0';
  ctx.beginPath(); ctx.arc(X(s), ringY(Math.log10(ringNow)), 3, 0, 2 * Math.PI); ctx.fill();
  ctx.beginPath(); ctx.arc(X(s), kY(Math.asinh(Know / K0)), 3, 0, 2 * Math.PI); ctx.fill();

  ctx.fillStyle = '#c9bfa6';
  ctx.fillText(`ring ${fmtYr(ringNow)} (log)`, padL, top.y - 3);
  const kr = Know === 0 ? 'flat' : `${Know > 0 ? '+' : '−'}1/(${fmtYr(1 / Math.sqrt(Math.abs(Know)))})²`;
  ctx.fillText(`curvature ${kr}`, padL, bot.y - 3);
  ctx.fillStyle = '#9d957f';
  ctx.textAlign = 'right';
  ctx.fillText('ancient ¦ history', X(ANCIENT_FRAC) + 42, H - 2);
  ctx.textAlign = 'left';
  ctx.fillText('centre', X(0), H - 2);
  ctx.textAlign = 'right';
  ctx.fillText('edge', X(1), H - 2);
  ctx.textAlign = 'left';
}
