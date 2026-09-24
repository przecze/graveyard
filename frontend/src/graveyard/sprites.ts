// Procedurally painted top-down grave sprites (no external image assets).
// Every sprite covers one plot: SPRITE_W × SPRITE_H metres, outward (head
// end) at the top. Painted once into offscreen canvases at PX_PER_M.

export const PX_PER_M = 64;
export const SPRITE_W = 1.3;  // = layout pitch
export const SPRITE_H = 2.6;  // = layout row height

export type Style = 'mound' | 'cairn' | 'menhir' | 'stele' | 'cross' | 'headstone' | 'granite';

export const STYLES_BY_ERA: { until: number; styles: [Style, number][] }[] = [
  { until: -8000, styles: [['mound', 5], ['cairn', 3], ['menhir', 1]] },
  { until: -3000, styles: [['mound', 3], ['cairn', 3], ['menhir', 3]] },
  { until: 400,   styles: [['stele', 5], ['mound', 2], ['cairn', 1], ['menhir', 1]] },
  { until: 1700,  styles: [['cross', 5], ['stele', 2], ['mound', 2]] },
  { until: 1920,  styles: [['headstone', 5], ['cross', 3]] },
  { until: 1e9,   styles: [['granite', 5], ['headstone', 2], ['cross', 1]] },
];

const VARIANTS = 4;

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shade(hex: string, k: number): string {
  const n = parseInt(hex.slice(1), 16);
  const c = (v: number) => Math.max(0, Math.min(255, Math.round(v * k)));
  return `rgb(${c(n >> 16)},${c((n >> 8) & 255)},${c(n & 255)})`;
}

type Ctx = CanvasRenderingContext2D;

function shadow(ctx: Ctx, x: number, y: number, w: number, h: number) {
  ctx.fillStyle = 'rgba(0,0,0,0.35)';
  ctx.beginPath();
  ctx.ellipse(x, y, w, h, 0, 0, Math.PI * 2);
  ctx.fill();
}

function earthPlot(ctx: Ctx, r: () => number, grassy: number) {
  // soft oblong of turned earth, grass creeping in
  ctx.fillStyle = shade('#4a3b2a', 0.85 + r() * 0.3);
  ctx.beginPath();
  ctx.ellipse(0.65, 1.45, 0.42, 0.95, 0, 0, Math.PI * 2);
  ctx.fill();
  for (let i = 0; i < 40 * grassy; i++) {
    ctx.fillStyle = `rgba(${60 + r() * 30},${90 + r() * 40},${50 + r() * 20},0.6)`;
    ctx.fillRect(0.25 + r() * 0.8, 0.5 + r() * 1.9, 0.05, 0.05);
  }
}

function stoneSlab(ctx: Ctx, r: () => number, x: number, y: number, w: number, h: number,
                   color: string, top: 'flat' | 'round' | 'gable', thickness = 0.12) {
  shadow(ctx, x, y + h * 0.15 + thickness, w * 0.6, 0.12);
  const path = () => {
    ctx.beginPath();
    const l = x - w / 2, rr = x + w / 2, b = y + h / 2, t = y - h / 2;
    ctx.moveTo(l, b);
    if (top === 'round') {
      ctx.lineTo(l, t + w / 2);
      ctx.arc(x, t + w / 2, w / 2, Math.PI, 0);
    } else if (top === 'gable') {
      ctx.lineTo(l, t + w * 0.25);
      ctx.lineTo(x, t);
      ctx.lineTo(rr, t + w * 0.25);
    } else {
      ctx.lineTo(l, t);
      ctx.lineTo(rr, t);
    }
    ctx.lineTo(rr, b);
    ctx.closePath();
  };
  // top face (seen from above, offset upward) then front face
  ctx.save();
  ctx.translate(0, -thickness);
  ctx.fillStyle = shade(color, 1.18);
  path(); ctx.fill();
  ctx.restore();
  ctx.fillStyle = color;
  path(); ctx.fill();
  ctx.strokeStyle = shade(color, 0.6);
  ctx.lineWidth = 0.02;
  path(); ctx.stroke();
  // weathering
  for (let i = 0; i < 12; i++) {
    ctx.fillStyle = `rgba(${r() < 0.5 ? '40,60,30' : '255,255,255'},${0.05 + r() * 0.08})`;
    ctx.beginPath();
    ctx.arc(x + (r() - 0.5) * w * 0.8, y + (r() - 0.3) * h * 0.6, 0.02 + r() * 0.05, 0, Math.PI * 2);
    ctx.fill();
  }
}

function engraving(ctx: Ctx, x: number, y: number, w: number, lines: number, color: string) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 0.025;
  for (let i = 0; i < lines; i++) {
    const lw = w * (i === 0 ? 0.7 : 0.5);
    ctx.beginPath();
    ctx.moveTo(x - lw / 2, y + i * 0.07);
    ctx.lineTo(x + lw / 2, y + i * 0.07);
    ctx.stroke();
  }
}

const painters: Record<Style, (ctx: Ctx, r: () => number) => void> = {
  mound(ctx, r) {
    const g = ctx.createRadialGradient(0.6, 1.2, 0.05, 0.65, 1.4, 1.0);
    const hue = 70 + r() * 25;
    g.addColorStop(0, `hsl(${hue},28%,${30 + r() * 6}%)`);
    g.addColorStop(1, `hsla(${hue},30%,16%,0)`);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.ellipse(0.65, 1.4, 0.55, 1.05, 0, 0, Math.PI * 2);
    ctx.fill();
    // ring of small stones
    const n = 6 + Math.floor(r() * 6);
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 + r() * 0.3;
      ctx.fillStyle = shade('#8a8878', 0.7 + r() * 0.5);
      ctx.beginPath();
      ctx.arc(0.65 + Math.cos(a) * 0.45, 1.4 + Math.sin(a) * 0.95, 0.04 + r() * 0.04, 0, Math.PI * 2);
      ctx.fill();
    }
  },
  cairn(ctx, r) {
    earthPlot(ctx, r, 1);
    shadow(ctx, 0.7, 0.95, 0.42, 0.25);
    for (let i = 0; i < 18; i++) {
      const rad = 0.08 + r() * 0.08;
      const a = r() * Math.PI * 2, d = Math.sqrt(r()) * (0.35 - i * 0.012);
      ctx.fillStyle = shade('#9a968a', 0.55 + r() * 0.55 + i * 0.01);
      ctx.beginPath();
      ctx.ellipse(0.65 + Math.cos(a) * d, 0.75 + Math.sin(a) * d * 0.7 - i * 0.012, rad, rad * 0.8, r() * 3, 0, Math.PI * 2);
      ctx.fill();
    }
  },
  menhir(ctx, r) {
    painters.mound(ctx, r);
    shadow(ctx, 0.72, 0.62, 0.3, 0.12);
    ctx.fillStyle = shade('#7f7c70', 0.8 + r() * 0.3);
    ctx.beginPath();
    const pts = 7;
    for (let i = 0; i < pts; i++) {
      const a = (i / pts) * Math.PI * 2;
      const rx = 0.2 + r() * 0.07, ry = 0.3 + r() * 0.1;
      const x = 0.65 + Math.cos(a) * rx, y = 0.35 + Math.sin(a) * ry;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.12)';
    ctx.beginPath();
    ctx.ellipse(0.6, 0.25, 0.1, 0.16, -0.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(90,120,60,0.45)';
    for (let i = 0; i < 5; i++) ctx.fillRect(0.5 + r() * 0.3, 0.3 + r() * 0.3, 0.06, 0.04);
  },
  stele(ctx, r) {
    earthPlot(ctx, r, 1.5);
    const color = ['#b8a27a', '#a8997e', '#c2b08a', '#9b8c70'][Math.floor(r() * 4)];
    stoneSlab(ctx, r, 0.65, 0.42, 0.62, 0.52, color, r() < 0.5 ? 'gable' : 'flat', 0.14);
    engraving(ctx, 0.65, 0.3, 0.62, 4, shade(color, 0.55));
  },
  cross(ctx, r) {
    earthPlot(ctx, r, 2);
    const wood = r() < 0.55;
    const color = wood ? shade('#5b4128', 0.8 + r() * 0.4) : shade('#8d8a80', 0.8 + r() * 0.3);
    shadow(ctx, 0.7, 0.72, 0.28, 0.1);
    ctx.fillStyle = shade(wood ? '#5b4128' : '#8d8a80', 1.25);
    ctx.fillRect(0.6, 0.08, 0.1, 0.7);
    ctx.fillRect(0.4, 0.22, 0.5, 0.09);
    ctx.fillStyle = color;
    ctx.fillRect(0.6, 0.14, 0.1, 0.66);
    ctx.fillRect(0.4, 0.28, 0.5, 0.09);
    if (!wood && r() < 0.5) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 0.05;
      ctx.beginPath();
      ctx.arc(0.65, 0.32, 0.14, 0, Math.PI * 2);
      ctx.stroke();
    }
  },
  headstone(ctx, r) {
    earthPlot(ctx, r, 2.5);
    if (r() < 0.4) {
      ctx.strokeStyle = shade('#9d9a92', 0.9);
      ctx.lineWidth = 0.06;
      ctx.strokeRect(0.22, 0.75, 0.86, 1.7);
    }
    const color = ['#d4d2c8', '#bdbab0', '#a9a69d', '#c9c3b2'][Math.floor(r() * 4)];
    stoneSlab(ctx, r, 0.65, 0.42, 0.7, 0.6, color, r() < 0.7 ? 'round' : 'gable', 0.12);
    engraving(ctx, 0.65, 0.35, 0.7, 3, shade(color, 0.55));
  },
  granite(ctx, r) {
    // kerbed plot with gravel or a slab
    const kerb = shade('#6f6d6a', 0.8 + r() * 0.3);
    ctx.fillStyle = kerb;
    ctx.fillRect(0.2, 0.6, 0.9, 1.85);
    if (r() < 0.5) {
      ctx.fillStyle = shade('#2c2c2e', 0.9 + r() * 0.4);
      ctx.fillRect(0.25, 0.65, 0.8, 1.75);
      ctx.fillStyle = 'rgba(255,255,255,0.08)';
      ctx.fillRect(0.3, 0.7, 0.2, 1.65);
    } else {
      ctx.fillStyle = '#7b776d';
      ctx.fillRect(0.26, 0.66, 0.78, 1.73);
      for (let i = 0; i < 160; i++) {
        ctx.fillStyle = shade('#a8a396', 0.6 + r() * 0.6);
        ctx.fillRect(0.26 + r() * 0.76, 0.66 + r() * 1.7, 0.03, 0.03);
      }
    }
    const color = ['#1f1f22', '#3a3534', '#4a4f55', '#6b5d57'][Math.floor(r() * 4)];
    stoneSlab(ctx, r, 0.65, 0.4, 0.8, 0.55, color, r() < 0.6 ? 'flat' : 'round', 0.1);
    engraving(ctx, 0.65, 0.3, 0.8, 3, 'rgba(210,190,140,0.8)');
  },
};

export type SpriteSet = Record<Style, HTMLCanvasElement[]>;

export function buildSprites(): SpriteSet {
  const out = {} as SpriteSet;
  let seed = 1;
  for (const style of Object.keys(painters) as Style[]) {
    out[style] = [];
    for (let v = 0; v < VARIANTS; v++) {
      const c = document.createElement('canvas');
      c.width = Math.round(SPRITE_W * PX_PER_M);
      c.height = Math.round(SPRITE_H * PX_PER_M);
      const ctx = c.getContext('2d')!;
      ctx.scale(PX_PER_M, PX_PER_M);
      painters[style](ctx, rng(seed++ * 7919));
      out[style].push(c);
    }
  }
  return out;
}

/** a signpost for year markers: small stone with plaque */
export function buildMarker(color = '#c8b27a'): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = c.height = 2 * PX_PER_M;
  const ctx = c.getContext('2d')!;
  ctx.scale(PX_PER_M, PX_PER_M);
  shadow(ctx, 1.05, 1.25, 0.6, 0.25);
  ctx.fillStyle = '#5c574c';
  ctx.beginPath();
  ctx.roundRect(0.45, 0.7, 1.1, 0.6, 0.08);
  ctx.fill();
  ctx.fillStyle = '#77715f';
  ctx.beginPath();
  ctx.roundRect(0.45, 0.62, 1.1, 0.5, 0.08);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.fillRect(0.6, 0.72, 0.8, 0.28);
  return c;
}
