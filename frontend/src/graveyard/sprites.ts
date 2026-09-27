// Procedurally painted grave sprites (no external image assets). One sprite
// per grave: the monument, seen from the front with a sliver of its top,
// outlined, with a cast shadow, standing inside its own CELL × CELL m square
// (head end at the top). Drawn rotated with the grid; no occlusion sorting.
// Painted once into offscreen canvases at PX_PER_M.

export const PX_PER_M = 64;
/** a grave plot is a square CELL × CELL metres */
export const CELL = 1.2;
export const SPRITE_W = CELL;
export const SPRITE_H = CELL;
// monuments are painted in their own frame (foot of the stone, front centre, at
// STONE_AX, STONE_AY) and fitted into the square: foot at FOOT_Y, heights × TILT
const STONE_AX = 0.8;
const STONE_AY = 2.2;
const FOOT_Y = 1.08;
const TILT = 0.65;

// Generic memorials: no religious or culture-specific symbols, no claim about
// what graves looked like in any era. Variety comes from the shape (by hash)
// and from age: older graves are weathered and mossy, recent ones crisp.
export type Style = 'rounded' | 'flat' | 'gabled' | 'block' | 'obelisk' | 'rough';
export const STYLES: Style[] = ['rounded', 'flat', 'gabled', 'block', 'obelisk', 'rough'];
/** 0 ancient era, 1 older history, 2 recent: only changes weathering and material */
export type Age = 0 | 1 | 2;
export const AGES: Age[] = [0, 1, 2];

const VARIANTS = 5;

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
const pick = <T,>(r: () => number, xs: T[]) => xs[Math.floor(r() * xs.length)];

type Ctx = CanvasRenderingContext2D;
type PathFn = (ctx: Ctx) => void;

const OUTLINE = 'rgba(14,12,9,0.9)';
const X = STONE_AX, Y = STONE_AY; // foot of the stone

// ── stone helpers (front view) ─────────────────────────────────────────────

/** cast shadow of a silhouette standing on y = base, falling back-right */
function castShadow(ctx: Ctx, path: PathFn, base = Y) {
  ctx.save();
  ctx.transform(1, 0, -0.55, 0.22, 0.55 * base, 0.78 * base);
  ctx.fillStyle = 'rgba(0,0,0,0.42)';
  path(ctx); ctx.fill();
  ctx.restore();
}

/**
 * a solid seen from the front: its top surface (the same silhouette raised by
 * depth·0.35, lighter) peeks out above the front face; lit from the left
 */
function solid(ctx: Ctx, path: PathFn, color: string, depth: number, x0: number, x1: number) {
  const up = depth * 0.35;
  ctx.save(); ctx.translate(0, -up);
  ctx.fillStyle = shade(color, 1.28); path(ctx); ctx.fill();
  ctx.strokeStyle = OUTLINE; ctx.lineWidth = 0.028; path(ctx); ctx.stroke();
  ctx.restore();
  const g = ctx.createLinearGradient(x0, 0, x1, 0);
  g.addColorStop(0, shade(color, 1.1)); g.addColorStop(0.55, color); g.addColorStop(1, shade(color, 0.72));
  ctx.fillStyle = g; path(ctx); ctx.fill();
  ctx.strokeStyle = OUTLINE; ctx.lineWidth = 0.028; path(ctx); ctx.stroke();
}

const rect = (x: number, y: number, w: number, h: number): PathFn => ctx => { ctx.beginPath(); ctx.rect(x, y, w, h); };
/** box standing on yb, centred on X; returns its top */
function box(ctx: Ctx, yb: number, w: number, h: number, depth: number, color: string): number {
  solid(ctx, rect(X - w / 2, yb - h, w, h), color, depth, X - w / 2, X + w / 2);
  return yb - h;
}

function weathering(ctx: Ctx, r: () => number, x0: number, y0: number, w: number, h: number, n: number) {
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = r() < 0.5 ? 'rgba(0,0,0,0.12)' : 'rgba(255,255,255,0.08)';
    ctx.beginPath(); ctx.arc(x0 + r() * w, y0 + r() * h, 0.01 + r() * 0.025, 0, Math.PI * 2); ctx.fill();
  }
}
function moss(ctx: Ctx, r: () => number, x0: number, y0: number, w: number, h: number, n: number) {
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = `rgba(${80 + r() * 30},${110 + r() * 40},${50 + r() * 20},${0.45 + r() * 0.3})`;
    ctx.beginPath(); ctx.arc(x0 + r() * w, y0 + h * Math.pow(r(), 0.5), 0.015 + r() * 0.03, 0, Math.PI * 2); ctx.fill();
  }
}
function textLines(ctx: Ctx, x: number, y: number, w: number, n: number, color: string, gap = 0.055) {
  ctx.fillStyle = color;
  for (let i = 0; i < n; i++) {
    const lw = w * (i === 0 ? 0.8 : 0.45 + ((i * 37) % 5) * 0.08);
    ctx.fillRect(x - lw / 2, y + i * gap, lw, 0.018);
  }
}
function flowers(ctx: Ctx, r: () => number, x: number, y: number, n: number, hue: number) {
  for (let i = 0; i < n; i++) {
    const a = r() * Math.PI - Math.PI, d = 0.03 + r() * 0.07;
    ctx.fillStyle = 'rgba(60,100,50,0.9)';
    ctx.fillRect(x + Math.cos(a) * d * 0.6 - 0.005, y - 0.02 + Math.sin(a) * d, 0.01, 0.06);
    ctx.fillStyle = `hsl(${hue + r() * 40 - 20},${60 + r() * 30}%,${55 + r() * 20}%)`;
    ctx.beginPath(); ctx.arc(x + Math.cos(a) * d, y - 0.03 + Math.sin(a) * d, 0.022 + r() * 0.012, 0, Math.PI * 2); ctx.fill();
  }
}

type Material = { color: string; moss: number; wear: number; text: string; lines: number; flowers: boolean };

function material(age: Age, r: () => number): Material {
  if (age === 0) return { color: pick(r, ['#9a978b', '#8c8a80', '#a39e8e', '#a8977a', '#958f7c']), moss: 22, wear: 40, text: 'rgba(0,0,0,0.18)', lines: 2, flowers: false };
  if (age === 1) return { color: pick(r, ['#a5a194', '#b9a27b', '#9f9c93', '#b3ad9f', '#a8977a']), moss: 10, wear: 22, text: 'rgba(0,0,0,0.32)', lines: 3, flowers: false };
  const dark = r() < 0.5;
  return {
    color: dark ? pick(r, ['#2d2c30', '#3c4148', '#4f4543', '#35393f']) : pick(r, ['#b8b5ad', '#a9a69f', '#c4c0b6', '#8e8b85']),
    moss: 0, wear: 4, text: dark ? 'rgba(225,200,140,0.85)' : 'rgba(0,0,0,0.4)', lines: 3, flowers: r() < 0.45,
  };
}

/** age marks, inscription and (sometimes) flowers on a slab spanning x0..x0+w, y0..y1 */
function finish(ctx: Ctx, r: () => number, m: Material, x0: number, y0: number, w: number, h: number, textY: number) {
  weathering(ctx, r, x0, y0, w, h, m.wear);
  textLines(ctx, X, textY, w * 0.62, m.lines, m.text);
  moss(ctx, r, x0, y0 + h * 0.3, w, h * 0.7, m.moss);
  if (m.flowers) flowers(ctx, r, X + (r() < 0.5 ? -1 : 1) * (w / 2 + 0.02), Y - 0.08, 5, pick(r, [0, 45, 300, 200, 20]));
}

const stones: Record<Style, (ctx: Ctx, r: () => number, age: Age) => void> = {
  rounded(ctx, r, age) {
    const m = material(age, r), w = 0.52 + r() * 0.1, h = 0.7 + r() * 0.16, top = Y - 0.12;
    const slab: PathFn = ctx => {
      ctx.beginPath(); ctx.moveTo(X - w / 2, top); ctx.lineTo(X - w / 2, top - h + w / 2);
      ctx.arc(X, top - h + w / 2, w / 2, Math.PI, 0); ctx.lineTo(X + w / 2, top); ctx.closePath();
    };
    castShadow(ctx, ctx => { rect(X - w / 2 - 0.1, top, w + 0.2, 0.12)(ctx); slab(ctx); });
    box(ctx, Y, w + 0.2, 0.12, 0.3, shade(m.color, 0.85));
    solid(ctx, slab, m.color, 0.12, X - w / 2, X + w / 2);
    finish(ctx, r, m, X - w / 2, top - h, w, h, top - h + w / 2 + 0.08);
  },
  flat(ctx, r, age) {
    const m = material(age, r), w = 0.6 + r() * 0.12, h = 0.5 + r() * 0.14, top = Y - 0.1;
    castShadow(ctx, rect(X - w / 2 - 0.08, top - h, w + 0.16, h + 0.1));
    box(ctx, Y, w + 0.16, 0.1, 0.34, shade(m.color, 0.85));
    const y = box(ctx, top, w, h, 0.12, m.color);
    finish(ctx, r, m, X - w / 2, y, w, h, y + 0.1);
  },
  gabled(ctx, r, age) {
    const m = material(age, r), w = 0.5 + r() * 0.1, h = 0.75 + r() * 0.15, top = Y - 0.12;
    const slab: PathFn = ctx => {
      ctx.beginPath(); ctx.moveTo(X - w / 2, top); ctx.lineTo(X - w / 2, top - h + w * 0.28);
      ctx.lineTo(X, top - h); ctx.lineTo(X + w / 2, top - h + w * 0.28); ctx.lineTo(X + w / 2, top); ctx.closePath();
    };
    castShadow(ctx, ctx => { rect(X - w / 2 - 0.1, top, w + 0.2, 0.12)(ctx); slab(ctx); });
    box(ctx, Y, w + 0.2, 0.12, 0.3, shade(m.color, 0.85));
    solid(ctx, slab, m.color, 0.12, X - w / 2, X + w / 2);
    finish(ctx, r, m, X - w / 2, top - h, w, h, top - h + w * 0.28 + 0.08);
  },
  block(ctx, r, age) {
    // low, wide memorial block with a bevelled top: reads well from above
    const m = material(age, r), w = 0.78 + r() * 0.1, h = 0.3 + r() * 0.08;
    castShadow(ctx, rect(X - w / 2, Y - h, w, h));
    const y = box(ctx, Y, w, h, 0.55, m.color);
    solid(ctx, rect(X - w / 2 + 0.05, y - 0.05, w - 0.1, 0.05), shade(m.color, 1.08), 0.45, X - w / 2, X + w / 2);
    finish(ctx, r, m, X - w / 2, y, w, h, y + 0.07);
  },
  obelisk(ctx, r, age) {
    const m = material(age, r), h = 0.85 + r() * 0.2, b = 0.24, t = 0.16;
    const base = Y - 0.2;
    const shaft: PathFn = ctx => {
      ctx.beginPath(); ctx.moveTo(X - b / 2, base); ctx.lineTo(X - t / 2, base - h); ctx.lineTo(X, base - h - 0.1);
      ctx.lineTo(X + t / 2, base - h); ctx.lineTo(X + b / 2, base); ctx.closePath();
    };
    castShadow(ctx, ctx => { rect(X - 0.25, base, 0.5, 0.2)(ctx); shaft(ctx); });
    let y = box(ctx, Y, 0.5, 0.1, 0.4, shade(m.color, 0.82));
    y = box(ctx, y, 0.36, 0.1, 0.3, shade(m.color, 0.92));
    solid(ctx, shaft, m.color, 0.2, X - b / 2, X + b / 2);
    finish(ctx, r, m, X - b / 2, base - h, b, h, base - 0.3);
  },
  rough(ctx, r, age) {
    // an uncut upright stone set in pebbles
    const m = material(age, r), w = 0.4 + r() * 0.14, h = 0.62 + r() * 0.22;
    const pts: [number, number][] = [[X - w / 2, Y - 0.03]];
    for (let i = 0; i <= 9; i++) {
      const a = Math.PI * (1 - i / 9);
      pts.push([X + Math.cos(a) * w / 2 * (0.94 + r() * 0.1), Y - h + w * 0.35 - Math.sin(a) * w * 0.35 * (0.8 + r() * 0.4)]);
    }
    pts.push([X + w / 2 * (0.95 + r() * 0.08), Y - 0.03]);
    const path: PathFn = ctx => { ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); };
    const c = age === 2 ? pick(r, ['#9a978b', '#8c8a80', '#a39e8e']) : m.color;
    castShadow(ctx, path);
    solid(ctx, path, c, 0.2, X - w / 2, X + w / 2);
    finish(ctx, r, { ...m, color: c, lines: Math.min(m.lines, 2) }, X - w / 2, Y - h, w, h, Y - h * 0.6);
    for (let i = 0; i < 6; i++) {
      const px = X + (i - 2.5) * 0.11 + (r() - 0.5) * 0.03, pr = 0.04 + r() * 0.025;
      solid(ctx, cx => { cx.beginPath(); cx.ellipse(px, Y - pr * 0.5, pr, pr * 0.65, 0, 0, Math.PI * 2); }, shade(c, 0.75 + r() * 0.3), 0.04, px - pr, px + pr);
    }
  },
};

// ── build ──────────────────────────────────────────────────────────────────

export type SpriteSet = Record<Style, HTMLCanvasElement[][]>; // [age][variant]

function canvas(w: number, h: number, paint: (ctx: Ctx) => void): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = Math.round(w * PX_PER_M); c.height = Math.round(h * PX_PER_M);
  const ctx = c.getContext('2d')!;
  ctx.scale(PX_PER_M, PX_PER_M);
  ctx.lineJoin = 'round';
  paint(ctx);
  return c;
}

export function buildSprites(): SpriteSet {
  const out = {} as SpriteSet;
  let seed = 1;
  for (const style of STYLES) {
    out[style] = AGES.map(age => Array.from({ length: VARIANTS }, () => {
      const r = rng(seed++ * 7919);
      return canvas(CELL, CELL, ctx => {
        ctx.beginPath(); ctx.rect(0, 0, CELL, CELL); ctx.clip();
        ctx.translate(CELL / 2, FOOT_Y); ctx.scale(1, TILT); ctx.translate(-STONE_AX, -STONE_AY);
        stones[style](ctx, r, age);
      });
    }));
  }
  return out;
}

/** a signpost for year markers: small stone with plaque */
export function buildMarker(color = '#c8b27a'): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = c.height = 2 * PX_PER_M;
  const ctx = c.getContext('2d')!;
  ctx.scale(PX_PER_M, PX_PER_M);
  ctx.fillStyle = 'rgba(0,0,0,0.35)';
  ctx.beginPath(); ctx.ellipse(1.05, 1.25, 0.6, 0.25, 0, 0, Math.PI * 2); ctx.fill();
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
