// Ready-made experiences, defined by what you feel:
//   minutes    the whole walk, centre → edge
//   pace       graves passed per second walking straight out
//   viewYr     view width
// From those: speed = walk length / time; rows per year = pace / speed, which
// fixes the grave density (fewer graves per year for a short walk, so they do
// not flash by). Ancient share is 10%.
import { DEFAULT_SHAPE, T_END, type Shape } from './surface';
import { DEFAULT_MOTION, type Motion } from './motion';
import { DEFAULT_LAYOUT, gridMetrics } from './grid';

export type Preset = { name: string; blurb: string; shape: Shape; motion: Motion; viewYr: number };

const sig = (x: number) => Number(x.toPrecision(3));
const GM = gridMetrics(DEFAULT_LAYOUT);

function preset(name: string, minutes: number, pace: number, viewYr: number, ancientShare = 0.1): Preset {
  const walkYr = (T_END - DEFAULT_SHAPE.switchYear) / (1 - ancientShare);
  const yrPerSec = walkYr / (minutes * 60);
  const rowsPerYr = pace / yrPerSec;
  const density = sig(GM.sigma * (rowsPerYr * GM.rowH) ** 2); // v = rows·rowH, density = σv²
  const blurb = `≈ ${minutes < 60 ? `${minutes} min` : `${minutes / 60} h`} · ${pace} graves/s · ${viewYr} yr view (a screen every ${Math.round(viewYr / yrPerSec)} s)`;
  return {
    name, blurb,
    shape: { ...DEFAULT_SHAPE, ancientShare, density },
    motion: { ...DEFAULT_MOTION, yrPerSec: sig(yrPerSec) },
    viewYr,
  };
}

export const PRESETS: Preset[] = [
  preset('Short walk', 20, 2, 80),
  preset('Quick look', 8, 3, 160),
  preset('Long walk', 60, 1.5, 40),
  (p => ({ ...p, blurb: 'debug: ⅓ view per second, no collisions', motion: { ...p.motion, mode: 'explore' as const, collide: false } }))(
    preset('Explore', 20, 2, 80)),
];

export const DEFAULT_PRESET = PRESETS[0];
