// Walker motion settings (live) and shared formatting.

export type Motion = {
  mode: 'fixed' | 'explore'; // fixed yr/s, or a fraction of the view per second
  yrPerSec: number;
  screensPerSec: number;
  boost: number;             // Shift multiplier
  collide: boolean;          // graves block the walker (off for debugging)
};
export const DEFAULT_MOTION: Motion = { mode: 'fixed', yrPerSec: 1, screensPerSec: 1 / 3, boost: 5, collide: true };

/** effective speed in yr/s for a view `viewYr` wide */
export const speedYr = (m: Motion, viewYr: number) => (m.mode === 'fixed' ? m.yrPerSec : viewYr * m.screensPerSec);

export const fmtDur = (s: number) =>
  !isFinite(s) ? '—' : s < 90 ? `${s.toFixed(0)} s` : s < 5400 ? `${(s / 60).toFixed(0)} min` : s < 172800 ? `${(s / 3600).toFixed(1)} h` : `${(s / 86400).toFixed(1)} days`;
