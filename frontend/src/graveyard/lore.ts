// Per-grave look and helpers. The look is a pure function of the global grave
// id (and its age), so a grave is the same on every visit without storing
// anything. No invented facts: a grave has an id and an approximate year.

import type { Age, Style } from './sprites';

/** 32-bit hash of a (possibly > 2^32) integer id and a salt */
export function hash32(id: number, salt = 0): number {
  const lo = id % 4294967296, hi = Math.floor(id / 4294967296);
  let h = Math.imul(lo ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(hi + salt * 0x27d4eb2f, 0xc2b2ae35);
  h ^= h >>> 16; h = Math.imul(h, 0x7feb352d);
  h ^= h >>> 15; h = Math.imul(h, 0x846ca68b);
  h ^= h >>> 16;
  return h >>> 0;
}

export const hash01 = (id: number, salt = 0) => hash32(id, salt) / 4294967296;

export function formatYear(t: number, approx = false): string {
  // model time: t = −8000 is 8000 BCE; the missing year 0 is ignored
  const y = Math.floor(t);
  const s = y <= 0 ? `${Math.max(1, -y).toLocaleString('en-US')} BCE` : `${y} CE`;
  return approx ? `c. ${s}` : s;
}

const STYLE_WEIGHTS: [Style, number][] = [['rounded', 3], ['flat', 3], ['gabled', 2], ['block', 2], ['obelisk', 1], ['rough', 1]];

/** a grave's memorial shape: by hash only (nothing is claimed about eras) */
export function styleFor(id: number): Style {
  let x = hash01(id, 2) * 12;
  for (const [s, w] of STYLE_WEIGHTS) { if ((x -= w) < 0) return s; }
  return 'rounded';
}

/** how weathered a grave looks: ancient era, more than ~250 years old, recent */
export function ageOf(year: number, switchYear: number): Age {
  return year < switchYear ? 0 : year < 1775 ? 1 : 2;
}

/** what we know about a grave: its place in the order of deaths, so a year (approximate) */
export type GraveInfo = { id: number; year: number };

export type Landmark = { year: number; title: string; text: string };
// landmark texts are built from the model in Walker (landmarks()), so their numbers stay true
