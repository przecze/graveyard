// Procedural per-grave details and fixed landmarks.
// Everything is a pure function of the global grave id, so a grave looks and
// "is" the same on every visit without storing anything.

import { STYLES_BY_ERA, type Style } from './sprites';

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

export function styleFor(id: number, year: number): Style {
  // blur era borders so styles blend
  const jitter = (hash01(id, 1) - 0.5) * (year < 1700 ? 300 : 40);
  const t = year + jitter;
  const era = STYLES_BY_ERA.find(e => t < e.until) ?? STYLES_BY_ERA[STYLES_BY_ERA.length - 1];
  const total = era.styles.reduce((a, [, w]) => a + w, 0);
  let x = hash01(id, 2) * total;
  for (const [s, w] of era.styles) { if ((x -= w) < 0) return s; }
  return era.styles[0][0];
}

export type GraveInfo = { id: number; year: number; age: number; sex: 'female' | 'male'; ageText: string };

/** rough, era-dependent age-at-death distribution (illustrative only) */
export function graveInfo(id: number, year: number): GraveInfo {
  const a = hash01(id, 3), b = hash01(id, 4), c = hash01(id, 5);
  const modern = Math.min(1, Math.max(0, (year - 1880) / 120)); // 0 before 1880 → 1 by 2000
  const pInfant = 0.26 * (1 - modern) + 0.04 * modern;
  const pChild = 0.2 * (1 - modern) + 0.03 * modern;
  let age: number;
  if (a < pInfant) age = b * b; // most infant deaths in the first weeks
  else if (a < pInfant + pChild) age = 1 + b * 14;
  else {
    const mode = 35 + 38 * modern, spread = 20 - 5 * modern;
    const g = (b + c + hash01(id, 6) - 1.5) * 2; // ≈ normal
    age = Math.min(105, Math.max(15, mode + g * spread));
  }
  const ageText = age < 1 ? (age < 1 / 12 ? `${Math.max(1, Math.round(age * 365))} days` : `${Math.round(age * 12)} months`)
    : `${Math.floor(age)} years`;
  return { id, year, age, sex: hash01(id, 7) < 0.515 ? 'male' : 'female', ageText };
}

export type Landmark = { year: number; title: string; text: string };

export const LANDMARKS: Landmark[] = [
  { year: -40000, title: 'Deep time', text: 'Inside the Ancient Circle time is not linear: its ~9 billion graves are packed in a flat disc, and the rings get longer the further out you walk.' },
  { year: -8000, title: 'Rim of the Ancient Circle', text: 'About 9 billion people died before 8000 BCE. From here outward, every 2.4 m walked is one year, and the ground starts to curve (negative curvature) so each year can hold all of its dead.' },
  { year: -3000, title: 'Writing', text: 'Around here the first written records appear. Everyone behind you lived and died before anyone could write their name.' },
  { year: 1, title: 'Year 1', text: 'World population ≈ 300 million. About 55 billion people have died before this ring — roughly half of everyone who ever lived.' },
  { year: 1200, title: 'Medieval world', text: 'PRB data resolution here is centuries: the Black Death (1347–1351) is inside the smoothed 1200–1650 period, so the rings do not show its spike.' },
  { year: 1650, title: '500 million alive', text: 'Deaths per year start rising steadily from here. Rings get longer much faster than a flat plane allows, which is what bends the ground.' },
  { year: 1850, title: 'Industrial age', text: 'Over a billion people alive. A ring around the graveyard here is ~65,000 km long (1.6× Earth\'s equator), yet you are only 122 km from the centre. On a flat plane it would be 770 km.' },
  { year: 1918, title: 'Pandemic & war', text: 'The 1900–1950 PRB period (2.5 billion deaths) includes the 1918 flu and both world wars; the model spreads it smoothly.' },
  { year: 1950, title: 'Yearly data', text: 'From here the model follows Our World in Data yearly deaths (5-year bins, smoothed). Around 48 million deaths per year.' },
  { year: 2020, title: 'COVID-19', text: '2020–2022 saw several million excess deaths per year. You are now close to the edge: the living are just ahead.' },
];
