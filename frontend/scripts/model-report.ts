// Prints the death model fit and derived geometry.  npm run model-report
import { Surface } from '../src/graveyard/surface';

const S = new Surface();
const B = (n: number) => `${(n / 1e9).toFixed(3)} B`;
console.log('period'.padEnd(30), 'data'.padStart(10), 'model rel. err');
for (const p of S.model.periods)
  console.log(p.source.padEnd(30), B(p.target).padStart(10), ((p.fitted - p.target) / p.target).toExponential(1));

const r = S.report;
console.log(`\nflatten window ${JSON.stringify(S.model.flattenWindow ?? null)}, D(switch) ${(S.model.D(S.t0) / 1e6).toFixed(2)} M/yr`);
console.log(`shape ${JSON.stringify(S.shape)}  σ = ${S.sigma.toFixed(4)} graves/m²`);
const Y = (m: number) => `${Math.round(S.yr(m)).toLocaleString('en-US')} yr`;
console.log(`ancient era (undated): ${B(S.ancientGraves)} graves, radius ${Y(S.rho0)} (floor ${Y(r.floorLen)}), positive curvature ${r.bulge ? `${Y(r.bulge[0])}–${Y(r.bulge[1])}, tightest radius ${Y(1 / Math.sqrt(r.maxBulgeK))}` : 'none'}${r.problem ? ` · PROBLEM: ${r.problem}` : ''}`);
console.log(`history ${Y(S.rhoEndTable - S.rho0)} (${S.rowsPerYear.toFixed(2)} rows/yr, render ${S.v.toFixed(2)} m/yr) · ancient:history ${(r.ancientLen / r.historyLen).toFixed(3)} · ancient share ${(100 * r.ancientShare).toFixed(1)}% (floor ${(100 * r.floorShare).toFixed(1)}%)`);
console.log(`rows ${S.nRows}, graves by 2030: ${B(S.rowStart[S.nRows])}, tightest ancient curvature radius ${Y(r.minKRadius)}\n`);
console.log('year'.padStart(7), 'ρ yr'.padStart(9), 'D M/yr'.padStart(8), 'ring yr'.padStart(10), 'K radius yr'.padStart(12));
for (const t of [-3000, -2000, -1000, 1, 1000, 1500, 1700, 1800, 1850, 1900, 1950, 2000, 2020, 2026]) {
  const rho = S.rhoAtTime(t), K = S.gaussK(rho);
  console.log(String(t).padStart(7), S.yr(rho).toFixed(0).padStart(9), (S.model.D(t) / 1e6).toFixed(2).padStart(8), S.yr(2 * Math.PI * S.f(rho)).toExponential(2).padStart(10),
    (K ? (K < 0 ? '−' : '+') + S.yr(1 / Math.sqrt(Math.abs(K))).toFixed(1) : 'flat').padStart(12));
}
