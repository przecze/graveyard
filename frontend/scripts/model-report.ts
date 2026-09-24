// Prints the death model fit and derived geometry.  npm run model-report
import { Surface } from '../src/graveyard/surface';

const S = new Surface();
const B = (n: number) => `${(n / 1e9).toFixed(3)} B`;
console.log('period'.padEnd(30), 'data'.padStart(10), 'model rel. err');
for (const p of S.model.periods)
  console.log(p.source.padEnd(30), B(p.target).padStart(10), ((p.fitted - p.target) / p.target).toExponential(1));

const r = S.report;
console.log(`\nshape ${JSON.stringify(S.shape)}  σ = ${S.sigma.toFixed(4)} graves/m²`);
console.log(`ancient era (undated): ${B(S.ancientGraves)} graves, ${(S.rho0 / 1000).toFixed(1)} km (flat core ${(S.coreR / 1000).toFixed(1)} km), ring ${(r.plateauRing / 1000).toFixed(0)} km`);
console.log(`history ${((S.rhoEndTable - S.rho0) / 1000).toFixed(1)} km at ${S.v} m/yr · ancient share ${(100 * r.ancientShare).toFixed(1)}% (floor ${(100 * r.floorShare).toFixed(1)}%)`);
console.log(`rows ${S.nRows}, graves by 2030: ${B(S.rowStart[S.nRows])}, tightest ancient curvature radius ${r.minKRadius.toFixed(0)} m\n`);
console.log('year'.padStart(7), 'ρ km'.padStart(9), 'D M/yr'.padStart(8), 'ring km'.padStart(10), 'K radius km'.padStart(12));
for (const t of [-3000, -2000, -1000, 1, 1000, 1500, 1700, 1800, 1850, 1900, 1950, 2000, 2020, 2026]) {
  const rho = S.rhoAtTime(t), K = S.gaussK(rho);
  console.log(String(t).padStart(7), (rho / 1000).toFixed(3).padStart(9), (S.model.D(t) / 1e6).toFixed(2).padStart(8), (2 * Math.PI * S.f(rho) / 1000).toFixed(0).padStart(10),
    (K ? (K < 0 ? '−' : '+') + (1 / Math.sqrt(Math.abs(K)) / 1000).toFixed(2) : 'flat').padStart(12));
}
