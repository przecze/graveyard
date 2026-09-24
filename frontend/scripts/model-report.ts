// Prints the death model fit and derived geometry.  npm run model-report
import { surface as S, T0 } from '../src/graveyard/surface';

const B = (n: number) => `${(n / 1e9).toFixed(3)} B`;
console.log('period'.padEnd(30), 'data'.padStart(10), 'model rel. err');
for (const p of S.model.periods)
  console.log(p.source.padEnd(30), B(p.target).padStart(10), ((p.fitted - p.target) / p.target).toExponential(1));

console.log(`\nancient deaths ${B(S.model.ancient.N)}, σ = ${S.sigma.toFixed(4)} graves/m²`);
console.log(`ancient circle radius ρ0 = ${(S.rho0 / 1000).toFixed(2)} km, v = ${S.v.toFixed(3)} m/yr`);
console.log(`rows ${S.nRows}, graves by ${T0 + S.model.D.length - 1}: ${B(S.rowStart[S.nRows])}\n`);
console.log('year'.padStart(7), 'ρ km'.padStart(9), 'D M/yr'.padStart(8), 'ring km'.padStart(10), 'K radius m'.padStart(11));
for (const t of [-40000, -20000, -8000, -5000, -3000, 1, 1000, 1500, 1700, 1800, 1850, 1900, 1950, 2000, 2020, 2026]) {
  const r = S.rhoAtTime(t), K = S.gaussK(r);
  console.log(String(t).padStart(7), (r / 1000).toFixed(3).padStart(9),
    (S.f(r) * 2 * Math.PI * S.sigma * S.v / 1e6).toFixed(2).padStart(8),
    (2 * Math.PI * S.f(r) / 1000).toFixed(0).padStart(10),
    (K ? (K < 0 ? '−' : '+') + (1 / Math.sqrt(Math.abs(K))).toFixed(0) : 'flat').padStart(11));
}
