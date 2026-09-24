import { useEffect, useRef, useState } from 'react';

const SPEED = 1;
const HALF_VIEW = 25;
const RHO_EDGE = 1000; // flat (K=0) for ρ < RHO_EDGE, hyperbolic (K=-1/R²) for ρ ≥ RHO_EDGE

export default function PolarWalker() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hudRef    = useRef<HTMLDivElement>(null);
  const [R, setR] = useState(10_000);
  const Rref = useRef(R);

  useEffect(() => { Rref.current = R; }, [R]);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx    = canvas.getContext('2d')!;
    const hud    = hudRef.current!;

    let W: number, H: number, cx: number, cy: number;
    function resize() {
      W = canvas.width  = window.innerWidth;
      H = canvas.height = window.innerHeight;
      cx = W / 2; cy = H / 2;
    }
    resize();
    window.addEventListener('resize', resize);

    let rho = 1000;
    let phi = 0;

    const keys: Record<string, boolean> = {};
    function onDown(e: KeyboardEvent) { keys[e.key] = true;  e.preventDefault(); }
    function onUp  (e: KeyboardEvent) { keys[e.key] = false; }
    window.addEventListener('keydown', onDown);
    window.addEventListener('keyup',   onUp);

    function update(dt: number) {
      const k = 1 / Rref.current;
      if (keys['ArrowUp'])   rho = Math.max(1, rho + SPEED * dt);
      if (keys['ArrowDown']) rho = Math.max(1, rho - SPEED * dt);
      // arc-length speed in φ direction = cf(rho) · dφ/dt = SPEED
      const d = rho - RHO_EDGE;
      const fr = rho <= RHO_EDGE
        ? rho
        : RHO_EDGE * Math.cosh(k * d) + Math.sinh(k * d) / k;
      const dphi = SPEED / Math.max(1e-6, fr) * dt;
      if (keys['ArrowLeft'])  phi -= dphi;
      if (keys['ArrowRight']) phi += dphi;
      phi = ((phi % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    }

    // Metric: ds² = dρ² + cf(ρ)²dφ²
    // K=0 for ρ≤RHO_EDGE (flat), K=-1/R² for ρ>RHO_EDGE (hyperbolic), C¹ match at boundary.

    function draw() {
      const R = Rref.current;
      const k = 1 / R;

      ctx.fillStyle = '#080c09';
      ctx.fillRect(0, 0, W, H);

      const scale = Math.min(W, H) / 2 / HALF_VIEW;

      // Circumference-radius: ds² = dρ² + cf(ρ)²dφ²
      function cf(r: number): number {
        if (r <= RHO_EDGE) return Math.max(1e-9, r);
        const d = r - RHO_EDGE;
        return RHO_EDGE * Math.cosh(k * d) + Math.sinh(k * d) / k;
      }
      const rMin = Math.max(1, Math.floor(rho - HALF_VIEW) - 1);
      const rMax = Math.ceil(rho + HALF_VIEW) + 1;

      // Isothermal radial coordinate: u(r) = ∫_1^r dρ/cf(ρ)
      // Flat part analytic: u(r) = ln(r) for r ≤ RHO_EDGE
      // Hyper part: u(r) = ln(RHO_EDGE) + ∫_{RHO_EDGE}^r dρ/cf(ρ), precomputed via Simpson
      const uStep = 0.5;
      const uArrStart = Math.max(RHO_EDGE, rMin - 1);
      const uArrCount = Math.max(2, Math.ceil((rMax + 1 - uArrStart) / uStep) + 2);
      const uArr = new Float64Array(uArrCount);
      for (let i = 1; i < uArrCount; i++) {
        const r1 = uArrStart + (i-1)*uStep, r2 = r1 + uStep, rm = (r1+r2)/2;
        uArr[i] = uArr[i-1] + (1/cf(r1) + 4/cf(rm) + 1/cf(r2)) * uStep/6;
      }
      function isoCoord(r: number): number {
        if (r <= RHO_EDGE) return Math.log(Math.max(1e-9, r));
        const idx = (r - uArrStart) / uStep, i = Math.floor(idx);
        const off = i <= 0 ? uArr[0] : i >= uArrCount-1 ? uArr[uArrCount-1]
          : uArr[i] + (idx-i)*(uArr[i+1]-uArr[i]);
        return Math.log(RHO_EDGE) + off;
      }

      // Precomputed player quantities
      const frho = cf(rho);
      const up   = isoCoord(rho);

      // Conformal map at player (rho,phi) → tangent vector (tx=east, ty=north) toward (rq,aq)
      // Isothermal coords (u,φ): ds²=cf²(du²+dφ²). Map: exp(Δu+iΔφ)−1, scaled by cf(rho).
      // Reduces to exact Euclidean in flat region; compresses hyperbolic infinity to finite disk.
      function conformalMap(rq: number, aq: number): [number, number] | null {
        const du   = isoCoord(rq) - up;
        const dphi = ((aq - phi) % (2*Math.PI) + 3*Math.PI) % (2*Math.PI) - Math.PI;
        const rt   = Math.exp(du);
        const tx   = rt * Math.sin(dphi) * frho;
        const ty   = (rt * Math.cos(dphi) - 1) * frho;
        if (tx*tx + ty*ty > (HALF_VIEW * 1.6)**2) return null;
        return [tx, ty];
      }

      function toScr(tx: number, ty: number): [number, number] {
        return [cx + tx * scale, cy - ty * scale];
      }

      function drawPath(pts: ([number, number] | null)[]) {
        ctx.beginPath();
        let pen = false;
        for (const p of pts) {
          if (!p) { pen = false; continue; }
          const [sx, sy] = toScr(p[0], p[1]);
          if (!pen) { ctx.moveTo(sx, sy); pen = true; }
          else        ctx.lineTo(sx, sy);
        }
        ctx.stroke();
      }

      // --- ρ circles ---
      for (let r = rMin; r <= rMax; r++) {
        const dRho = Math.abs(r - rho);
        if (dRho > HALF_VIEW + 0.5) continue;
        const tangBudget = Math.sqrt(Math.max(0, HALF_VIEW*HALF_VIEW - dRho*dRho));
        const fMin = Math.max(1, Math.min(frho, cf(r)));
        const arcPhi = Math.min(Math.PI, tangBudget / fMin + 0.02);
        if (arcPhi <= 0) continue;

        const major = r % 10 === 0, mid = r % 5 === 0;
        ctx.strokeStyle = major ? '#2a6638' : mid ? '#184228' : '#0f2318';
        ctx.lineWidth   = major ? 1.2 : 0.5;

        const N = Math.max(8, Math.round(arcPhi * 120));
        const pts: ([number, number] | null)[] = [];
        for (let i = 0; i <= N; i++)
          pts.push(conformalMap(r, phi - arcPhi + (i / N) * 2 * arcPhi));
        drawPath(pts);

        if (major) {
          const lm = conformalMap(r, phi - 0.05);
          if (lm) {
            const [sx, sy] = toScr(lm[0], lm[1]);
            if (sx > 0 && sx < W && sy > 0 && sy < H) {
              ctx.fillStyle = '#3d9952'; ctx.font = '11px monospace'; ctx.textAlign = 'left';
              ctx.fillText(String(r), sx + 4, sy - 3);
            }
          }
        }
      }

      // --- φ radial lines ---
      const maxDphi = Math.asin(Math.min(1, HALF_VIEW * 1.1 / Math.max(1, frho)));

      for (let deg = 0; deg < 360; deg++) {
        const a = deg * Math.PI / 180;
        let da = ((a - phi) % (2*Math.PI) + 2*Math.PI) % (2*Math.PI);
        if (da > Math.PI) da -= 2*Math.PI;
        if (Math.abs(da) > maxDphi + 0.1) continue;

        const major = deg % 10 === 0, mid = deg % 5 === 0;
        ctx.strokeStyle = major ? '#2a6638' : mid ? '#184228' : '#0f2318';
        ctx.lineWidth   = major ? 1.2 : 0.5;

        const N = 80;
        const pts: ([number, number] | null)[] = [];
        for (let i = 0; i <= N; i++)
          pts.push(conformalMap(rMin + (i / N) * (rMax - rMin), a));
        drawPath(pts);

        if (major) {
          const lm = conformalMap(Math.min(rho + HALF_VIEW * 0.78, rMax - 1), a);
          if (lm) {
            const [sx, sy] = toScr(lm[0], lm[1]);
            if (sx > 8 && sx < W-8 && sy > 8 && sy < H-8) {
              ctx.fillStyle = '#3d9952'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
              ctx.fillText(deg + '°', sx, sy);
            }
          }
        }
      }

      // --- player dot ---
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, 2*Math.PI);
      ctx.fillStyle = '#88ffaa'; ctx.shadowColor = '#88ffaa'; ctx.shadowBlur = 8;
      ctx.fill(); ctx.shadowBlur = 0;

      // east tick
      const tickLen = 14 / scale;
      const [ax, ay] = toScr(tickLen, 0);
      ctx.strokeStyle = '#88ffaa55'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ax, ay); ctx.stroke();

      // Ring area: 2π ∫ cf(r) dr, piecewise analytical
      function ringAreaFn(ri: number, ro: number): number {
        let area = 0;
        const flatHi = Math.min(ro, RHO_EDGE);
        if (ri < flatHi)
          area += Math.PI * (flatHi*flatHi - ri*ri);
        if (ro > RHO_EDGE) {
          const dMax = ro - RHO_EDGE, dMin = Math.max(0, ri - RHO_EDGE);
          const hyperInteg = (d: number) =>
            RHO_EDGE * Math.sinh(k*d) / k + (Math.cosh(k*d) - 1) / (k*k);
          area += 2*Math.PI * (hyperInteg(dMax) - hyperInteg(dMin));
        }
        return area;
      }

      // --- chunk for player's ring ---
      const ringW    = 10;
      const rInner   = Math.floor(rho / ringW) * ringW;
      const rOuter   = rInner + ringW;
      const ringArea = ringAreaFn(Math.max(1, rInner), rOuter);
      const nChunks  = Math.max(1, Math.ceil(ringArea / 100));
      const dPhi     = (2*Math.PI) / nChunks;
      const chunkIdx = Math.floor(phi / dPhi);
      const chunkA0  = chunkIdx * dPhi;
      const chunkA1  = chunkA0 + dPhi;

      ctx.strokeStyle = '#ffcc44'; ctx.lineWidth = 1.5;
      for (const borderAngle of [chunkA0, chunkA1]) {
        const pts: ([number, number] | null)[] = [];
        for (let i = 0; i <= 60; i++)
          pts.push(conformalMap(rInner + (i/60)*(rOuter-rInner), borderAngle));
        drawPath(pts);
      }
      ctx.strokeStyle = '#ffcc4466'; ctx.lineWidth = 1.0;
      for (const arcR of [rInner, rOuter]) {
        const N = Math.max(8, Math.round(Math.abs(chunkA1-chunkA0)*80));
        const pts: ([number, number] | null)[] = [];
        for (let i = 0; i <= N; i++)
          pts.push(conformalMap(arcR, chunkA0 + (i/N)*(chunkA1-chunkA0)));
        drawPath(pts);
      }
      for (let ir = 0; ir < 10; ir++) {
        const gr = rInner + (ir+0.5)/10*(rOuter-rInner);
        for (let ip = 0; ip < 10; ip++) {
          const ga = chunkA0 + (ip+0.5)/10*(chunkA1-chunkA0);
          const lm = conformalMap(gr, ga); if (!lm) continue;
          const [sx, sy] = toScr(lm[0], lm[1]);
          ctx.beginPath(); ctx.arc(sx, sy, 2.5, 0, 2*Math.PI);
          ctx.fillStyle = '#ffcc4499'; ctx.fill();
        }
      }

      // --- viewport chunk debug + render ---
      const ringDebugLines: string[] = [];
      const firstRing = Math.floor(rMin / ringW) * ringW;
      for (let ri = firstRing; ri < rMax; ri += ringW) {
        const ro = ri + ringW;
        const dRho_mid = Math.abs((ri+ro)/2 - rho);
        const tangBudget = Math.sqrt(Math.max(0, HALF_VIEW*HALF_VIEW - Math.min(HALF_VIEW,dRho_mid)**2));
        const fMid = Math.max(1, cf((ri+ro)/2));
        const ringArcPhi = Math.min(Math.PI, tangBudget/fMid*1.3 + 0.05);
        if (ringArcPhi <= 0) continue;

        const rArea = ringAreaFn(Math.max(0, ri), ro);
        const nC    = Math.max(1, Math.ceil(rArea/100));
        const dpC   = (2*Math.PI) / nC;
        const overlapping: number[] = [];
        for (let kk = 0; kk < nC; kk++) {
          const midA = (kk+0.5)*dpC;
          let da = Math.abs(midA-phi) % (2*Math.PI);
          if (da > Math.PI) da = 2*Math.PI - da;
          if (da - dpC/2 <= ringArcPhi) overlapping.push(kk);
        }

        ctx.strokeStyle = '#00ccff'; ctx.lineWidth = 1.0;
        const riClamped = Math.max(1, ri);
        for (const kk of overlapping) {
          const a0 = kk*dpC, a1 = a0+dpC;
          for (const ba of [a0, a1]) {
            const pts: ([number, number] | null)[] = [];
            for (let i = 0; i <= 40; i++)
              pts.push(conformalMap(riClamped + (i/40)*(ro-riClamped), ba));
            drawPath(pts);
          }
          for (const arcR of [riClamped, ro]) {
            const N = Math.max(4, Math.round(Math.abs(a1-a0)*60));
            const pts: ([number, number] | null)[] = [];
            for (let i = 0; i <= N; i++)
              pts.push(conformalMap(arcR, a0+(i/N)*(a1-a0)));
            drawPath(pts);
          }
          for (let ir2 = 0; ir2 < 10; ir2++) {
            const gr = riClamped + (ir2+0.5)/10*(ro-riClamped);
            for (let ip = 0; ip < 10; ip++) {
              const ga = a0 + (ip+0.5)/10*(a1-a0);
              const lm = conformalMap(gr, ga); if (!lm) continue;
              const [sx, sy] = toScr(lm[0], lm[1]);
              ctx.beginPath(); ctx.arc(sx, sy, 2.5, 0, 2*Math.PI);
              ctx.fillStyle = '#00ccff99'; ctx.fill();
            }
          }
        }
        const phiLoDeg = ((phi-ringArcPhi)*180/Math.PI).toFixed(1);
        const phiHiDeg = ((phi+ringArcPhi)*180/Math.PI).toFixed(1);
        ringDebugLines.push(
          `[${ri},${ro}] φ∈[${phiLoDeg}°,${phiHiDeg}°] n=${nC} ids:[${overlapping.join(',')}]`
        );
      }

      const circ = (2*Math.PI*cf(rho)).toFixed(1);
      hud.innerHTML =
        `ρ = ${rho.toFixed(2)}&nbsp;&nbsp; φ = ${(phi*180/Math.PI).toFixed(2)}°&nbsp;&nbsp; C = ${circ}<br>` +
        `ring: [${rInner}, ${rOuter}]<br>` +
        `area = 2π·∫cf(ρ)dρ = ${ringArea.toFixed(2)}<br>` +
        `n_chunks = ⌈${ringArea.toFixed(2)}/100⌉ = ${nChunks}<br>` +
        `δφ = 2π/${nChunks} = ${(dPhi*180/Math.PI).toFixed(3)}°<br>` +
        `chunk #${chunkIdx}  [${(chunkA0*180/Math.PI).toFixed(2)}°, ${(chunkA1*180/Math.PI).toFixed(2)}°]<br>` +
        `<br>viewport chunks:<br>` +
        ringDebugLines.join('<br>');
    }

    let animId: number, last = 0;
    function loop(ts: number) {
      const dt = Math.min((ts - last) / 1000, 0.05);
      last = ts;
      update(dt);
      draw();
      animId = requestAnimationFrame(loop);
    }
    animId = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
      window.removeEventListener('keydown', onDown);
      window.removeEventListener('keyup',   onUp);
    };
  }, []);

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden', background: '#080c09' }}>
      <canvas ref={canvasRef} style={{ display: 'block' }} />
      <div ref={hudRef} style={{
        position: 'fixed', top: 14, left: 14,
        font: '13px monospace', color: '#4bba66',
        textShadow: '0 0 6px #4bba6633',
        lineHeight: '1.6',
      }} />
      <div style={{
        position: 'fixed', bottom: 14, left: 14,
        font: '11px monospace', color: '#2a5c38',
        display: 'flex', flexDirection: 'column', gap: 6,
      }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#3d9952' }}>
          R = {R}
          <input
            type="range" min={10} max={500} step={1} value={R}
            onChange={e => setR(Number(e.target.value))}
            style={{ accentColor: '#4bba66', width: 120 }}
          />
        </label>
        <span>↑ ↓  ρ    ←  →  φ</span>
      </div>
    </div>
  );
}
