import { useEffect, useRef, useState } from 'react';

const SPEED = 1;      // metric units / sec
const HALF_VIEW = 25; // metric viewport radius

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
      const R = Rref.current;
      if (keys['ArrowUp'])   rho = Math.max(1, rho + SPEED * dt);
      if (keys['ArrowDown']) rho = Math.max(1, rho - SPEED * dt);
      // arc-length speed = R·sinh(ρ/R)·dφ/dt = SPEED
      const dphi = SPEED / (R * Math.sinh(rho / R)) * dt;
      if (keys['ArrowLeft'])  phi -= dphi;
      if (keys['ArrowRight']) phi += dphi;
      phi = ((phi % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    }

    // Hyperboloid model: H² in ℝ^{2,1} with signature (−,+,+)
    // P(r,a) = (R·cosh(r/R), R·sinh(r/R)·cos(a), R·sinh(r/R)·sin(a))
    // Minkowski: ⟨u,v⟩ = −u₀v₀ + u₁v₁ + u₂v₂
    //
    // Log map at player P maps Q to tangent coords (tx, ty):
    //   tx = ê_φ direction (east, increasing φ)
    //   ty = ê_ρ direction (outward, increasing ρ)

    function draw() {
      const R = Rref.current;
      ctx.fillStyle = '#080c09';
      ctx.fillRect(0, 0, W, H);

      const scale  = Math.min(W, H) / 2 / HALF_VIEW;
      const shrp   = Math.sinh(rho / R);
      const chrp   = Math.cosh(rho / R);
      const cosp   = Math.cos(phi);
      const sinp   = Math.sin(phi);
      const Pz = R * chrp, Px = R * shrp * cosp, Py = R * shrp * sinp;
      const coshHV = Math.cosh(HALF_VIEW / R);
      const sinhHV = Math.sinh(HALF_VIEW / R);

      function logMap(r: number, a: number): [number, number] | null {
        const shr = Math.sinh(r / R), chr = Math.cosh(r / R);
        const Qz = R * chr, Qx = R * shr * Math.cos(a), Qy = R * shr * Math.sin(a);
        const coshD = Math.max(1, (Pz * Qz - Px * Qx - Py * Qy) / (R * R));
        if (coshD > coshHV * 1.6) return null;
        const dR    = Math.acosh(coshD);
        const f     = dR < 1e-9 ? 1.0 : dR / Math.sinh(dR);
        const vz = f * (Qz - coshD * Pz);
        const vx = f * (Qx - coshD * Px);
        const vy = f * (Qy - coshD * Py);
        // ê_φ = (0, −sinp, cosp)  ê_ρ = (shrp, chrp·cosp, chrp·sinp)  [indices: z,x,y]
        const tx = -vx * sinp + vy * cosp;
        const ty = -vz * shrp + vx * chrp * cosp + vy * chrp * sinp;
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
      const rMin = Math.max(1, Math.floor(rho - HALF_VIEW) - 1);
      const rMax = Math.ceil(rho + HALF_VIEW) + 1;

      for (let r = rMin; r <= rMax; r++) {
        const shr = Math.sinh(r / R), chr = Math.cosh(r / R);
        const denom = shrp * shr;
        let arcPhi: number;
        if (denom < 1e-10) {
          arcPhi = Math.PI;
        } else {
          const cosT = (chrp * chr - coshHV) / denom;
          if (cosT > 1) continue;
          arcPhi = cosT < -1 ? Math.PI : Math.acos(cosT);
        }

        const major = r % 10 === 0, mid = r % 5 === 0;
        ctx.strokeStyle = major ? '#2a6638' : mid ? '#184228' : '#0f2318';
        ctx.lineWidth   = major ? 1.2 : 0.5;

        const N = Math.max(8, Math.round(arcPhi * 120));
        const pts: ([number, number] | null)[] = [];
        for (let i = 0; i <= N; i++)
          pts.push(logMap(r, phi - arcPhi + (i / N) * 2 * arcPhi));
        drawPath(pts);

        if (major) {
          const lm = logMap(r, phi - 0.05);
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
      // min dist to line at angle a: R·arcsinh(shrp·|sin Δφ|) ≤ HALF_VIEW
      const maxSinDphi = shrp > 1e-10 ? Math.min(1, sinhHV / shrp) : 1;
      const maxDphi    = Math.asin(maxSinDphi);

      for (let deg = 0; deg < 360; deg++) {
        const a  = deg * Math.PI / 180;
        let   da = ((a - phi) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
        if (da > Math.PI) da -= 2 * Math.PI;
        if (Math.abs(da) > maxDphi + 0.1) continue;

        const major = deg % 10 === 0, mid = deg % 5 === 0;
        ctx.strokeStyle = major ? '#2a6638' : mid ? '#184228' : '#0f2318';
        ctx.lineWidth   = major ? 1.2 : 0.5;

        const N = 80;
        const pts: ([number, number] | null)[] = [];
        for (let i = 0; i <= N; i++)
          pts.push(logMap(rMin + (i / N) * (rMax - rMin), a));
        drawPath(pts);

        if (major) {
          const lm = logMap(Math.min(rho + HALF_VIEW * 0.78, rMax - 1), a);
          if (lm) {
            const [sx, sy] = toScr(lm[0], lm[1]);
            if (sx > 8 && sx < W - 8 && sy > 8 && sy < H - 8) {
              ctx.fillStyle = '#3d9952'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
              ctx.fillText(deg + '°', sx, sy);
            }
          }
        }
      }

      // --- player dot ---
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
      ctx.fillStyle = '#88ffaa'; ctx.shadowColor = '#88ffaa'; ctx.shadowBlur = 8;
      ctx.fill(); ctx.shadowBlur = 0;

      // east tick (increasing φ = tx direction)
      const tickLen   = 14 / scale;
      const [ax, ay]  = toScr(tickLen, 0);
      ctx.strokeStyle = '#88ffaa55'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ax, ay); ctx.stroke();

      // --- chunk calculation for the ring the player is in ---
      const ringW    = 10;
      const rInner   = Math.floor(rho / ringW) * ringW;
      const rOuter   = rInner + ringW;
      const ringArea = 2 * Math.PI * R * R * (Math.cosh(rOuter / R) - Math.cosh(rInner / R));
      const nChunks  = Math.max(1, Math.ceil(ringArea / 100));
      const dPhi     = (2 * Math.PI) / nChunks;
      const chunkIdx = Math.floor(phi / dPhi);
      const chunkA0  = chunkIdx * dPhi;
      const chunkA1  = chunkA0 + dPhi;

      // draw chunk borders (two radial lines at chunkA0 and chunkA1)
      ctx.strokeStyle = '#ffcc44';
      ctx.lineWidth   = 1.5;
      for (const borderAngle of [chunkA0, chunkA1]) {
        const N = 60;
        const pts: ([number, number] | null)[] = [];
        for (let i = 0; i <= N; i++)
          pts.push(logMap(rInner + (i / N) * (rOuter - rInner), borderAngle));
        drawPath(pts);
      }
      // draw inner/outer arc of current chunk
      ctx.strokeStyle = '#ffcc4466';
      ctx.lineWidth   = 1.0;
      for (const arcR of [rInner, rOuter]) {
        const N = Math.max(8, Math.round(Math.abs(chunkA1 - chunkA0) * 80));
        const pts: ([number, number] | null)[] = [];
        for (let i = 0; i <= N; i++)
          pts.push(logMap(arcR, chunkA0 + (i / N) * (chunkA1 - chunkA0)));
        drawPath(pts);
      }

      // --- chunk grid: 10 ρ × 10 φ sample points ---
      const NR = 10, NP = 10;
      for (let ir = 0; ir < NR; ir++) {
        const gr = rInner + (ir + 0.5) / NR * (rOuter - rInner);
        for (let ip = 0; ip < NP; ip++) {
          const ga = chunkA0 + (ip + 0.5) / NP * (chunkA1 - chunkA0);
          const lm = logMap(gr, ga);
          if (!lm) continue;
          const [sx, sy] = toScr(lm[0], lm[1]);
          ctx.beginPath();
          ctx.arc(sx, sy, 2.5, 0, 2 * Math.PI);
          ctx.fillStyle = '#ffcc4499';
          ctx.fill();
        }
      }

      // --- viewport chunk debug + render ---
      const ringDebugLines: string[] = [];
      const firstRing = Math.floor(rMin / ringW) * ringW;
      for (let ri = firstRing; ri < rMax; ri += ringW) {
        const ro = ri + ringW;
        // conservative arcPhi for the ring: max over inner, mid, outer radii
        let ringArcPhi = 0;
        for (const r of [Math.max(1, ri), (Math.max(1, ri) + ro) / 2, ro]) {
          const shr = Math.sinh(r / R), chr = Math.cosh(r / R);
          const den = shrp * shr;
          let ap: number;
          if (den < 1e-10) { ap = Math.PI; }
          else {
            const cosT = (chrp * chr - coshHV) / den;
            ap = cosT >= 1 ? 0 : cosT <= -1 ? Math.PI : Math.acos(cosT);
          }
          if (ap > ringArcPhi) ringArcPhi = ap;
        }
        if (ringArcPhi === 0) continue;

        const rArea  = 2 * Math.PI * R * R * (Math.cosh(ro / R) - Math.cosh(Math.max(0, ri) / R));
        const nC     = Math.max(1, Math.ceil(rArea / 100));
        const dpC    = (2 * Math.PI) / nC;

        const overlapping: number[] = [];
        for (let k = 0; k < nC; k++) {
          const midA = (k + 0.5) * dpC;
          let da = Math.abs(midA - phi) % (2 * Math.PI);
          if (da > Math.PI) da = 2 * Math.PI - da;
          if (da - dpC / 2 <= ringArcPhi) overlapping.push(k);
        }

        // render visible chunks: border + grid, cyan
        ctx.strokeStyle = '#00ccff';
        ctx.lineWidth = 1.0;
        const riClamped = Math.max(1, ri);
        for (const k of overlapping) {
          const a0 = k * dpC;
          const a1 = a0 + dpC;
          // radial edges
          for (const borderAngle of [a0, a1]) {
            const N = 40;
            const pts: ([number, number] | null)[] = [];
            for (let i = 0; i <= N; i++)
              pts.push(logMap(riClamped + (i / N) * (ro - riClamped), borderAngle));
            drawPath(pts);
          }
          // arc edges
          for (const arcR of [riClamped, ro]) {
            const N = Math.max(4, Math.round(Math.abs(a1 - a0) * 60));
            const pts: ([number, number] | null)[] = [];
            for (let i = 0; i <= N; i++)
              pts.push(logMap(arcR, a0 + (i / N) * (a1 - a0)));
            drawPath(pts);
          }
          // grid dots
          const NR = 10, NP = 10;
          for (let ir2 = 0; ir2 < NR; ir2++) {
            const gr = riClamped + (ir2 + 0.5) / NR * (ro - riClamped);
            for (let ip = 0; ip < NP; ip++) {
              const ga = a0 + (ip + 0.5) / NP * (a1 - a0);
              const lm = logMap(gr, ga);
              if (!lm) continue;
              const [sx, sy] = toScr(lm[0], lm[1]);
              ctx.beginPath();
              ctx.arc(sx, sy, 2.5, 0, 2 * Math.PI);
              ctx.fillStyle = '#00ccff99';
              ctx.fill();
            }
          }
        }

        const phiLoDeg = ((phi - ringArcPhi) * 180 / Math.PI).toFixed(1);
        const phiHiDeg = ((phi + ringArcPhi) * 180 / Math.PI).toFixed(1);
        ringDebugLines.push(
          `[${ri},${ro}] φ∈[${phiLoDeg}°,${phiHiDeg}°] n=${nC} ids:[${overlapping.join(',')}]`
        );
      }

      const circ = (2 * Math.PI * R * Math.sinh(rho / R)).toFixed(1);
      hud.innerHTML =
        `ρ = ${rho.toFixed(2)}&nbsp;&nbsp; φ = ${(phi * 180 / Math.PI).toFixed(2)}°&nbsp;&nbsp; C = ${circ}<br>` +
        `ring: [${rInner}, ${rOuter}]<br>` +
        `area = 2π·R²·(cosh(${rOuter}/R)−cosh(${rInner}/R))<br>` +
        `     = 2π·${R}²·(${Math.cosh(rOuter/R).toFixed(4)}−${Math.cosh(rInner/R).toFixed(4)})<br>` +
        `     = ${ringArea.toFixed(2)}<br>` +
        `n_chunks = ⌈${ringArea.toFixed(2)}/100⌉ = ${nChunks}<br>` +
        `δφ = 2π/${nChunks} = ${(dPhi * 180 / Math.PI).toFixed(3)}°<br>` +
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
