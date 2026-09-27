// Procedural night soundscape (Web Audio, no assets): wind in the grass, a low
// drone that deepens in the ancient era, crickets, an occasional owl, and gravel
// footsteps while walking. Must be started from a user gesture.

export type SoundState = {
  moving: boolean;
  stepRate: number; // steps per second
  ancient: number;  // 0 in history … 1 deep in the ancient era
};

export class Soundscape {
  private ctx: AudioContext | null = null;
  private master!: GainNode;
  private noise!: AudioBuffer;
  private drone!: GainNode;
  private droneFilter!: BiquadFilterNode;
  private stepClock = 0;
  private nextChirp = 0;
  private nextOwl = 0;

  get running(): boolean { return !!this.ctx; }

  start() {
    if (this.ctx) return;
    const ctx = this.ctx = new AudioContext();
    this.master = ctx.createGain();
    this.master.gain.value = 0;
    this.master.gain.linearRampToValueAtTime(0.7, ctx.currentTime + 2);
    this.master.connect(ctx.destination);

    // 4 s of brown-ish noise, looped
    const len = ctx.sampleRate * 4;
    this.noise = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = this.noise.getChannelData(0);
    let last = 0;
    for (let i = 0; i < len; i++) { last = 0.97 * last + 0.03 * (Math.random() * 2 - 1); d[i] = last * 6; }

    // wind: low-passed noise, cutoff and level drifting slowly
    const wind = this.loopNoise();
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass'; lp.frequency.value = 420; lp.Q.value = 0.7;
    const wg = ctx.createGain();
    wg.gain.value = 0.22;
    this.lfo(0.05, 260, lp.frequency);
    this.lfo(0.09, 0.08, wg.gain);
    wind.connect(lp).connect(wg).connect(this.master);
    // grass rustle: faint high band
    const rustle = this.loopNoise(1.3);
    const hp = ctx.createBiquadFilter();
    hp.type = 'bandpass'; hp.frequency.value = 3200; hp.Q.value = 0.6;
    const rg = ctx.createGain();
    rg.gain.value = 0.025;
    this.lfo(0.13, 0.02, rg.gain);
    rustle.connect(hp).connect(rg).connect(this.master);

    // drone: two soft low sines through a lowpass
    this.drone = ctx.createGain();
    this.drone.gain.value = 0.04;
    this.droneFilter = ctx.createBiquadFilter();
    this.droneFilter.type = 'lowpass'; this.droneFilter.frequency.value = 300;
    for (const [f, g] of [[55, 1], [82.4, 0.6], [110.3, 0.25]]) {
      const o = ctx.createOscillator();
      o.frequency.value = f;
      const og = ctx.createGain();
      og.gain.value = g;
      o.connect(og).connect(this.droneFilter);
      o.start();
    }
    this.droneFilter.connect(this.drone).connect(this.master);

    this.nextChirp = ctx.currentTime + 1;
    this.nextOwl = ctx.currentTime + 8 + Math.random() * 20;
  }

  stop() {
    const ctx = this.ctx;
    if (!ctx) return;
    this.ctx = null;
    this.master.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.4);
    setTimeout(() => ctx.close(), 500);
  }

  update(dt: number, s: SoundState) {
    const ctx = this.ctx;
    if (!ctx) return;
    const t = ctx.currentTime;
    this.drone.gain.setTargetAtTime(0.03 + 0.07 * s.ancient, t, 1.5);
    this.droneFilter.frequency.setTargetAtTime(380 - 200 * s.ancient, t, 1.5);

    if (s.moving) {
      this.stepClock += dt * s.stepRate;
      if (this.stepClock >= 1) { this.stepClock %= 1; this.crunch(t); }
    } else this.stepClock = 0.7; // first step comes quickly

    if (t >= this.nextChirp) {
      this.chirp(t, (1 - s.ancient * 0.7));
      this.nextChirp = t + 0.25 + Math.random() * (s.ancient > 0.5 ? 4 : 1.8);
    }
    if (t >= this.nextOwl) {
      this.owl(t);
      this.nextOwl = t + 25 + Math.random() * 50;
    }
  }

  // ── pieces ────────────────────────────────────────────────────────────────

  private loopNoise(rate = 1): AudioBufferSourceNode {
    const src = this.ctx!.createBufferSource();
    src.buffer = this.noise;
    src.loop = true;
    src.playbackRate.value = rate;
    src.start(0, Math.random() * 4);
    return src;
  }

  private lfo(freq: number, depth: number, target: AudioParam) {
    const ctx = this.ctx!;
    const o = ctx.createOscillator();
    o.frequency.value = freq * (0.8 + Math.random() * 0.4);
    const g = ctx.createGain();
    g.gain.value = depth;
    o.connect(g).connect(target);
    o.start();
  }

  private panner(x: number): StereoPannerNode {
    const p = this.ctx!.createStereoPanner();
    p.pan.value = x;
    p.connect(this.master);
    return p;
  }

  /** a footstep on gravel: short band-passed noise burst with a soft low thump */
  private crunch(t: number) {
    const ctx = this.ctx!;
    const src = ctx.createBufferSource();
    src.buffer = this.noise;
    src.playbackRate.value = 3 + Math.random() * 1.5;
    const bp = ctx.createBiquadFilter();
    bp.type = 'bandpass'; bp.frequency.value = 1500 + Math.random() * 900; bp.Q.value = 0.9;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.5, t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.12 + Math.random() * 0.05);
    src.connect(bp).connect(g).connect(this.panner((Math.random() - 0.5) * 0.2));
    src.start(t, Math.random() * 3.5, 0.2);
  }

  /** a cricket: a few 4–5 kHz pulses */
  private chirp(t: number, level: number) {
    const ctx = this.ctx!;
    const f = 4200 + Math.random() * 600;
    const pan = this.panner(Math.random() * 1.6 - 0.8);
    const pulses = 2 + Math.floor(Math.random() * 3);
    const vol = (0.012 + Math.random() * 0.02) * level;
    for (let k = 0; k < pulses; k++) {
      const o = ctx.createOscillator();
      o.frequency.value = f;
      const g = ctx.createGain();
      const t0 = t + k * 0.045;
      g.gain.setValueAtTime(0, t0);
      g.gain.linearRampToValueAtTime(vol, t0 + 0.006);
      g.gain.linearRampToValueAtTime(0, t0 + 0.028);
      o.connect(g).connect(pan);
      o.start(t0);
      o.stop(t0 + 0.03);
    }
  }

  /** a distant owl: two soft falling hoots */
  private owl(t: number) {
    const ctx = this.ctx!;
    const pan = this.panner(Math.random() * 1.4 - 0.7);
    const base = 360 + Math.random() * 60;
    for (const [dt, dur] of [[0, 0.5], [0.75, 0.9]]) {
      const o = ctx.createOscillator();
      o.type = 'sine';
      o.frequency.setValueAtTime(base * 1.04, t + dt);
      o.frequency.exponentialRampToValueAtTime(base * 0.94, t + dt + dur);
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t + dt);
      g.gain.linearRampToValueAtTime(0.05, t + dt + 0.08);
      g.gain.linearRampToValueAtTime(0, t + dt + dur);
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass'; lp.frequency.value = 900;
      o.connect(lp).connect(g).connect(pan);
      o.start(t + dt);
      o.stop(t + dt + dur + 0.05);
    }
  }
}
