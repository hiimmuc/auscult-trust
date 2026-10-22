// Canvas waveform panel: shared time view, playback cursor, prediction overlays in packed rows, 2-class and annotation lanes.

import type { Segment } from "./api";
import { conformalOn } from "./prefs";

export class TimeView {
  t0 = 0;
  t1 = 1;
  duration = 1;
  private subs = new Set<() => void>();

  on(f: () => void) { this.subs.add(f); return () => this.subs.delete(f); }
  private emit() { this.subs.forEach((f) => f()); }

  setDuration(d: number, keep = false) {
    this.duration = Math.max(d, 0.5);
    if (!keep || this.t1 > this.duration) { this.t0 = 0; this.t1 = this.duration; }
    this.emit();
  }
  set(t0: number, t1: number) {
    const span = Math.min(Math.max(t1 - t0, 0.25), this.duration);
    this.t0 = Math.min(Math.max(t0, 0), this.duration - span);
    this.t1 = this.t0 + span;
    this.emit();
  }
  fit() { this.set(0, this.duration); }
  zoom(f: number, at = (this.t0 + this.t1) / 2) { this.set(at - (at - this.t0) * f, at + (this.t1 - at) * f); }
  pan(dt: number) { this.set(this.t0 + dt, this.t1 + dt); }
}

export class Player {
  private src: AudioBufferSourceNode | null = null;
  private startedAt = 0;
  private offset = 0;
  playing = false;
  onchange = () => {};

  constructor(private ctx: AudioContext, public buffer: AudioBuffer) {}

  position() {
    return this.playing ? Math.min(this.offset + this.ctx.currentTime - this.startedAt, this.buffer.duration) : this.offset;
  }
  async play(from = this.position()) {
    this.stopSource();
    if (this.ctx.state === "suspended") await this.ctx.resume();
    if (from >= this.buffer.duration - 0.05) from = 0;
    const s = this.ctx.createBufferSource();
    s.buffer = this.buffer;
    s.connect(this.ctx.destination);
    s.onended = () => { if (this.src === s) { this.playing = false; this.offset = 0; this.src = null; this.onchange(); } };
    s.start(0, from);
    this.src = s; this.startedAt = this.ctx.currentTime; this.offset = from; this.playing = true;
    this.onchange();
  }
  pause() { this.offset = this.position(); this.stopSource(); this.playing = false; this.onchange(); }
  seek(t: number) { this.playing ? this.play(t) : (this.offset = Math.max(0, Math.min(t, this.buffer.duration)), this.onchange()); }
  toggle() { this.playing ? this.pause() : this.play(); }
  private stopSource() { if (this.src) { const s = this.src; this.src = null; try { s.stop(); } catch { /* already stopped */ } } }
  dispose() { this.stopSource(); this.playing = false; }
}

const WAVE_H = 148, LANE_H = 16, TRUTH_H = 10, AXIS_H = 22, GAP = 4;
const BLOCK = 256;

export function css(name: string) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

export function packRows<T extends { start: number; end: number }>(items: T[]): number[] {
  const ends: number[] = [];
  return items.map((s) => {
    let r = ends.findIndex((e) => e <= s.start + 1e-3);
    if (r < 0) { r = ends.length; ends.push(0); }
    ends[r] = s.end;
    return r;
  });
}

function niceStep(span: number, px: number) {
  const target = span / Math.max(px / 90, 1);
  return [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30].find((s) => s >= target) ?? 60;
}

export interface WaveOptions {
  describe: (seg: Segment) => string; // tooltip HTML
  onSelect?: (index: number) => void;
  label: string; // accessible name of the canvas
}

export class WaveView {
  readonly canvas = document.createElement("canvas");
  private samples: Float32Array | null = null;
  private sr = 16000;
  private mins = new Float32Array(0);
  private maxs = new Float32Array(0);
  private peak = 1;
  gain: number | null = null; // shared display scale (1 / peak); null = own peak
  private segs: Segment[] = [];
  private rows: number[] = [];
  private nRows = 1;
  private truth: { start: number; end: number; label: string }[] = [];
  selected = -1;
  player: Player | null = null;
  private raf = 0;
  private drag: { x0: number; x1: number } | null = null;
  private hover = -1;
  private liveLength = 0;

  constructor(host: HTMLElement, public view: TimeView, private opts: WaveOptions) {
    this.canvas.className = "wave-canvas";
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute("role", "img");
    this.canvas.setAttribute("aria-label", opts.label);
    host.append(this.canvas);
    new ResizeObserver(() => this.draw()).observe(host);
    view.on(() => this.draw());
    for (const [t, e] of [[matchMedia("(prefers-color-scheme: dark)"), "change"], [window, "themechange"]] as const) t.addEventListener(e, () => this.draw());
    this.bind();
  }

  get height() { return WAVE_H + GAP + LANE_H + (this.truth.length ? GAP + TRUTH_H : 0) + AXIS_H; }

  setAudio(samples: Float32Array, sr: number) {
    this.samples = samples; this.sr = sr; this.liveLength = 0;
    const n = Math.ceil(samples.length / BLOCK);
    this.mins = new Float32Array(n); this.maxs = new Float32Array(n);
    let peak = 1e-6;
    for (let b = 0; b < n; b++) {
      let lo = 1, hi = -1;
      for (let i = b * BLOCK, e = Math.min(i + BLOCK, samples.length); i < e; i++) { const v = samples[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
      this.mins[b] = lo; this.maxs[b] = hi; peak = Math.max(peak, -lo, hi);
    }
    this.peak = peak;
    this.draw();
  }

  get peakAmplitude() { return this.peak; }

  /** Live recording: `samples` grows; only the first `length` samples are drawn. */
  setLive(samples: Float32Array, length: number, sr: number) {
    this.samples = samples; this.sr = sr; this.liveLength = length; this.mins = new Float32Array(0);
    this.draw();
  }

  setSegments(segs: Segment[], truth: { start: number; end: number; label: string }[] = []) {
    this.segs = segs; this.rows = packRows(segs); this.nRows = Math.max(1, ...this.rows.map((r) => r + 1));
    this.truth = truth;
    this.draw();
  }

  attachPlayer(p: Player | null) {
    this.player?.dispose();
    this.player = p;
    if (p) p.onchange = () => { this.draw(); if (p.playing) this.loop(); };
    this.draw();
  }

  private loop() {
    cancelAnimationFrame(this.raf);
    const tick = () => {
      if (!this.player?.playing) return this.draw();
      const t = this.player.position();
      const span = this.view.t1 - this.view.t0;
      if (t > this.view.t1 || t < this.view.t0) this.view.set(t - span * 0.1, t + span * 0.9); // follow the cursor
      else this.draw();
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  private x(t: number, w: number) { return ((t - this.view.t0) / (this.view.t1 - this.view.t0)) * w; }
  private t(x: number, w: number) { return this.view.t0 + (x / w) * (this.view.t1 - this.view.t0); }

  draw() {
    const c = this.canvas, dpr = devicePixelRatio || 1;
    const w = c.parentElement?.clientWidth || 600, h = this.height;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.width = `${w}px`; c.style.height = `${h}px`;
    }
    const g = c.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const ink = css("--ink"), surface = css("--surface"), ink3 = css("--ink-3"), line = css("--line"), wave = css("--wave");
    const cls = (k: string) => css(`--c-${k}`);
    const font = "12px 'IBM Plex Sans', system-ui, sans-serif";

    const cf = conformalOn();
    // prediction boxes behind the waveform, one band per packed row
    const rowH = WAVE_H / this.nRows;
    const labels: (() => void)[] = []; // drawn over the waveform
    this.segs.forEach((s, i) => {
      const x0 = this.x(s.start, w), x1 = this.x(s.end, w);
      if (x1 < 0 || x0 > w) return;
      const y = this.rows[i] * rowH, col = cls(s.pred4);
      g.globalAlpha = i === this.hover ? 0.3 : 0.17;
      g.fillStyle = col; g.fillRect(x0, y, x1 - x0, rowH);
      g.globalAlpha = 1;
      g.fillRect(x0, y, x1 - x0, 3);
      if (i === this.selected) { g.strokeStyle = ink; g.lineWidth = 2; g.strokeRect(x0 + 1, y + 1, x1 - x0 - 2, rowH - 2); }
      const p = `${Math.round(100 * s.probs4[s.pred4])}%`;
      const txt = x1 - x0 > 120 ? `${s.pred4} ${p}` + (cf && s.set4.length !== 1 ? ` · set ${s.set4.length}` : "")
        : x1 - x0 > 40 ? `${s.pred4[0].toUpperCase()} ${p}` : "";
      if (txt) labels.push(() => {
        g.font = font; g.textBaseline = "top";
        g.save(); g.beginPath(); g.rect(x0, y, x1 - x0, rowH); g.clip();
        g.fillStyle = surface; g.globalAlpha = 0.8; g.fillRect(x0 + 3, y + 5, g.measureText(txt).width + 5, 16); g.globalAlpha = 1;
        g.fillStyle = ink; g.fillText(txt, x0 + 5, y + 7); g.restore();
      });
    });

    // waveform
    g.strokeStyle = line; g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, WAVE_H / 2 + 0.5); g.lineTo(w, WAVE_H / 2 + 0.5); g.stroke();
    if (this.samples) {
      const scale = (this.gain ?? 1 / this.peak) * (WAVE_H / 2 - 6);
      const live = this.liveLength > 0;
      const n = live ? this.liveLength : this.samples.length;
      const spp = ((this.view.t1 - this.view.t0) * this.sr) / w;
      g.fillStyle = wave;
      for (let px = 0; px < w; px++) {
        const a = Math.floor((this.view.t0 * this.sr) + px * spp), b = Math.min(Math.floor(a + Math.max(spp, 1)), n);
        if (a >= n) break;
        let lo = 1, hi = -1;
        if (spp > BLOCK && !live && this.mins.length) {
          for (let k = Math.floor(a / BLOCK), e = Math.ceil(b / BLOCK); k < e; k++) { lo = Math.min(lo, this.mins[k]); hi = Math.max(hi, this.maxs[k]); }
        } else {
          for (let i = Math.max(a, 0); i < b; i++) { const v = this.samples[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
        }
        if (hi < lo) continue;
        const y0 = WAVE_H / 2 - Math.min(hi * scale, WAVE_H / 2), y1 = WAVE_H / 2 - Math.max(lo * scale, -WAVE_H / 2);
        g.fillRect(px, y0, 1, Math.max(y1 - y0, 1));
      }
    }

    labels.forEach((f) => f());

    // 2-class lane: solid = singleton conformal set, hatched = both classes, outline = empty set; without conformal, the top class
    let y = WAVE_H + GAP;
    const laneH = LANE_H / this.nRows; // overlapping segments split the lane like the overlay rows
    this.segs.forEach((s, i) => {
      const x0 = this.x(s.start, w), x1 = this.x(s.end, w);
      if (x1 < 0 || x0 > w) return;
      const y = WAVE_H + GAP + this.rows[i] * laneH;
      if (!cf) { g.fillStyle = css(`--c2-${s.pred2}`); g.fillRect(x0 + 0.5, y, x1 - x0 - 1, laneH); }
      else if (s.set2.length === 1) { g.fillStyle = css(`--c2-${s.set2[0]}`); g.fillRect(x0 + 0.5, y, x1 - x0 - 1, laneH); }
      else if (s.set2.length === 2) { g.fillStyle = hatch(g, css("--c2-normal"), css("--c2-abnormal")); g.fillRect(x0 + 0.5, y, x1 - x0 - 1, laneH); }
      else { g.strokeStyle = ink3; g.lineWidth = 1; g.strokeRect(x0 + 1, y + 0.5, x1 - x0 - 2, laneH - 1); }
    });
    if (this.truth.length) {
      y += LANE_H + GAP;
      this.truth.forEach((s) => {
        const x0 = this.x(s.start, w), x1 = this.x(s.end, w);
        g.fillStyle = cls(s.label); g.fillRect(x0 + 0.5, y + 3, x1 - x0 - 1, TRUTH_H - 6);
      });
      y += TRUTH_H;
    } else y += LANE_H;

    // axis
    y += 2;
    const step = niceStep(this.view.t1 - this.view.t0, w);
    g.font = "11px 'IBM Plex Mono', ui-monospace, monospace"; g.fillStyle = ink3; g.strokeStyle = line; g.textBaseline = "top";
    for (let t = Math.ceil(this.view.t0 / step) * step; t <= this.view.t1 + 1e-9; t += step) {
      const x = Math.round(this.x(t, w)) + 0.5;
      g.beginPath(); g.moveTo(x, y); g.lineTo(x, y + 4); g.stroke();
      const label = `${step < 1 ? t.toFixed(1) : t.toFixed(0)} s`;
      g.fillText(label, Math.min(Math.max(x - g.measureText(label).width / 2, 0), w - g.measureText(label).width), y + 6);
    }

    // drag selection and cursor
    if (this.drag) {
      g.fillStyle = ink; g.globalAlpha = 0.08;
      g.fillRect(Math.min(this.drag.x0, this.drag.x1), 0, Math.abs(this.drag.x1 - this.drag.x0), WAVE_H);
      g.globalAlpha = 1;
    }
    const pos = this.player?.position();
    if (pos !== undefined && (this.player!.playing || pos > 0)) {
      const x = Math.round(this.x(pos, w)) + 0.5;
      g.strokeStyle = css("--cursor"); g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, WAVE_H + GAP + LANE_H); g.stroke();
    }
  }

  private hit(px: number, py: number): number {
    const w = this.canvas.clientWidth, t = this.t(px, w);
    if (py < WAVE_H) {
      const row = Math.floor(py / (WAVE_H / this.nRows));
      return this.segs.findIndex((s, i) => this.rows[i] === row && s.start <= t && t <= s.end);
    }
    if (py < WAVE_H + GAP + LANE_H) return this.segs.findIndex((s) => s.start <= t && t <= s.end);
    return -1;
  }

  private bind() {
    const c = this.canvas, tip = document.getElementById("tooltip")!;
    const pt = (e: MouseEvent) => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    let down: [number, number] | null = null;
    c.addEventListener("pointerdown", (e) => { down = pt(e) as [number, number]; c.setPointerCapture(e.pointerId); });
    c.addEventListener("pointermove", (e) => {
      const [x, yy] = pt(e);
      if (down && Math.abs(x - down[0]) > 4) { this.drag = { x0: down[0], x1: x }; tip.hidden = true; this.draw(); return; }
      const i = this.hit(x, yy);
      if (i !== this.hover) { this.hover = i; this.draw(); }
      if (i >= 0) {
        tip.innerHTML = this.opts.describe(this.segs[i]); tip.hidden = false;
        const tw = tip.offsetWidth;
        tip.style.left = `${Math.min(e.clientX + 14, innerWidth - tw - 8)}px`; tip.style.top = `${e.clientY + 14}px`;
      } else tip.hidden = true;
    });
    c.addEventListener("pointerleave", () => { tip.hidden = true; if (this.hover !== -1) { this.hover = -1; this.draw(); } });
    c.addEventListener("pointerup", (e) => {
      const [x, yy] = pt(e), w = c.clientWidth;
      if (this.drag) {
        const a = this.t(Math.min(this.drag.x0, this.drag.x1), w), b = this.t(Math.max(this.drag.x0, this.drag.x1), w);
        this.drag = null; this.view.set(a, b);
      } else if (down) {
        const i = this.hit(x, yy);
        if (i >= 0) { this.selected = i; this.opts.onSelect?.(i); }
        this.player?.seek(this.t(x, w));
        this.draw();
      }
      down = null;
    });
    c.addEventListener("dblclick", () => this.view.fit());
    c.addEventListener("wheel", (e) => {
      const w = c.clientWidth, span = this.view.t1 - this.view.t0;
      if (e.ctrlKey || e.metaKey) { e.preventDefault(); this.view.zoom(Math.exp(e.deltaY * 0.01), this.t(pt(e)[0], w)); }
      else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) { e.preventDefault(); this.view.pan(((e.shiftKey ? e.deltaY : e.deltaX) / w) * span); }
    }, { passive: false });
    c.addEventListener("keydown", (e) => {
      const span = this.view.t1 - this.view.t0, p = this.player;
      const keys: Record<string, () => void> = {
        " ": () => p?.toggle(),
        ArrowLeft: () => p?.seek(p.position() - (e.shiftKey ? 5 : 1)),
        ArrowRight: () => p?.seek(p.position() + (e.shiftKey ? 5 : 1)),
        "+": () => this.view.zoom(0.5), "=": () => this.view.zoom(0.5), "-": () => this.view.zoom(2), "0": () => this.view.fit(),
        "[": () => this.view.pan(-span / 4), "]": () => this.view.pan(span / 4),
      };
      if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
    });
  }
}

function hatch(g: CanvasRenderingContext2D, a: string, b: string) {
  const p = document.createElement("canvas");
  p.width = p.height = 8;
  const q = p.getContext("2d")!;
  q.fillStyle = a; q.fillRect(0, 0, 8, 8);
  q.strokeStyle = b; q.lineWidth = 3;
  q.beginPath(); q.moveTo(-2, 10); q.lineTo(10, -2); q.moveTo(6, 10); q.lineTo(10, 6); q.moveTo(-2, 2); q.lineTo(2, -2); q.stroke();
  return g.createPattern(p, "repeat")!;
}
