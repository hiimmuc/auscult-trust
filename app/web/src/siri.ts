// Live recording visual: a few sine curves, pinched to a point at both ends, whose height follows the microphone level.

import { dB } from "./mic";
import { css } from "./wave";

// cycles across the width, drift speed (rad/s), share of the full height, line width
const CURVES = [[1.1, 1.4, 0.35, 1], [1.7, -1.9, 0.55, 1.2], [2.4, 2.4, 0.75, 1.5], [1.4, -1.1, 1, 2.2]];
const still = matchMedia("(prefers-reduced-motion: reduce)").matches;

export class Siri {
  private canvas = document.createElement("canvas");
  private level = 0;
  private target = 0;
  private raf = 0;

  constructor(host: HTMLElement, private height = 220) {
    this.canvas.className = "siri-canvas";
    this.canvas.setAttribute("aria-hidden", "true");
    host.append(this.canvas);
  }

  /** Feed the microphone RMS (linear); −65 dB maps to a flat line, −25 dB to full height (lung sounds are quiet), eased so soft breaths still move. */
  push(rms: number) { this.target = Math.max(0, Math.min(1, (dB(rms) + 65) / 40)) ** 0.7; }

  start() {
    this.level = this.target = 0;
    const tick = (ms: number) => {
      this.level += (this.target - this.level) * (this.target > this.level ? 0.35 : 0.07); // quick attack, slow release
      this.draw(still ? 0 : ms / 1000);
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  stop() { cancelAnimationFrame(this.raf); }

  private draw(t: number) {
    const c = this.canvas, dpr = devicePixelRatio || 1, w = c.parentElement?.clientWidth || 600, h = this.height;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.width = `${w}px`; c.style.height = `${h}px`;
    }
    const g = c.getContext("2d")!, ink = css("--ink"), mid = h / 2, amp = 2 + this.level * (mid - 8);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    g.strokeStyle = css("--line"); g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, mid + 0.5); g.lineTo(w, mid + 0.5); g.stroke();
    g.strokeStyle = g.fillStyle = ink;
    CURVES.forEach(([cycles, speed, share, lw], k) => {
      g.beginPath();
      for (let x = 0; x <= w; x += 3) {
        const u = x / w, pinch = (4 * u * (1 - u)) ** 2;
        const y = mid + Math.sin(u * 2 * Math.PI * cycles + t * speed) * amp * share * pinch;
        x ? g.lineTo(x, y) : g.moveTo(x, y);
      }
      g.globalAlpha = 0.18 + 0.82 * (k / (CURVES.length - 1)) ** 2;
      g.lineWidth = lw; g.stroke();
      if (k === CURVES.length - 1) { g.globalAlpha = 0.07; g.fill(); }
    });
    g.globalAlpha = 1;
  }
}
