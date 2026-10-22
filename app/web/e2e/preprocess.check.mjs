// Self-check of the browser preprocessing on synthetic audio (Node strips the TypeScript types itself):
//   node e2e/preprocess.check.mjs   (Node 23 or newer)
import assert from "node:assert/strict";
import { denoise, trimSilence } from "../src/preprocess.ts";

const sr = 16000, rng = (() => { let s = 1; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32) - 0.5; })();
const rms = (x) => Math.sqrt(x.reduce((t, v) => t + v * v, 0) / x.length);
const tone = (n, a, f = 220) => Float32Array.from({ length: n }, (_, i) => a * Math.sin((2 * Math.PI * f * i) / sr));

// noise reduction: breaths (a tone 1 s on, 1 s off) buried in white noise come out cleaner and keep their level
const clean = tone(8 * sr, 0.2).map((v, i) => (Math.floor(i / sr) % 2 ? 0 : v)), noisy = clean.map((v) => v + 0.1 * rng());
const out = denoise(noisy);
assert.equal(out.length, noisy.length);
const snr = (y) => 20 * Math.log10(rms(clean) / rms(y.map((v, i) => v - clean[i])));
assert.ok(snr(out) > snr(noisy) + 4, `SNR ${snr(noisy).toFixed(1)} -> ${snr(out).toFixed(1)} dB`);
assert.ok(Math.abs(20 * Math.log10(rms(out) / rms(clean))) < 2, "level preserved");

// no-op input (silence) stays silent and finite
assert.ok(denoise(new Float32Array(5000)).every((v) => v === 0));

// trimming: 2 s of room noise, 3 s of sound, 2 s of room noise -> about 3 s + 2 x 0.3 s margin
const room = (n) => Float32Array.from({ length: n }, () => 0.002 * rng());
const take = Float32Array.from([...room(2 * sr), ...tone(3 * sr, 0.2).map((v) => v + 0.002 * rng()), ...room(2 * sr)]);
const t = trimSilence(take, sr, 1);
assert.ok(Math.abs(t.x.length / sr - 3.6) < 0.1, `kept ${t.x.length / sr} s`);
// steady noise has nothing to trim and a quiet take is never cut below the minimum
assert.equal(trimSilence(room(5 * sr), sr, 1).x.length, 5 * sr);
assert.ok(trimSilence(Float32Array.from([...room(2 * sr), ...tone(sr / 4, 0.2), ...room(2 * sr)]), sr, 1).x.length >= sr);
console.log("preprocess OK");
