// Optional clean-up applied in the browser before upload: spectral noise reduction, then trimming of leading and trailing silence.

const N = 1024, H = N / 2, BINS = N / 2 + 1;
const WIN = Float32Array.from({ length: N }, (_, i) => Math.sin((Math.PI * (i + 0.5)) / N)); // sine window: squares sum to 1 at 50 % overlap
const COS = Float32Array.from({ length: H }, (_, i) => Math.cos((2 * Math.PI * i) / N));
const SIN = Float32Array.from({ length: H }, (_, i) => Math.sin((2 * Math.PI * i) / N));

/** In-place radix-2 FFT of size N; `inv` conjugates the twiddles (unscaled). */
function fft(re: Float32Array, im: Float32Array, inv: boolean) {
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= N; len <<= 1) {
    const step = N / len;
    for (let i = 0; i < N; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = COS[k * step], wi = inv ? SIN[k * step] : -SIN[k * step];
        const a = i + k, b = a + len / 2;
        const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }
}

/**
 * Spectral subtraction. The noise floor of every frequency bin is its 15th-percentile magnitude over the whole take, so
 * breathing (present in only part of the frames) is not mistaken for noise. A sound that fills the whole take would be. Each bin is scaled by max(FLOOR, 1 - OVER * noise / |X|).
 */
const OVER = 1.5, FLOOR = 0.2, QUANTILE = 0.15;

export function denoise(x: Float32Array): Float32Array {
  const frames = Math.ceil(x.length / H) + 1;
  const pad = new Float32Array((frames + 1) * H); // H of silence in front so every sample is covered twice
  pad.set(x, H);
  const re = new Float32Array(N), im = new Float32Array(N);
  const mag = new Float32Array(frames * BINS), phase = new Float32Array(frames * BINS);
  for (let f = 0; f < frames; f++) {
    for (let i = 0; i < N; i++) { re[i] = pad[f * H + i] * WIN[i]; im[i] = 0; }
    fft(re, im, false);
    for (let b = 0; b < BINS; b++) { mag[f * BINS + b] = Math.hypot(re[b], im[b]); phase[f * BINS + b] = Math.atan2(im[b], re[b]); }
  }
  const noise = new Float32Array(BINS), col = new Float32Array(frames);
  for (let b = 0; b < BINS; b++) {
    for (let f = 0; f < frames; f++) col[f] = mag[f * BINS + b];
    noise[b] = col.sort()[Math.floor(QUANTILE * (frames - 1))];
  }
  const out = new Float32Array(pad.length);
  for (let f = 0; f < frames; f++) {
    for (let b = 0; b < BINS; b++) {
      const m = mag[f * BINS + b], g = m > 0 ? Math.max(FLOOR, 1 - (OVER * noise[b]) / m) : FLOOR;
      re[b] = g * m * Math.cos(phase[f * BINS + b]); im[b] = g * m * Math.sin(phase[f * BINS + b]);
      if (b > 0 && b < N / 2) { re[N - b] = re[b]; im[N - b] = -im[b]; }
    }
    fft(re, im, true);
    for (let i = 0; i < N; i++) out[f * H + i] += (re[i] / N) * WIN[i];
  }
  return out.slice(H, H + x.length);
}

/**
 * Cuts leading and trailing silence only (pauses between breaths carry information and stay). A 20 ms frame is "active" when it
 * is louder than the quiet floor (10th percentile) by 6 dB or a quarter of the floor-to-loud range; 0.3 s of margin stays on each side.
 */
const FRAME_SEC = 0.02, MARGIN_SEC = 0.3;

export function trimSilence(x: Float32Array, sr: number, minSec: number): { x: Float32Array; note: string } {
  const fl = Math.round(FRAME_SEC * sr), n = Math.floor(x.length / fl);
  const db = Float32Array.from({ length: n }, (_, k) => {
    let s = 0;
    for (let i = k * fl; i < (k + 1) * fl; i++) s += x[i] * x[i];
    return 10 * Math.log10(s / fl + 1e-12);
  });
  if (n < 10) return { x, note: "too short to trim" };
  const sorted = Float32Array.from(db).sort(), floor = sorted[Math.floor(0.1 * (n - 1))], loud = sorted[Math.floor(0.95 * (n - 1))];
  const thr = floor + Math.max(6, 0.25 * (loud - floor));
  const first = db.findIndex((v) => v > thr), last = n - 1 - [...db].reverse().findIndex((v) => v > thr);
  if (first < 0) return { x, note: "no silence found to trim" };
  let a = Math.max(0, Math.round((first * fl) - MARGIN_SEC * sr)), b = Math.min(x.length, Math.round(((last + 1) * fl) + MARGIN_SEC * sr));
  const need = Math.ceil(minSec * sr);
  if (b - a < need) { // keep the shortest length the API accepts, centred on the sound
    const mid = (a + b) >> 1;
    a = Math.max(0, Math.min(mid - (need >> 1), x.length - need)); b = Math.min(x.length, a + need);
  }
  const cut = (x.length - (b - a)) / sr;
  return cut < 0.1 ? { x, note: "no silence to trim" } : { x: x.slice(a, b), note: `trimmed ${cut.toFixed(1)} s of silence` };
}
