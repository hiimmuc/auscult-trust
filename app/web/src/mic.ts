// Microphone plumbing shared by Real and Demo mode: device list, level meter, permission errors, raw capture through the worklet.

import { esc } from "./ui";

export const MIC_HTML = `
  <label class="field">Microphone
    <select class="mic-select"><option value="">Default microphone</option></select>
  </label>
  <div class="meter-row"><span class="meter" aria-hidden="true"><span></span></span><span class="meter-db num muted">— dB</span>
    <button type="button" class="btn small ghost mic-test">Test microphone</button></div>`;

export const dB = (v: number) => 20 * Math.log10(Math.max(v, 1e-5));

export function micAlert(e: any) {
  const n = e?.name;
  const text = !window.isSecureContext || !navigator.mediaDevices ? "The microphone needs a secure page (https:// or localhost)."
    : n === "NotAllowedError" ? "Microphone access was blocked. Allow it in the browser's site settings, then try again."
    : n === "NotFoundError" || n === "OverconstrainedError" ? "No microphone found. Plug one in or choose another, then try again."
    : n === "NotReadableError" ? "The microphone is in use by another application."
    : `Microphone error: ${e?.message ?? e}`;
  return `<div class="alert error" role="alert">${esc(text)}</div>`;
}

/** Wires the `MIC_HTML` block inside `root`. `open` asks for the raw microphone; `level` drives the meter (and `extra`) from a stream. */
export function micPanel(root: HTMLElement, ctx: () => AudioContext, msg: HTMLElement) {
  const q = <T extends HTMLElement>(s: string) => root.querySelector<T>(s)!;
  const select = q<HTMLSelectElement>(".mic-select"), bar = q(".meter span"), db = q(".meter-db"), test = q<HTMLButtonElement>(".mic-test");

  async function open() {
    const id = select.value;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: {
      ...(id ? { deviceId: { exact: id } } : {}), channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    if (select.options.length <= 1) { // labels are only readable after the first permission
      const mics = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput" && d.deviceId);
      select.innerHTML = `<option value="">Default microphone</option>` + mics.map((d, i) => `<option value="${esc(d.deviceId)}">${esc(d.label || `Microphone ${i + 1}`)}</option>`).join("");
      select.value = id;
    }
    return stream;
  }

  function level(stream: MediaStream, extra?: (rms: number) => void) {
    const c = ctx(), an = c.createAnalyser(), src = c.createMediaStreamSource(stream), buf = new Float32Array(2048);
    an.fftSize = buf.length;
    src.connect(an);
    let raf = 0;
    const tick = () => {
      an.getFloatTimeDomainData(buf);
      let pk = 0, sq = 0;
      for (const v of buf) { pk = Math.max(pk, Math.abs(v)); sq += v * v; }
      const d = dB(pk);
      bar.style.width = `${Math.max(0, Math.min(100, (d + 60) / 60 * 100))}%`;
      bar.classList.toggle("hot", d > -1);
      db.textContent = `${d.toFixed(0)} dB`;
      extra?.(Math.sqrt(sq / buf.length));
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => { cancelAnimationFrame(raf); src.disconnect(); bar.style.width = "0"; db.textContent = "— dB"; };
  }

  let stopTest: (() => void) | null = null;
  const endTest = () => { stopTest?.(); stopTest = null; test.textContent = "Test microphone"; };
  test.onclick = async () => {
    if (stopTest) return endTest();
    try {
      const s = await open(), stop = level(s);
      stopTest = () => { stop(); s.getTracks().forEach((t) => t.stop()); };
      test.textContent = "Stop test";
    } catch (e) { msg.innerHTML = micAlert(e); }
  };
  return { select, open, level, endTest };
}

let worklet: Promise<void> | null = null;

/** Streams raw 128-frame blocks, each with its audio-context frame index. Returns the function that stops the capture. */
export async function capture(c: AudioContext, stream: MediaStream, onBlock: (frame: number, data: Float32Array) => void) {
  worklet ??= c.audioWorklet.addModule(`${import.meta.env.BASE_URL}recorder-worklet.js`);
  try { await worklet; } catch (e: any) { worklet = null; throw new Error(`Audio worklet failed to load: ${e?.message ?? e}`); }
  const node = new AudioWorkletNode(c, "recorder"), src = c.createMediaStreamSource(stream), mute = c.createGain();
  mute.gain.value = 0; // the worklet needs a path to the destination to run; nothing is heard
  node.port.onmessage = (e) => onBlock(e.data.frame, e.data.data);
  src.connect(node); node.connect(mute); mute.connect(c.destination);
  node.port.postMessage("start");
  return () => { node.port.postMessage("stop"); src.disconnect(); node.disconnect(); mute.disconnect(); };
}
