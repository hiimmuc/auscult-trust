// Demo mode: play an ICBHI reference through the speaker, record it through the microphone on the same audio clock,
// then compare the model's predictions on both takes cycle by cycle.

import { api, type Class4, type InferResult, type ModelInfo, type Sample } from "./api";
import { capture, MIC_HTML, micAlert, micPanel } from "./mic";
import { onConformal } from "./prefs";
import { attachAudio, CLASS_LABEL, comparison, conformalToggle, describe, errorBox, esc, legend, panel, pipelineSwitch, pipelineTag, truthOf } from "./ui";
import { encodeWav } from "./wav";
import { TimeView } from "./wave";

const LEAD = 0.4; // s between arming the recorder and starting playback
const TAIL = 1.2; // s recorded after the reference ends (speaker-to-mic latency, room decay)

type Stage = "idle" | "loading" | "ready" | "permission" | "recording" | "analysing" | "done" | "error";
const STEPS: [Stage, string][] = [["permission", "Microphone access"], ["recording", "Play and record"], ["analysing", "Align and classify"], ["done", "Compare"]];

export function mountDemo(root: HTMLElement, m: ModelInfo, ctx: () => AudioContext) {
  const names = Object.keys(m.pipelines);
  let shown = names[0];
  let samples: Sample[] = [];
  let sample: Sample | null = null;
  let refAudio: AudioBuffer | null = null;
  let refResult: InferResult | null = null;
  let micResult: InferResult | null = null;
  let stage: Stage = "idle";
  let abort: (() => void) | null = null;
  const view = new TimeView();

  root.innerHTML = `
    <div class="demo-grid">
        <header class="title">
          <h1>Speaker → microphone</h1>
          <p class="muted">The reference plays through your speaker while the microphone records it. Both takes go through the same model, so you see what the replay chain changes.</p>
        </header>
      <aside class="side">
        <label class="field">Sound class
          <select id="demo-class">${m.classes.map((c) => `<option value="${c}">${CLASS_LABEL[c]}</option>`).join("")}</select>
        </label>
        <fieldset class="samples"><legend>Recording</legend><div class="sample-list" role="radiogroup"></div></fieldset>
        ${MIC_HTML}
        <div class="row">
          <button class="btn primary" id="demo-start" disabled>Start demo</button>
          <button class="btn ghost" id="demo-cancel" hidden>Cancel</button>
        </div>
        <ol class="steps"></ol>
        <div class="msg" aria-live="polite"></div>
        <details class="advanced"><summary>Room tips</summary>
          <ul class="tips"><li>Use loudspeakers, not headphones: the microphone must hear the sound.</li>
          <li>Set the volume so the meter peaks around −20 dB, without touching 0 dB.</li>
          <li>Keep the room quiet; speech and fans get recorded too.</li>
          <li>Browser echo cancellation is switched off for this take. The page never plays the microphone back, so there is no feedback loop.</li></ul>
        </details>
      </aside>
      <div class="main-col">
        <div class="toolbar"><label class="check"><input type="checkbox" id="demo-scale" /> Same amplitude scale</label></div>
        <div class="panels"></div>
        <div class="legend-host"></div>
        <div class="compare"></div>
      </div>
    </div>`;
  const $ = <T extends HTMLElement>(s: string) => root.querySelector<T>(s)!;
  const msg = $(".msg"), start = $<HTMLButtonElement>("#demo-start"), cancel = $<HTMLButtonElement>("#demo-cancel");
  const mic = micPanel(root, ctx, msg);
  const desc = (who: string) => (s: any) => describe(s, `${who} · ${m.pipelines[shown].label}`, m);
  const refP = panel($(".panels"), view, "Reference", "Choose a recording", "ref", desc("Reference"), (i) => select(i));
  const micP = panel($(".panels"), view, "Microphone take", "Not recorded yet", "mic", desc("Microphone"), (i) => select(i));
  pipelineSwitch($(".toolbar"), m, shown, (v) => { shown = v; paint(); });
  conformalToggle($(".toolbar"));
  onConformal(() => paint());
  $<HTMLInputElement>("#demo-scale").onchange = () => scale();

  let reached = -1; // last step index reached, kept when a step fails
  function setStage(s: Stage) {
    stage = s;
    const at = s === "error" ? reached : STEPS.findIndex((x) => x[0] === s);
    reached = at;
    $(".steps").innerHTML = s === "idle" || s === "loading" || s === "ready" ? "" : STEPS.map(([, t], i) =>
      `<li class="${i < at || s === "done" ? "ok" : i === at ? (s === "error" ? "err" : "now") : ""}">${t}</li>`).join("");
    start.disabled = !(s === "ready" || s === "done" || s === "error") || !refAudio;
    start.textContent = s === "done" || s === "error" ? "Run again" : "Start demo";
    cancel.hidden = !(s === "permission" || s === "recording" || s === "analysing");
    $<HTMLSelectElement>("#demo-class").disabled = mic.select.disabled = !cancel.hidden;
    root.querySelectorAll<HTMLInputElement>(".sample-list input").forEach((i) => (i.disabled = !cancel.hidden));
    micP.root.classList.toggle("recording", s === "recording");
  }

  function select(i: number) {
    refP.wave.selected = micP.wave.selected = i;
    refP.wave.draw(); micP.wave.draw();
    root.querySelectorAll(".compare tr.sel").forEach((r) => r.classList.remove("sel"));
    root.querySelector(`.compare tr[data-i="${i}"]`)?.classList.add("sel");
  }

  function scale() {
    const same = $<HTMLInputElement>("#demo-scale").checked;
    const g = same ? 1 / Math.max(refP.wave.peakAmplitude, micP.wave.peakAmplitude) : null;
    refP.wave.gain = micP.wave.gain = g;
    refP.wave.draw(); micP.wave.draw();
  }

  function paint() {
    if (refResult) {
      const s = refResult.pipelines[shown].segments;
      refP.wave.setSegments(s, truthOf(s));
      $(".legend-host").innerHTML = legend(m, true);
    }
    micP.wave.setSegments(micResult ? micResult.pipelines[shown].segments : [], micResult ? truthOf(micResult.pipelines[shown].segments) : []);
    $(".compare").innerHTML = refResult && micResult ? `
      <h3 class="sec-title">Reference vs microphone · ${esc(m.pipelines[shown].label)} ${pipelineTag(m, shown)}</h3>
      ${alignNote()}
      ${comparison(refResult, micResult, m, shown)}
      <p class="muted">Other pipeline: ${names.filter((n) => n !== shown).map((n) => {
        const a = refResult!.pipelines[n].segments, b = micResult!.pipelines[n].segments;
        return `${esc(m.pipelines[n].label)} changed the top class on ${a.filter((s, i) => s.pred4 !== b[i].pred4).length} of ${a.length} cycles`;
      }).join("; ")}. Switch above to see it.</p>` : "";
    root.querySelectorAll<HTMLTableRowElement>(".compare tr[data-i]").forEach((tr) => (tr.onclick = () => {
      const i = +tr.dataset.i!; select(i);
      const s = refResult!.pipelines[shown].segments[i];
      if (s.end < view.t0 || s.start > view.t1) view.set(s.start - 1, s.end + 1);
      refP.wave.player?.seek(s.start);
    }));
  }

  function alignNote() {
    const a = micResult?.alignment;
    if (!a) return "";
    const ms = Math.round(a.lag_sec * 1000);
    return a.reliable
      ? `<p class="muted">Microphone take shifted by ${ms} ms to line up with the reference (speaker and microphone latency). Cycle boundaries are the reference annotations.</p>`
      : `<div class="alert warn" role="status">Alignment is unreliable (no clear match between the takes, peak score ${a.peak_score}). The microphone may have heard mostly noise; cycle-by-cycle comparison may be off.</div>`;
  }

  function renderSamples() {
    const cls = $<HTMLSelectElement>("#demo-class").value as Class4;
    const list = samples.filter((s) => s.class === cls);
    $(".sample-list").innerHTML = list.map((s, i) => `
      <label class="sample"><input type="radio" name="demo-sample" value="${esc(s.id)}" ${i === 0 ? "checked" : ""} />
        <span class="s-main">${esc(s.device)} · ${esc(s.chest_location)} <span class="muted">patient ${esc(s.patient.replace("icbhi_", ""))}</span></span>
        <span class="s-sub">${s.class_cycles} of ${s.cycles_total} cycles annotated ${CLASS_LABEL[cls].toLowerCase()}</span>
      </label>`).join("") || `<p class="muted">No sample for this class.</p>`;
    root.querySelectorAll<HTMLInputElement>(".sample-list input").forEach((r) => (r.onchange = () => load(r.value)));
    if (list[0]) load(list[0].id);
  }

  let loadSeq = 0;
  async function load(id: string) {
    const seq = ++loadSeq;
    sample = samples.find((s) => s.id === id)!;
    refAudio = null; refResult = micResult = null;
    micP.wave.attachPlayer(null); micP.wave.setLive(new Float32Array(0), 0, 16000); micP.play.disabled = true;
    micP.root.querySelector(".sub")!.textContent = "Not recorded yet";
    setStage("loading"); paint();
    refP.root.querySelector(".sub")!.textContent = `${sample.id} · ${sample.device} · ${sample.cycles.length} cycles`;
    refP.status.textContent = "Loading";
    msg.innerHTML = "";
    try {
      const [buf, res] = await Promise.all([api.sampleAudio(id), api.sampleResult(id)]);
      if (seq !== loadSeq) return;
      refAudio = await ctx().decodeAudioData(buf);
      refResult = res;
      view.setDuration(refAudio.duration);
      attachAudio(refP, ctx(), refAudio);
      paint(); scale();
      setStage("ready");
    } catch (e) {
      if (seq !== loadSeq) return;
      msg.innerHTML = errorBox(e); setStage("error");
    } finally {
      if (seq === loadSeq) refP.status.textContent = "";
    }
  }

  async function runDemo() {
    if (!refAudio || !sample) return;
    mic.endTest();
    micResult = null; paint(); msg.innerHTML = "";
    const c = ctx();
    if (c.state === "suspended") await c.resume();
    refP.wave.player?.pause(); micP.wave.attachPlayer(null);
    let cancelled = false;
    const cleanup: (() => void)[] = [];
    abort = () => { cancelled = true; cleanup.forEach((f) => f()); };

    setStage("permission");
    let stream: MediaStream;
    try { stream = await mic.open(); } catch (e) { msg.innerHTML = micAlert(e); return setStage("error"); }
    cleanup.push(() => stream.getTracks().forEach((t) => t.stop()));
    if (cancelled) return;
    const warn: string[] = [];
    const st = stream.getAudioTracks()[0].getSettings();
    if (st.echoCancellation) warn.push("The browser kept echo cancellation on; it may erase part of the played sound.");

    // capture is indexed from the playback start, on the audio context clock
    const sr = c.sampleRate, dur = refAudio.duration;
    const cap = new Float32Array(Math.ceil((dur + TAIL) * sr));
    let filled = 0;
    let startFrame = Infinity; // set once the recorder runs, so a slow first worklet load cannot eat into the lead-in
    try {
      const stopCap = await capture(c, stream, (frame, data) => {
        const off = frame - startFrame;
        for (let i = 0; i < data.length; i++) { const k = off + i; if (k >= 0 && k < cap.length) cap[k] = data[i]; }
        filled = Math.max(filled, Math.min(off + data.length, cap.length));
      });
      cleanup.push(stopCap);
    } catch (e) { msg.innerHTML = errorBox(e); cleanup.forEach((f) => f()); return setStage("error"); }
    cleanup.push(mic.level(stream));
    const startAt = c.currentTime + LEAD;
    startFrame = Math.round(startAt * sr);

    const play = c.createBufferSource();
    play.buffer = refAudio; play.connect(c.destination); play.start(startAt);
    cleanup.push(() => { try { play.stop(); } catch { /* not started */ } });
    setStage("recording");
    micP.root.querySelector(".sub")!.textContent = "Recording…";
    micP.status.innerHTML = `<span class="rec-dot" aria-hidden="true"></span>Recording`;

    const done = await new Promise<boolean>((resolve) => {
      const tick = () => {
        if (cancelled) return resolve(false);
        const t = c.currentTime - startAt;
        micP.wave.setLive(cap, Math.max(filled, 0), sr);
        refP.status.textContent = t < 0 ? "Starting" : t < dur ? `Playing ${t.toFixed(1)} / ${dur.toFixed(1)} s` : "Recording the tail";
        if (t >= dur + TAIL) return resolve(true);
        requestAnimationFrame(tick);
      };
      tick();
    });
    cleanup.forEach((f) => f());
    refP.status.textContent = ""; micP.status.textContent = "";
    if (!done) { micP.root.querySelector(".sub")!.textContent = "Cancelled"; msg.innerHTML = `<p class="muted">Cancelled. Nothing was sent.</p>`; return setStage("ready"); }

    let pk = 0, clip = 0;
    for (let i = 0; i < filled; i++) { const v = Math.abs(cap[i]); if (v > pk) pk = v; if (v > 0.999) clip++; }
    const pkDb = 20 * Math.log10(Math.max(pk, 1e-6));
    if (pkDb < -50) warn.push(`The microphone barely heard the speaker (peak ${pkDb.toFixed(0)} dBFS). Raise the volume or move the microphone closer.`);
    if (clip / Math.max(filled, 1) > 0.001) warn.push("The microphone clipped. Lower the speaker volume.");
    micP.root.querySelector(".sub")!.textContent = `${(filled / sr).toFixed(1)} s at ${sr} Hz · peak ${pkDb.toFixed(0)} dBFS`;

    setStage("analysing");
    micP.status.textContent = "Aligning and classifying";
    const ctl = new AbortController();
    abort = () => ctl.abort();
    try {
      micResult = await api.infer({ file: encodeWav(cap.subarray(0, filled), sr), reference_sample: sample.id }, ctl.signal);
    } catch (e: any) {
      micP.status.textContent = "";
      msg.innerHTML = e.code === "cancelled" ? `<p class="muted">Cancelled.</p>` : errorBox(e);
      return setStage(e.code === "cancelled" ? "ready" : "error");
    }
    micP.status.textContent = "";
    // show the take as the model saw it: shifted by the estimated lag, cut to the reference length
    const lag = Math.round((micResult.alignment?.lag_sec ?? 0) * sr);
    const n = Math.round(dur * sr), aligned = c.createBuffer(1, n, sr), ch = aligned.getChannelData(0);
    for (let i = 0; i < n; i++) { const k = i + lag; ch[i] = k >= 0 && k < filled ? cap[k] : 0; }
    attachAudio(micP, c, aligned);
    msg.innerHTML = warn.map((w) => `<div class="alert warn" role="status">${esc(w)}</div>`).join("");
    paint(); scale();
    setStage("done");
  }

  start.onclick = runDemo;
  cancel.onclick = () => abort?.();
  $<HTMLSelectElement>("#demo-class").onchange = renderSamples;

  (async () => {
    try { samples = await api.samples(); renderSamples(); }
    catch (e) { msg.innerHTML = errorBox(e); setStage("error"); }
  })();
  setStage(stage);
}
