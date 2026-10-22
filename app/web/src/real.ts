// Real mode: one uploaded WAV, validated in the browser and on the API, analysed by every pipeline.

import { api, type InferResult, type ModelInfo } from "./api";
import { capture, dB, MIC_HTML, micAlert, micPanel } from "./mic";
import { conformalOn, onConformal } from "./prefs";
import { denoise, trimSilence } from "./preprocess";
import { Siri } from "./siri";
import { attachAudio, conformalToggle, describe, errorBox, esc, legend, panel, pipelineSwitch, pipelineTag, proposed, scCurve, screening, segSwitch, segmentTable, truthOf } from "./ui";
import { encodeWav, readWavHeader } from "./wav";
import { TimeView } from "./wave";

/** One analysable take: a decoded upload or a live recording, as mono samples. */
interface Source { name: string; note: string; pcm: Float32Array; sr: number; file?: File; rows: [string, string][] }

const clock = (t: number) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;

export function mountReal(root: HTMLElement, m: ModelInfo, ctx: () => AudioContext) {
  const names = Object.keys(m.pipelines);
  let shown = names[0];
  let src: Source | null = null;
  let send: File | null = null; // what Run uploads: the original file, or the preprocessed samples as WAV
  let annotation: File | null = null;
  let result: InferResult | null = null;
  let busy: AbortController | null = null;
  let stopLive: ((keep: boolean) => void) | null = null;
  const view = new TimeView();
  const L = m.limits;

  root.innerHTML = `
    <div class="real-grid">
        <header class="title">
          <h1>Analyse a recording</h1>
          <p class="muted">Per-segment class, conformal sets at ${Math.round(100 * (1 - m.alpha))}% target coverage and a screening decision, with and without spectrum correction.</p>
        </header>
      <aside class="side">
        <div class="source"></div>
        <div class="file-block">
          <input type="file" multiple accept=".wav,audio/wav,audio/x-wav,.txt,text/plain" class="visually-hidden" id="real-file" />
          <button type="button" class="btn" id="real-pick">Choose .wav (and .txt)</button>
          <span class="hint">Mono or stereo · ${L.sample_rate[0] / 1000}–${L.sample_rate[1] / 1000} kHz · ${L.min_sec}–${L.max_sec} s · up to ${L.max_bytes >> 20} MB</span>
        </div>
        <div class="mic-block" hidden>
          ${MIC_HTML}
          <button type="button" class="btn primary" id="real-rec">Record</button>
          <span class="hint">Echo cancellation, noise suppression and auto gain are off, so the model hears the raw signal. Up to ${L.max_sec} s.</span>
        </div>
        <div class="meta" hidden></div>
        <fieldset class="prep">
          <legend>Preprocess</legend>
          <label class="switch"><input type="checkbox" role="switch" id="real-noise" /><span class="track" aria-hidden="true"></span><span>Reduce noise</span></label>
          <label class="switch"><input type="checkbox" role="switch" id="real-trim" /><span class="track" aria-hidden="true"></span><span>Trim silence</span></label>
          <span class="hint">Applied in the browser before upload; the waveform shows what is sent. The model was trained on unprocessed recordings, so leave both off unless the take is noisy.</span>
        </fieldset>
        <label class="field">Stethoscope
          <select id="real-device">
            <option value="">Unknown (estimate from recording)</option>
            ${m.devices.map((d) => `<option>${esc(d)}</option>`).join("")}
          </select>
          <span class="hint">A known ICBHI device uses its saved spectrum for correction.</span>
        </label>
        <details class="advanced">
          <summary>Segments</summary>
          <label class="field">ICBHI annotation (.txt, optional)
            <input type="file" accept=".txt,text/plain" id="real-ann" />
            <span class="hint" id="real-ann-note"></span>
            <span class="hint">Tip: choose or drop the .wav and its .txt together. With cycle times the model scores each breathing cycle, the unit it was trained on. Without, it scores sliding windows.</span>
          </label>
          <div class="row">
            <label class="field">Window (s) <input type="number" id="real-win" min="1" max="8" step="0.5" value="8" /></label>
            <label class="field">Step (s) <input type="number" id="real-hop" min="0.25" max="8" step="0.25" value="4" /></label>
          </div>
        </details>
        <div class="row">
          <button class="btn primary" id="real-run" disabled>Run analysis</button>
          <button class="btn ghost" id="real-cancel" hidden>Cancel</button>
        </div>
        <div class="msg" aria-live="polite"></div>
      </aside>
      <div class="main-col">
        <div class="empty" id="real-empty">
          <div class="drop" id="real-drop" tabindex="0" role="button">
            <span class="drop-main"></span><span class="drop-sub"></span>
          </div>
          <p class="muted">Each segment gets one of four labels:</p>
          <dl class="key">
            <div><dt><i class="sw c4-normal"></i>Normal</dt><dd>No adventitious sound.</dd></div>
            <div><dt><i class="sw c4-crackle"></i>Crackle</dt><dd>Short, discontinuous clicks.</dd></div>
            <div><dt><i class="sw c4-wheeze"></i>Wheeze</dt><dd>Continuous, musical tone.</dd></div>
            <div><dt><i class="sw c4-both"></i>Crackle + wheeze</dt><dd>Both in the same segment.</dd></div>
          </dl>
        </div>
        <section class="live" hidden aria-label="Live recording">
          <div class="live-top"><span class="rec-dot" aria-hidden="true"></span><b>Recording</b>
            <span class="live-time num">0:00.0</span><span class="muted num">of ${clock(L.max_sec)}</span></div>
          <div class="siri"></div>
          <div class="live-foot">
            <span class="muted">Breathe slowly and deeply through the mouth. A few full breaths are enough.</span>
            <span class="row"><button type="button" class="btn ghost" id="live-discard">Discard</button>
              <button type="button" class="btn primary" id="live-stop">Stop and review</button></span>
          </div>
        </section>
        <div class="work" hidden>
          <div class="toolbar"></div>
          <div class="verdict"></div>
          <div class="panels"></div>
          <div class="legend-host"></div>
          <div class="results"></div>
        </div>
      </div>
    </div>`;

  const $ = <T extends HTMLElement>(s: string) => root.querySelector<T>(s)!;
  const msg = $(".msg"), meta = $(".meta"), run = $<HTMLButtonElement>("#real-run"), cancel = $<HTMLButtonElement>("#real-cancel");
  const results = $(".results"), verdict = $(".verdict"), live = $(".live"), noise = $<HTMLInputElement>("#real-noise"), trim = $<HTMLInputElement>("#real-trim");
  const siri = new Siri($(".siri"));
  const mic = micPanel($(".mic-block"), ctx, msg);
  const P = panel($(".panels"), view, "Recording", "", "plain", (s) => describe(s, m.pipelines[shown].label, m), (i) => highlight(i));
  pipelineSwitch($(".toolbar"), m, shown, (v) => { shown = v; paint(); });
  conformalToggle($(".toolbar"));
  onConformal(() => { paint(); renderResults(); });

  function highlight(i: number) {
    results.querySelectorAll("tr.sel").forEach((r) => r.classList.remove("sel"));
    results.querySelector(`tr[data-i="${i}"]`)?.classList.add("sel");
  }

  function paint() {
    if (!result) { P.wave.setSegments([]); $(".legend-host").innerHTML = ""; return; }
    const segs = result.pipelines[shown].segments;
    P.wave.setSegments(segs, truthOf(segs));
    $(".legend-host").innerHTML = legend(m, segs.some((s) => s.truth4));
  }

  /** With an annotation: the recording's true label (ICBHI rule: abnormal if any cycle has a crackle or a wheeze) and each pipeline's agreement. */
  function truthNote(r: InferResult) {
    const segs = r.pipelines[names[0]].segments;
    if (!segs.some((x) => x.truth4)) return "";
    const abn = segs.filter((x) => x.truth4 && x.truth4 !== "normal").length;
    const hit = names.map((n) => `${esc(r.pipelines[n].label)}: ${r.pipelines[n].segments.filter((x) => x.pred4 === x.truth4).length} of ${segs.length}`);
    return `<p class="explain truth"><b>Annotation:</b> this recording is <b>${abn ? "abnormal" : "normal"}</b> (${abn} of ${segs.length} cycles with a crackle or a wheeze). Top class equals the annotated label in ${hit.join("; ")} cycles.</p>`;
  }

  function renderResults() {
    if (!result) { results.innerHTML = verdict.innerHTML = ""; return; }
    const r = result;
    const seg = r.segmentation.mode === "cycles" ? `${r.pipelines[names[0]].segments.length} annotated cycles`
      : `${r.pipelines[names[0]].segments.length} windows of ${r.segmentation.window_sec} s, step ${r.segmentation.hop_sec} s`;
    verdict.innerHTML = `
      <div class="verdict-head"><h2>Screening decision</h2>
        <span class="muted done">Done in ${(r.timing_ms / 1000).toFixed(1)} s · ${seg} · request ${esc(r.request_id)}</span></div>
      ${truthNote(r)}
      <p class="explain">${conformalOn()
        ? `The <b>top class</b> is the most probable label. The <b>conformal set</b> holds every label the model cannot rule out; across recordings like the calibration ones it contains the true label ${Math.round(100 * (1 - m.alpha))}% of the time. A set with both labels means “uncertain”.`
        : `Conformal prediction is off: only the <b>top class</b> is shown, with no statement of how sure the model is allowed to be. Switch it on above to see the sets.`}</p>
      <div class="cards">${names.map((n) => `<article class="card"><h4>${esc(r.pipelines[n].label)} ${pipelineTag(m, n)}</h4>${screening(r.pipelines[n], m.pipelines[n].thresholds.screen)}</article>`).join("")}</div>`;
    results.innerHTML = `
      <h3 class="sec-title">Spectrum correction ${proposed()}</h3>
      <div class="cards">${names.filter((n) => r.pipelines[n].sc.applied).map((n) => `<article class="card wide"><h4>Gain applied by ${esc(r.pipelines[n].label)}</h4>${scCurve(r.pipelines[n], m.sc_freqs_hz, r.device.declared)}</article>`).join("")}</div>
      <h3 class="sec-title">Per segment</h3>
      ${segmentTable(r, m, names)}`;
    results.querySelectorAll<HTMLTableRowElement>("tr[data-i]").forEach((tr) => (tr.onclick = () => {
      const i = +tr.dataset.i!;
      P.wave.selected = i; highlight(i);
      const s = result!.pipelines[shown].segments[i];
      if (s.end < view.t0 || s.start > view.t1) view.set(s.start - 1, s.end + 1);
      P.wave.player?.seek(s.start);
    }));
  }

  const mono = (a: AudioBuffer) => {
    const x = new Float32Array(a.length);
    for (let c = 0; c < a.numberOfChannels; c++) a.getChannelData(c).forEach((v, i) => (x[i] += v / a.numberOfChannels));
    return x;
  };

  /** Clears the previous take and its results. */
  function reset() {
    result = null; renderResults(); paint();
    msg.innerHTML = ""; run.disabled = true;
  }

  const fail = (t: string) => { msg.innerHTML = `<div class="alert error" role="alert">${esc(t)}</div>`; send = src = null; meta.hidden = true; showStage(); };

  function showStage() {
    $("#real-empty").hidden = !!src || !live.hidden; $(".work").hidden = !src || !live.hidden;
  }

  const stem = (n: string) => n.replace(/\.[^.]+$/, "");

  /** Attaches (or clears) the ICBHI annotation; its cycles give the true labels shown next to the predictions. */
  async function setAnnotation(f: File | null) {
    annotation = f;
    const note = $("#real-ann-note"), box = $<HTMLInputElement>("#real-ann");
    if (!f) { box.value = ""; note.textContent = ""; return; }
    const rows = (await f.text()).split("\n").filter((l) => l.trim()).length;
    if (annotation !== f) return;
    note.textContent = `${f.name}: ${rows} annotated cycle${rows === 1 ? "" : "s"}, true labels are shown with the predictions.`;
    $<HTMLDetailsElement>(".advanced").open = true;
  }

  /** One or several files: the .wav, and optionally its ICBHI .txt (same name), as when picking both from the dataset folder. */
  async function choose(files: File[]) {
    const f = files.find((x) => /\.wav$/i.test(x.name)), txt = files.find((x) => /\.txt$/i.test(x.name));
    reset();
    if (!f) return txt ? setAnnotation(txt) : fail("Only .wav files are accepted.");
    if (txt) await setAnnotation(txt);
    else if (annotation && stem(annotation.name) !== stem(f.name)) await setAnnotation(null); // an annotation of another recording would mislabel this one
    const buf = await f.arrayBuffer();
    let info;
    try { info = readWavHeader(buf); } catch (e: any) { return fail(e.message); }
    if (info.channels > L.channels[1]) return fail(`${info.channels} channels; use mono or stereo.`);
    if (info.sampleRate < L.sample_rate[0] || info.sampleRate > L.sample_rate[1]) return fail(`Sample rate ${info.sampleRate} Hz is outside ${L.sample_rate[0]}–${L.sample_rate[1]} Hz.`);
    if (info.durationSec < L.min_sec || info.durationSec > L.max_sec) return fail(`Duration ${info.durationSec.toFixed(1)} s is outside ${L.min_sec}–${L.max_sec} s.`);
    let audio: AudioBuffer;
    try { audio = await ctx().decodeAudioData(buf); } catch { return fail("The browser could not decode this WAV (unusual encoding?)."); }
    const dev = m.devices.find((d) => f.name.endsWith(`_${d}.wav`)); // ICBHI names end with the stethoscope
    $<HTMLSelectElement>("#real-device").value = dev ?? "";
    use({ name: f.name, note: "resampled to 16 kHz for the model", pcm: mono(audio), sr: audio.sampleRate, file: f, rows: [
      ["File", esc(f.name)], ["Duration", `${info.durationSec.toFixed(2)} s`], ["Sample rate", `${info.sampleRate} Hz`],
      ["Channels", String(info.channels)], ["Encoding", `${info.format} ${info.bitsPerSample}-bit`], ["Size", `${(f.size / 1024).toFixed(0)} KB`],
      ...(dev ? [["Stethoscope", `${dev} (from the file name)`] as [string, string]] : [])] });
  }

  function use(s: Source) {
    src = s;
    meta.hidden = false;
    meta.innerHTML = `<dl class="kv">${s.rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>`;
    showStage();
    prepare();
  }

  /** Applies the preprocessing switches to the current take and shows exactly what will be uploaded. */
  function prepare() {
    if (!src) return;
    let x = src.pcm;
    const notes = [src.note];
    if (noise.checked) { x = denoise(x); notes.push("noise reduced"); }
    if (trim.checked) { const t = trimSilence(x, src.sr, L.min_sec); x = t.x; notes.push(t.note); }
    const buf = ctx().createBuffer(1, x.length, src.sr);
    buf.getChannelData(0).set(x);
    send = x === src.pcm && src.file ? src.file : new File([encodeWav(x, src.sr)], src.name.replace(/\.wav$/i, "") + ".wav", { type: "audio/wav" });
    P.root.querySelector("h3")!.textContent = src.name;
    P.root.querySelector(".sub")!.textContent = `${buf.duration.toFixed(1)} s · ${notes.join(" · ")}`;
    view.setDuration(buf.duration);
    attachAudio(P, ctx(), buf);
    result = null; renderResults(); paint();
    run.disabled = false;
  }

  async function record() {
    msg.innerHTML = "";
    let stream: MediaStream;
    try { stream = await mic.open(); } catch (e) { msg.innerHTML = micAlert(e); return; }
    const c = ctx(), chunks: Float32Array[] = [];
    let n = 0, raf = 0, release = () => {};
    try {
      if (c.state === "suspended") await c.resume();
      const stopCap = await capture(c, stream, (_, d) => { chunks.push(d); n += d.length; });
      const stopLevel = mic.level(stream, (rms) => siri.push(rms));
      release = () => { cancelAnimationFrame(raf); stopCap(); stopLevel(); stream.getTracks().forEach((t) => t.stop()); siri.stop(); };
    } catch (e: any) { stream.getTracks().forEach((t) => t.stop()); msg.innerHTML = errorBox(e); return; }

    P.wave.player?.pause();
    live.hidden = false; showStage(); siri.start();
    run.disabled = recBtn.disabled = true;
    const time = $(".live-time");
    const tick = () => {
      const t = n / c.sampleRate;
      time.textContent = clock(t);
      if (t >= L.max_sec) return stopLive?.(true);
      raf = requestAnimationFrame(tick);
    };
    tick();

    stopLive = (keep) => {
      stopLive = null; release(); live.hidden = true; recBtn.disabled = false;
      if (!keep) { run.disabled = !send; msg.innerHTML = `<p class="muted">Discarded. Nothing was sent.</p>`; return showStage(); }
      const pcm = new Float32Array(n);
      let o = 0;
      for (const d of chunks) { pcm.set(d, o); o += d.length; }
      const dur = n / c.sampleRate;
      reset(); setAnnotation(null);
      if (dur < L.min_sec) return fail(`The recording is ${dur.toFixed(1)} s; at least ${L.min_sec} s is needed.`);
      const stamp = new Date().toTimeString().slice(0, 8).replaceAll(":", "-");
      use({ name: `microphone-${stamp}.wav`, note: "recorded live", pcm, sr: c.sampleRate, rows: [
        ["Source", "Microphone"], ["Duration", `${dur.toFixed(2)} s`], ["Sample rate", `${c.sampleRate} Hz`], ["Channels", "1"],
        ["Peak", `${dB(pcm.reduce((m, v) => Math.max(m, Math.abs(v)), 0)).toFixed(0)} dBFS`]] });
    };
  }

  async function analyse() {
    if (!send || busy) return;
    busy = new AbortController();
    run.disabled = true; cancel.hidden = false;
    msg.innerHTML = `<div class="progress" role="status"><span class="spinner" aria-hidden="true"></span>Analysing with ${names.length} pipelines…</div>`;
    P.status.textContent = "Analysing";
    try {
      const fields: Record<string, Blob | string> = { file: send, device: $<HTMLSelectElement>("#real-device").value,
        window_sec: $<HTMLInputElement>("#real-win").value, hop_sec: $<HTMLInputElement>("#real-hop").value };
      if (annotation) fields.annotation = annotation;
      result = await api.infer(fields, busy.signal);
      msg.innerHTML = "";
      paint(); renderResults();
    } catch (e: any) {
      msg.innerHTML = e.code === "cancelled" ? `<p class="muted">Cancelled.</p>` : errorBox(e);
    } finally {
      busy = null; run.disabled = !send; cancel.hidden = true; P.status.textContent = "";
    }
  }

  const input = $<HTMLInputElement>("#real-file"), drop = $("#real-drop"), recBtn = $<HTMLButtonElement>("#real-rec");
  let source = "file";
  const aim = () => (source === "file" ? input.click() : record());
  const limits = `${L.sample_rate[0] / 1000}–${L.sample_rate[1] / 1000} kHz · ${L.min_sec}–${L.max_sec} s · up to ${L.max_bytes >> 20} MB`;
  function setSource(v: string) {
    source = v;
    $(".file-block").hidden = v !== "file"; $(".mic-block").hidden = v !== "mic"; mic.endTest();
    $(".drop-main").innerHTML = v === "file" ? "Drop a <b>.wav</b> recording here, or click to choose. Add its ICBHI <b>.txt</b> to see the true labels." : "Click to <b>record</b> with the microphone";
    $(".drop-sub").textContent = v === "file" ? limits : `Up to ${L.max_sec} s · raw signal, no echo cancellation`;
  }
  setSource("file");
  input.onchange = () => input.files?.length && choose([...input.files]);
  $("#real-pick").onclick = () => input.click();
  drop.onclick = aim;
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); aim(); } });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); const fs = [...(e.dataTransfer?.files ?? [])]; if (fs.length) choose(fs); });
  segSwitch($(".source"), [{ value: "file", html: "Upload file" }, { value: "mic", html: "Record live" }], "file", "Audio source", setSource);
  recBtn.onclick = record;
  $("#live-stop").onclick = () => stopLive?.(true);
  $("#live-discard").onclick = () => stopLive?.(false);
  noise.onchange = trim.onchange = prepare;
  $<HTMLInputElement>("#real-ann").onchange = (e) => setAnnotation((e.target as HTMLInputElement).files?.[0] ?? null);
  run.onclick = analyse;
  cancel.onclick = () => busy?.abort();
}
