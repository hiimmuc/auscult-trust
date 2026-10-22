// Shared view pieces: panel shell, legend, tooltip text, screening card, SC curve, segment and comparison tables.

import type { Class4, InferResult, ModelInfo, PipelineResult, Segment } from "./api";
import { conformalOn, onConformal, setConformal } from "./prefs";
import { Player, TimeView, WaveView } from "./wave";

export const esc = (s: unknown) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
export const pct = (p: number) => `${(100 * p).toFixed(p >= 0.995 || p < 0.005 ? 1 : 0)}%`;
export const sec = (t: number) => `${t.toFixed(2)} s`;

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls = "", html = ""): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html) e.innerHTML = html;
  return e;
}

export const CLASS_LABEL: Record<string, string> = { normal: "Normal", crackle: "Crackle", wheeze: "Wheeze", both: "Crackle + wheeze", abnormal: "Abnormal" };

const SHORT: Record<string, string> = { ...CLASS_LABEL, both: "Both" };

/** Marks the methods this project proposes: spectrum correction (the SC pipeline) and conformal prediction. */
export const proposed = (on = true) => (on ? `<span class="tag" title="Proposed method">Proposed</span>` : "");
export const pipelineTag = (m: ModelInfo, name: string) => proposed(m.pipelines[name].spectrum_correction);

/** Switch that shows or hides the conformal layer; several instances (one per mode) stay in sync. */
export function conformalToggle(host: HTMLElement) {
  const l = el("label", "switch", `<input type="checkbox" role="switch" /><span class="track" aria-hidden="true"></span>
    <span>Conformal prediction</span>${proposed()}`);
  const box = l.querySelector("input")!;
  const sync = () => { box.checked = conformalOn(); };
  sync();
  box.onchange = () => setConformal(box.checked);
  onConformal(sync);
  host.append(l);
}

export function chip(c: string, on: boolean, two = false) {
  return `<span class="chip ${two ? "c2" : "c4"}-${c} ${on ? "on" : "off"}" title="${esc(CLASS_LABEL[c])}: ${on ? "in" : "not in"} the conformal set">${esc(SHORT[c])}</span>`;
}

export function setChips(set: string[], all: readonly string[], two = false) {
  if (!set.length) return `<span class="chip empty">empty set</span>`;
  return all.map((c) => (set.includes(c) ? chip(c, true, two) : "")).join("");
}

export function describe(seg: Segment, label: string, m: ModelInfo) {
  const cf = conformalOn();
  const rows = m.classes.map((c) => `<tr class="${c === seg.pred4 ? "top" : ""}"><td><i class="sw c4-${c}"></i>${CLASS_LABEL[c]}</td>
    <td class="num">${pct(seg.probs4[c])}</td>${cf ? `<td>${seg.set4.includes(c) ? "in set" : ""}</td>` : ""}</tr>`).join("");
  return `<div class="tt-head">${esc(label)} · segment ${seg.index + 1}</div>
    <div class="tt-sub">${sec(seg.start)} – ${sec(seg.end)}${seg.truth4 ? ` · annotated <b>${CLASS_LABEL[seg.truth4]}</b>` : ""}</div>
    <table class="tt">${rows}</table>
    <div class="tt-sub">Normal vs abnormal: <b>${CLASS_LABEL[seg.pred2]}</b> ${pct(seg.probs2[seg.pred2])}${cf ? ` · set {${seg.set2.map((c) => CLASS_LABEL[c]).join(", ") || "empty"}}` : ""}</div>`;
}

export function legend(m: ModelInfo, truth: boolean) {
  return `<div class="legend" aria-label="Legend">
    <span class="lg-title">Overlay, 4 classes</span>${m.classes.map((c) => `<span><i class="sw c4-${c}"></i>${CLASS_LABEL[c]}</span>`).join("")}
    <span class="lg-title">Lane, 2 classes</span>
    <span><i class="sw c2-normal"></i>Normal</span><span><i class="sw c2-abnormal"></i>Abnormal</span>
    ${conformalOn() ? `<span><i class="sw hatch"></i>Set holds both</span><span><i class="sw outline"></i>Empty set</span>` : ""}
    ${truth ? `<span class="lg-title">Thin lane</span><span>Annotated label (ICBHI)</span>` : ""}
  </div>`;
}

export interface Panel {
  root: HTMLElement;
  wave: WaveView;
  status: HTMLElement;
  play: HTMLButtonElement;
}

export function panel(host: HTMLElement, view: TimeView, title: string, sub: string, tone: "ref" | "mic" | "plain",
                      describeFn: (s: Segment) => string, onSelect?: (i: number) => void): Panel {
  const root = el("section", `panel tone-${tone}`);
  root.innerHTML = `<header class="panel-head">
      <div><h3>${esc(title)}</h3><p class="sub">${sub}</p></div>
      <div class="panel-tools">
        <span class="status" aria-live="polite"></span>
        <button class="btn small play" disabled aria-label="Play ${esc(title)}">Play</button>
        <span class="zoom" role="group" aria-label="Zoom">
          <button class="btn small ghost" data-z="in" aria-label="Zoom in">+</button>
          <button class="btn small ghost" data-z="out" aria-label="Zoom out">−</button>
          <button class="btn small ghost" data-z="fit">Fit</button>
        </span>
      </div>
    </header><div class="wave-host"></div>`;
  host.append(root);
  const wave = new WaveView(root.querySelector(".wave-host")!, view, { describe: describeFn, onSelect, label: `${title} waveform` });
  const play = root.querySelector<HTMLButtonElement>(".play")!;
  play.onclick = () => wave.player?.toggle();
  root.querySelectorAll<HTMLButtonElement>("[data-z]").forEach((b) => (b.onclick = () =>
    b.dataset.z === "in" ? view.zoom(0.5) : b.dataset.z === "out" ? view.zoom(2) : view.fit()));
  return { root, wave, status: root.querySelector(".status")!, play };
}

export function attachAudio(p: Panel, ctx: AudioContext, buf: AudioBuffer) {
  const player = new Player(ctx, buf);
  p.wave.attachPlayer(player);
  const prev = player.onchange;
  player.onchange = () => { prev(); p.play.textContent = player.playing ? "Pause" : "Play"; };
  p.play.disabled = false;
  p.wave.setAudio(buf.getChannelData(0), buf.sampleRate);
}

const TRIAGE: Record<string, [string, string]> = {
  refer: ["Refer", "Only “abnormal” is in the set."],
  no_referral: ["No referral", "Only “normal” is in the set."],
  uncertain: ["Uncertain", "Both classes are in the set: the model cannot rule either out."],
  rerecord: ["Re-record", "The set is empty: the recording looks unlike the calibration data."],
};

/** Recording-level screening: p(abnormal) on a 0-1 bar with the zones the conformal threshold creates. */
export function screening(r: PipelineResult, thr: number) {
  const rec = r.recording;
  if (!conformalOn()) {
    const abn = rec.p_abnormal >= 0.5;
    return `<div class="screen">
    <div class="screen-head"><span class="triage ${abn ? "t-refer" : "t-no_referral"}">${abn ? "Abnormal" : "Normal"}</span>
      <span class="pab">p(abnormal) <b class="num">${rec.p_abnormal.toFixed(3)}</b></span></div>
    <div class="bar" role="img" aria-label="p abnormal ${rec.p_abnormal.toFixed(3)}"><span class="mark" style="left:${100 * rec.p_abnormal}%"></span></div>
    <div class="bar-scale"><span>0 · normal</span><span>0.5</span><span>abnormal · 1</span></div>
    <p class="why">Top class only, no uncertainty. Mean over ${rec.windows.length} window${rec.windows.length > 1 ? "s" : ""} of 8 s.</p>
  </div>`;
  }
  const lo = Math.min(1 - thr, thr), hi = Math.max(1 - thr, thr);
  const mid = thr >= 0.5 ? ["uncertain", "both in set"] : ["rerecord", "empty set"];
  const [word, why] = TRIAGE[rec.triage];
  return `<div class="screen">
    <div class="screen-head"><span class="triage t-${rec.triage}">${word}</span>
      <span class="pab">p(abnormal) <b class="num">${rec.p_abnormal.toFixed(3)}</b></span></div>
    <div class="bar" role="img" aria-label="p abnormal ${rec.p_abnormal.toFixed(3)}; normal-only below ${lo.toFixed(3)}, abnormal-only above ${hi.toFixed(3)}">
      <span class="z z-no" style="width:${100 * lo}%"></span><span class="z z-${mid[0]}" style="width:${100 * (hi - lo)}%"></span><span class="z z-refer" style="width:${100 * (1 - hi)}%"></span>
      <span class="mark" style="left:${100 * rec.p_abnormal}%"></span>
    </div>
    <div class="bar-scale"><span>0 · normal only</span><span>${mid[1]}</span><span>abnormal only · 1</span></div>
    <p class="why">${why} Mean over ${rec.windows.length} window${rec.windows.length > 1 ? "s" : ""} of 8 s.</p>
  </div>`;
}

/** SC gain per frequency bin, 0-4 kHz, as an SVG line. */
export function scCurve(r: PipelineResult, freqs: number[], device: string | null) {
  if (!r.sc.applied) return `<p class="muted">No spectrum correction in this pipeline.</p>`;
  const W = 540, H = 150, P = { l: 54, r: 26, t: 10, b: 26 };
  const n = freqs.findIndex((f) => f > 4000);
  const g = r.sc.gain_db.slice(0, n < 0 ? undefined : n), f = freqs.slice(0, g.length);
  const lim = Math.max(3, Math.ceil(Math.max(...g.map(Math.abs)) / 3) * 3);
  const X = (hz: number) => P.l + (hz / 4000) * (W - P.l - P.r), Y = (db: number) => P.t + ((lim - db) / (2 * lim)) * (H - P.t - P.b);
  const path = g.map((d, i) => `${i ? "L" : "M"}${X(f[i]).toFixed(1)},${Y(d).toFixed(1)}`).join("");
  const xt = [0, 1000, 2000, 3000, 4000].map((hz) => `<text x="${X(hz)}" y="${H - 8}" text-anchor="middle">${hz / 1000} kHz</text>`).join("");
  const yt = [-lim, 0, lim].map((d) => `<text x="${P.l - 6}" y="${Y(d) + 4}" text-anchor="end">${d > 0 ? "+" : ""}${d} dB</text><line x1="${P.l}" x2="${W - P.r}" y1="${Y(d)}" y2="${Y(d)}" class="${d ? "grid" : "zero"}"/>`).join("");
  const band = r.sc.mode === "static" ? `<rect x="${X(50)}" y="${P.t}" width="${X(2000) - X(50)}" height="${H - P.t - P.b}" class="band"/>` : "";
  const src = r.sc.coefficient_source.startsWith("saved")
    ? `From the saved spectrum of unlabelled ICBHI ${esc(device)} recordings.`
    : "Estimated from this recording alone (device unknown), so part of the curve can be the sound itself rather than the device.";
  return `<svg viewBox="0 0 ${W} ${H}" class="sc-svg" role="img" aria-label="Spectrum correction gain per frequency">
      ${band}${yt}${xt}<path d="${path}" class="curve"/></svg>
    <p class="muted">${src} ${r.sc.mode === "static" ? "Static mode: corrected between 50 Hz and 2 kHz (shaded), untouched elsewhere." : "Dynamic mode: every bin clipped to ±20 dB."}</p>`;
}

const flip = (a: string, b: string) => (a !== b ? `<span class="flip" title="changed">changed</span>` : "");

/** Real mode: one row per segment, every pipeline side by side. */
export function segmentTable(res: InferResult, m: ModelInfo, names: string[]) {
  const segs = res.pipelines[names[0]].segments;
  const truth = segs.some((s) => s.truth4);
  const cf = conformalOn();
  const head = `<tr><th>#</th><th>Time</th>${truth ? "<th>Annotated</th>" : ""}${names.map((n) =>
    `<th colspan="${cf ? 2 : 1}">${esc(res.pipelines[n].label)} ${pipelineTag(m, n)}</th>`).join("")}${names.length === 2 ? "<th></th>" : ""}</tr>`;
  const body = segs.map((s, i) => {
    const cells = names.map((n) => {
      const q = res.pipelines[n].segments[i];
      return `<td><i class="sw c4-${q.pred4}"></i>${CLASS_LABEL[q.pred4]} <span class="num muted">${pct(q.probs4[q.pred4])}</span></td>
        ${cf ? `<td class="sets">${setChips(q.set4, m.classes)}</td>` : ""}`;
    }).join("");
    const d = names.length === 2 ? `<td>${flip(res.pipelines[names[0]].segments[i].pred4, res.pipelines[names[1]].segments[i].pred4)}</td>` : "";
    return `<tr data-i="${i}"><td class="num">${i + 1}</td><td class="num">${s.start.toFixed(2)}–${s.end.toFixed(2)}</td>
      ${truth ? `<td>${s.truth4 ? `<i class="sw c4-${s.truth4}"></i>${CLASS_LABEL[s.truth4]}` : ""}</td>` : ""}${cells}${d}</tr>`;
  }).join("");
  return `<div class="table-wrap"><table class="segs"><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
}

export function agreement(a: Segment[], b: Segment[]) {
  const n = a.length;
  return {
    n,
    flip4: a.filter((s, i) => s.pred4 !== b[i].pred4).length,
    flip2: a.filter((s, i) => s.pred2 !== b[i].pred2).length,
    setChange: a.filter((s, i) => s.set4.join() !== b[i].set4.join()).length,
    size: (x: Segment[]) => x.reduce((t, s) => t + s.set4.length, 0) / Math.max(x.length, 1),
    correct: (x: Segment[]) => x.filter((s) => s.truth4 && s.pred4 === s.truth4).length,
  };
}

/** Demo mode: reference vs microphone, per pipeline summary and per-cycle rows. */
export function comparison(ref: InferResult, mic: InferResult, m: ModelInfo, name: string) {
  const R = ref.pipelines[name], M = mic.pipelines[name];
  const a = agreement(R.segments, M.segments);
  const cf = conformalOn();
  const tri = (r: PipelineResult) => !cf
    ? `<span class="triage small ${r.recording.p_abnormal >= 0.5 ? "t-refer" : "t-no_referral"}">${r.recording.p_abnormal >= 0.5 ? "Abnormal" : "Normal"}</span>`
    : `<span class="triage small t-${r.recording.triage}">${TRIAGE[r.recording.triage][0]}</span>`;
  const truth = R.segments.some((s) => s.truth4);
  const stats = `<dl class="stats">
      <div><dt>Top class changed</dt><dd class="num">${a.flip4} / ${a.n}</dd><dd class="muted">cycles (flip rate ${pct(a.flip4 / a.n)})</dd></div>
      <div><dt>Normal ↔ abnormal changed</dt><dd class="num">${a.flip2} / ${a.n}</dd></div>
      ${cf ? `<div><dt>Conformal set changed</dt><dd class="num">${a.setChange} / ${a.n}</dd><dd class="muted">mean size ${a.size(R.segments).toFixed(2)} → ${a.size(M.segments).toFixed(2)}</dd></div>` : ""}
      ${truth ? `<div><dt>Matches annotation</dt><dd class="num">${a.correct(R.segments)} → ${a.correct(M.segments)}</dd><dd class="muted">of ${a.n} cycles</dd></div>` : ""}
      <div><dt>Recording screening</dt><dd>${tri(R)} → ${tri(M)}</dd><dd class="muted num">p(abn.) ${R.recording.p_abnormal.toFixed(3)} → ${M.recording.p_abnormal.toFixed(3)}</dd></div>
    </dl>`;
  const rows = R.segments.map((s, i) => {
    const q = M.segments[i];
    return `<tr data-i="${i}"><td class="num">${i + 1}</td><td class="num">${s.start.toFixed(2)}–${s.end.toFixed(2)}</td>
      ${truth ? `<td>${s.truth4 ? `<i class="sw c4-${s.truth4}"></i>${CLASS_LABEL[s.truth4]}` : ""}</td>` : ""}
      <td><i class="sw c4-${s.pred4}"></i>${CLASS_LABEL[s.pred4]} <span class="num muted">${pct(s.probs4[s.pred4])}</span></td>${cf ? `<td class="sets">${setChips(s.set4, m.classes)}</td>` : ""}
      <td><i class="sw c4-${q.pred4}"></i>${CLASS_LABEL[q.pred4]} <span class="num muted">${pct(q.probs4[q.pred4])}</span></td>${cf ? `<td class="sets">${setChips(q.set4, m.classes)}</td>` : ""}
      <td>${flip(s.pred4, q.pred4)}</td></tr>`;
  }).join("");
  return `${stats}<div class="table-wrap"><table class="segs"><thead><tr><th>#</th><th>Time</th>${truth ? "<th>Annotated</th>" : ""}
    <th colspan="${cf ? 2 : 1}" class="ref">Reference</th><th colspan="${cf ? 2 : 1}" class="mic">Microphone</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

/** Radio-style segmented control: arrow keys move the selection, `html` is trusted markup. */
export function segSwitch(host: HTMLElement, items: { value: string; html: string }[], value: string, label: string, onChange: (v: string) => void) {
  const g = el("div", "seg-switch");
  g.setAttribute("role", "radiogroup");
  g.setAttribute("aria-label", label);
  const pick = (v: string) => { render(v); onChange(v); g.querySelector<HTMLButtonElement>(`[data-v="${v}"]`)!.focus(); };
  const render = (v: string) => {
    g.innerHTML = items.map((i) => `<button type="button" role="radio" aria-checked="${i.value === v}" data-v="${i.value}" tabindex="${i.value === v ? 0 : -1}">${i.html}</button>`).join("");
    g.querySelectorAll<HTMLButtonElement>("button").forEach((b) => {
      b.onclick = () => pick(b.dataset.v!);
      b.onkeydown = (e) => {
        if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
        const at = items.findIndex((i) => i.value === b.dataset.v);
        pick(items[(at + (e.key === "ArrowRight" ? 1 : items.length - 1)) % items.length].value);
      };
    });
  };
  render(value);
  host.append(g);
}

export const pipelineSwitch = (host: HTMLElement, m: ModelInfo, value: string, onChange: (v: string) => void) =>
  segSwitch(host, Object.keys(m.pipelines).map((n) => ({ value: n, html: `${esc(m.pipelines[n].label)}${pipelineTag(m, n)}` })),
            value, "Pipeline shown on the waveforms", onChange);

export function errorBox(e: any) {
  const rid = e?.requestId ? ` <span class="muted">Request ${esc(e.requestId)}</span>` : "";
  return `<div class="alert error" role="alert"><b>${esc(e?.code === "network" ? "Connection problem" : "Something failed")}</b> ${esc(e?.message ?? e)}${rid}</div>`;
}

export const truthOf = (segs: Segment[]) => segs.filter((s) => s.truth4).map((s) => ({ start: s.start, end: s.end, label: s.truth4 as Class4 }));
