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

/** The recording-level decision of one pipeline: wording, colour class, bar with the zones the conformal threshold creates, and the reading. */
function triage(r: PipelineResult, thr: number) {
  const rec = r.recording, p = rec.p_abnormal, n = rec.windows.length;
  const windows = `Mean over ${n} window${n > 1 ? "s" : ""} of 8 s.`;
  if (!conformalOn()) {
    const abn = p >= 0.5;
    return { word: abn ? "Abnormal" : "Normal", cls: abn ? "t-refer" : "t-no_referral", p,
      bar: `<div class="bar" role="img" aria-label="p abnormal ${p.toFixed(3)}"><span class="mark" style="left:${100 * p}%"></span></div>`,
      scale: ["0 · normal", "0.5", "abnormal · 1"], why: `Top class only, no uncertainty. ${windows}` };
  }
  const lo = Math.min(1 - thr, thr), hi = Math.max(1 - thr, thr);
  const mid = thr >= 0.5 ? ["uncertain", "both in set"] : ["rerecord", "empty set"];
  const [word, why] = TRIAGE[rec.triage];
  return { word, cls: `t-${rec.triage}`, p,
    bar: `<div class="bar" role="img" aria-label="p abnormal ${p.toFixed(3)}; normal-only below ${lo.toFixed(3)}, abnormal-only above ${hi.toFixed(3)}">
      <span class="z z-no" style="width:${100 * lo}%"></span><span class="z z-${mid[0]}" style="width:${100 * (hi - lo)}%"></span><span class="z z-refer" style="width:${100 * (1 - hi)}%"></span>
      <span class="mark" style="left:${100 * p}%"></span></div>`,
    scale: ["0 · normal only", mid[1], "abnormal only · 1"], why: `${why} ${windows}` };
}

const scale = (t: string[]) => `<div class="bar-scale"><span>${t[0]}</span><span>${t[1]}</span><span>${t[2]}</span></div>`;

/** Full card of one pipeline's recording-level decision. */
export function screening(r: PipelineResult, thr: number) {
  const t = triage(r, thr);
  return `<div class="screen">
    <div class="screen-head"><span class="triage ${t.cls}">${t.word}</span><span class="pab">p(abnormal) <b class="num">${t.p.toFixed(3)}</b></span></div>
    ${t.bar}${scale(t.scale)}<p class="why">${t.why}</p></div>`;
}

/**
 * Compact decision strip: the proposed pipeline's verdict and bar, the others as one "says" line, and the full cards
 * and the reading guide behind a disclosure.
 */
export function verdictStrip(res: InferResult, m: ModelInfo, names: string[], meta: string, note: string, explain: string) {
  const main = names.find((n) => m.pipelines[n].spectrum_correction) ?? names[0];
  const others = names.filter((n) => n !== main);
  const T = (n: string) => triage(res.pipelines[n], m.pipelines[n].thresholds.screen);
  const t = T(main);
  const says = others.map((n) => {
    const o = T(n);
    return `<p class="says"><span class="muted">${esc(res.pipelines[n].label)} says</span> <b class="${o.cls}">${o.word}</b>
      <span class="num muted">p ${o.p.toFixed(3)}</span> <span class="muted">· ${o.word === t.word ? "same decision" : "different decision"}</span></p>`;
  }).join("");
  return `<section class="strip">
    <div class="strip-main">
      <h2>${esc(res.pipelines[main].label)} ${pipelineTag(m, main)}</h2>
      <div class="screen-head"><span class="triage ${t.cls}">${t.word}</span><span class="pab">p(abnormal) <b class="num">${t.p.toFixed(3)}</b></span></div>
      ${t.bar}${scale(t.scale)}
    </div>
    <div class="strip-side">${says}${note}<p class="muted strip-meta">${meta}</p></div>
    <details class="strip-more"><summary>How to read this, and the full decisions</summary>
      <p class="explain">${explain}</p>
      <div class="cards">${names.map((n) => `<article class="card"><h4>${esc(res.pipelines[n].label)} ${pipelineTag(m, n)}</h4>${screening(res.pipelines[n], m.pipelines[n].thresholds.screen)}</article>`).join("")}</div>
    </details>
  </section>`;
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

const set = (q: Segment) => (q.set4.length > 1
  ? ` <span class="set" title="Conformal set: ${q.set4.map((c) => CLASS_LABEL[c]).join(", ")}">${q.set4.length > 2 ? `set of ${q.set4.length}` : `or ${CLASS_LABEL[q.set4.find((c) => c !== q.pred4)!].toLowerCase()}`}</span>` : "");
const top = (q: Segment, cf: boolean, miss = false) =>
  `<td${miss ? ' class="miss"' : ""}><i class="sw c4-${q.pred4}"></i>${CLASS_LABEL[q.pred4]} <span class="num muted">${pct(q.probs4[q.pred4])}</span>${cf ? set(q) : ""}</td>`;

let onlyFlagged = false;

/** Wraps the table with its "only differing rows" filter. Call `bindTable` on the container after inserting the markup. */
const tableBox = (flagged: number, total: number, label: string, table: string) => `<div class="segs-box${onlyFlagged ? " only" : ""}">
  <div class="table-tools"><span class="muted num">${flagged} of ${total} rows ${label}</span>
    <label class="check"><input type="checkbox" class="only-flag" ${onlyFlagged ? "checked" : ""} /> Only these rows</label></div>
  <div class="table-wrap">${table}</div></div>`;

export function bindTable(root: HTMLElement) {
  const box = root.querySelector<HTMLElement>(".segs-box"), cb = root.querySelector<HTMLInputElement>(".only-flag");
  if (box && cb) cb.onchange = () => { onlyFlagged = cb.checked; box.classList.toggle("only", onlyFlagged); };
}

/** Real mode: one row per segment, top class (and the conformal set when it holds more than one label) per pipeline. */
export function segmentTable(res: InferResult, m: ModelInfo, names: string[]) {
  const segs = res.pipelines[names[0]].segments;
  const truth = segs.some((s) => s.truth4);
  const cf = conformalOn();
  const head = `<tr><th>#</th><th>Time</th>${truth ? "<th>Annotated</th>" : ""}${names.map((n) =>
    `<th>${esc(res.pipelines[n].label)} ${pipelineTag(m, n)}</th>`).join("")}<th></th></tr>`;
  let flagged = 0;
  const body = segs.map((s, i) => {
    const qs = names.map((n) => res.pipelines[n].segments[i]);
    const miss = qs.map((q) => !!s.truth4 && q.pred4 !== s.truth4);
    const differs = qs.some((q) => q.pred4 !== qs[0].pred4);
    const flag = miss.some(Boolean) || differs;
    if (flag) flagged++;
    return `<tr data-i="${i}"${flag ? ' data-flag="1"' : ""}><td class="num">${i + 1}</td><td class="num">${s.start.toFixed(2)}–${s.end.toFixed(2)}</td>
      ${truth ? `<td>${s.truth4 ? `<i class="sw c4-${s.truth4}"></i>${CLASS_LABEL[s.truth4]}` : ""}</td>` : ""}${qs.map((q, k) => top(q, cf, miss[k])).join("")}
      <td class="flag">${differs ? `<span class="flip">differs</span>` : ""}</td></tr>`;
  }).join("");
  return tableBox(flagged, segs.length, truth ? "differ from the annotation or between pipelines" : "differ between pipelines",
    `<table class="segs"><thead>${head}</thead><tbody>${body}</tbody></table>`);
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

/** Demo mode: summary tiles, then one row per cycle with the reference and the microphone result. */
export function comparison(ref: InferResult, mic: InferResult, name: string) {
  const R = ref.pipelines[name], M = mic.pipelines[name];
  const a = agreement(R.segments, M.segments);
  const cf = conformalOn();
  const word = (r: PipelineResult) => (cf ? TRIAGE[r.recording.triage][0] : r.recording.p_abnormal >= 0.5 ? "Abnormal" : "Normal");
  const truth = R.segments.some((s) => s.truth4);
  const tile = (label: string, big: string, sub = "") => `<div><dt>${label}</dt><dd class="num">${big}</dd>${sub ? `<dd class="muted">${sub}</dd>` : ""}</div>`;
  const stats = `<dl class="stats">
      ${tile("Top class changed", `${a.flip4}<small> of ${a.n}</small>`, `flip rate ${pct(a.flip4 / a.n)}`)}
      ${tile("Normal vs abnormal changed", `${a.flip2}<small> of ${a.n}</small>`)}
      ${cf ? tile("Conformal set changed", `${a.setChange}<small> of ${a.n}</small>`, `mean size ${a.size(R.segments).toFixed(2)} → ${a.size(M.segments).toFixed(2)}`) : ""}
      ${truth ? tile("Matches annotation", `${a.correct(R.segments)} → ${a.correct(M.segments)}`, `of ${a.n} cycles`) : ""}
      <div><dt>Recording screening</dt><dd class="lines"><span>Reference <b>${word(R)}</b> <span class="num muted">p ${R.recording.p_abnormal.toFixed(3)}</span></span>
        <span>Microphone <b>${word(M)}</b> <span class="num muted">p ${M.recording.p_abnormal.toFixed(3)}</span></span></dd></div>
    </dl>`;
  let changed = 0;
  const rows = R.segments.map((s, i) => {
    const q = M.segments[i];
    const flag = s.pred4 !== q.pred4;
    if (flag) changed++;
    return `<tr data-i="${i}"${flag ? ' data-flag="1"' : ""}><td class="num">${i + 1}</td><td class="num">${s.start.toFixed(2)}–${s.end.toFixed(2)}</td>
      ${truth ? `<td>${s.truth4 ? `<i class="sw c4-${s.truth4}"></i>${CLASS_LABEL[s.truth4]}` : ""}</td>` : ""}
      ${top(s, cf)}${top(q, cf, flag)}<td class="flag">${flag ? `<span class="flip">changed</span>` : ""}</td></tr>`;
  }).join("");
  return `${stats}${tableBox(changed, a.n, "changed their top class",
    `<table class="segs"><thead><tr><th>#</th><th>Time</th>${truth ? "<th>Annotated</th>" : ""}<th>Reference</th><th>Microphone</th><th></th></tr></thead><tbody>${rows}</tbody></table>`)}`;
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

/** Shows the text of every `[data-tip]` element in the page tooltip on hover and keyboard focus. */
export function bindTips(root: HTMLElement) {
  const tip = document.getElementById("tooltip")!;
  root.querySelectorAll<HTMLElement>("[data-tip]").forEach((e) => {
    const show = () => {
      const r = e.getBoundingClientRect();
      tip.innerHTML = esc(e.dataset.tip!); tip.hidden = false;
      tip.style.left = `${Math.max(8, Math.min(r.left, innerWidth - tip.offsetWidth - 8))}px`; tip.style.top = `${r.bottom + 8}px`;
    };
    e.addEventListener("pointerenter", show); e.addEventListener("focus", show);
    e.addEventListener("pointerleave", () => (tip.hidden = true)); e.addEventListener("blur", () => (tip.hidden = true));
  });
}
