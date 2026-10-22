// Display preference shared by both modes: show the conformal layer (sets, 2-class set lane, triage) or only top-class predictions.

const KEY = "auscult-trust.conformal";
let on = true;
try { on = localStorage.getItem(KEY) !== "off"; } catch { /* storage blocked: default on */ }
const subs = new Set<() => void>();

export const conformalOn = () => on;

export function setConformal(v: boolean) {
  on = v;
  try { localStorage.setItem(KEY, v ? "on" : "off"); } catch { /* not persisted */ }
  subs.forEach((f) => f());
}

export const onConformal = (f: () => void) => { subs.add(f); };
