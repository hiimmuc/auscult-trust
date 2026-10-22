// Light / dark / follow-the-system switch; the choice is kept in localStorage and set as `data-theme` on <html>.

const KEY = "auscult-trust.theme";
const MODES = ["auto", "light", "dark"] as const;
type Mode = (typeof MODES)[number];

export function initTheme(btn: HTMLElement) {
  let mode: Mode = "auto";
  try { const v = localStorage.getItem(KEY); if (v === "light" || v === "dark") mode = v; } catch { /* storage blocked: follow the system */ }
  const apply = () => {
    if (mode === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = mode;
    btn.textContent = `Theme: ${mode[0].toUpperCase()}${mode.slice(1)}`;
    window.dispatchEvent(new Event("themechange")); // canvases read the colours when they draw
  };
  btn.onclick = () => {
    mode = MODES[(MODES.indexOf(mode) + 1) % MODES.length];
    try { localStorage.setItem(KEY, mode); } catch { /* not persisted */ }
    apply();
  };
  apply();
}
