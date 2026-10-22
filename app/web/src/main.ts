import "./style.css";
import { activeBase, api, ApiError, candidateBases, connection, onFallback, saveConnection, useBase, type ModelInfo } from "./api";
import { mountDemo } from "./demo";
import { mountReal } from "./real";
import { initTheme } from "./theme";
import { errorBox, esc } from "./ui";

let audio: AudioContext | null = null;
const ctx = () => (audio ??= new AudioContext({ latencyHint: "interactive" }));

const status = document.getElementById("api-status")!;
const dialog = document.getElementById("settings") as HTMLDialogElement;
const form = dialog.querySelector("form")!;
const views = { real: document.getElementById("view-real")!, demo: document.getElementById("view-demo")! };
const tabs = { real: document.getElementById("tab-real")!, demo: document.getElementById("tab-demo")! };
let mounted = { real: false, demo: false };
let model: ModelInfo | null = null;

function show(mode: "real" | "demo", focus = false) {
  (Object.keys(views) as ("real" | "demo")[]).forEach((k) => {
    views[k].hidden = k !== mode;
    tabs[k].setAttribute("aria-selected", String(k === mode));
    tabs[k].tabIndex = k === mode ? 0 : -1;
  });
  if (focus) tabs[mode].focus();
  if (model && !mounted[mode]) { (mode === "real" ? mountReal : mountDemo)(views[mode], model, ctx); mounted[mode] = true; }
  history.replaceState(null, "", mode === "demo" ? "#demo" : "#");
}
tabs.real.onclick = () => show("real");
tabs.demo.onclick = () => show("demo");
document.querySelector(".modes")!.addEventListener("keydown", (e) => {
  const k = (e as KeyboardEvent).key;
  if (k === "ArrowRight" || k === "ArrowLeft") show(tabs.real.getAttribute("aria-selected") === "true" ? "demo" : "real", true);
});

function setStatus(state: "ok" | "down" | "wait", text: string) {
  status.dataset.state = state;
  status.querySelector(".txt")!.textContent = text;
  status.title = `${activeBase()} · click to change`;
}

async function connect() {
  setStatus("wait", "Connecting");
  const waking = setTimeout(() => setStatus("wait", "Waking the API"), 5000);
  try {
    let first: any = null;
    for (const base of candidateBases()) {  // the chosen API, then the fallback
      try {
        const h = await api.health(base);
        if (h.status !== "ok") throw new ApiError("model_unavailable", h.reason || "The API has no model loaded.");
        useBase(base);
        model = await api.model();
        first = null;
        break;
      } catch (e) { first ??= e; }
    }
    if (first) throw first;
    setStatus("ok", onFallback() ? "Fallback API ready" : "API ready");
    mounted = { real: false, demo: false };
    views.real.innerHTML = views.demo.innerHTML = "";
    show(location.hash === "#demo" ? "demo" : "real");
    return null;
  } catch (e: any) {
    const noModel = e.code === "model_unavailable";
    setStatus("down", noModel ? "Model unavailable" : e.code === "unauthorized" ? "Token needed" : "API offline");
    const box = noModel
      ? `<div class="offline"><h1>No model loaded</h1>${errorBox(e)}<p>The API is running, but it cannot classify anything and shows no estimate in its place.</p>
      <div class="row"><button class="btn primary" data-retry>Try again</button><button class="btn ghost" data-open>Connection settings</button></div></div>`
      : `<div class="offline"><h1>Connect to the API</h1>${errorBox(e)}<p>Set the API address and token, then try again.</p>
      <button class="btn primary" data-open>Connection settings</button></div>`;
    views.real.innerHTML = views.demo.innerHTML = box;
    document.querySelectorAll<HTMLButtonElement>("[data-open]").forEach((b) => (b.onclick = openSettings));
    document.querySelectorAll<HTMLButtonElement>("[data-retry]").forEach((b) => (b.onclick = () => void connect()));
    return e;
  } finally {
    clearTimeout(waking);
  }
}

function openSettings() {
  const c = connection();
  (form.elements.namedItem("base") as HTMLInputElement).value = c.base;
  (form.elements.namedItem("token") as HTMLInputElement).value = c.token;
  document.getElementById("settings-msg")!.textContent = "";
  dialog.showModal();
}
status.onclick = openSettings;
form.addEventListener("submit", async (e) => {
  if ((e.submitter as HTMLButtonElement)?.value !== "save") return;
  e.preventDefault();
  saveConnection({ base: (form.elements.namedItem("base") as HTMLInputElement).value.trim(),
                   token: (form.elements.namedItem("token") as HTMLInputElement).value.trim() });
  const err = await connect();
  if (err) document.getElementById("settings-msg")!.innerHTML = esc(err.message);
  else dialog.close();
});

initTheme(document.getElementById("theme")!);
connect();
