import "./style.css";
import { api, connection, saveConnection, type ModelInfo } from "./api";
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
  status.title = `${connection().base} · click to change`;
}

async function connect() {
  setStatus("wait", "Connecting");
  try {
    const h = await api.health();
    if (h.status !== "ok") throw new Error("The API is still loading the model.");
    model = await api.model();
    setStatus("ok", "API ready");
    mounted = { real: false, demo: false };
    views.real.innerHTML = views.demo.innerHTML = "";
    show(location.hash === "#demo" ? "demo" : "real");
    return null;
  } catch (e: any) {
    setStatus("down", e.code === "unauthorized" ? "Token needed" : "API offline");
    const box = `<div class="offline"><h1>Connect to the API</h1>${errorBox(e)}<p>Set the API address and token, then try again.</p>
      <button class="btn primary" data-open>Connection settings</button></div>`;
    views.real.innerHTML = views.demo.innerHTML = box;
    document.querySelectorAll<HTMLButtonElement>("[data-open]").forEach((b) => (b.onclick = openSettings));
    return e;
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
