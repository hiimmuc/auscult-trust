// Conformal switch check: upload a sample, run, screenshot with the switch on and off.
//   E2E_API=<url> node e2e/toggle.mjs <outdir> [web url]
import puppeteer from "puppeteer-core";
const [S, BASE = "http://127.0.0.1:5173/"] = process.argv.slice(2);
const SAMPLES = new URL("../../bundle/samples/", import.meta.url).pathname;
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || "/usr/bin/google-chrome", headless: "new", args: ["--window-size=1440,1000"] });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 1000 });
const errs = [];
page.on("pageerror", (e) => errs.push(e.message));
if (process.env.E2E_API) {
  await page.evaluateOnNewDocument((c) => { localStorage.setItem("auscult-trust.connection", c); localStorage.removeItem("auscult-trust.conformal"); },
    JSON.stringify({ base: process.env.E2E_API, token: process.env.E2E_TOKEN || "" }));
}
await page.goto(BASE, { waitUntil: "networkidle0" });
await page.waitForSelector("#real-file", { timeout: 30000 });
await (await page.$("#real-file")).uploadFile(SAMPLES + "178_1b2_Pl_mc_AKGC417L.wav");
await page.waitForFunction(() => !document.querySelector("#real-run").disabled, { timeout: 20000 });
await page.click("#real-run");
await page.waitForSelector(".results .cards", { timeout: 120000 });
const state = () => page.evaluate(() => ({
  on: document.querySelector(".toolbar .switch input").checked,
  chips: document.querySelectorAll(".results td.sets").length,
  sets: document.querySelectorAll(".results .chip").length,
  triage: [...document.querySelectorAll(".results .triage")].map((e) => e.textContent),
  tags: document.querySelectorAll(".tag").length,
  legendHatch: !!document.querySelector(".legend .hatch"),
}));
const on = await state();
await page.screenshot({ path: `${S}/toggle_on.png`, fullPage: true });
await page.click(".toolbar .switch");
await new Promise((r) => setTimeout(r, 300));
const off = await state();
await page.screenshot({ path: `${S}/toggle_off.png`, fullPage: true });
console.log("on ", JSON.stringify(on)); console.log("off", JSON.stringify(off));
await page.click(".toolbar .switch");
await page.click("#tab-demo");
await new Promise((r) => setTimeout(r, 500));
console.log("sync", await page.evaluate(() => [...document.querySelectorAll(".toolbar .switch input")].map((i) => i.checked)));
await browser.close();
const ok = on.on && !off.on && on.chips > 0 && off.chips === 0 && off.sets === 0 && !off.legendHatch && on.legendHatch && !errs.length;
console.log(errs.length ? errs : "", ok ? "TOGGLE OK" : "TOGGLE FAIL");
process.exit(ok ? 0 : 1);
