// Headless Chrome run of both modes with a fake microphone (a WAV file Chrome loops as capture); screenshots to <outdir>.
//   node e2e/e2e.mjs <outdir> <fake_mic.wav> [web url]
// E2E_API / E2E_TOKEN override the page's saved API connection (e.g. a token-protected or public API).
import puppeteer from "puppeteer-core";
const [S, MIC, BASE = "http://127.0.0.1:5173/"] = process.argv.slice(2);
const SAMPLES = new URL("../../bundle/samples/", import.meta.url).pathname;
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || "/usr/bin/google-chrome", headless: "new",
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-audio-capture=${MIC}`,
         "--autoplay-policy=no-user-gesture-required", "--window-size=1440,1000"] });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 1000 });
const logs = [];
page.on("console", (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on("pageerror", (e) => logs.push(`pageerror: ${e.message}`));
if (process.env.E2E_API) {
  const conn = JSON.stringify({ base: process.env.E2E_API, token: process.env.E2E_TOKEN || "" });
  await page.evaluateOnNewDocument((c) => localStorage.setItem("auscult-trust.connection", c), conn);
}
await page.goto(BASE, { waitUntil: "networkidle0" });
await page.waitForSelector("#real-file", { timeout: 30000 });
const input = await page.$("#real-file");
await input.uploadFile(SAMPLES + "187_1b1_Ll_sc_Meditron.wav");
await page.waitForFunction(() => !document.querySelector("#real-run").disabled, { timeout: 20000 });
await page.screenshot({ path: `${S}/real_1_loaded.png` });
await page.click("#real-run");
await page.waitForSelector(".results .cards", { timeout: 120000 });
await page.screenshot({ path: `${S}/real_2_results.png`, fullPage: true });
// hover a segment for tooltip
const box = await (await page.$(".wave-canvas")).boundingBox();
await page.mouse.move(box.x + 150, box.y + 60);
await page.mouse.move(box.x + 160, box.y + 62);
await new Promise((r) => setTimeout(r, 300));
console.log("tooltip", await page.evaluate(() => { const t = document.getElementById("tooltip"); return [t.hidden, t.innerText.slice(0, 80)]; }));
await page.screenshot({ path: `${S}/real_3_tooltip.png`, clip: { x: 0, y: 0, width: 1440, height: 700 } });
// demo
await page.click("#tab-demo");
await page.select("#demo-class", "crackle");
await page.waitForFunction(() => !document.querySelector("#demo-start").disabled, { timeout: 60000 });
await page.screenshot({ path: `${S}/demo_1_ready.png`, fullPage: true });
await page.click("#demo-start");
await new Promise((r) => setTimeout(r, 6000));
await page.screenshot({ path: `${S}/demo_2_recording.png` });
await page.waitForSelector(".compare .stats", { timeout: 120000 });
await page.screenshot({ path: `${S}/demo_3_done.png`, fullPage: true });
const align = await page.evaluate(() => document.querySelector(".compare")?.innerText.slice(0, 400));
console.log(align);
console.log(logs.join("\n"));
const failed = !/shifted by|unreliable/.test(align || "") || logs.some((l) => l.startsWith("pageerror"));
await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
await page.setViewport({ width: 900, height: 1100 });
await new Promise((r) => setTimeout(r, 500));
await page.screenshot({ path: `${S}/demo_4_dark_tablet.png` });
await browser.close();
if (failed) { console.error("E2E FAILED"); process.exit(1); }
console.log("E2E OK");
