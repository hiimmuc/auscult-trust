// Real mode, live recording: record from Chrome's fake microphone, check the Siri view, the preprocess switches and an analysis.
//   E2E_API=<url> node e2e/record.mjs <outdir> <fake_mic.wav> [web url]
import puppeteer from "puppeteer-core";
const [S, MIC, BASE = "http://127.0.0.1:5173/"] = process.argv.slice(2);
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || "/usr/bin/google-chrome", headless: "new",
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-audio-capture=${MIC}`, "--window-size=1440,1000"] });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 1000 });
const errs = [];
page.on("pageerror", (e) => errs.push(e.message));
if (process.env.E2E_API) {
  await page.evaluateOnNewDocument((c) => localStorage.setItem("auscult-trust.connection", c), JSON.stringify({ base: process.env.E2E_API, token: process.env.E2E_TOKEN || "" }));
}
await page.goto(BASE, { waitUntil: "networkidle0" });
await page.waitForSelector("#real-rec", { timeout: 30000, visible: false });
await page.click('.source [data-v="mic"]');
await page.screenshot({ path: `${S}/rec_0_idle.png` });
await page.click("#real-rec");
await page.waitForSelector(".live:not([hidden])");
await new Promise((r) => setTimeout(r, 7000));
const time = await page.$eval(".live-time", (e) => e.textContent);
await page.screenshot({ path: `${S}/rec_1_live.png` });
await page.click("#live-stop");
await page.waitForFunction(() => !document.querySelector("#real-run").disabled, { timeout: 20000 });
const raw = await page.$eval(".panel .sub", (e) => e.textContent);
await page.screenshot({ path: `${S}/rec_2_review.png` });
await page.click("#prep-state"); // the Preprocess group is collapsed
await page.click("#real-noise + .track"); await page.click("#real-trim + .track");
await page.waitForFunction(() => /noise reduced/.test(document.querySelector(".panel .sub").textContent));
const cleaned = await page.$eval(".panel .sub", (e) => e.textContent);
await page.click("#real-run");
await page.waitForSelector(".results .cards", { timeout: 120000 });
await page.screenshot({ path: `${S}/rec_3_results.png`, fullPage: true });
// discard keeps the previous take
await page.click("#real-rec"); await page.waitForSelector(".live:not([hidden])");
await page.click("#live-discard");
await page.waitForSelector(".work:not([hidden])");
console.log({ time, raw, cleaned, results: await page.$$eval(".results .card", (c) => c.length), errs });
await browser.close();
if (errs.length || !/recorded live/.test(raw) || !/noise reduced/.test(cleaned) || parseFloat(time.split(":")[1]) < 5) { console.error("RECORD FAILED"); process.exit(1); }
console.log("RECORD OK");
