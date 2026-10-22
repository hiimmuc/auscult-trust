// Dataset files: pick an ICBHI .wav together with its .txt, check the true labels, the stethoscope taken from the name, and the theme switch.
//   E2E_API=<url> node e2e/labels.mjs <outdir> <annotation.txt> [web url]      (the .wav comes from app/bundle/samples)
import puppeteer from "puppeteer-core";
const [S, TXT, BASE = "http://127.0.0.1:5173/"] = process.argv.slice(2);
const WAV = new URL("../../bundle/samples/187_1b1_Ll_sc_Meditron.wav", import.meta.url).pathname;
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || "/usr/bin/google-chrome", headless: "new", args: ["--window-size=1440,1000"] });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 1000 });
const errs = [];
page.on("pageerror", (e) => errs.push(e.message));
if (process.env.E2E_API) await page.evaluateOnNewDocument((c) => localStorage.setItem("auscult-trust.connection", c), JSON.stringify({ base: process.env.E2E_API, token: process.env.E2E_TOKEN || "" }));
await page.goto(BASE, { waitUntil: "networkidle0" });
await page.waitForSelector("#real-file", { timeout: 30000 });
await (await page.$("#real-file")).uploadFile(WAV, TXT);
await page.waitForFunction(() => !document.querySelector("#real-run").disabled, { timeout: 20000 });
const device = await page.$eval("#real-device", (e) => e.value);
const note = await page.$eval("#real-ann-note", (e) => e.textContent);
await page.click("#real-run");
await page.waitForSelector(".results .cards", { timeout: 120000 });
const truth = await page.$eval(".verdict .truth", (e) => e.textContent);
const col = await page.$$eval("table.segs th", (t) => t.some((x) => /Annotated/.test(x.textContent)));
await page.screenshot({ path: `${S}/labels_dark_or_auto.png`, fullPage: true });
// theme: Auto -> Light -> Dark -> Auto (each click shows the new state)
const themes = [];
for (let i = 0; i < 3; i++) {
  await page.click("#theme");
  themes.push(await page.evaluate(() => [document.documentElement.dataset.theme ?? "auto", getComputedStyle(document.body).backgroundColor]));
}
await page.click("#theme"); await page.screenshot({ path: `${S}/labels_light.png`, fullPage: true });
console.log({ device, note, truth, col, themes, errs });
await browser.close();
const ok = device === "Meditron" && /true labels/.test(note) && /Annotation/.test(truth) && col && !errs.length && themes[0][0] === "light" && themes[1][0] === "dark" && themes[2][0] === "auto";
console.log(ok ? "LABELS OK" : "LABELS FAIL"); process.exit(ok ? 0 : 1);
