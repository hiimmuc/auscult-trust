// The primary API does not answer, the build's fallback does: the page must connect to the fallback and still analyse.
//   E2E_API=<dead or no-model API> node e2e/fallback.mjs <outdir> <web url of a build made with VITE_API_FALLBACK=<working API>>
import puppeteer from "puppeteer-core";
import assert from "node:assert/strict";
const [S, BASE] = process.argv.slice(2);
const SAMPLES = new URL("../../bundle/samples/", import.meta.url).pathname;
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || "/usr/bin/google-chrome", headless: "new" });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });
await page.evaluateOnNewDocument((c) => localStorage.setItem("auscult-trust.connection", c), JSON.stringify({ base: process.env.E2E_API, token: "" }));
await page.goto(BASE, { waitUntil: "networkidle0" });
await page.waitForSelector("#real-file", { timeout: 20000 });
assert.equal(await page.$eval("#api-status .txt", (e) => e.textContent), "Fallback API ready");
await (await page.$("#real-file")).uploadFile(SAMPLES + "187_1b1_Ll_sc_Meditron.wav");
await page.waitForFunction(() => !document.querySelector("#real-run").disabled, { timeout: 20000 });
await page.click("#real-run");
await page.waitForSelector(".results .cards", { timeout: 120000 });
await page.screenshot({ path: `${S}/fallback_results.png` });
await browser.close();
console.log("FALLBACK OK");
