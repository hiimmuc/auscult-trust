// The page against an API that is up but has no model: status chip, unavailable panel, retry, no horizontal scroll.
//   E2E_API=<url of an API without a model> node e2e/unavailable.mjs <outdir> [web url]
import puppeteer from "puppeteer-core";
import assert from "node:assert/strict";
const [S, BASE = "http://127.0.0.1:5173/"] = process.argv.slice(2);
const browser = await puppeteer.launch({ executablePath: process.env.CHROME || "/usr/bin/google-chrome", headless: "new" });
const page = await browser.newPage();
const logs = [];
page.on("pageerror", (e) => logs.push(e.message));
await page.evaluateOnNewDocument((c) => localStorage.setItem("auscult-trust.connection", c), JSON.stringify({ base: process.env.E2E_API, token: "" }));
for (const [w, h] of [[1440, 900], [390, 800]]) {
  await page.setViewport({ width: w, height: h });
  await page.goto(BASE, { waitUntil: "networkidle0" });
  await page.waitForSelector("[data-retry]", { timeout: 10000 });
  assert.equal(await page.$eval("#api-status .txt", (e) => e.textContent), "Model unavailable");
  assert.match(await page.$eval("#view-real", (e) => e.textContent), /No model bundle found/);
  assert.ok(await page.$eval("#view-real", (e) => !e.querySelector("#real-file")), "no analysis controls without a model");
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `horizontal overflow at ${w}`);
  await page.screenshot({ path: `${S}/unavailable_${w}.png` });
}
await page.focus("[data-retry]");
await page.keyboard.press("Enter");
await page.waitForFunction(() => document.querySelector("#api-status .txt").textContent === "Model unavailable" && document.querySelector("[data-retry]"));
assert.deepEqual(logs, []);
await browser.close();
console.log("UNAVAILABLE OK");
