const { test, expect } = require("@playwright/test");

const base = process.env.PAPER_PAGES_BASE_URL || "http://127.0.0.1:4000/al-folio";

for (const [slug, title, arxiv] of [
  ["half-truths", "Half-Truths Break Similarity-Based Retrieval", "2602.23906"],
  ["conformal-elo", "From Uncertain Judgments to Calibrated Rankings: Conformal Elo Estimation for LLM Evaluation", "2606.13221"],
]) {
  test(`${title}: resources, sharing, citation, and responsive layout`, async ({ page }) => {
    await page.addInitScript(() => {
      Object.defineProperty(navigator, "clipboard", {
        value: {
          writeText: async (text) => {
            window.copiedCitation = text;
          },
        },
      });
    });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(`${base}/${slug}/`);
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(title);
    await expect(page.getByRole("link", { name: "Paper", exact: true })).toHaveAttribute("href", `https://arxiv.org/pdf/${arxiv}`);
    await expect(page.locator('meta[property="og:url"]')).toHaveAttribute("content", new RegExp(`/${slug}/$`));
    const preview = await page.locator('meta[property="og:image"]').getAttribute("content");
    expect((await page.request.get(`${base}/assets/img/papers/${slug}/social.png`)).ok()).toBeTruthy();
    expect(preview).toMatch(new RegExp(`/assets/img/papers/${slug}/social.png$`));
    await page.getByRole("button", { name: "Copy citation" }).click();
    await expect(page.locator("#copy-status")).toHaveText("Citation copied to clipboard.");
    expect(await page.evaluate(() => window.copiedCitation)).toContain(arxiv);
    await expect(page.getByRole("heading", { name: "Abstract", exact: true })).toBeVisible();
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
    }
    if (slug === "half-truths") await expect(page.getByRole("heading", { name: "Compare all 16 models", exact: true })).toBeVisible();
    for (const img of await page.locator("img").all()) {
      await img.scrollIntoViewIfNeeded();
      await expect.poll(() => img.evaluate((node) => node.complete && node.naturalWidth > 0)).toBeTruthy();
    }
    expect(errors).toEqual([]);
  });
}

test("Citation copy failure remains usable", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      value: {
        writeText: async () => {
          throw new Error("Clipboard unavailable");
        },
      },
    });
  });
  await page.goto(`${base}/half-truths/`);
  await page.getByRole("button", { name: "Copy citation" }).click();
  await expect(page.locator("#copy-status")).toHaveText("Could not copy. Select and copy the citation below.");
  await expect(page.locator("#bibtex")).toContainText("kargi2026halftruths");
});

test("Short project pages preserve results without JavaScript", async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto(`${base}/half-truths/`);
  await expect(page.getByRole("region", { name: "Half-truth accuracy summary" })).toContainText("76.4");
  await expect(page.locator("#half-image")).toBeVisible();
  await page.goto(`${base}/conformal-elo/`);
  await expect(page.locator("#abstract")).toContainText("17.9 Elo mean absolute error");
  await expect(page.locator("#elo-pipeline")).toBeVisible();
  await expect(page.locator("#elo-rating")).toHaveText("1,218");
  await expect(page.locator(".elo-example-controls")).toBeHidden();
  await expect(page.locator("#method math")).toHaveCount(5);
  await context.close();
});

test("Measured examples autoplay and retain exact scores, pause, and reduced motion", async ({ page }) => {
  await page.goto(`${base}/half-truths/`);
  await page.locator("#half-demo").scrollIntoViewIfNeeded();
  await expect(page.getByRole("button", { name: "Pause animation", exact: true })).toBeVisible();
  const examples = await (await page.request.get(`${base}/assets/img/papers/half-truths/examples.json`)).json();
  expect(examples[1].item_id).toBe(967);
  expect(examples[1].full).toBe("cook is in kitchen");
  expect(examples[1].half).toBe("cook is outside kitchen");
  // It advances without pressing Play.
  await expect(page.getByLabel("Correct detail", { exact: true })).toBeChecked({ timeout: 7000 });
  await expect(page.locator("#half-clip-score")).toHaveText(examples[0].scores.CLIP.s_full.toFixed(3));
  await page.getByRole("button", { name: "Pause animation", exact: true }).click();
  await page.waitForTimeout(4500);
  await expect(page.getByLabel("Correct detail", { exact: true })).toBeChecked();
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(page.locator("#half-pause")).toBeHidden();
  for (const [index, example] of examples.entries()) {
    await page.getByLabel("Example", { exact: true }).selectOption(String(index));
    for (const [kind, label] of [
      ["anchor", "Original caption"],
      ["full", "Correct detail"],
      ["half", "False detail"],
    ]) {
      await page.getByLabel(label, { exact: true }).check();
      await expect(page.locator("#half-caption-text")).toHaveText(example[kind]);
      await expect(page.locator("#half-clip-score")).toHaveText(example.scores.CLIP[`s_${kind}`].toFixed(3));
      await expect(page.locator("#half-cs-score")).toHaveText(example.scores["CS-CLIP"][`s_${kind}`].toFixed(3));
    }
    expect(example.scores.CLIP.s_half).toBeGreaterThan(example.scores.CLIP.s_anchor);
    expect(example.scores["CS-CLIP"].s_full).toBeGreaterThan(example.scores["CS-CLIP"].s_anchor);
    expect(example.scores["CS-CLIP"].s_anchor).toBeGreaterThan(example.scores["CS-CLIP"].s_half);
  }
});

test("Soft-Elo pipeline connects scores, probabilities, fitted ratings and intervals", async ({ page }) => {
  await page.goto(`${base}/conformal-elo/`);
  await expect(page.locator("#method mjx-container")).toHaveCount(5);
  await expect(page.locator("#elo-rating")).toHaveText("1,218");
  await expect(page.locator("#elo-p-1")).toHaveText("52.5%");
  await expect(page.locator("#elo-fitted-1")).toHaveText("52.6%");
  await expect(page.locator("#elo-band-label")).toHaveText("1,218 ± 50 Elo");
  await page.getByLabel("Clear preference", { exact: true }).check();
  await expect(page.locator("#elo-p-1")).toHaveText("85.8%");
  await expect(page.locator("#elo-fitted-1")).toHaveText("86.6%");
  await expect(page.locator("#elo-rating")).toHaveText("1,524");
  await expect(page.locator("#elo-band-label")).toHaveText("1,524 ± 50 Elo");
  await page.getByLabel("95%", { exact: true }).check();
  await expect(page.locator("#elo-band-label")).toHaveText("1,524 ± 70 Elo");
  await expect(page.locator("#elo-quantile")).toHaveText("19th of 19 errors → q = 2.8");
  await page.locator("#elo-beta").fill("0");
  await expect(page.locator("#elo-p-1")).toHaveText("50.0%");
  await expect(page.locator("#elo-rating")).toHaveText("1,200");
  await page.getByRole("button", { name: "Reset example" }).click();
  await expect(page.locator("#elo-rating")).toHaveText("1,218");
  await expect(page.locator("#elo-band-caption")).toContainText("width 100 Elo");
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.getByLabel("Clear preference", { exact: true }).check();
  await expect(page.locator("#elo-rating")).toHaveText("1,524");
});

test("All-model table and charts match the release CSV", async ({ page }) => {
  await page.goto(`${base}/half-truths/`);
  await expect(page.getByRole("heading", { name: "Compare all 16 models", exact: true })).toBeVisible();
  const tableRows = page.locator("#half-results tbody tr");
  await expect(tableRows).toHaveCount(16);
  const csvResponse = await page.request.get(`${base}/assets/img/papers/half-truths/results.csv`);
  expect(csvResponse.ok()).toBeTruthy();
  const csv = (await csvResponse.text())
    .trim()
    .split(/\r?\n/)
    .slice(1)
    .map((line) => line.split(","));
  expect(csv).toHaveLength(48);
  await expect(page.locator('link[href*="bulma"]')).toHaveCount(0);
  for (const [metric, label] of [
    [0, "Overall"],
    [1, "Entities"],
    [2, "Relations"],
  ]) {
    await page.getByLabel(label, { exact: true }).check();
    for (const bar of await page.locator(".result-bar-row").all()) {
      const source = await bar.getAttribute("data-dataset");
      const model = await bar.getAttribute("data-model");
      const record = csv.find((row) => row[0] === source && row[1] === model);
      await expect(bar.locator("output")).toHaveText(Number(record[3 + metric]).toFixed(1));
      expect(await bar.locator(".result-bar").evaluate((el) => parseFloat(el.style.width))).toBeCloseTo(Number(record[3 + metric]), 4);
    }
  }

  await page.getByLabel("Compare", { exact: true }).selectOption("0");
  for (let model = 0; model < 16; model++) {
    for (let dataset = 0; dataset < 3; dataset++) {
      const record = csv[model * 3 + dataset];
      expect(Number(record[2])).toBe([509, 454, 474][dataset]);
      await expect(tableRows.nth(model).locator("td").nth(dataset)).toHaveText(Number(record[3]).toFixed(1));
    }
  }
  await page.getByLabel("Compare", { exact: true }).selectOption("2");
  await expect(page.getByLabel("Relations", { exact: true })).toBeChecked();
  await expect(page.locator("#half-table-caption")).toHaveText("Relation accuracy (%)");
  await expect(tableRows.last().locator("td").first()).toHaveText("79.2");
  await page.setViewportSize({ width: 320, height: 900 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});

test("Half-Truths results work without JavaScript", async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto(`${base}/half-truths/`);
  await expect(page.getByRole("region", { name: "Half-truth accuracy summary" })).toContainText("76.4");
  await expect(page.locator("#half-image")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Compare all 16 models", exact: true })).toBeVisible();
  await expect(page.locator("#half-results tbody tr")).toHaveCount(16);
  await context.close();
});
