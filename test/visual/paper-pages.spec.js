const { test, expect } = require("@playwright/test");

const base = process.env.PAPER_PAGES_BASE_URL || "http://127.0.0.1:4000/al-folio";

for (const [slug, title, arxiv] of [
  ["half-truths", "Half-Truths", "2602.23906"],
  ["conformal-elo", "Conformal Elo", "2606.13221"],
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
    await expect(page.getByRole("link", { name: "Read the paper" })).toHaveAttribute("href", `https://arxiv.org/pdf/${arxiv}`);
    await expect(page.locator('meta[property="og:url"]')).toHaveAttribute("content", new RegExp(`/${slug}/$`));
    const preview = await page.locator('meta[property="og:image"]').getAttribute("content");
    expect((await page.request.get(`${base}/assets/img/papers/${slug}/social.png`)).ok()).toBeTruthy();
    expect(preview).toMatch(new RegExp(`/assets/img/papers/${slug}/social.png$`));
    await page.getByRole("button", { name: "Copy citation" }).click();
    await expect(page.locator("#copy-status")).toHaveText("Citation copied to clipboard.");
    expect(await page.evaluate(() => window.copiedCitation)).toContain(arxiv);
    await page.getByText(/What does (this result measure|the uncertainty guarantee mean)/).click();
    await expect(page.locator("details")).toHaveAttribute("open", "");
    for (const width of [1440, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
    }
    for (const img of await page.locator("img").all()) {
      await img.scrollIntoViewIfNeeded();
      await expect.poll(() => img.evaluate((node) => node.complete && node.naturalWidth > 0)).toBeTruthy();
    }
    expect(errors).toEqual([]);
  });
}

test("Soft-Elo demo keeps direction, ties, and strength of preference", async ({ page }) => {
  await page.goto(`${base}/conformal-elo/`);
  const slider = page.getByLabel("Judge score difference, A − B");
  await expect(slider).toBeEnabled();
  await expect(page.locator("#win-probability")).toHaveText("62.2%");
  for (const [gap, probability] of [
    [0, "50.0%"],
    [6, "95.3%"],
    [-6, "4.7%"],
  ]) {
    await slider.fill(String(gap));
    await expect(page.locator("#win-probability")).toHaveText(probability);
  }
});

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
