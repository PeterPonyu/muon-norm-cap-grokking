import { expect, test, type Page } from "@playwright/test";

const ROUTES = ["./", "dose/", "floor/", "lmc/", "boundary/", "reproduce/"];

const FORBIDDEN = [
  /\.json\b/i,
  /\.tex\b/i,
  /\.md\b/i,
  /\.py\b/i,
  /\.r\b/i,
  /figure-index/i,
  /pipeline/i,
  /warehouse/i,
  /\bpaper\b/i,
  /\bjournal\b/i,
  /\bdocument\b/i,
  /\bmanuscript\b/i,
  /\bsubmission\b/i,
  /\bpreprint\b/i,
  /\bpublication\b/i,
  /\bvenue\b/i,
  /\bE1\b/,
  /main\.tex/i,
  /papers\/A/i,
  /\bA_/,
];

const SCIENCE = [
  /Cap/,
  /Dose/,
  /Floor/,
  /LMC/,
  /Boundary/,
  /dose response|ceiling series|hidden-norm|linear-mode|group ladder/i,
];

async function visibleText(page: Page): Promise<string> {
  return page.locator("body").innerText();
}

function leakHits(text: string): string[] {
  return FORBIDDEN.filter((re) => re.test(text)).map((re) => String(re));
}

test.describe("science door", () => {
  test("home loads public summaries and shows structure", async ({ page }) => {
    await page.goto("./");
    await expect(page.locator('[data-live="ready"]').first()).toBeVisible({ timeout: 20000 });
    await expect(page.locator("[data-live=questions]").first()).toBeVisible();
    const text = await visibleText(page);
    for (const re of SCIENCE) {
      expect(text, `missing ${re}`).toMatch(re);
    }
    expect(text).toMatch(/ceilings|panels|seeds|faces/i);
    expect(leakHits(text), `forbidden tokens in visible copy: ${leakHits(text).join(", ")}`).toEqual([]);
  });

  for (const route of ROUTES) {
    test(`route ${route} is 200 and leak-clean`, async ({ page }) => {
      const res = await page.goto(route);
      expect(res?.ok(), `${route} status`).toBeTruthy();
      await expect(page.locator('[data-live="ready"]').first()).toBeVisible({ timeout: 20000 });
      const text = await visibleText(page);
      expect(leakHits(text), `${route} leaks ${leakHits(text).join(", ")}`).toEqual([]);
    });
  }
});
