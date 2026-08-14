import { defineConfig } from "@playwright/test";

const live = (process.env.LIVE_URL ?? "https://peterponyu.github.io/muon-norm-cap-grokking").replace(
  /\/$/,
  "",
);

export default defineConfig({
  testDir: "./e2e",
  timeout: 30000,
  use: {
    baseURL: `${live}/`,
    viewport: { width: 1440, height: 900 },
  },
});
