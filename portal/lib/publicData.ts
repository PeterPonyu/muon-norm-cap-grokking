export const BASE_PATH = "/muon-norm-cap-grokking";

/** Public JSON the Pages export already ships. Paths are fetch-only — never render them. */
const INDEX_CANDIDATES = ["figures.json", "FIGURE-INDEX.json"] as const;

const SUMMARY_BY_FACE: Record<string, string> = {
  dose: "figs/summaries/A_gap_normcap.json",
  timecourse: "figs/summaries/A_normctl_timecourse.json",
  discriminator: "figs/summaries/A_norm_discriminator.json",
  lmc: "figs/summaries/A_lmc.json",
  floor: "figs/summaries/A_floor.json",
  sink: "figs/summaries/A_sink.json",
  plasticity: "figs/summaries/A_plasticity.json",
  synth: "figs/summaries/A_synth.json",
  cap: "figs/summaries/A_normctl.json",
};

export type LiveFace = {
  questions: string[];
  panelCount: number | null;
  families: string[];
  ceilings: string[];
  seeds: number | null;
};

export type LiveScience = {
  faceCount: number;
  faces: Record<string, LiveFace>;
};

function publicUrl(rel: string): string {
  return `${BASE_PATH}/data/${rel.replace(/^\/+/, "")}`;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

const FAMILY_LABEL: Record<string, string> = {
  muon: "Muon",
  adamw: "AdamW",
  sgdm: "SGDM",
};

const CEILING_LABEL: Record<string, string> = {
  kinf: "∞",
  k3: "3",
  k2: "2",
  k1p5: "1.5",
  k1: "1",
};

async function readJson(url: string): Promise<unknown | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

function collectFamilies(node: unknown, into: Set<string>, depth = 0): void {
  if (depth > 6 || node == null) return;
  if (Array.isArray(node)) {
    for (const item of node) collectFamilies(item, into, depth + 1);
    return;
  }
  const rec = asRecord(node);
  if (!rec) return;
  if (typeof rec.optimizer === "string") {
    const label = FAMILY_LABEL[rec.optimizer.toLowerCase()];
    if (label) into.add(label);
  }
  if (asRecord(rec.by_optimizer)) {
    for (const key of Object.keys(rec.by_optimizer as object)) {
      const label = FAMILY_LABEL[key.toLowerCase()];
      if (label) into.add(label);
    }
  }
  if (asRecord(rec.grok_counts)) {
    const first = Object.values(rec.grok_counts as object)[0];
    const row = asRecord(first);
    if (row) {
      for (const key of Object.keys(row)) {
        const label = FAMILY_LABEL[key.toLowerCase()];
        if (label) into.add(label);
      }
    }
  }
  for (const value of Object.values(rec)) {
    if (value && typeof value === "object") collectFamilies(value, into, depth + 1);
  }
}

function collectCeilings(node: unknown, into: string[], depth = 0): void {
  if (depth > 6 || node == null) return;
  const rec = asRecord(node);
  if (!rec) return;
  const rows = asRecord(rec.rows);
  if (rows) {
    for (const key of Object.keys(rows)) {
      const label = CEILING_LABEL[key];
      if (label && !into.includes(label)) into.push(label);
    }
  }
  for (const value of Object.values(rec)) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      collectCeilings(value, into, depth + 1);
    }
  }
}

function collectSeeds(node: unknown): number | null {
  const rec = asRecord(node);
  if (!rec) return null;
  const panels = Array.isArray(rec.panels) ? rec.panels : [];
  for (const panel of panels) {
    const metrics = asRecord(asRecord(panel)?.metrics);
    if (!metrics) continue;
    for (const block of Object.values(metrics)) {
      const rows = asRecord(asRecord(block)?.rows);
      if (!rows) continue;
      for (const row of Object.values(rows)) {
        const n = asRecord(row)?.n;
        if (typeof n === "number" && Number.isFinite(n)) return n;
      }
    }
    const byOpt = asRecord(metrics.by_optimizer);
    if (byOpt) {
      for (const arm of Object.values(byOpt)) {
        const n = asRecord(arm)?.n;
        if (typeof n === "number" && Number.isFinite(n)) return n;
      }
    }
  }
  return null;
}

export function faceFromSummary(raw: unknown): LiveFace {
  const root = asRecord(raw);
  const panels = Array.isArray(root?.panels) ? root.panels : [];
  const questions: string[] = [];
  for (const panel of panels) {
    const question = asRecord(panel)?.question;
    if (typeof question === "string" && question.trim()) questions.push(question.trim());
  }
  const families = new Set<string>();
  collectFamilies(raw, families);
  const ceilings: string[] = [];
  collectCeilings(raw, ceilings);
  const panelCount = typeof root?.panel_count === "number" ? root.panel_count : panels.length || null;
  return {
    questions,
    panelCount,
    families: ["Muon", "AdamW", "SGDM"].filter((name) => families.has(name)),
    ceilings,
    seeds: collectSeeds(raw),
  };
}

export async function loadLiveScience(): Promise<LiveScience | null> {
  let index: unknown = null;
  for (const name of INDEX_CANDIDATES) {
    index = await readJson(publicUrl(name));
    if (index) break;
  }
  const figures = asRecord(index)?.figures;
  const faceCount = Array.isArray(figures) ? figures.length : 0;

  const entries = await Promise.all(
    Object.entries(SUMMARY_BY_FACE).map(async ([id, rel]) => {
      const raw = await readJson(publicUrl(rel));
      return [id, raw] as const;
    }),
  );

  const faces: Record<string, LiveFace> = {};
  let loaded = 0;
  for (const [id, raw] of entries) {
    if (!raw) continue;
    faces[id] = faceFromSummary(raw);
    loaded += 1;
  }
  if (!index && loaded === 0) return null;
  return { faceCount: faceCount || loaded, faces };
}
