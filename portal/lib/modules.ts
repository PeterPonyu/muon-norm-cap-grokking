export type ModuleId = "cap" | "dose" | "floor" | "lmc" | "boundary" | "reproduce";

export type FaceSpec = {
  id: string;
  title: string;
  role: string;
  panels: string[];
  relates: string[];
};

export const INSTRUMENT = {
  modules: 6,
  faces: 10,
  families: ["Muon", "AdamW", "SGDM"] as const,
  ceilings: ["∞", "3", "2", "1.5", "1"] as const,
  panelsPerFace: 4,
};

export const FACES: Record<string, FaceSpec> = {
  dose: {
    id: "dose",
    title: "Dose",
    role: "ceiling series",
    panels: [
      "dose response",
      "paired interval",
      "defloored control",
      "cross-task sensitivity",
    ],
    relates: ["feeds Cap"],
  },
  timecourse: {
    id: "timecourse",
    title: "Timecourse",
    role: "hidden-norm path",
    panels: [
      "hidden-norm trajectory",
      "fraction grokked",
      "per-seed timing",
      "validation trajectory",
    ],
    relates: ["meets the Cap ceiling"],
  },
  discriminator: {
    id: "discriminator",
    title: "Discriminator",
    role: "optimizer-family split",
    panels: [
      "ladder norm growth",
      "per-seed distribution",
      "optimizer-family control",
      "non-monotonicity sensitivity",
    ],
    relates: ["splits Cap by family"],
  },
  lmc: {
    id: "lmc",
    title: "LMC",
    role: "linear-mode connectivity",
    panels: [
      "barrier distribution",
      "optimizer comparison",
      "spawn extension",
      "seed-count robustness",
    ],
    relates: ["tests Cap basins"],
  },
  floor: {
    id: "floor",
    title: "Floor",
    role: "family trainability floor",
    panels: [
      "group ladder",
      "per-seed accuracy",
      "time-to-grok",
      "decay-free control",
    ],
    relates: ["bounds Boundary"],
  },
  sink: {
    id: "sink",
    title: "Sink",
    role: "activation concentration",
    panels: [
      "sink trajectories",
      "seed variation",
      "learning-rate control",
      "depth localization",
    ],
    relates: ["Boundary probe"],
  },
  plasticity: {
    id: "plasticity",
    title: "Plasticity",
    role: "retention split",
    panels: [
      "retention dissociation",
      "retention gap",
      "task support",
      "permuted-MNIST boundary",
    ],
    relates: ["Boundary probe"],
  },
  synth: {
    id: "synth",
    title: "Synthesis",
    role: "scope bound",
    panels: [
      "growth signature",
      "causal cap",
      "basin lock",
      "diagnostic boundary",
    ],
    relates: ["Boundary probe"],
  },
};

export const MODULE_FACES: Record<Exclude<ModuleId, "reproduce">, string[]> = {
  cap: ["dose", "timecourse", "discriminator", "lmc"],
  dose: ["dose"],
  floor: ["floor"],
  lmc: ["lmc"],
  boundary: ["sink", "plasticity", "synth"],
};

export const RELATIONS = [
  { from: "Dose", to: "Cap", via: "ceiling series" },
  { from: "Timecourse", to: "Cap", via: "hidden-norm path" },
  { from: "Discriminator", to: "Cap", via: "family split" },
  { from: "LMC", to: "Cap", via: "basin connectivity" },
  { from: "Floor", to: "Boundary", via: "trainability floor" },
  { from: "Sink", to: "Boundary", via: "activation probe" },
  { from: "Plasticity", to: "Boundary", via: "retention probe" },
];
