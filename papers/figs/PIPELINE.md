# Figure pipeline (warehouse pointer)

Figures on this remote are **pointers**, not venue-flat PDFs.

| Tier | Build product (untracked) | Manuscript include |
|------|---------------------------|--------------------|
| plot | `papers/figs/tex/<name>.tex` | `\figtikz{<name>}` |
| heatmap | `papers/figs/vec/<name>.pdf` | preamble-routed `\includegraphics{<name>.pdf}` |

`tex/` and `vec/` are gitignored. Do not commit `A_*.pdf` next to `papers/A/main.tex`.
The portal reads `papers/FIGURE-INDEX.json` and `papers/figs/summaries/*.json`.
