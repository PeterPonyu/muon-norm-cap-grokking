# Muon-accelerated grokking: causal norm-cap test — code, data, pointer TeX

Warehouse: **https://github.com/PeterPonyu/muon-norm-cap-grokking**

Reproducibility archive for the causal Frobenius-norm cap on Muon grokking.
Concept DOI: [10.5281/zenodo.21020291](https://doi.org/10.5281/zenodo.21020291).

## Contents
- `experiments/<study>/` — runner / analysis code per sub-experiment.
- `experiments/results/` — per-run logs (JSON/JSONL) behind every reported number.
- `papers/A/main.tex` — full pointer manuscript (`\input{../figs/figpreamble.tex}`).
- `papers/figs/` — generators + JSON summaries. Compiled `tex/` and `vec/` are gitignored.
- `papers/FIGURE-INDEX.json` — portal figure contract (papers/-relative paths).
- `portal/` — Next.js instrument door (`output: 'export'`, `basePath` `/muon-norm-cap-grokking`). Excluded from Zenodo `git archive`.

## Reproducing
Committed per-run logs are the recorded outputs. To re-run a study from
scratch (GPU recommended): `python experiments/<study>/run_*.py`. Rebuild
figures with `papers/figs/PIPELINE.md`.

## Portal
Graphite instrument UI. Static export (no LaTeX, not a preprint):

```bash
bash portal/build.sh
```

Writes `portal/out/` and copies it to `_site/`. Project Pages (after deploy from `main`):
https://peterponyu.github.io/muon-norm-cap-grokking/

## License
Code: MIT (`LICENSE`). Result logs and figures: CC BY 4.0. See `CITATION.cff`.
