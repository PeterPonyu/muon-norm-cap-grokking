# Muon-accelerated grokking: causal norm-cap test — code & data

Repository: https://github.com/PeterPonyu/muon-norm-cap-grokking

Reproducibility archive: experiment code, per-run result logs, and a **pointer**
manuscript tree (`papers/`) that references untracked TikZ/PDF build products.
Venue-flat figure PDFs and compiled `main.pdf` are not included.

## Contents
- `experiments/<study>/` — runner / analysis code per sub-experiment.
- `experiments/results/` — per-run logs (JSON/JSONL) behind every reported number.
- `papers/` — pointer `main.tex`, `FIGURE-INDEX.json`, and summary JSON.
- `portal/` — instrument-chrome source (GitHub Pages is not enabled).

## Reproducing
The committed per-run logs are the recorded outputs. To re-run a study from
scratch (GPU recommended): `python experiments/<study>/run_*.py`. Runs are seeded
(seed lists appear in result-log filenames). Dependencies: Python 3.11+, PyTorch,
numpy. All inputs are synthetic and fully specified in the code, except large
standard datasets (MNIST / WikiText) which are not bundled.

## Scale-hardening additions (v1.3, 2026-07)
- `experiments/s5_normctl/run_20260708_capscale.py` + `experiments/results/s5_normctl_scale/` (112 runs):
  norm-cap dose-response across modulus rungs p in {97,251,337} (capscale, 52 runs) and the
  S5 width {128,256,512} x depth {2,4} x ceiling {inf,1} grid (capwd, 60 runs).
- `experiments/group_complexity/run_20260708_adam_nowd.py` + `experiments/results/group_complexity_nowd/`
  (30 runs): plain-Adam (lambda=0) taxonomy arm over the group ladder.

## License
Code: MIT (`LICENSE`). Result logs: CC BY 4.0. See `CITATION.cff`.
