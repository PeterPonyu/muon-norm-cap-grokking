# Figure generators (Paper A)

Pointers only. Generator scripts live with the lab figure pipeline; this warehouse
does not vendor compiled PDFs.

| Artifact | Generator |
|----------|-----------|
| `A_gap_normlaw`, `A_gap_normcap` | `figs/make_gap20260705_figs_r.R` |
| `A_normctl`, `A_floor`, `A_lmc`, `A_sink`, `A_plasticity`, `A_synth` | `figs/make_A_figs_r.R` |
| `A_normctl_timecourse`, `A_norm_discriminator` | `figs/make_A_new_figs_r.R` |
| `A_landscape` | `figs/make_A_landscape.py` |

## scheme

`A_scheme` is a documented schematic (`*_scheme`). It is emitted by the lab TikZ
pipeline, not a journal-flat `A_scheme.pdf` sitting beside `main.tex`.
