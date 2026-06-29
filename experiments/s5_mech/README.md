# Direction 002 — S5 Mechanism Scaffold

Thin wrapper over the grokking codebase that runs the two-route mechanism
experiment on S5 (symmetric group composition) instead of modular arithmetic.

## Smoke check (no files written, <60 s)

```
python run_s5_mech.py --smoke
```

## Dry run (prints planned cells, no training)

```
python run_s5_mech.py --dry-run
```

## Real grid (run when ready — do NOT launch yet)

```
python run_s5_mech.py
```

Grid: optimizer ∈ {muon, adamw, sgdm} × init_scale ∈ {1.0, 3.0} × wd=0.01
× seeds 0-4. Results land in experiments/results/s5_mech/.

## Reference

See directions/002-s5-mech-two-route.md for the full research write-up.
