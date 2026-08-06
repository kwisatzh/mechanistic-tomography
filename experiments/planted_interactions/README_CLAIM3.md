# Claim 3 planted-reach experiment

This package tests the regime that Step 0 says the NT/MI story needs: not a single smooth belief coordinate where AtP plus one scalar gain is enough, but planted mechanisms where scalar calibration fails.

The default plant includes four structures:

1. **Two heterogeneous finite-effect coordinates**: two main-effect components saturate at different epsilon scales, so a single scalar gain cannot calibrate AtP.
2. **Pure interaction pair**: two components have zero singleton/gradient effects but a nonzero joint effect.
3. **Redundant / self-repair pair**: another pair is visible only jointly, modeling a Hydra-style total-effect failure.
4. **Off-path confound**: a non-causal coordinate can be aliased with a causal coordinate if the measurement design is bad.

The script compares:

- `raw_atp`: infinitesimal first-order map.
- `scalar_cal_atp`: AtP times one gain fit from finite aggregate measurements.
- `multigain_atp`: stronger baseline with one gain per known main-effect coordinate.
- `finite_single`: exhaustive singleton finite patching.
- `subset_ridge`: random subset regression / KernelSHAP-style linear fit.
- `first_order_omp`: sparse first-order tomography.
- `lifted_omp`: sparse lifted tomography over main and pair terms.

## M4 run

From the package folder:

```bash
python claim3_planted_reach.py \
  --outdir runs/claim3_combined \
  --mode combined \
  --seeds 0 1 2 \
  --epsilons 0.6 1.2 2 5 8 \
  --measurements 256 \
  --holdout-measurements 512
```

This should run comfortably on an Apple M4 Max CPU. It does not need a GPU.

For a smoke test:

```bash
python claim3_planted_reach.py --quick --outdir runs/claim3_smoke
```

## Bad-design confound test

To deliberately alias an off-path confound with causal coordinate 0 in the train measurement design:

```bash
python claim3_planted_reach.py \
  --mode combined \
  --confound-correlation 0.85 \
  --outdir runs/claim3_bad_design_confound
```

A good designed-measurement story should show that independent masks suppress the confound, while a correlated/bad design causes false positives.

## Outputs

Each run writes:

- `claim3_results.csv`: method-level metrics.
- `claim3_coefficients.csv`: recovered main and pair coefficients.
- `claim3_summary.csv`: mean/std across seeds.
- `claim3_metadata.json`: plant and ground-truth details.
- `claim3_heldout_r2_by_epsilon.png`: prediction quality.
- `claim3_pair_recovery.png`: top-k pair recovery.
- `claim3_confound_false_positive.png`: off-path coefficient.
- `claim3_final_epsilon_bar.png`: final epsilon comparison.

## Expected interpretation

If the default combined run behaves as intended, scalar-calibrated AtP and finite singleton patching will perform poorly at high epsilon because they cannot represent pair interactions. `lifted_omp` should recover the planted pairs and dominate held-out subset prediction. That is the Claim 3 reach result: designed lifted measurements see what singleton and first-order designs cannot.

If `lifted_omp` fails, check:

- mask density: pair coverage scales roughly like density^2;
- `--measurements`: lifted recovery needs enough rows;
- `--lifted-omp-max-k`: must exceed the number of planted nonzero main/pair terms;
- `claim3_coefficients.csv`: verify whether the true pair terms are selected.
