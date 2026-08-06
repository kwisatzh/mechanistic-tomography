# Step 0 v3: calibrated-AtP gain-budget sweep

This patch adds the `m_gain` experiment that was missing from v2.

The question is:

> After one backward pass gives the AtP map, how many finite aggregate forward measurements are needed to fit a single scalar gain that predicts held-out finite aggregate interventions?

This is the right baseline for the white-box regime: AtP gives the direction cheaply; a few finite measurements may calibrate the operating-scale gain.

## New arguments

```bash
--gain-budgets 1 2 4 8 16 32 64
--gain-budget-repeats 50
--no-gain-budget-sweep
```

Budget `0` is automatically included as raw uncalibrated AtP. Budgets larger than the available non-holdout aggregate measurements are skipped.

## Recommended local run

Run the packaged script from the HMM experiment directory:

```bash
cd experiments/hmm
cp step0_atp_patch_v3/attribution_vs_finite_step0.py .
```

Run a focused budget sweep on the most informative epsilons:

```bash
python attribution_vs_finite_step0.py \
  --run-dir runs/m4_baseline_seed7 \
  --device mps \
  --seed 7 \
  --epsilons 0.6 1.2 2 5 8 \
  --measurements 128 \
  --batch-size 256 \
  --mask-density 0.30 \
  --gain-budgets 1 2 4 8 16 32 64 96 \
  --gain-budget-repeats 50 \
  --outdir runs/m4_baseline_seed7/step0_v3_gain_budget_seed7
```

For three seeds:

```bash
for s in 7 8 9
 do
  python attribution_vs_finite_step0.py \
    --run-dir runs/m4_baseline_seed7 \
    --device mps \
    --seed $s \
    --epsilons 0.6 1.2 2 5 8 \
    --measurements 128 \
    --batch-size 256 \
    --mask-density 0.30 \
    --gain-budgets 1 2 4 8 16 32 64 96 \
    --gain-budget-repeats 50 \
    --outdir runs/m4_baseline_seed7/step0_v3_gain_budget_seed$s
 done
```

## Outputs

The new files are:

```text
step0_gain_budget_results.csv
step0_gain_budget_summary.csv
step0_gain_budget_summary_with_refs.csv
step0_gain_budget_r2_vs_m_gain.png
step0_gain_budget_gap_to_omp.png
```

Interpretation:

- If small `m_gain` such as 2--8 reaches OMP/ridge held-out R2, then in this harness the winner is AtP plus a tiny finite calibration set.
- If OMP/ridge stays ahead even after 16--64 calibration measurements, then finite tomography is learning more than a scalar gain correction.
- If the answer changes with epsilon, that is useful: it identifies where the finite effect map stops being a scalar-rescaled AtP map.
