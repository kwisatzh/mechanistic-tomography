# IOI Stage 2d: per-pair decomposition with count-additive control

This is a post-processing analysis for the Stage 2c IOI primary-stratified run.
It does not rerun GPT-2. It consumes the Stage 2c files:

- `ioi_stage2c_subset_design.csv`
- `ioi_stage2c_head_records.csv`
- `ioi_stage2c_per_prompt_drops.csv`

It asks whether the Stage 2c `group_count_interaction` win is really cross-group interaction, and which pair term earns it.

## Models

- `additive_head`: intercept + one coefficient per candidate head.
- `count_additive`: `additive_head` plus nonlinear within-group count-bin controls for `nP`, `nB`, and `nE`; no cross terms.
- `count_plus_PB_count`: `count_additive + (nP/|P|)(nB/|B|)`.
- `count_plus_PE_count`: `count_additive + (nP/|P|)(nE/|E|)`.
- `count_plus_BE_count`: `count_additive + (nB/|B|)(nE/|E|)`.
- `count_plus_all_pairs`: `count_additive` plus all three pair count terms.

The primary statistic is paired bootstrap improvement over `count_additive`:

`Delta MAE = MAE(count_additive) - MAE(candidate)`.

Positive means the pair term improves prediction. A term is robust only if the 5th percentile of paired `Delta MAE` is above zero.

## Run

```bash
PYTHONPATH=. python scripts/ioi_stage2d_per_pair_decomposition.py \
  --input-run /path/to/ioi_stage2c_primary_stratified_mean_end \
  --outdir runs/ioi_stage2d_per_pair \
  --k-folds 5 \
  --bootstrap-repeats 300
```

For your current run:

```bash
PYTHONPATH=. python scripts/ioi_stage2d_per_pair_decomposition.py \
  --input-run runs/ioi_stage2c_primary_stratified_mean_end \
  --outdir runs/ioi_stage2d_per_pair
```

## Outputs

- `ioi_stage2d_report.md`
- `ioi_stage2d_fit_summary.csv`
- `ioi_stage2d_bootstrap_summary.csv`
- `ioi_stage2d_paired_delta_vs_count_additive.csv`
- `ioi_stage2d_paired_delta_vs_additive.csv`
- `ioi_stage2d_coefficients.csv`
- `ioi_stage2d_kfold_predictions.csv`
- `ioi_stage2d_diagnostics.json`
- plots for MAE and paired deltas.
