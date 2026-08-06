# IOI Stage 2b: head-level subset prediction

This package implements the head-level version of the IOI ObserverBench real-circuit test.

The group-level Stage 2 result showed strong non-additivity on GPT-2-small IOI, but only had 8 group-subset conditions. This Stage 2b script samples many subsets of individual heads from the published IOI head groups and evaluates whether interaction-aware observers generalize to held-out interventions.

## Head groups

Defaults:

- `P`: primary Name Mover heads: `9.9,9.6,10.0`
- `B`: Backup Name Mover heads: `9.0,9.7,10.1,10.2,10.6,10.10,11.2,11.9`
- `E`: Negative Name Mover heads: `10.7,11.10`

## Models compared

- `additive_head`: intercept + one coefficient per head.
- `pb_group_interaction`: additive head model + group occupancy term `P_B`.
- `group_interaction`: additive head model + group occupancy terms `P_B`, `P_E`, `B_E`.
- Optional: `head_pair_sparse_ridge` via `--include-head-pairs`, mainly diagnostic; not the primary comparison.

The main comparison is `additive_head` vs. `group_interaction`, with `pb_group_interaction` included to test whether the narrow primary-backup story is enough.

## Primary run

```bash
cd ioi_stage2b_v0
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

PYTHONPATH=. python scripts/ioi_stage2b_head_subset_prediction.py \
  --outdir runs/ioi_stage2b_mean_end \
  --device auto \
  --n-prompts 256 \
  --n-reference 512 \
  --n-subsets 160 \
  --batch-size 32 \
  --ablation mean \
  --positions end \
  --k-folds 5 \
  --bootstrap-repeats 200
```

## Outputs

Key files:

- `ioi_stage2b_report.md`
- `ioi_stage2b_fit_summary.csv`
- `ioi_stage2b_kfold_predictions.csv`
- `ioi_stage2b_subset_measurements.csv`
- `ioi_stage2b_coefficients.csv`
- `ioi_stage2b_bootstrap_summary.csv`
- `ioi_stage2b_prediction_scatter.png`
- `ioi_stage2b_mae_bar.png`
- `ioi_stage2b_group_occupancy_errors.csv`

## Success criterion

Primary success: interaction-aware model has lower held-out MAE than the singleton-additive model on random head-level subset interventions.

Strong success: gains are largest on held-out subsets containing heads from multiple groups, especially subsets with both P and B or P and E.
