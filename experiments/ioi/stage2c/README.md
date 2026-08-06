# IOI Stage 2c: primary-stratified head-level subset prediction

This package implements the single pre-registered follow-up to the Stage 2b IOI null.

Stage 2b fixed the underpowered 8-condition group-subset design by sampling random subsets of individual heads. It showed that the conditional IOI self-repair effect is real, but that a per-head additive observer already predicts most random subset interventions. Stage 2c asks the one mechanism-grounded follow-up question:

> If we stratify the subset design toward high primary-name-mover coverage, where self-repair is predicted to fire, do interaction terms improve held-out prediction?

## Head groups

Defaults follow the published GPT-2-small IOI taxonomy:

- `P`: primary Name Mover heads: `9.9,9.6,10.0`
- `B`: Backup Name Mover heads: `9.0,9.7,10.1,10.2,10.6,10.10,11.2,11.9`
- `E`: Negative Name Mover heads: `10.7,11.10`

## Primary design

The default sampler is `--sampling-mode primary_stratified`. It balances coverage over:

- exact primary ablation count `nP ∈ {0,1,2,3}`;
- backup count bins: `0`, `1`, `2–3`, `4–5`, `6–8`;
- exact negative-name-mover count `nE ∈ {0,1,2}`.

It still includes anchors: clean, all singletons, and all whole-group masks.

## Models compared

- `additive_head`: intercept + one coefficient per candidate head.
- `pb_occupancy_interaction`: additive + binary `P_B` occupancy term. Diagnostic.
- `pb_count_interaction`: additive + normalized count interaction `(nP/|P|)(nB/|B|)`.
- `group_count_interaction`: additive + normalized `P×B`, `P×E`, and `B×E` count interactions.

The primary comparison is `additive_head` vs `pb_count_interaction` and `group_count_interaction`, with success gated by paired bootstrap ΔMAE, not by a single point estimate.

## Primary run

```bash
cd ioi_stage2c_v0
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

PYTHONPATH=. python scripts/ioi_stage2c_primary_stratified.py \
  --outdir runs/ioi_stage2c_primary_stratified_mean_end \
  --device auto \
  --n-prompts 256 \
  --n-reference 512 \
  --n-subsets 240 \
  --sampling-mode primary_stratified \
  --batch-size 32 \
  --ablation mean \
  --positions end \
  --k-folds 5 \
  --bootstrap-repeats 200
```

## Quick smoke test

```bash
PYTHONPATH=. python scripts/ioi_stage2c_primary_stratified.py \
  --outdir runs/ioi_stage2c_quick \
  --device auto \
  --quick
```

## Outputs

Key files:

- `ioi_stage2c_report.md`
- `ioi_stage2c_fit_summary.csv`
- `ioi_stage2c_bootstrap_summary.csv`
- `ioi_stage2c_paired_delta_mae.csv`
- `ioi_stage2c_primary_count_errors.csv`
- `ioi_stage2c_diagnostics.json`
- `ioi_stage2c_prediction_scatter.png`
- `ioi_stage2c_bootstrap_mae.png`
- `ioi_stage2c_paired_delta_mae.png`
- `ioi_stage2c_primary_coverage_errors.png`

## Success criterion

This is a boundary-finding experiment. Report whichever outcome happens.

Strong success:

- `pb_count_interaction` or `group_count_interaction` has positive paired ΔMAE with `delta_mae_q05 > 0`.

Weak success:

- `p_delta_gt_0 ≥ 0.95` but `delta_mae_q05` crosses zero.

Null:

- paired ΔMAE overlaps or falls below zero. This means the per-head additive observer is still sufficient even in the primary-targeted subset regime.
