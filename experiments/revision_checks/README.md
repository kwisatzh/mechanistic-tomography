# NTMI v21 targeted revision experiments

These public checks address the review concerns about Section 5.2 seed
stability and Section 5.5 IOI robustness. They compose the experiment harnesses
elsewhere in this repository and preserve the frozen outputs used by paper v2.

## Result 3: calibration factorial

`calibration_seed_factorial.py` composes the existing HMM Step-0 harness. It
fixes the trained checkpoint and belief-direction estimate, then crosses three
disjoint evaluation-batch seeds (17--19) with three measurement-design seeds
at `epsilon = 5, 8`.

Main result: at `epsilon = 8`, raw AtP has mean held-out R-squared `0.818`, a
scalar gain fitted on 72 masks raises it to `0.960`, and OMP reaches `0.969`.
At `epsilon = 5`, the corresponding values are `0.943`, `0.963`, and `0.965`.
The scalar correction does not match OMP uniformly across evaluation-batch and
design cells. This finding narrowed the paper's former matching claim.

Official outputs: `frozen/calibration_seed_factorial_disjoint/`.

## Result 6: IOI direct-effect robustness

`ioi_template_ablation_robustness.py` composes the packaged Stage 2c IOI
measurement primitives. It uses eight balanced lexical/order templates, 128
matched name pairs per template, the 21 direct anchor masks, and both
template-conditioned mean and zero ablation.

The predeclared gate passed. All 16 lexical/order-by-ablation cells order the
direct interactions `PE > PB > BE`. `ioi_crossed_bootstrap_reanalysis.py`
then resamples the four lexical frames, retains both orders, and uses one shared
name-pair resample per replicate. Its 95% intervals for `PE - PB` are
`[0.562, 0.934]` under mean ablation and `[0.207, 0.611]` under zero ablation.

Outputs: `frozen/ioi_template_ablation_robustness/`.

## Result 6: consistent-fold predictive audit

`ioi_predictive_consistent_seed.py` reuses the packaged Stage 2d analysis and
makes no model queries. It holds the fold assignment fixed for the point
estimate and 2,000 paired prompt bootstraps, then audits 200 alternative fold
assignments.

The paired Delta-MAE improvements over the count-additive control are:

- `PB`: `0.00829 [0.00642, 0.01009]`;
- `PE`: `0.12654 [0.11953, 0.13376]`;
- `BE`: `-0.00023 [-0.00149, 0.00107]`;
- all pairs: `0.14993 [0.14252, 0.15776]`.

`PE` improves under all 200 audited fold assignments, `PB` under 199, and `BE`
under 154. The paper therefore calls `PE` dominant and split-stable, `PB`
smaller, and `BE` split-sensitive.

Outputs: `frozen/ioi_predictive_consistent_seed/`.

## Figure regeneration

- `regenerate_result2_sample_efficiency.py` replaces the stale Result 2 panel
  with the saved OMP/ridge sparse-recovery sweep.
- `regenerate_saved_output_figures.py` rebuilds the observer, planted-reach,
  calibration-gain, and saved noise-sweep panels with paper-readable labels.
- `regenerate_claim3_budget_figure.py` corrects the planted support from five to
  four and labels `4 log(N/4) = 25.02` as orientation only.
- `regenerate_ioi_predictive_figure.py` plots the consistent-fold central 95%
  paired intervals used in the revised IOI figure.

Every experiment or figure script contains the requested authorship line:

> Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
