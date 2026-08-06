# Preregistration: Qwen population finite-effect response-surface gate

Status: design frozen before any Qwen intervention result is inspected.

Scientific status: this document records the design frozen before the completed
Qwen study was inspected. Paper v2 reports the result as a weak-ground-truth
extension. It is not a modern-scale replication of the paper's ground-truthed
validations and does not convert behavioral outcomes into mechanistic ground
truth.

## Question and scope

On a modern open-weight instruction model, does a lifted finite-effect response
surrogate predict held-out layerwise steering interventions better than a
calibrated additive surrogate, and does that improvement change fixed-budget
action selection?

The primary model is the pinned BF16 Qwen2.5-7B-Instruct revision in the config.
The experiment tests a declared layerwise harmful-versus-benign content-contrast
basis and a refusal-propensity endpoint. The contrast direction is not assumed
to be a refusal direction. This is not a circuit-discovery test, a claim that
refusal is one-dimensional, a prompt-state observer, a first use of feedback
control for activation steering, or a scaling study.

The locked outcomes are ground truth only for the declared behavioral response
functional `F(u)`: they are direct measurements of that metric under the chosen
actions. They are not ground truth for a represented variable, sparse support,
compiler-defined mechanism, causal head group, or semantic notion of refusal.
Accordingly, a positive result supports predictive transfer under weak ground
truth, not mechanistic identification.

## Estimand

For a distribution of prompts and an action vector `u` over eight fixed layers,
the estimand is the mean finite change in refusal propensity

    F(u) = E[margin(prompt; u) - margin(prompt; 0)].

The deterministic margin is the mean length-normalized log probability of four
fixed refusal stems minus the corresponding mean for four fixed neutral-
compliance stems. The exact strings, system prompt, tokenizer, model revision,
layers, and edit position are frozen in the config.

## Measurement design

Contrastive directions are computed once on a direction-construction split and
then frozen. At each selected layer, the actuator adds `u_l d_l` at the final
prompt token. Every direction has norm five percent of that layer's median clean
residual norm. The same directions and normalizations are used by every response
surrogate.

Random signed action rows use balanced densities of 0.25, 0.5, 0.75, and 1.0.
Each row is L2-normalized before its physical scale is applied. Calibration and
ridge-selection validation use scales 0.5 and 1.0; the locked test alone uses
the held-out within-range scale 0.75. Zero and signed singleton actions are
calibration-only. Every support/sign pattern is unique across action splits.
Prompt IDs, normalized prompt texts, and families cannot cross construction,
fitting, locked-test, or collateral splits where a shared population could leak.

Thirty-two HarmBench behaviors and two complete XSTest families are reserved
exclusively for the engineering pilot. The full primary split excludes that
reserve and uses 32 harmful and 32 benign construction prompts, 112 harmful
surrogate-fitting prompts, 224 locked harmful test prompts, and 150 benign
XSTest collateral prompts. Pilot and full prompt families are therefore
disjoint. Across the reserve and full design, every canonical HarmBench behavior
is assigned at most once. Any StrongREJECT or JailbreakBench transfer study is a
separately frozen confirmatory run.

## Response-surrogate families

- Calibrated additive: intercept, eight action coordinates, eight per-site
  self-curvature terms `u_i^2`, density, density-squared, and total action scale,
  fit by ridge regression.
- Lifted NT: the identical additive features plus all preregistered cross-site
  pair products `u_i u_j`, fit by ridge regression.

Ridge strength is chosen separately for each family on validation actions and
then the model is refit on calibration plus validation. Both families therefore
receive the same self-curvature nuisance terms; the lifted family differs only
by cross-site products. A raw contrastive score is not the headline baseline.

## Primary test and gate

The locked-test statistic is

    Delta_MAE = MAE(calibrated additive) - MAE(lifted).

Predictions are frozen before locked outcomes are inspected. A two-way paired
bootstrap resamples locked prompt families and locked actions, using identical
draws for both surrogates. The reported point estimate is the observed locked
contrast, not the bootstrap mean. The Stage-A interaction gate passes only if
the lower bound of the 95-percent relative-improvement interval is strictly
above five percent. MAE, RMSE, held-out
R-squared, calibration, density slices, and coefficient stability are secondary
diagnostics.

If the gate does not pass, the default conclusion is "no lifted advantage
detected" on this model/task/basis/scale/design. Practical equivalence is
reported only when the entire relative-improvement interval is contained in
[-5%, +5%]. An upper bound below +5% can rule out a practically meaningful
lifted advantage without establishing equivalence. None of these outcomes is a
claim that interactions are absent or that mechanistic tomography has failed.

## Secondary fixed-budget action selector

For each preregistered positive setpoint, the selector chooses from the same
locked, equal-norm action library by minimizing predicted squared target error.
Setpoints are quantiles of positive calibration effects; if no positive effect
exists, no selector result is reported. Only the response surrogate changes.
The plant, directions, action library, setpoints, prompts, and seeds are shared.

Outcomes are actual setpoint error on harmful prompts and change in refusal
margin on held-out benign prompts. The actual-response oracle is a diagnostic
ceiling only. This stage supports a narrow surrogate-to-decision claim, not a
claim of online, closed-loop, or PID control.

## Run order

1. CPU-only synthetic falsifier.
2. M4 0.5B execution smoke; no scientific claim.
3. H200 7B pilot on the reserved prompts; use it only to validate throughput,
   sign, and nondegeneracy.
4. Freeze any operational correction that does not use locked outcomes.
5. Run the disjoint full locked 7B design once.

The full run may be stopped for a documented engineering failure, degenerate
clean margin, or unsafe activation norm, but not because an interim effect is
small. A positive gate triggers preregistered robustness work: leave-one-stem-out
scoring, blinded generated-response or human validation, random-direction or
layer controls, a generic nonlinear capacity control, and collateral-effect
intervals at matched achieved targets. These secondary checks do not replace
the deterministic primary endpoint.
