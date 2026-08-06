# Qwen-2.5-7B follow-up for Mechanistic Tomography

This package implements the modern-open-weight follow-up to *Mechanistic
Tomography: Designed Measurement for Control-Oriented Interpretability*. It
tests the framework's measurement decision on Qwen-2.5-7B and includes a
public-facing Colab notebook, a frozen result bundle, and a resumable GPU path.

Paper 1 validates against ground truth: analytic posteriors, planted supports,
compiler labels, and documented causal head groups. Qwen2.5 supplies no
comparable ground truth. This follow-up can test whether two response surrogates
predict a directly measured behavioral intervention surface, but it cannot
establish that either surrogate identifies the represented state, support,
circuit, or mechanism. Scale does not repair that validation gap.

The experiment fixes a layerwise harmful-versus-benign content-contrast basis
and physical action library. It compares a calibrated additive population
finite-effect surrogate with an NT lifted surrogate on held-out prompts and
held-out intervention combinations. A secondary fixed-budget selector consumes
each surrogate. This Stage-A experiment is not a prompt-state observer, a
feedback loop, or a PID controller.

## What is ready

- CPU-only planted-interaction falsifier for design, fitting, two-way paired
  bootstrap, and action-selection plumbing.
- Pinned M4 execution-smoke config for Qwen2.5-0.5B-Instruct.
- Pinned A100/H100-class pilot and full configs for Qwen2.5-7B-Instruct.
- Disjoint pilot/full prompt profiles from pinned HarmBench and XSTest sources.
- Resumable direction construction and finite-effect measurement with strict
  cache, source, environment, and artifact fingerprints.
- Deterministic refusal-stem margin with no LLM judge in the primary endpoint.
- Prompt-family-by-action paired bootstrap, a five-percent practical-effect
  gate, and explicit null/equivalence interpretations.

## Completed result

The full A100 run measured 401 designed actions and evaluated 128 held-out
actions over 224 held-out prompts. The calibrated additive observer reached
MAE 0.003790 and R2 0.9829. The lifted observer reached MAE 0.003801 and R2
0.9835. The estimated relative lifted MAE improvement was -0.29%, with a 95%
paired two-way-bootstrap interval of [-3.56%, 5.65%]. No lifted advantage was
detected, and the interval narrowly failed to rule out the preregistered 5%
practical threshold.

This is a positive result for the measurement procedure: finite calibration
made the additive map adequate on the declared Qwen surface, so the held-out
test did not justify the more expensive interaction family. It is not evidence
that interactions never matter in larger models.

Start with
[`notebooks/mechanistic_tomography_qwen_colab.ipynb`](notebooks/mechanistic_tomography_qwen_colab.ipynb).
Its default path reproduces the result from frozen measurements without loading
Qwen or using a GPU. The optional path reruns the measurements with Drive-backed
checkpoints.

## Install and verify the CPU path

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
mechtomo toy --outdir artifacts/generated/toy_smoke
~~~

The toy run should report a positive held-out
`calibrated_additive_mae_minus_lifted_mae` interval because it contains planted
pair terms. That validates the pipeline, not a claim about Qwen or Paper 1.

## Prepare locked prompt data

Download the official HarmBench behavior CSV and XSTest prompt CSV at the pinned
commits listed in `docs/DATA.md`. The Colab notebook contains the exact pilot and
full preparation commands. Preparation assigns each prompt family to at most one
split/profile and writes a checksum manifest next to each JSONL.

## Rerun the follow-up

Every expensive stage is resumable. On an A100/H100-class GPU:

~~~bash
pip install -e '.[qwen,analysis]'

mechtomo qwen \
  --config configs/qwen2_5_7b_h200_pilot.json \
  --outdir artifacts/runs/qwen2_5_7b_h200_pilot \
  --stage directions

mechtomo qwen \
  --config configs/qwen2_5_7b_h200_pilot.json \
  --outdir artifacts/runs/qwen2_5_7b_h200_pilot \
  --stage measure

mechtomo qwen \
  --config configs/qwen2_5_7b_h200_pilot.json \
  --outdir artifacts/runs/qwen2_5_7b_h200_pilot \
  --stage analyze
~~~

The measurement stage checkpoints after every action. Analysis is CPU-only and
can be rerun without loading Qwen.

## Primary interpretation

The primary statistic is held-out

`MAE(calibrated additive) - MAE(lifted)`.

The interaction gate passes only if the lower bound of its two-way-bootstrap
95% relative-improvement interval exceeds five percent. If it does not pass,
the result is **no lifted advantage detected** in this declared regime unless a
more specific preregistered null status applies. Practical equivalence is
reported only when the entire relative-effect interval lies inside the
preregistered minus-five-to-plus-five-percent band.

Even a positive result permits only a weak-ground-truth claim: better prediction
of the declared behavioral finite-effect surface and better fixed-budget action
selection in the declared basis. PID Steering evaluates behavioral feedback
control on modern Qwen, Gemma, and Llama models; this complementary study asks
whether response-model family selection transfers when mechanistic ground truth
is unavailable.

## Repository boundary

This follow-up package does not replace the Paper 1 code repository. The Paper 1
repository should reproduce the existing R1--R6 ground-truthed results and
regenerate every figure/table from frozen raw-enough artifacts. ObserverBench
should own reusable observer/task/controller interfaces and benchmark machinery.
See `docs/REPOSITORY_BOUNDARY.md`.

The frozen Qwen archive is small enough to ship with the public artifact. Model
weights remain external and are referenced by immutable Hugging Face revision.

The description in `pyproject.toml` retains the pre-run phrase "Private Qwen
weak-ground-truth follow-up." The file is part of the source fingerprint stored
with the frozen measurement run, so the public release preserves it byte for
byte. This repository and README supersede that historical packaging label.
