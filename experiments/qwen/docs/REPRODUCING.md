# Reproducing the Qwen weak-ground-truth response-surface follow-up

## Execution levels

1. `mechtomo toy`: CPU-only falsifier with a planted interaction.
2. M4 smoke: pinned 0.5B model, tiny reviewed data, tiny action design.
3. H200 pilot: pinned 7B model with a reserved 16+16 construction split, 8
   fitting harmful prompts, 8 locked harmful prompts, and 25 locked benign
   prompts.
4. H200 full: pinned 7B model, excluding every pilot family, with 32+32
   construction prompts, 112 fitting harmful prompts, 224 locked harmful
   prompts, and 150 family-separated benign XSTest prompts. Pilot plus full
   exhaust the 400 canonical HarmBench behaviors without reuse.
   A StrongREJECT/JailbreakBench OOD run is a separate artifact, not a source
   silently spliced into the locked split.

This is the weak-ground-truth extension reported in paper v2. It measures
behavioral finite-effect prediction and fixed-budget action selection; it does
not turn a behavioral response surface into mechanistic ground truth. Levels
1--3 may reveal engineering defects, degenerate scores, wrong direction sign,
and throughput, but cannot be used to choose a favorable observer family or
endpoint.

## Frozen inputs

The runner fingerprints the config, prepared JSONL, ordered prompt records,
directions, action matrix, source tree, exact NumPy/PyTorch/Transformers/
Accelerate versions, and resolved accelerator details. Direction construction
also records its system prompt, construction records, layer/fraction/position,
tokenizer revision, and capture environment. Resuming with a changed fingerprint
or a cache that predates its fingerprint is refused. The data-preparation
manifest records raw and prepared hashes plus the reserved-profile selection.

## Outputs

- `directions.npz`: fixed contrastive directions and layer norm scales.
- `directions_inputs.json`: construction, tokenizer, source, and environment provenance.
- `measurement_inputs.json`: immutable cache/resume fingerprint.
- `clean_refusal_margin.npy`: clean deterministic margins.
- `effects_all.npy`: prompt-by-action finite effects, checkpointed by action.
- `surface_measurements.npz`: portable frozen analysis input.
- `measurement_complete.json`: hashes binding the verified surface to its inputs.
- `analysis/summary.json`: response-surrogate metrics and primary confidence interval.
- `analysis/test_predictions.csv`: locked action-surface predictions.
- `analysis/selector_choices.csv`: fixed-budget selector choices and outcomes.
- `analysis/bootstrap_delta_mae.npy`: bootstrap draws for the primary contrast.
- `analysis/bootstrap_relative_mae_improvement.npy`: relative-improvement bootstrap draws.

The portable `surface_measurements.npz` is the minimum raw-enough frozen
artifact. Do not retain only the summary or plotted points.
