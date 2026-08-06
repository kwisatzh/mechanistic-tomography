# Tracr Mechanistic Tomography Harness v0

This package starts the real-circuit bridge for the formalism paper.

It has two backends:

1. `--backend tracr`: a first-day feasibility check against `google-deepmind/tracr`. It compiles a simple first-order predicate and a bigram/interaction-style RASP program, runs example sequences, and inventories the Tracr outputs (`decoded`, `residuals`, `layer_outputs`, `attn_logits`). It does **not** yet implement residual patching inside the Haiku model.
2. `--backend surrogate`: a runnable known-circuit surrogate that exercises the same method comparison (AtP, scalar-calibrated AtP, finite singleton, first-order OMP, lifted OMP). This is a code-path sanity check, not a Tracr result.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional Tracr install:

```bash
git clone https://github.com/google-deepmind/tracr
cd tracr
python -m pip install .
```

The upstream Tracr repo was archived in 2025 and is read-only, so expect some dependency pinning on modern Python/JAX. Use a fresh environment if possible.

## Run the surrogate sanity check

```bash
python tracr_r1_harness.py \
  --backend surrogate \
  --outdir runs/surrogate_r1_smoke
```

Expected outputs:

```text
surrogate_results.csv
surrogate_summary.csv
surrogate_r2.png
surrogate_pair_recall.png
surrogate_metadata.json
```

## Run the Tracr feasibility check

```bash
python tracr_r1_harness.py \
  --backend tracr \
  --programs first_order_is_A interaction_bigram_AB shuffle_dyck2 \
  --outdir runs/tracr_feasibility
```

If Tracr is not installed, the script writes a report with install hints instead of crashing.

## What to inspect in `tracr_feasibility_report.json`

- Which programs compile?
- Does `model.apply` return sensible `decoded` outputs?
- Are `residuals`, `layer_outputs`, and `attn_logits` present?
- Are their objects dictionaries, lists, arrays, or named tuples?

This decides the adapter work needed for the real R1/R2 experiments.

## R1/R2 target after feasibility

- R1: Tracr interaction circuit (bigram / conditional / AND-like program). Compare singleton patching, AtP/AtP*, first-order OMP, lifted forward masks, and HVP where available, scored against compiled circuit ground truth.
- R2: Tracr first-order/null circuit. Show AtP plus small finite calibration matches lifted tomography in a real compiled circuit.

The next code milestone is `TracrActivationAdapter.patch`: a way to intervene on residual-stream subspaces exposed by Tracr's compiled model.
