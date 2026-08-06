# Claim 3 HVP Baseline

This package adds the white-box baseline a reviewer is likely to ask for:
recovering the sparse second-order / pair map from designed Hessian-vector products (HVPs).

The earlier lifted forward-mask result compares against exhaustive pair probing in a gradient-free / forward-only regime. This HVP baseline answers the white-box question: if gradients are available, does one still need designed measurement for interactions? Yes, but the primitive becomes HVPs rather than forward subset masks.

## Install

The script needs only NumPy, pandas, and Matplotlib. From this directory, install
the small dependency set before running it:

```bash
cd experiments/hvp_interactions
python -m pip install -r requirements.txt
```

## Smoke test

```bash
python claim3_hvp_baseline.py \
  --quick \
  --outdir runs/claim3_hvp_smoke
```

## Main run

Dense signed HVP vectors are the clean white-box baseline:

```bash
python claim3_hvp_baseline.py \
  --mode combined \
  --seeds 0 1 2 \
  --epsilons 5 8 \
  --hvp-budgets 1 2 4 8 12 16 24 32 48 64 \
  --hvp-kind signed \
  --hvp-density 1.0 \
  --hvp-noise-std 0.01 \
  --forward-noise-std 0.01 \
  --holdout-measurements 512 \
  --outdir runs/claim3_hvp_dense
```

To match the forward mask density more closely, also run sparse HVP vectors:

```bash
python claim3_hvp_baseline.py \
  --mode combined \
  --seeds 0 1 2 \
  --epsilons 5 8 \
  --hvp-budgets 1 2 4 8 12 16 24 32 48 64 \
  --hvp-kind signed \
  --hvp-density 0.30 \
  --hvp-noise-std 0.01 \
  --forward-noise-std 0.01 \
  --holdout-measurements 512 \
  --outdir runs/claim3_hvp_sparse030
```

## Outputs

Each run writes:

- `claim3_hvp_results.csv`
- `claim3_hvp_summary.csv`
- `claim3_hvp_summary_flat.csv`
- `claim3_hvp_thresholds.csv`
- `claim3_hvp_metadata.json`
- `claim3_hvp_report.md`
- `claim3_hvp_r2.png`
- `claim3_hvp_pair_recall.png`
- `claim3_hvp_selected_k.png`

## How to read

- `hvp_pairs_atp_main`: AtP first-order main effects + HVP-recovered pair terms.
- `hvp_pairs_finite_main`: finite-calibrated main effects + HVP-recovered pair terms. This is an upper-bound diagnostic that isolates pair recovery from finite main-effect calibration.
- `hvp_pairs_only`: pair terms alone.

If a few HVP queries recover the pair support, the paper should say: the interactional map is not free with gradients; it still requires designed measurement. But the white-box primitive is HVPs, while lifted forward masks are the gradient-free / nondifferentiable version.
