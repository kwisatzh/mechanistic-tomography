# Qwen-2.5-7B result

## Declared question

For a fixed Qwen-2.5-7B refusal-margin readout, layerwise contrast basis, action
library, prompt distribution, and intervention regime, does a lifted pairwise
response model improve held-out finite-effect prediction over a calibrated
additive response model?

The primary contrast is held-out

`MAE(calibrated additive) - MAE(lifted)`.

The interaction gate passes only if the lower 95% bound on relative MAE
improvement exceeds the preregistered 5% threshold.

## Result

The full design contains 401 actions. The primary test uses 128 actions at the
held-out scale 0.75 and 224 held-out prompts.

| Response model | Parameters | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| Calibrated additive | 20 | 0.00378964 | 0.00469899 | 0.982851 |
| Lifted | 48 | 0.00380062 | 0.00461288 | 0.983473 |

The estimated relative lifted MAE improvement is -0.2896%. Its paired
prompt-family-by-action bootstrap 95% interval is [-3.5618%, 5.6470%]. The
status is **no lifted advantage detected**. The interval narrowly crosses the
5% practical threshold, so the experiment does not establish practical
equivalence and does not rule out every meaningful lifted advantage.

## Interpretation for mechanistic tomography

The result supports the measurement procedure rather than a claim that
interactions disappear at scale. Finite calibration makes the additive map an
accurate predictor of the declared Qwen intervention surface. The held-out test
therefore does not justify escalating to pairwise measurements in this regime.

This is also evidence that the operational procedure transfers beyond
GPT-2-small: a 7B instruction-tuned model can be evaluated by declaring a basis,
measurement design, operating scale, residual test, and escalation rule. It is
not a scaling law and does not establish ground-truth mechanism recovery.

## Descriptive density breakdown

The balanced test design contains 32 actions at each density. This breakdown is
post-hoc and descriptive; it does not replace the aggregate primary endpoint.

| Density | Additive MAE | Lifted MAE | Relative lifted improvement |
|---:|---:|---:|---:|
| 0.25 | 0.004093 | 0.004082 | 0.29% |
| 0.50 | 0.004058 | 0.004001 | 1.39% |
| 0.75 | 0.003353 | 0.003307 | 1.36% |
| 1.00 | 0.003655 | 0.003812 | -4.31% |

The pattern is small and non-monotone. It gives no evidence that denser actions
systematically require pairwise lifting.

## Provenance

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Hardware: NVIDIA A100-SXM4-40GB
- Archive: `artifacts/frozen/qwen2_5_7b_a100_full_results.zip`
- Archive SHA-256: `aca53bf0c108a0de1812edbbbf98ece0612a304a151f50cf3f13e109ac01544e`
- Source fingerprint: `74afa289364cda34e89d22ad065fb8a6332f548c2c92c546a6fc44593e09bf6d`
