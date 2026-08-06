# Frozen Qwen-2.5-7B result

`qwen2_5_7b_a100_full_results.zip` contains the complete minimally processed
measurements and analysis outputs for the Qwen weak-ground-truth follow-up.

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Model revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Hardware: NVIDIA A100-SXM4-40GB
- Designed actions: 401
- Held-out actions: 128
- Held-out prompts: 224
- Primary status: no lifted advantage detected
- Calibrated additive MAE: 0.0037896413
- Lifted MAE: 0.0038006164
- Relative lifted MAE improvement: -0.2896%
- Paired two-way-bootstrap 95% interval: [-3.5618%, 5.6470%]

The archive contains the action design, prompt-level finite effects, directions,
source/config/environment fingerprints, predictions, bootstrap draws, selector
outputs, and result summary. It does not contain model weights or raw harmful
prompt text.

The public notebook verifies the archive hash before reading it and recomputes
the reported MAEs from `test_predictions.csv`.
