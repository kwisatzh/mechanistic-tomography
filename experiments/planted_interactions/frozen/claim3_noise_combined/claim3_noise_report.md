# Claim 3 Noise Sweep Summary
## Inputs
- num_input_files: 5
- noise_std values: [np.float64(0.01), np.float64(0.05), np.float64(0.1), np.float64(0.2), np.float64(0.3)]
- epsilons: [np.float64(5.0), np.float64(8.0)]
- measurements: 96
- holdout_measurements: 512
- n_components: 64
- mode: combined
- mask_density: 0.3

## Threshold robustness
Thresholds: held-out R² ≥ 0.95; pair recall ≥ 0.99.

### epsilon = 5

| method          |   epsilon |   max_noise_r2_ge_0.95 |   max_noise_pair_ge_0.99 |   max_noise_both_ge_r2_0.95_pair_0.99 |   max_heldout_r2_mean |   max_pair_topk_recall_mean |
|:----------------|----------:|-----------------------:|-------------------------:|--------------------------------------:|----------------------:|----------------------------:|
| finite_single   |         5 |                 nan    |                   nan    |                                nan    |             0.577676  |                           0 |
| first_order_omp |         5 |                 nan    |                   nan    |                                nan    |             0.656269  |                           0 |
| lifted_omp      |         5 |                   0.01 |                     0.05 |                                  0.01 |             0.997695  |                           1 |
| multigain_atp   |         5 |                 nan    |                   nan    |                                nan    |             0.684345  |                           0 |
| raw_atp         |         5 |                 nan    |                   nan    |                                nan    |             0.671846  |                           0 |
| scalar_cal_atp  |         5 |                 nan    |                   nan    |                                nan    |             0.678793  |                           0 |
| subset_ridge    |         5 |                 nan    |                   nan    |                                nan    |             0.0730268 |                           0 |

### epsilon = 8

| method          |   epsilon |   max_noise_r2_ge_0.95 |   max_noise_pair_ge_0.99 |   max_noise_both_ge_r2_0.95_pair_0.99 |   max_heldout_r2_mean |   max_pair_topk_recall_mean |
|:----------------|----------:|-----------------------:|-------------------------:|--------------------------------------:|----------------------:|----------------------------:|
| finite_single   |         8 |                 nan    |                    nan   |                                nan    |              0.329028 |                           0 |
| first_order_omp |         8 |                 nan    |                    nan   |                                nan    |              0.375874 |                           0 |
| lifted_omp      |         8 |                   0.05 |                      0.2 |                                  0.05 |              0.998319 |                           1 |
| multigain_atp   |         8 |                 nan    |                    nan   |                                nan    |              0.415275 |                           0 |
| raw_atp         |         8 |                 nan    |                    nan   |                                nan    |              0.382987 |                           0 |
| scalar_cal_atp  |         8 |                 nan    |                    nan   |                                nan    |              0.401822 |                           0 |
| subset_ridge    |         8 |                 nan    |                    nan   |                                nan    |             -0.720547 |                           0 |

## Reading guide
- `lifted_omp` is the key interaction-recovery method.
- If pair recall remains high while R² falls, support is recovered but coefficient/effect-size estimation has degraded.
- If both pair recall and R² collapse, the noise level exceeds the current design/solver's useful regime.
