# Claim 3 Budget Sweep Summary

## Setup inferred from inputs

- **n_components**: 64
- **n_pairs**: 2016
- **lifted_dim**: 2080
- **support_k**: 5
- **cs_heuristic**: 30.153426301306318
- **exhaustive_pair_budget**: 2016
- **num_input_files**: 13

## Threshold crossings

Thresholds: held-out R² ≥ 0.95; pair recall ≥ 0.99.

### epsilon = 5

| method          |   first_budget_r2_ge_0.95 |   first_budget_pair_ge_0.99 |   first_budget_both_ge_r2_0.95_pair_0.99 |   max_heldout_r2_mean |   max_pair_topk_recall_mean |
|:----------------|--------------------------:|----------------------------:|-----------------------------------------:|----------------------:|----------------------------:|
| finite_single   |                       nan |                         nan |                                      nan |              0.609475 |                           0 |
| first_order_omp |                       nan |                         nan |                                      nan |              0.721642 |                           0 |
| lifted_omp      |                        64 |                          64 |                                       64 |              0.998264 |                           1 |
| multigain_atp   |                       nan |                         nan |                                      nan |              0.721635 |                           0 |
| raw_atp         |                       nan |                         nan |                                      nan |              0.711098 |                           0 |
| scalar_cal_atp  |                       nan |                         nan |                                      nan |              0.714455 |                           0 |
| subset_ridge    |                       nan |                         nan |                                      nan |              0.62309  |                           0 |

### epsilon = 8

| method          |   first_budget_r2_ge_0.95 |   first_budget_pair_ge_0.99 |   first_budget_both_ge_r2_0.95_pair_0.99 |   max_heldout_r2_mean |   max_pair_topk_recall_mean |
|:----------------|--------------------------:|----------------------------:|-----------------------------------------:|----------------------:|----------------------------:|
| finite_single   |                       nan |                         nan |                                      nan |              0.370602 |                           0 |
| first_order_omp |                       nan |                         nan |                                      nan |              0.472853 |                           0 |
| lifted_omp      |                        72 |                          24 |                                       72 |              0.998382 |                           1 |
| multigain_atp   |                       nan |                         nan |                                      nan |              0.472846 |                           0 |
| raw_atp         |                       nan |                         nan |                                      nan |              0.444638 |                           0 |
| scalar_cal_atp  |                       nan |                         nan |                                      nan |              0.455246 |                           0 |
| subset_ridge    |                       nan |                         nan |                                      nan |              0.283937 |                           0 |


## How to read this

- `lifted_omp` is the key Claim-3 method: it estimates main effects plus pair terms.
- `first_order_omp`, AtP variants, and `finite_single` cannot recover pure pair terms by construction.
- Compare the first budget where `lifted_omp` reaches both thresholds to the CS heuristic `k log(N/k)` and the exhaustive-pair baseline `C(n,2)`.
