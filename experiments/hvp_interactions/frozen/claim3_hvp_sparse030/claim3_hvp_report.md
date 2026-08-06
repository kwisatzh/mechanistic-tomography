# Claim 3 Designed-HVP Baseline

## Setup

- **n_components**: 64
- **n_pairs**: 2016
- **lifted_dim**: 2080
- **support_k**: 5
- **hvp_budgets**: [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
- **epsilons**: [5.0, 8.0]
- **seeds**: [0, 1, 2]

## Thresholds

|   epsilon | method                |   first_hvp_budget_r2_ge_0.95 |   first_hvp_budget_pair_ge_0.99 |   first_hvp_budget_both_ge_r2_0.95_pair_0.99 |   max_heldout_r2_mean |   max_pair_topk_recall_mean |
|----------:|:----------------------|------------------------------:|--------------------------------:|---------------------------------------------:|----------------------:|----------------------------:|
|         5 | hvp_pairs_atp_main    |                            12 |                              12 |                                           12 |              0.986886 |                           1 |
|         5 | hvp_pairs_finite_main |                           nan |                              12 |                                          nan |              0.893011 |                           1 |
|         5 | hvp_pairs_only        |                           nan |                              12 |                                          nan |              0.357554 |                           1 |
|         8 | hvp_pairs_atp_main    |                            24 |                              24 |                                           24 |              0.969945 |                           1 |
|         8 | hvp_pairs_finite_main |                           nan |                              24 |                                          nan |              0.91108  |                           1 |
|         8 | hvp_pairs_only        |                           nan |                              24 |                                          nan |              0.595882 |                           1 |

## Reading guide

- `hvp_pairs_atp_main` uses AtP first-order main effects plus HVP-recovered pair terms.
- `hvp_pairs_finite_main` is an upper-bound diagnostic: finite-calibrated main effects plus HVP-recovered pair terms. It isolates pair recovery from first-order finite calibration.
- If HVP recovers pairs in a few backward passes, the white-box interactional regime still needs designed measurements, but the primitive is HVP rather than forward subset masks.