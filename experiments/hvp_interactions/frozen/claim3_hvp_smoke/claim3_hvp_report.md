# Claim 3 Designed-HVP Baseline

## Setup

- **n_components**: 64
- **n_pairs**: 2016
- **lifted_dim**: 2080
- **support_k**: 5
- **hvp_budgets**: [1, 2, 4, 8]
- **epsilons**: [5.0, 8.0]
- **seeds**: [0]

## Thresholds

|   epsilon | method                |   first_hvp_budget_r2_ge_0.95 |   first_hvp_budget_pair_ge_0.99 |   first_hvp_budget_both_ge_r2_0.95_pair_0.99 |   max_heldout_r2_mean |   max_pair_topk_recall_mean |
|----------:|:----------------------|------------------------------:|--------------------------------:|---------------------------------------------:|----------------------:|----------------------------:|
|         5 | hvp_pairs_atp_main    |                             1 |                               1 |                                            1 |              0.987517 |                           1 |
|         5 | hvp_pairs_finite_main |                           nan |                               1 |                                          nan |              0.900942 |                           1 |
|         5 | hvp_pairs_only        |                           nan |                               1 |                                          nan |              0.374779 |                           1 |
|         8 | hvp_pairs_atp_main    |                             1 |                               1 |                                            1 |              0.976647 |                           1 |
|         8 | hvp_pairs_finite_main |                           nan |                               1 |                                          nan |              0.925394 |                           1 |
|         8 | hvp_pairs_only        |                           nan |                               1 |                                          nan |              0.626696 |                           1 |

## Reading guide

- `hvp_pairs_atp_main` uses AtP first-order main effects plus HVP-recovered pair terms.
- `hvp_pairs_finite_main` is an upper-bound diagnostic: finite-calibrated main effects plus HVP-recovered pair terms. It isolates pair recovery from first-order finite calibration.
- If HVP recovers pairs in a few backward passes, the white-box interactional regime still needs designed measurements, but the primitive is HVP rather than forward subset masks.