# IOI Stage 2d per-pair decomposition

This is a post-processing analysis of a Stage 2c primary-stratified run. It adds a count-additive control and decomposes the bundled group-count interaction into single-pair terms.

## Setup

| input_run                                                      |   n_subsets |   n_heads |   n_prompts |   k_folds |   bootstrap_repeats |
|:---------------------------------------------------------------|------------:|----------:|------------:|----------:|--------------------:|
| ../ioi_stage2c_v0/runs/ioi_stage2c_primary_stratified_mean_end |         240 |        13 |         256 |         5 |                 500 |


## K-fold held-out fit summary

| model                |   n_rows |      mae |     rmse |       r2 | columns                                                                                                                                                                                                                                            |   n_params |
|:---------------------|---------:|---------:|---------:|---------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------:|
| additive_head        |      239 | 0.37933  | 0.454974 | 0.759141 | intercept,P:9.9,P:9.6,P:10.0,B:9.0,B:9.7,B:10.1,B:10.2,B:10.6,B:10.10,B:11.2,B:11.9,E:10.7,E:11.10                                                                                                                                                 |         14 |
| count_additive       |      239 | 0.385506 | 0.46142  | 0.752268 | intercept,P:9.9,P:9.6,P:10.0,B:9.0,B:9.7,B:10.1,B:10.2,B:10.6,B:10.10,B:11.2,B:11.9,E:10.7,E:11.10,P_count_eq_1,P_count_eq_2,P_count_eq_3,B_count_eq_1,B_count_2_3,B_count_4_5,B_count_6_8,E_count_eq_1,E_count_eq_2                               |         23 |
| count_plus_PB_count  |      239 | 0.377234 | 0.440863 | 0.77385  | intercept,P:9.9,P:9.6,P:10.0,B:9.0,B:9.7,B:10.1,B:10.2,B:10.6,B:10.10,B:11.2,B:11.9,E:10.7,E:11.10,P_count_eq_1,P_count_eq_2,P_count_eq_3,B_count_eq_1,B_count_2_3,B_count_4_5,B_count_6_8,E_count_eq_1,E_count_eq_2,P_B_count                     |         24 |
| count_plus_PE_count  |      239 | 0.259037 | 0.311709 | 0.886945 | intercept,P:9.9,P:9.6,P:10.0,B:9.0,B:9.7,B:10.1,B:10.2,B:10.6,B:10.10,B:11.2,B:11.9,E:10.7,E:11.10,P_count_eq_1,P_count_eq_2,P_count_eq_3,B_count_eq_1,B_count_2_3,B_count_4_5,B_count_6_8,E_count_eq_1,E_count_eq_2,P_E_count                     |         24 |
| count_plus_BE_count  |      239 | 0.385804 | 0.459786 | 0.754019 | intercept,P:9.9,P:9.6,P:10.0,B:9.0,B:9.7,B:10.1,B:10.2,B:10.6,B:10.10,B:11.2,B:11.9,E:10.7,E:11.10,P_count_eq_1,P_count_eq_2,P_count_eq_3,B_count_eq_1,B_count_2_3,B_count_4_5,B_count_6_8,E_count_eq_1,E_count_eq_2,B_E_count                     |         24 |
| count_plus_all_pairs |      239 | 0.235563 | 0.282239 | 0.907312 | intercept,P:9.9,P:9.6,P:10.0,B:9.0,B:9.7,B:10.1,B:10.2,B:10.6,B:10.10,B:11.2,B:11.9,E:10.7,E:11.10,P_count_eq_1,P_count_eq_2,P_count_eq_3,B_count_eq_1,B_count_2_3,B_count_4_5,B_count_6_8,E_count_eq_1,E_count_eq_2,P_B_count,P_E_count,B_E_count |         26 |


## Bootstrap summary

| model                |   n_bootstrap |   mae_mean |   mae_median |    mae_std |   mae_q05 |   mae_q95 |   rmse_mean |   rmse_median |   rmse_std |   rmse_q05 |   rmse_q95 |   r2_mean |   r2_median |     r2_std |   r2_q05 |   r2_q95 |
|:---------------------|--------------:|-----------:|-------------:|-----------:|----------:|----------:|------------:|--------------:|-----------:|-----------:|-----------:|----------:|------------:|-----------:|---------:|---------:|
| additive_head        |           500 |   0.383244 |     0.383586 | 0.0110472  |  0.365762 |  0.400617 |    0.452592 |      0.453092 | 0.0127384  |   0.432416 |   0.471391 |  0.762179 |    0.762537 | 0.00827511 | 0.748112 | 0.774759 |
| count_additive       |           500 |   0.386333 |     0.386755 | 0.011365   |  0.368371 |  0.403846 |    0.457462 |      0.457749 | 0.0130915  |   0.436639 |   0.476351 |  0.75703  |    0.757438 | 0.00875041 | 0.742059 | 0.770126 |
| count_plus_BE_count  |           500 |   0.383296 |     0.383788 | 0.0113938  |  0.36531  |  0.40101  |    0.452217 |      0.452565 | 0.0129656  |   0.431573 |   0.471342 |  0.762569 |    0.762864 | 0.00857112 | 0.747968 | 0.775504 |
| count_plus_PB_count  |           500 |   0.37628  |     0.37671  | 0.0111203  |  0.358272 |  0.393742 |    0.438333 |      0.438743 | 0.012642   |   0.418191 |   0.457189 |  0.776895 |    0.777379 | 0.00890123 | 0.761851 | 0.790767 |
| count_plus_PE_count  |           500 |   0.256381 |     0.256813 | 0.00808656 |  0.243359 |  0.269374 |    0.311347 |      0.311833 | 0.00957661 |   0.295958 |   0.326633 |  0.887462 |    0.887478 | 0.004193   | 0.880549 | 0.893712 |
| count_plus_all_pairs |           500 |   0.229313 |     0.229425 | 0.00728959 |  0.217585 |  0.240994 |    0.283187 |      0.283365 | 0.00891422 |   0.268629 |   0.297433 |  0.906876 |    0.907142 | 0.00414016 | 0.899911 | 0.913516 |


## Paired bootstrap improvement vs count-additive control

| baseline       | model                |   n_bootstrap |   delta_mae_mean |   delta_mae_median |   delta_mae_std |   delta_mae_q05 |   delta_mae_q95 |   p_delta_gt_0 | strict_success_q05_gt_0   | weak_success_p_gt_0_ge_0.95   |
|:---------------|:---------------------|--------------:|-----------------:|-------------------:|----------------:|----------------:|----------------:|---------------:|:--------------------------|:------------------------------|
| count_additive | additive_head        |           500 |       0.00308893 |         0.00308742 |     0.000465502 |      0.00233118 |      0.00383707 |              1 | True                      | True                          |
| count_additive | count_plus_BE_count  |           500 |       0.00303726 |         0.00305222 |     0.000661642 |      0.0018917  |      0.00397861 |              1 | True                      | True                          |
| count_additive | count_plus_PB_count  |           500 |       0.0100535  |         0.00997701 |     0.00115776  |      0.00824599 |      0.0119292  |              1 | True                      | True                          |
| count_additive | count_plus_PE_count  |           500 |       0.129952   |         0.130108   |     0.00386327  |      0.123367   |      0.136173   |              1 | True                      | True                          |
| count_additive | count_plus_all_pairs |           500 |       0.15702    |         0.157249   |     0.00437599  |      0.149691   |      0.163706   |              1 | True                      | True                          |


## Paired bootstrap improvement vs additive-head baseline

| baseline      | model                |   n_bootstrap |   delta_mae_mean |   delta_mae_median |   delta_mae_std |   delta_mae_q05 |   delta_mae_q95 |   p_delta_gt_0 | strict_success_q05_gt_0   | weak_success_p_gt_0_ge_0.95   |
|:--------------|:---------------------|--------------:|-----------------:|-------------------:|----------------:|----------------:|----------------:|---------------:|:--------------------------|:------------------------------|
| additive_head | count_additive       |           500 |     -0.00308893  |       -0.00308742  |     0.000465502 |     -0.00383707 |     -0.00233118 |          0     | False                     | False                         |
| additive_head | count_plus_BE_count  |           500 |     -5.16665e-05 |        1.71805e-05 |     0.000924108 |     -0.00170428 |      0.0013781  |          0.506 | False                     | False                         |
| additive_head | count_plus_PB_count  |           500 |      0.00696459  |        0.0069663   |     0.00123779  |      0.00495065 |      0.00888319 |          1     | True                      | True                          |
| additive_head | count_plus_PE_count  |           500 |      0.126863    |        0.126974    |     0.00357068  |      0.120847   |      0.132715   |          1     | True                      | True                          |
| additive_head | count_plus_all_pairs |           500 |      0.153931    |        0.154158    |     0.00409647  |      0.147118   |      0.160411   |          1     | True                      | True                          |


## Coefficients for pair terms

| model                | term      |     coef |
|:---------------------|:----------|---------:|
| count_plus_PB_count  | P_B_count | 1.39296  |
| count_plus_PE_count  | P_E_count | 2.24539  |
| count_plus_BE_count  | B_E_count | 0.754789 |
| count_plus_all_pairs | P_B_count | 1.15906  |
| count_plus_all_pairs | P_E_count | 2.16447  |
| count_plus_all_pairs | B_E_count | 0.521502 |


## Reading guide

- If `count_additive` is close to `count_plus_all_pairs`, the Stage 2c win was mostly within-group count curvature, not cross-group interaction.

- If a single-pair model has positive paired Delta MAE with q05 > 0 versus count-additive, that pair earns a robust predictive gain.

- The self-repair-specific term is `count_plus_PB_count`. The name-mover/negative-name-mover cancellation term is `count_plus_PE_count`.
