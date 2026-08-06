# IOI Stage 2c head-level subset-prediction report

This is the pre-registered Stage-2c variant: primary-stratified random subsets of individual heads. It tests whether interaction terms help when the subset design actually contains the high-primary-coverage regimes where IOI self-repair is mechanistically predicted to matter.

## Setup

|   n_subsets |   n_heads |   n_prompts | ablation   | position_mode   |   k_folds |   bootstrap_repeats |
|------------:|----------:|------------:|:-----------|:----------------|----------:|--------------------:|
|         240 |        13 |         256 | mean       | end             |         5 |                 200 |


## K-fold held-out fit summary

| model                    |   n_rows |      mae |     rmse |       r2 | columns                                                                                                                          |   n_params |
|:-------------------------|---------:|---------:|---------:|---------:|:---------------------------------------------------------------------------------------------------------------------------------|-----------:|
| additive_head            |      239 | 0.37933  | 0.454974 | 0.759141 | intercept,P:9.9,P:9.6,P:10.0,B:9.0,B:9.7,B:10.1,B:10.2,B:10.6,B:10.10,B:11.2,B:11.9,E:10.7,E:11.10                               |         14 |
| pb_occupancy_interaction |      239 | 0.379864 | 0.454339 | 0.759813 | intercept,P:9.9,P:9.6,P:10.0,B:9.0,B:9.7,B:10.1,B:10.2,B:10.6,B:10.10,B:11.2,B:11.9,E:10.7,E:11.10,P_B_occ                       |         15 |
| pb_count_interaction     |      239 | 0.369496 | 0.433361 | 0.781481 | intercept,P:9.9,P:9.6,P:10.0,B:9.0,B:9.7,B:10.1,B:10.2,B:10.6,B:10.10,B:11.2,B:11.9,E:10.7,E:11.10,P_B_count                     |         15 |
| group_count_interaction  |      239 | 0.229304 | 0.279415 | 0.909157 | intercept,P:9.9,P:9.6,P:10.0,B:9.0,B:9.7,B:10.1,B:10.2,B:10.6,B:10.10,B:11.2,B:11.9,E:10.7,E:11.10,P_B_count,P_E_count,B_E_count |         17 |


## Bootstrap summary

| model                    |   n_bootstrap |   mae_mean |   mae_median |    mae_std |   mae_q05 |   mae_q95 |   rmse_mean |   rmse_median |   rmse_std |   rmse_q05 |   rmse_q95 |   r2_mean |   r2_median |     r2_std |   r2_q05 |   r2_q95 |
|:-------------------------|--------------:|-----------:|-------------:|-----------:|----------:|----------:|------------:|--------------:|-----------:|-----------:|-----------:|----------:|------------:|-----------:|---------:|---------:|
| additive_head            |           200 |   0.383131 |     0.382985 | 0.0116044  |  0.365229 |  0.402481 |    0.45242  |      0.452236 | 0.0134517  |   0.432244 |   0.475826 |  0.762659 |    0.762237 | 0.00811728 | 0.749367 | 0.774812 |
| group_count_interaction  |           200 |   0.223608 |     0.223335 | 0.00745429 |  0.211476 |  0.236124 |    0.277901 |      0.277727 | 0.00887006 |   0.263941 |   0.292932 |  0.910434 |    0.910315 | 0.00380833 | 0.904283 | 0.91682  |
| pb_count_interaction     |           200 |   0.37003  |     0.37015  | 0.0113382  |  0.352353 |  0.38907  |    0.43199  |      0.431734 | 0.0129131  |   0.411853 |   0.452909 |  0.783582 |    0.783334 | 0.00827556 | 0.769692 | 0.79666  |
| pb_occupancy_interaction |           200 |   0.384015 |     0.383969 | 0.0116616  |  0.36594  |  0.403387 |    0.452424 |      0.452223 | 0.0135148  |   0.432249 |   0.476026 |  0.762655 |    0.762223 | 0.00820953 | 0.74925  | 0.775123 |


## Paired bootstrap improvement vs additive

| baseline      | model                    |   n_bootstrap |   delta_mae_mean |   delta_mae_median |   delta_mae_std |   delta_mae_q05 |   delta_mae_q95 |   p_delta_gt_0 | strict_success_q05_gt_0   | weak_success_p_gt_0_ge_0.95   |
|:--------------|:-------------------------|--------------:|-----------------:|-------------------:|----------------:|----------------:|----------------:|---------------:|:--------------------------|:------------------------------|
| additive_head | group_count_interaction  |           200 |      0.159523    |        0.159584    |     0.00444057  |      0.152373   |     0.166134    |              1 | True                      | True                          |
| additive_head | pb_count_interaction     |           200 |      0.0131008   |        0.0130666   |     0.00115695  |      0.0113858  |     0.0148793   |              1 | True                      | True                          |
| additive_head | pb_occupancy_interaction |           200 |     -0.000884223 |       -0.000901142 |     0.000126776 |     -0.00106127 |    -0.000669904 |              0 | False                     | False                         |


## Group-mask diagnostics

|   drop_P |   drop_B |   drop_E |   drop_P+B |   drop_P+E |   drop_B+E |   drop_P+B+E |   pb_interaction_group_mask |   pb_interaction_fraction_of_joint_group_mask |   backup_effect_primary_intact_group_mask |   backup_effect_primary_ablated_group_mask |   backup_conditional_amplification_group_mask |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|-------------:|----------------------------:|----------------------------------------------:|------------------------------------------:|-------------------------------------------:|----------------------------------------------:|
|    0.463 | 0.350237 | -2.15007 |    1.86852 |   0.161238 |   -1.37944 |      2.32678 |                     1.05529 |                                       0.56477 |                                  0.350237 |                                    1.40552 |                                       4.01306 |


## Occupancy-sliced errors

| model                    | slice   |   value |   n |      mae |     rmse |
|:-------------------------|:--------|--------:|----:|---------:|---------:|
| additive_head            | P_B     |       0 |  62 | 0.44252  | 0.515477 |
| additive_head            | P_B     |       1 | 177 | 0.357196 | 0.431781 |
| additive_head            | P_E     |       0 | 107 | 0.409779 | 0.47836  |
| additive_head            | P_E     |       1 | 132 | 0.354649 | 0.435096 |
| additive_head            | B_E     |       0 |  98 | 0.376045 | 0.444675 |
| additive_head            | B_E     |       1 | 141 | 0.381614 | 0.461997 |
| group_count_interaction  | P_B     |       0 |  62 | 0.197013 | 0.255202 |
| group_count_interaction  | P_B     |       1 | 177 | 0.240615 | 0.287415 |
| group_count_interaction  | P_E     |       0 | 107 | 0.202426 | 0.252343 |
| group_count_interaction  | P_E     |       1 | 132 | 0.251092 | 0.29957  |
| group_count_interaction  | B_E     |       0 |  98 | 0.210575 | 0.255433 |
| group_count_interaction  | B_E     |       1 | 141 | 0.242322 | 0.294937 |
| pb_count_interaction     | P_B     |       0 |  62 | 0.423793 | 0.486964 |
| pb_count_interaction     | P_B     |       1 | 177 | 0.350477 | 0.412942 |
| pb_count_interaction     | P_E     |       0 | 107 | 0.404513 | 0.470327 |
| pb_count_interaction     | P_E     |       1 | 132 | 0.341111 | 0.400903 |
| pb_count_interaction     | B_E     |       0 |  98 | 0.379626 | 0.438603 |
| pb_count_interaction     | B_E     |       1 | 141 | 0.362455 | 0.42968  |
| pb_occupancy_interaction | P_B     |       0 |  62 | 0.450902 | 0.52531  |
| pb_occupancy_interaction | P_B     |       1 | 177 | 0.35498  | 0.426696 |
| pb_occupancy_interaction | P_E     |       0 | 107 | 0.413235 | 0.482635 |
| pb_occupancy_interaction | P_E     |       1 | 132 | 0.352813 | 0.430038 |
| pb_occupancy_interaction | B_E     |       0 |  98 | 0.378561 | 0.446904 |
| pb_occupancy_interaction | B_E     |       1 | 141 | 0.38077  | 0.459435 |


## Primary-coverage sliced errors

| model                    |   n_P |   n |      mae |     rmse |   has_B |
|:-------------------------|------:|----:|---------:|---------:|--------:|
| additive_head            |     0 |  42 | 0.507275 | 0.573337 |     nan |
| additive_head            |     1 |  43 | 0.338956 | 0.431709 |     nan |
| additive_head            |     2 |  87 | 0.305832 | 0.348477 |     nan |
| additive_head            |     3 |  67 | 0.420477 | 0.505024 |     nan |
| additive_head            |     0 |   3 | 0.410084 | 0.484585 |       0 |
| additive_head            |     0 |  39 | 0.514751 | 0.579602 |       1 |
| additive_head            |     1 |   8 | 0.274988 | 0.345502 |       0 |
| additive_head            |     1 |  35 | 0.353577 | 0.449096 |       1 |
| additive_head            |     2 |   8 | 0.307782 | 0.354895 |       0 |
| additive_head            |     2 |  79 | 0.305634 | 0.34782  |       1 |
| additive_head            |     3 |   4 | 0.367141 | 0.420057 |       0 |
| additive_head            |     3 |  63 | 0.423863 | 0.509941 |       1 |
| group_count_interaction  |     0 |  42 | 0.171193 | 0.236146 |     nan |
| group_count_interaction  |     1 |  43 | 0.287323 | 0.326843 |     nan |
| group_count_interaction  |     2 |  87 | 0.271574 | 0.316922 |     nan |
| group_count_interaction  |     3 |  67 | 0.173608 | 0.211092 |     nan |
| group_count_interaction  |     0 |   3 | 0.193657 | 0.294412 |       0 |
| group_count_interaction  |     0 |  39 | 0.169465 | 0.231056 |       1 |
| group_count_interaction  |     1 |   8 | 0.228563 | 0.249277 |       0 |
| group_count_interaction  |     1 |  35 | 0.300754 | 0.342112 |       1 |
| group_count_interaction  |     2 |   8 | 0.275095 | 0.322681 |       0 |
| group_count_interaction  |     2 |  79 | 0.271218 | 0.316333 |       1 |
| group_count_interaction  |     3 |   4 | 0.248863 | 0.302372 |       0 |
| group_count_interaction  |     3 |  63 | 0.16883  | 0.203921 |       1 |
| pb_count_interaction     |     0 |  42 | 0.486815 | 0.544306 |     nan |
| pb_count_interaction     |     1 |  43 | 0.334124 | 0.406189 |     nan |
| pb_count_interaction     |     2 |  87 | 0.305419 | 0.346551 |     nan |
| pb_count_interaction     |     3 |  67 | 0.401857 | 0.471553 |     nan |
| pb_count_interaction     |     0 |   3 | 0.494832 | 0.596061 |       0 |
| pb_count_interaction     |     0 |  39 | 0.486199 | 0.540119 |       1 |
| pb_count_interaction     |     1 |   8 | 0.263212 | 0.308046 |       0 |
| pb_count_interaction     |     1 |  35 | 0.350333 | 0.425455 |       1 |
| pb_count_interaction     |     2 |   8 | 0.285443 | 0.338744 |       0 |
| pb_count_interaction     |     2 |  79 | 0.307442 | 0.347332 |       1 |
| pb_count_interaction     |     3 |   4 | 0.359914 | 0.381417 |       0 |
| pb_count_interaction     |     3 |  63 | 0.404521 | 0.476701 |       1 |
| pb_occupancy_interaction |     0 |  42 | 0.509542 | 0.57831  |     nan |
| pb_occupancy_interaction |     1 |  43 | 0.340899 | 0.426001 |     nan |
| pb_occupancy_interaction |     2 |  87 | 0.308064 | 0.349418 |     nan |
| pb_occupancy_interaction |     3 |  67 | 0.416813 | 0.501686 |     nan |
| pb_occupancy_interaction |     0 |   3 | 0.405665 | 0.485872 |       0 |
| pb_occupancy_interaction |     0 |  39 | 0.517532 | 0.584816 |       1 |
| pb_occupancy_interaction |     1 |   8 | 0.280174 | 0.334541 |       0 |
| pb_occupancy_interaction |     1 |  35 | 0.354779 | 0.444271 |       1 |
| pb_occupancy_interaction |     2 |   8 | 0.351614 | 0.399535 |       0 |
| pb_occupancy_interaction |     2 |  79 | 0.303654 | 0.343936 |       1 |
| pb_occupancy_interaction |     3 |   4 | 0.375219 | 0.471696 |       0 |
| pb_occupancy_interaction |     3 |  63 | 0.419454 | 0.503529 |       1 |