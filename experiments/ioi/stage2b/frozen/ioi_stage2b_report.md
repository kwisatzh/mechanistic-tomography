# IOI Stage 2b head-level subset-prediction report

This is the powered Stage-2 design: random subsets of individual heads, not only 8 group-level subsets.

## Setup

|   n_subsets |   n_heads |   n_prompts | ablation   | position_mode   |   k_folds |   bootstrap_repeats |
|------------:|----------:|------------:|:-----------|:----------------|----------:|--------------------:|
|         160 |        13 |         256 | mean       | end             |         5 |                 200 |


## K-fold held-out fit summary

| model                |   n_rows |      mae |     rmse |       r2 | columns                                                                                                        |   n_params |
|:---------------------|---------:|---------:|---------:|---------:|:---------------------------------------------------------------------------------------------------------------|-----------:|
| additive_head        |      159 | 0.324851 | 0.40944  | 0.748701 | intercept,P:9.9,P:9.6,P:10.0,B:9.0,B:9.7,B:10.1,B:10.2,B:10.6,B:10.10,B:11.2,B:11.9,E:10.7,E:11.10             |         14 |
| pb_group_interaction |      159 | 0.323897 | 0.407502 | 0.751073 | intercept,P:9.9,P:9.6,P:10.0,B:9.0,B:9.7,B:10.1,B:10.2,B:10.6,B:10.10,B:11.2,B:11.9,E:10.7,E:11.10,P_B         |         15 |
| group_interaction    |      159 | 0.307643 | 0.395446 | 0.765585 | intercept,P:9.9,P:9.6,P:10.0,B:9.0,B:9.7,B:10.1,B:10.2,B:10.6,B:10.10,B:11.2,B:11.9,E:10.7,E:11.10,P_B,P_E,B_E |         17 |


## Bootstrap summary

| model                |   n_bootstrap |   mae_mean |   mae_median |   mae_std |   mae_q05 |   mae_q95 |   rmse_mean |   rmse_median |   rmse_std |   rmse_q05 |   rmse_q95 |   r2_mean |   r2_median |     r2_std |   r2_q05 |   r2_q95 |
|:---------------------|--------------:|-----------:|-------------:|----------:|----------:|----------:|------------:|--------------:|-----------:|-----------:|-----------:|----------:|------------:|-----------:|---------:|---------:|
| additive_head        |           200 |   0.338727 |     0.33876  | 0.0100458 |  0.323141 |  0.355027 |    0.418338 |      0.418081 |  0.0122514 |   0.399751 |   0.438741 |  0.739014 |    0.738943 | 0.00799245 | 0.725798 | 0.752577 |
| group_interaction    |           200 |   0.32882  |     0.328937 | 0.0100696 |  0.313521 |  0.344998 |    0.409147 |      0.409532 |  0.0121557 |   0.391217 |   0.428636 |  0.750347 |    0.750327 | 0.00813588 | 0.737488 | 0.764096 |
| pb_group_interaction |           200 |   0.339233 |     0.339394 | 0.0101681 |  0.32346  |  0.356135 |    0.421412 |      0.421312 |  0.0124246 |   0.402592 |   0.442274 |  0.735167 |    0.735091 | 0.00814762 | 0.721825 | 0.748587 |


## Group-mask diagnostics

|   drop_P |   drop_B |   drop_E |   drop_P+B |   drop_P+E |   drop_B+E |   drop_P+B+E |   pb_interaction_group_mask |   pb_interaction_fraction_of_joint_group_mask |   backup_effect_primary_intact_group_mask |   backup_effect_primary_ablated_group_mask |   backup_conditional_amplification_group_mask |   mae_reduction_group_vs_additive |
|---------:|---------:|---------:|-----------:|-----------:|-----------:|-------------:|----------------------------:|----------------------------------------------:|------------------------------------------:|-------------------------------------------:|----------------------------------------------:|----------------------------------:|
|    0.463 | 0.350237 | -2.15007 |    1.86852 |   0.161238 |   -1.37944 |      2.32678 |                     1.05529 |                                       0.56477 |                                  0.350237 |                                    1.40552 |                                       4.01306 |                         0.0529723 |


## Occupancy-sliced errors

| model                | slice   |   value |   n |      mae |     rmse |
|:---------------------|:--------|--------:|----:|---------:|---------:|
| additive_head        | P_B     |       0 |  60 | 0.299563 | 0.34434  |
| additive_head        | P_B     |       1 |  99 | 0.340177 | 0.444276 |
| additive_head        | P_E     |       0 |  89 | 0.299265 | 0.361214 |
| additive_head        | P_E     |       1 |  70 | 0.357381 | 0.463567 |
| additive_head        | B_E     |       0 |  68 | 0.301175 | 0.365973 |
| additive_head        | B_E     |       1 |  91 | 0.342542 | 0.439121 |
| group_interaction    | P_B     |       0 |  60 | 0.270445 | 0.327893 |
| group_interaction    | P_B     |       1 |  99 | 0.330186 | 0.431268 |
| group_interaction    | P_E     |       0 |  89 | 0.26924  | 0.321605 |
| group_interaction    | P_E     |       1 |  70 | 0.356468 | 0.472967 |
| group_interaction    | B_E     |       0 |  68 | 0.27737  | 0.340948 |
| group_interaction    | B_E     |       1 |  91 | 0.330264 | 0.431702 |
| pb_group_interaction | P_B     |       0 |  60 | 0.304314 | 0.360843 |
| pb_group_interaction | P_B     |       1 |  99 | 0.335765 | 0.433343 |
| pb_group_interaction | P_E     |       0 |  89 | 0.304424 | 0.371245 |
| pb_group_interaction | P_E     |       1 |  70 | 0.348655 | 0.449396 |
| pb_group_interaction | B_E     |       0 |  68 | 0.302438 | 0.370923 |
| pb_group_interaction | B_E     |       1 |  91 | 0.339931 | 0.432823 |