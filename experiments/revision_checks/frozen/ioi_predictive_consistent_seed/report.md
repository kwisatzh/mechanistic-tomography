# IOI predictive reanalysis with a consistent fold seed

This analysis makes no model queries. It uses fold seed 777 for the point estimate and all 2000 paired prompt-bootstrap replicates. Bootstrap sampling uses the separate seed 999.

Intervals below are central 95% prompt-bootstrap intervals. The fold-seed audit is a sensitivity analysis, not a confidence interval.

## Fixed-fold point metrics

```text
               model  n_rows      mae     rmse       r2  n_params
       additive_head     239 0.379330 0.454974 0.759141        14
      count_additive     239 0.385506 0.461420 0.752268        23
 count_plus_PB_count     239 0.377234 0.440863 0.773850        24
 count_plus_PE_count     239 0.259037 0.311709 0.886945        24
 count_plus_BE_count     239 0.385804 0.459786 0.754019        24
count_plus_all_pairs     239 0.235563 0.282239 0.907312        26
```

## Paired prompt-bootstrap delta MAE versus count-additive

```text
      baseline                model  n_bootstrap  delta_mae_mean  delta_mae_median  delta_mae_q025  delta_mae_q975  p_delta_gt_0  central_95_interval_excludes_zero
count_additive        additive_head         2000        0.006127          0.006144        0.004937        0.007217        1.0000                               True
count_additive  count_plus_PB_count         2000        0.008285          0.008300        0.006422        0.010089        1.0000                               True
count_additive  count_plus_PE_count         2000        0.126544          0.126591        0.119532        0.133755        1.0000                               True
count_additive  count_plus_BE_count         2000       -0.000231         -0.000243       -0.001486        0.001071        0.3595                              False
count_additive count_plus_all_pairs         2000        0.149930          0.149992        0.142519        0.157760        1.0000                               True
```

## Fold sensitivity across 200 alternative seeds

```text
pair               model  n_fold_seeds  delta_mae_min  delta_mae_q025  delta_mae_median  delta_mae_q975  delta_mae_max  positive_fraction  any_sign_flip  material_instability_positive_fraction_lt_0_95
  PB count_plus_PB_count           200      -0.000824        0.003236          0.009444        0.014547       0.017777              0.995           True                                           False
  PE count_plus_PE_count           200       0.115268        0.116993          0.129084        0.140302       0.149828              1.000          False                                           False
  BE count_plus_BE_count           200      -0.007764       -0.003235          0.001918        0.006967       0.008306              0.770           True                                            True
```

A positive delta MAE means that adding the named pair term improves held-out prediction over the count-additive model. `any_sign_flip` records even a rare reversal. `material_instability` means that fewer than 95% of audited splits improve.

## Stability flags

- PB: 99.5% of folds improve (1/200 reversals); a rare split reversal.
- PE: 100.0% of folds improve (0/200 reversals); no split reversal.
- BE: 77.0% of folds improve (46/200 reversals); material split instability.
