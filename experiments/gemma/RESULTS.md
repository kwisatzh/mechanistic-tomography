# Gemma: aggregate measurements help prediction, while selection exposes a trade-off

Verified 4 October 2026. The primary-scale experiment is complete: all 2,618 fixed checkpoints are retained, the frozen analysis has run, and a fresh Colab account listing confirms that no GPU runtime remains. Total charged occupancy, including both preflights, was **7.31 hours**, within the 14-hour allowance.

The predeclared primary result is positive. At 16 fitting interventions plus eight validation interventions, aggregate ridge has **21.6% lower held-out mean-response MSE** than coordinate ridge. The paired 95% family-bootstrap interval excludes zero. Selection and collateral measurements make this more informative than an additional prediction score: at this budget, OMP predicts more accurately than aggregate ridge but chooses a weaker intervention, while aggregate ridge's stronger target effect comes with greater disturbance on benign prompts.

## What was measured

The study uses Gemma-2-9B-it, revision `11c9b309abf73637e4b6f9a3fa1e92e615547819`, in FP32 with TF32 disabled. Its 32 declared directions span evenly distributed layers and were constructed from a separate set of 32 prompts. Fits use 32 harmful prompts; evaluation uses 96 held-out harmful prompts and 32 benign prompts. The intervention scale is 0.5. Secondary scales were deferred before acquisition.

The target is the change in a fixed, length-normalized refusal-versus-compliance log-probability margin. Each margin scores four refusal stems and four compliance stems. Larger target values mean a stronger refusal-stem margin on this readout, not a measured reduction in harmful generated behavior. All selection rules choose from the same 64 signed interventions plus doing nothing. The selection rule uses fitting data only and optimizes the target; collateral effects are measured afterward, not optimized away.

This is a cross-family, forward-only finite-effect measurement comparison in a constructed basis. It does not extend the closed ObserverBench internal-versus-output comparison, establish model safety, or demonstrate scaling to an unrestricted head- or feature-level basis.

## Primary result: better aggregate prediction at the fixed budget

The primary contrast is aggregate-ridge MSE minus coordinate-ridge MSE, averaged over the three frozen design seeds at fitting budget 16. Both methods pay for eight validation interventions, for 24 intervention measurements each.

| Quantity | Result |
| --- | ---: |
| Coordinate ridge mean-response MSE | 0.000379037 |
| Aggregate ridge mean-response MSE | 0.000297122 |
| Difference, aggregate minus coordinate | −0.000081915 |
| Paired 95% family-bootstrap interval | [−0.000098230, −0.000067489] |
| Relative MSE reduction, point estimate | 21.6% |

The interval uses the frozen 2,000 paired resamples over the 95 target source-family labels and keeps methods and intervention responses paired. It conditions on this model, constructed basis, and the three fixed design seeds; it does not estimate variability across trained models or new direction constructions. Aggregate ridge wins on two seeds and loses slightly on the third. The primary mean and interval should therefore be reported together with the seed results, not as a universal advantage for every design.

## Better prediction does not automatically choose a better intervention

The following values are arithmetic means over the three design seeds at fitting budget 16. Regret is the gap from the best intervention's mean response on the held-out menu, including zero; smaller is better. The table's secondary comparisons are descriptive point estimates, not newly introduced significance tests.

| Method | Charged interventions | Mean-response MSE | Selected target effect | Regret | Benign absolute margin change | Benign KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Coordinate ridge | 24 | 0.000379037 | 0.006964 | 0.032769 | 0.012158 | 0.000818 |
| Aggregate ridge | 24 | 0.000297122 | 0.029712 | 0.010021 | 0.037219 | 0.004144 |
| Aggregate OMP | 24 | 0.000148359 | 0.020140 | 0.019593 | 0.028981 | 0.001294 |
| Coordinate central differences | 16 | 0.000380483 | 0.001768 | 0.037965 | 0.015374 | 0.000919 |
| Direct menu search | 16 | — | 0.023974 | 0.015759 | 0.031238 | 0.002540 |
| Cost-matched direct menu search | 24 | — | 0.023974 | 0.015759 | 0.031238 | 0.002540 |
| Random menu choice | 0 | — | −0.020199 | 0.059932 | 0.020018 | 0.001344 |
| No intervention | 0 | — | 0.000000 | 0.039733 | 0.000000 | 0.000000 |

Direct search chooses among its measured fitting-menu candidates and does not estimate a complete response surface, so it has no map-prediction MSE. At this budget the 16- and 24-measurement searches select the same candidates in each seed. Aggregate ridge chooses the same candidate as cost-matched search in one seed, a stronger candidate in another, and a weaker candidate in the third. Its average target effect is higher, but so are its average collateral measurements. This supports a measured target–collateral trade-off, not an unqualified decision-efficiency win.

OMP's validation-selected support sizes are two, four, and four at this budget. Its mean prediction error is lower than aggregate ridge's in all three seeds, yet it chooses menu item 1 each time, with target effect 0.020140 and regret 0.019593. Aggregate ridge's average regret is 0.010021. The sparse fit is useful for predicting the surface here, but ranking its upper end is a different objective. These observations do not establish that the complete underlying map is sparse.

Benign signed margin changes at budget 16 are 0.004417 for coordinate ridge, 0.035212 for aggregate ridge, 0.028981 for OMP, and 0.031238 for cost-matched search. The report retains absolute changes as well, because signed averages can cancel. KL is `KL(P_clean || P_edited)` over the full vocabulary at the first assistant-token position. The benign split contains only two source families: its predeclared family intervals are retained in the full results, but they offer limited evidence about broader benign-prompt distributions. Neither KL nor this margin is a certified measure of behavioral harm.

## The complete budget curve

The primary budget was fixed at 16; the other budgets describe how the same methods develop as measurements increase. Entries below are mean-response MSE multiplied by one million, averaged over the three seeds.

| Method | k = 8 | k = 16 | k = 32 | k = 64 |
| --- | ---: | ---: | ---: | ---: |
| Coordinate ridge | 385.508 | 379.037 | 328.767 | 0.583 |
| Aggregate ridge | 381.408 | 297.122 | 14.612 | 0.600 |
| Aggregate OMP | 587.196 | 148.359 | 2.500 | 0.599 |
| Coordinate central differences | 385.034 | 380.483 | 327.285 | 0.623 |

The ridge and OMP methods use `k + 8` intervention measurements; central differences use `k`. Coordinate designs use positive/negative pairs, so `k = 16` covers eight of the 32 coordinates, and `k = 64` covers all of them. At the smallest budget, aggregate ridge and coordinate ridge are close and OMP is worse. At 32, aggregate methods predict much better than these partial-coordinate designs. At 64, the map estimators are close, with coordinate ridge slightly better by mean MSE. The result is about the value of aggregate measurements before full coordinate coverage, not an advantage at every budget.

| Method | Target effect, k = 8 | k = 16 | k = 32 | k = 64 |
| --- | ---: | ---: | ---: | ---: |
| Coordinate ridge | −0.002254 | 0.006964 | −0.004619 | 0.039733 |
| Aggregate ridge | 0.000569 | 0.029712 | 0.039045 | 0.039733 |
| Aggregate OMP | 0.002050 | 0.020140 | 0.039733 | 0.039733 |
| Coordinate central differences | −0.000812 | 0.001768 | −0.004619 | 0.039733 |
| Direct menu search | 0.014435 | 0.023974 | 0.028348 | 0.039733 |
| Cost-matched direct menu search | 0.023974 | 0.023974 | 0.037670 | 0.039733 |

Random choice and no intervention remain at −0.020199 and 0, respectively. Cost-matched search uses `min(k + 8, 64)` probes, because the menu contains 64 nonzero candidates. At `k = 32`, OMP selects the best held-out mean-response candidate in every seed, while aggregate ridge does so in two. At `k = 64`, all map and direct-search methods select that candidate. Its mean target effect is 0.039733, benign absolute margin change is 0.058089, and benign KL is 0.008323. The largest target effect in this menu therefore also has measurable collateral movement on the recorded readouts. No acceptability threshold for that movement was declared.

Mean-response prediction must also be distinguished from individual-prompt prediction. At `k = 64`, the map methods' per-prompt MSE is about 0.0000993, although their mean-response MSE is about 0.0000006. Fitting the population-average response accurately does not remove response variation across prompts.

## Checkpoints, integrity, and cost

All 2,618 checkpoint hashes, job identities, direction bindings, and frozen source hashes pass. Every resume prefix is byte-identical to the retained predecessor. All coefficient estimates, validation choices, menu predictions, and selected IDs were committed before every held-out measurement; recomputing that commitment from fitting data alone reproduces the choices exactly and the numerical values within the recorded tolerance. All retained arrays are finite. The token audit is identical across all four leases: 192 prompts, no truncation, and at most 52 prompt tokens. Each lease passed all six startup KL checks. The 24 checkpoint/analysis/controller CPU tests pass again at handoff.

| Charged interval | Minutes | End state |
| --- | ---: | --- |
| Both prior preflights | 22.12 | Preserved, included in allowance |
| Lease 001 | 86.15 | Infrastructure loss after 750 retained blocks |
| Lease 002 | 162.40 | Orderly boundary after 2,022 retained blocks |
| Lease 003 | 162.48 | Orderly boundary after 2,599 retained blocks |
| Lease 004 | 5.60 | Complete at 2,618 retained blocks |
| Total | 438.74 | 7.31 of 14 authorized hours |

There were four of at most six authorized leases and no simultaneous owned runtimes. The first lease's loss did not erase retained measurements; later leases resumed the unchanged checkpoint prefix. The final lease was explicitly released, its release receipt records account absence, and a new account listing at 10:14:21 UTC again returned zero active runtimes. The final host controller and its keep-awake helper have exited.

Retained measurements account for 19,112 batched forward calls, 152,896 sequence evaluations, 6,087,145 unpadded input tokens, and 2.31 hours of timed measurement work. Peak recorded CUDA allocation was 34.80 GiB. The 7.31-hour ledger also charges preflights, setup, transfers, acknowledgement waits, lost-session time, and cleanup; it is elapsed occupancy, not a provider billing statement. User-owned occupancy before the first preflight is unknown and is not silently assigned zero cost.

An intervention measurement in the method tables averages 32 fitting prompts, each with eight stem evaluations: 256 sequence evaluations, or 32 forward calls at batch size eight. The displayed method budgets exclude shared direction construction (32 sequence evaluations), clean fitting responses (256), and the held-out/collateral audit. All those shared and evaluation costs are included in the full acquisition ledger. No conversion from backward passes or HVPs is involved.

## Review record and next step

The complete 96 method/seed/budget rows, all declared metrics, and the frozen intervals are in [analysis/results.json](analysis/results.json). [analysis/descriptive_summary.json](analysis/descriptive_summary.json) provides all 32 method/budget summaries and seed ranges; ranges are descriptive, not confidence intervals. [analysis/integrity.json](analysis/integrity.json), [analysis/analysis_binding.json](analysis/analysis_binding.json), and [analysis/cpu_tests.json](analysis/cpu_tests.json) bind the verification and analysis. The acquisition and statistical sources remain frozen; the local audit and summarization wrappers add no model measurements or new fitting rule.

The complete comparison was verified on 4 October, before the 6 October, 23:59 America/New_York readiness cutoff. It therefore meets the result-independent inclusion rule. The recommended next step is independent review of the retained results, followed by one appendix figure or table and a short main-text cross-family comparison that reports prediction, selection, and collateral effects together. No additional GPU run is needed for this addition. The manuscript and public artifact have not been edited or published by this follow-up, and no CPU-panel work was resumed. The recurring follow-up is paused; its receipt and the final verification of both preflight freezes are in [analysis/handoff.json](analysis/handoff.json).

**Operational judgment: positive.** The experiment completed within its allowance, checkpoint recovery preserved prior work, and no owned GPU remains. **Scientific judgment: positive for the predeclared aggregate-versus-coordinate prediction comparison, with mixed decision evidence.** This supplies a bounded new Gemma result and a concrete example of why prediction quality and intervention choice must both be measured. It does not establish a general sparse-recovery advantage or an unconditional control or safety improvement. Stop acquisition here; review and report this complete comparison before considering any extension.
