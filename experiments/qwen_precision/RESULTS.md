# Qwen edit-fidelity audit: completed, tolerance not met

The authorized setup-only retry completed all 29 fixed actions on sixteen benign sequences using the original Qwen2.5-7B-Instruct hook, saved directions, and bfloat16 arithmetic. It did not regenerate the scientific response surface, fit a model, select interventions, or generate text.

## Result

Across 1,216 nonzero layer-sequence edits, maximum relative displacement error was **0.298017 (29.8%)**, and median error was **0.082358 (8.2%)**. The maximum exceeds the predeclared 0.20 tolerance, so the edit-fidelity qualification is negative. Relative error is the Euclidean norm of realized minus intended displacement, divided by the intended displacement norm; it is not an error in refusal probability or in the fitted prediction score.

| Declared scale | Nonzero edits checked | Median relative error | Maximum relative error |
| --- | ---: | ---: | ---: |
| 0.5 | 320 | 0.136971 | 0.298017 |
| 0.75 | 320 | 0.098414 | 0.207947 |
| 1.0 | 576 | 0.049129 | 0.124870 |

The scale-1 group includes the sixteen signed singleton controls; the scale summaries therefore cover different selected action geometries and are not a matched scale experiment. The full action-level records remain available.

Zero actions and inactive layers remained exactly unchanged. The frozen worker also checked that non-target token positions were unchanged, all captured arrays were finite, activation dtype was bfloat16, every required layer was observed, and all observation/edit hooks were removed. The independent CPU verification rechecked retained edited-position arrays, hashes, coverage, and every reported error, norm ratio, and cosine. Non-target full tensors were not retained, so their bitwise checks and hook removal rely on the frozen worker's receipts rather than a second array replay.

## Preservation, environment, and cost

All 29 checkpoint archives match their indexed hashes. The original and retry freezes verify, including byte-identical numerical audit, Qwen source, model revision, config, directions, fixtures, actions, requirements, and tests. The only remote setup change was creation of the system-site-packages environment with `--without-pip`; NumPy 2.0.2 then built successfully from source. The retained CPU qualification has three passing tests.

The runtime logged bfloat16 parameters, PyTorch 2.11.0+cu130, Transformers 5.14.0, NumPy 2.0.2, Accelerate 1.14.0, CUDA 13.0, and an A100-SXM4-80GB. The historical study recorded CUDA 12.8 and an A100 with 40 GB. The audit is prospective evidence for these fixtures and this original code path, not a reconstruction of historical activations or certification of all 401 original actions.

The failed first setup remains intact and cost 155.965546 seconds. This retry cost 363.681399 seconds. **Total charged occupancy was 519.646945 seconds (8.66 minutes), within the original one-hour allowance.** The worker exited naturally, no GPU process remained, the owned runtime was released, and a fresh account listing found zero active runtimes. The host monitor has exited.

## Interpretation and next step

Operational judgment: **positive**. The bounded retry completed, all records were retained and independently checked, and cleanup was verified.

Edit-fidelity judgment: **negative** for the predeclared 20% engineering tolerance on these selected fixtures. This audit does not reassess Qwen's prediction scores; they continue to describe the implemented bfloat16 response surface. It supplies no retrospective bound on historical displacement error and no claim about generated safety behavior.

Next: retain the original Qwen study unchanged and disclose the measured precision limitation in the private submission and review artifact. Do not rerun Qwen, change its precision, or spend the remaining allowance on additional checks for this submission. The private manuscript is updated; publishing updated artifact material remains a separate action.

Evidence: `verification.json`, `retained/report.json`, `retained/index.json`, the 29 `retained/checkpoints/` files, `retained/ready.json`, `retained/terminal.json`, `release.json`, `fresh_absence.json`, and the original failed attempt one directory above.
