# Gemma measurement efficiency and intervention selection

This private study asks whether a finite response map over 32 declared residual-stream directions supports accurate prediction and useful intervention choice with fewer measurements than direct alternatives. It extends MT's eight-direction Qwen experiment to Gemma-2-9B-it and a larger basis. It is a cross-family, larger-basis generalization study, not a parameter scaling law or an ObserverBench result.

the study author's 3 October 2026 “do it” approves preparing the fixed protocol and an adapter and cost preflight. The full comparison is not launched by this packet. The preflight must establish correctness, access, and a measured completion estimate first. No paper, public repository, closed experiment, or model outcome is changed here. This proposal revisits the earlier September 29 empirical cutoff explicitly; inclusion in the October 9 submission remains conditional on a complete, checked result and available page space.

## Prior evidence and the distinct claim

The retained Qwen-2.5-7B result has a calibrated additive held-out R-squared of 0.982851 over eight directions, with no detected lifted advantage. The HMM selector replay shows that lower subset variance can be outweighed by calibration bias; it does not validate an improved selector. Gemma/APPS found no supported residual-readout advantage over a capacity-matched full-output readout. That checker comparison stays closed. The ObserverBench cost appendix establishes model-loading and extraction infrastructure, not causal-intervention fidelity.

Here the measured system is the intervened Gemma model, the decision is a fixed residual edit chosen before its response is measured, and the outcome is a continuous response to that edit. The intended benefit is measurement efficiency, not detection of malicious behavior or information uniquely available internally. Every method observes the same scalar output response. No APPS records, checker labels, SAE features, or ObserverBench outcomes enter this study.

The closest source records are `experiments/qwen/docs/QWEN_RESULT.md` in the public MT source, `notes/TEST1_PAPER_INTEGRATION_2026-09-25.md` in the private submission, the DecisionAwareMT `experiments/apps_matched_closeout_2026_09_18/CLOSEOUT.md`, and ObserverBench `sections/observer_costs.tex`. Existing recovery and decoder studies are not resumed.

## Measurement and basis

The exact checkpoint, 32 evenly distributed layer indices, scales, seeds, and other constants are in `config.json`. Each direction is the harmful-minus-benign mean residual contrast at the last prompt token, computed using only the direction-construction split. The direction at each layer has norm 5 percent of that layer's median construction residual norm, following the existing MT convention. These are content-contrast directions, not certified refusal mechanisms.

A coefficient vector a of Euclidean norm one specifies an edit; scale s multiplies a. All coordinate, aggregate, and menu vectors use this normalization. This equates coefficient-space energy across designs, not physical response or cumulative state displacement across layers. Changing density changes where the perturbation is applied and can change finite bias; that is part of the comparison, not a nuisance to hide.

The response is the change from the unedited prompt in the mean length-normalized log probability of four fixed refusal continuations minus the corresponding mean for four fixed compliance continuations. It is a continuous language-model readout, not a refusal-rate or safety outcome. No free-form answers or LLM judges are generated. The fixed stems are inherited from MT; the Qwen-branded system message is replaced with the neutral sentence in the config and included in Gemma's single user turn, because Gemma does not support a separate system turn. The same rendering is used for every arm.

Direction prompts and comparison prompts are separated by source family and normalized text using the existing preparation functions. HarmBench and XSTest are pinned to the same public source revisions as MT. Their prior use in MT is disclosed; this is not a newly invented benchmark. No Gemma intervention outcomes are used to select prompts, directions, layers, or scales. Truncation counts and token lengths must be retained.

## Fixed comparison

The three scales are 0.25, 0.5, and 1.0. The primary contrast is at scale 0.5 and 16 measured interventions, chosen before outcomes. Learning curves use budgets 8, 16, 32, and 64. Three fixed design seeds share prompts and evaluation menus; they are design sensitivity runs, not three independently trained models.

1. Coordinate probing uses a random layer order and consecutive positive and negative unit edits. Budgets 8, 16, 32, and 64 therefore inspect 4, 8, 16, and all 32 coordinates. Report central-difference prediction and a ridge fit on exactly those measurements; the stronger coordinate result is not selected on the test set.
2. Aggregate ridge uses independent signed, unit-norm dense masks, with prefixes at each budget.
3. Aggregate orthogonal matching pursuit uses the same measurements as aggregate ridge. The sparse estimator must beat ridge to support a sparsity-specific gain; beating coordinate probing alone cannot establish that.
4. Direct search measures a random prefix of the candidate intervention menu and chooses the largest mean response on fitting prompts. It is a decision baseline, with no invented full-surface prediction. It avoids fitting a response map and must appear alongside the model-based choices.
5. Retaining the unedited state and a uniformly random menu choice are cost-zero decision references. Exhaustive menu search is a separately charged evaluation reference, not a feasible method with a free oracle.

Ridge penalties and OMP support use eight common validation interventions on fitting prompts, whose cost is added to every tuned method. Direct search is shown both at k interventions and k+8, capped at the menu size, so a fitting method cannot hide its tuning cost. Hyperparameters minimize validation mean squared error; ties choose the stronger regularization or smaller support. The final implementation and analysis must be frozen before any comparison measurement. Calibrated attribution is a conditional extension: the current forward-only runner has no verified gradient path. If omitted, the claim is explicitly limited to forward-only measurement efficiency, with no claim to beat gradient-access procedures.

## Splits and evaluation

Construction uses 16 harmful and 16 benign prompts. Measurement fitting uses 32 harmful prompts; testing uses 96 disjoint harmful prompts and 32 benign prompts for collateral response. The exact identities and source-family checks are frozen before collection. Fit response maps to the mean finite response across fitting prompts, not to test responses. This population-level map chooses one edit per design, scale, and budget; it is not a per-prompt adaptive controller.

The held-out menu contains 64 independent signed unit-norm vectors plus the zero edit. The same menu applies at every scale. No menu vector is a coordinate, aggregate-fitting, or validation intervention. Direct search alone sees its budgeted menu responses on fitting prompts; those responses are not given to the map estimators. The action may be supplied to a predictor, but its test response may not. Choosing a model or edit from test outcomes is forbidden.

Primary prediction reporting is held-out MSE and MAE over the mean response of the 96 test prompts across the fixed nonzero menu. Per-prompt errors are also reported to expose averaging. The decision endpoint is the difference between the best measured mean test-menu response and the mean test response of the action selected on fitting information. Report the selected action's absolute effect, its gap to zero, and its mean absolute response on benign prompts; a larger refusal-margin response alone is not a safety improvement.

Commit model coefficients, validation choices, predictions, and selected action IDs before collecting or opening test-menu responses. Keep raw per-prompt, per-action effects. Bootstrap test prompt families, keeping all methods and actions paired; also give each fixed design seed separately. Conditional intervals do not capture checkpoint or direction-construction uncertainty. No favorable seed, scale, budget, or estimator replaces the primary contrast.

## Construction check and possible outcomes

| Possible response surface | Available evidence before choice | Meaning of the comparison |
| --- | --- | --- |
| Approximately additive and compressible | The same budgeted scalar responses for ridge and sparse recovery | Aggregate sparse recovery may save measurements, but must beat aggregate ridge and direct search on their relevant endpoints |
| Approximately additive and dense | The same responses, with no hidden support supplied | Ridge may be best; that is a useful simple-method result, not sparse recovery success |
| Nonlinear or strongly prompt-dependent | Budgeted measurements and the fixed validation responses | Held-out prediction or selected interventions can fail; report the scope boundary without retuning the task |
| Little usable intervention effect | Same evidence, with the zero action available | Report low decision room; do not amplify scales or select a more responsive model after seeing outcomes |

The primary statistical contrast is aggregate ridge minus coordinate ridge MSE at scale 0.5 and budget 16, averaging the three fixed design seeds. Positive evidence requires its paired 95 percent interval to lie below zero. OMP versus aggregate ridge is secondary, as is comparison with central differences. Decision efficiency is a separate claim against cost-matched direct search. A curve alone does not demonstrate a specified percentage saving. Negative evidence includes supported worse performance against these ordinary methods. Intervals crossing zero are inconclusive for superiority. All results can inform the manuscript, but a weak or engineering-only result need not be added. No automatic Qwen switch, extra prompts, or interaction search follows a null.

## Cost accounting and preflight

An intervention evaluation is not one model forward: eight continuations are scored per prompt. Report unbatched prompt-continuation equivalents, actual batch forward calls, input tokens, wall time, peak memory, and common setup separately. Direction construction, clean baselines, fitting probes, validation, exhaustive evaluation, and analysis each receive a ledger entry. Shared research acquisition must not be charged repeatedly as measured work, nor omitted from the research total. Each hypothetical method's deployment ledger charges its own construction and use. No fixed backward-to-forward conversion is assumed.

The preflight uses eight benign engineering prompts outside all study data, including two near-cap inputs, and deterministic random directions, not the scientific contrast directions. It tests pinned configuration and tokenizer access, exact template behavior, all 32 edit locations, finite scores, zero-edit equivalence, restoration after edits, and batched versus single-item scores. Zero edits and restored baselines must agree within 0.000001 log-margin units; batch-one versus batch-eight margins must agree within 0.02. The actual dense edit must leave other token positions unchanged, with relative displacement rounding error no greater than 20 percent at every layer. This deliberately loose BF16 engineering bound detects missing or badly rounded edits; it is not a scientific error guarantee. All observed errors are retained. It times short and near-cap prompts, a coordinate edit, a dense edit, and a direction-capture pass; it records the cost and memory without comparing estimators or choosing a favorable scale.

Only after CPU tests, source hashes, and model-file access pass may one owned A100 runtime be allocated. No existing runtime is commandeered. Setup is bounded at 30 minutes, live checks at 15 minutes, and ownership at 60 minutes including cleanup. These are preflight bounds, not reinstatement of the withdrawn research envelope. No full-model download is started merely to discover missing licensed access. Credentials remain in the normal secret store, never in this packet or printed logs. A failed preflight is retained and does not authorize repeated allocations.

Before the full run, produce a measured time projection with a factor-of-two planning margin, a complete frozen collection and analysis program, and a decision on whether there is time to finish and verify before submission. This packet contains no automatic full-run launcher. The current judgment is positive for preparing this distinct measurement study, with scientific value unresolved until measurements exist.
