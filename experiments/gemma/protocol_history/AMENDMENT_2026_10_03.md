# Gemma inclusion and selection amendment

This amendment records the study author's feedback before any live Gemma preflight or scientific intervention measurement. It supersedes the original protocol's discretionary language about whether a weak scientific result is worth including. The original protocol, sources, and preflight freeze remain unchanged and available for comparison.

## Inclusion is decided by readiness

The cutoff is **6 October 2026 at 23:59 America/New_York**, equivalent to 7 October at 03:59 UTC. If the declared comparison is complete and its analysis and integrity checks have passed by that cutoff, report it in the SIGMETRICS submission regardless of the effect direction, magnitude, or significance: at least one appendix figure and one or two sentences in Section 4.6. A dense map, a ridge tie, no probe saving, or poor intervention selection is not a reason for omission. Engineering checks are not a scientific study and will not be inserted as one.

If completion and verification miss the cutoff, defer the entire scientific addition to v2. Retain the attempt, failures, and costs; do not select a promising partial endpoint for the submission. Do not lengthen the run, replace the checkpoint, change the scale, or add prompts in response to an unfavorable result. The comparison and primary endpoint remain those in the original protocol. The abstract changes only if a completed result changes the paper's substantive story, not merely because another model has been measured.

## Selection and collateral response

The menu is fixed at the existing **64 signed unit-norm actions plus the zero action**, separately evaluated at each of the three declared scales. Every map-based selector chooses the action with the largest predicted mean target response; direct search chooses the action with the largest measured mean target response on its permitted fitting subset. Include zero with effect zero. Resolve exact ties by choosing zero first, then the lexicographically smallest fixed action ID. Freeze the selected ID before test outcomes are opened. There is one selected action per method, design seed, scale, and budget, not an outcome-dependent choice for each test prompt.

The target remains the refusal-minus-compliance log-probability margin on harmful prompts. Collateral reporting includes both:

- **Benign margin disturbance**, already declared: the mean absolute change of that margin on the 32 disjoint benign prompts, plus its signed mean. Increased refusal on benign requests is not credited as target improvement.
- **Benign next-token distribution disturbance**, now explicit: mean `KL(P_clean || P_edited)` at the first assistant-token prediction position on the same 32 benign prompts. Compute log probabilities in float32, over the full vocabulary, without truncating to selected tokens or sampling a continuation. Retain per-prompt values for every action in the same 65-action menu. The zero action has zero KL by definition and is verified as a no-op.

Report target-selection loss and these collateral quantities side by side for the selected action, with paired prompt-family intervals. The selector is target-only, as requested; this experiment does not claim collateral-constrained optimization or safety. Do not choose a different action after inspecting collateral results. The common benign prompt set and menu permit fair comparison of target gains and collateral movement.

The additional KL acquisition costs 32 clean prompt forwards plus `64 × 32 × 3 = 6,144` edited prompt forwards. It uses prompt-only forwards, not the eight continuation scores, and is accounted separately. The original planning count of 449,824 becomes **456,000** mixed prompt/prompt-continuation evaluations. These are logical unbatched evaluations, not GPU launches or a conversion to elapsed time. The frozen engineering worker reports the original base-plan estimate; its report must be supplemented with the KL cost before deciding on the full comparison. No claim relies on an unmeasured backward-to-forward conversion.

## Bounded compute and licensed access

The proposed study ceiling is **12 allocated GPU-hours including setup, preflight, collection, failures, transfer, and cleanup**: at most one hour for this preflight, with the unused portion available only within the same total ceiling if the full comparison is later authorized. Reserve up to 11 hours for the full comparison as a planning allocation. Do not exceed 12 hours without a new decision, and do not treat model-generation time as the entire cost. This is a new study-specific bound, not a revival of the withdrawn first-32 research envelope. Completing the preflight does not authorize the full comparison automatically.

The user confirmed that Colab's `HF_TOKEN` secret is enabled. The visible notebook's access-only check read that secret and received HTTP 200 for the exact pinned Gemma configuration. This establishes current authorized file access, not a new acceptance of legal terms by Codex. The token remains in Colab's secret store and runtime environment; it is never copied to the host, an upload, or a result file.

The user supplied an already connected A100 notebook. Its displayed URL can retain an old backend identifier; reconcile it against the account inventory using a harmless notebook-written marker before adopting the idle runtime for the preflight. Do not infer ownership solely from the URL, touch a different runtime, or retry allocation. This branch allocates no new runtime. Record the start of study use; earlier user-owned allocation time is unknown and must not be represented as zero. No other workload may be running when the preflight starts.

## Paper space and interpretation

Reserve the new figure for the appendix. The one or two Section 4.6 sentences must replace at least as much main-text space: first trim the redundant Measurement budget and access paragraph; if necessary, fold the existing ObserverBench sentence into Related work. Rebuild and verify the 20-page technical limit before inclusion. This amendment plans that edit; it does not change the paper now.

A sparse-recovery advantage requires favorable structure such as compressibility; the arbitrary basis selection does not guarantee it. Aggregate ridge can also differ from coordinate probing through coverage and conditioning, so not every aggregate-method advantage establishes sparse structure. Report a dense map or a simple-method tie as the measured regime and the appropriate method choice. Retain the cross-family and 32-direction scope: a 9B checkpoint is not a scaling law, and this does not reopen the ObserverBench internal-versus-output comparison.

Judgment: positive for the amendment because it removes outcome-dependent inclusion and makes the decision endpoint reflect collateral movement. Scientific value remains unresolved. Next: finish the bounded engineering preflight, report its real timing and precision checks, and make the full-run readiness decision under this rule.
