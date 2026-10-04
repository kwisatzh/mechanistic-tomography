# Qwen edit accuracy qualification

This bounded engineering check measures how faithfully the original Qwen intervention code realizes its intended residual displacement. It does not rerun the response-surface experiment, collect scientific outcomes, or compare predictive models. A failure changes the disclosure, not the original results or the run plan.

## Authorization and limits

the study author approved the proposed Qwen edit-accuracy check with “do it please” on 4 October 2026. One owned Colab runtime may be used for at most 3,600 seconds, including allocation, setup, and cleanup. Setup, including model loading, is limited to 1,800 seconds; the worker stops by 3,300 seconds, reserving cleanup time. No retry, purchase, scientific rerun, or new model is authorized. Retain failed invocations and all costs. Never start a second supervisor or adopt an unidentified existing runtime.

## Fixed inputs

Use the retained Qwen2.5-7B-Instruct revision, original bfloat16 loader and edit hook, original eight saved directions, and original system prompt, continuation banks, and 384-token bound. Copy the original source unchanged. The host preparation selects actions using only saved design metadata: zero, all sixteen signed singleton actions, and the first action at each of the twelve scale/density combinations (scales 0.5, 0.75, and 1; densities 0.25, 0.5, 0.75, and 1). No saved response values are consulted.

Eight fixed benign engineering texts cover short and truncated contexts. Each is followed by the first refusal stem and the first compliance stem, producing sixteen sequences. No text is generated. The original last-prompt-token positioning and editing code is used, with observers immediately before and after the hook. This covers both continuation lengths without computing refusal margins or scientific endpoint scores.

## Measurements and gates

For every action, layer, and sequence, retain the residual vectors before and after editing, the intended displacement (saved float64 direction times coefficient), and the edited token position. Compute relative displacement error as the Euclidean norm of realized minus intended displacement divided by the intended norm. Report its maximum, median, and per-scale summaries. Also report realized/intended norm ratios and cosines. Zero actions and inactive layers require exact equality, and every non-target position must remain bitwise unchanged by the hook. Require finite vectors, exact layer/row coverage, correct activation dtype, and removal of every observation/edit hook.

The predeclared engineering tolerance is relative displacement error at most 0.20 for every nonzero edit, matching the Gemma preflight tolerance. A threshold failure is a completed negative qualification, not grounds for changing precision, directions, fixtures, or tolerance and rerunning. Structural or nonfinite failures stop immediately. Each completed action is checkpointed and retained before cleanup. The historical library versions are pinned; differences in CUDA build and hardware are recorded rather than concealed.

## Interpretation and reporting

This prospective audit qualifies only these benign fixtures and selected saved actions. It cannot retrospectively certify the original prompt activations or all 401 original actions. The original Qwen fit remains a fit to its implemented bfloat16 response surface. Report qualification and scientific interpretation separately, disclose the evidence boundary, and recommend no full Qwen rerun before submission. The paper's precision inventory distinguishes retained configuration, source inference, and directly measured edit accuracy.
