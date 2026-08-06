# Prompt-data contract

The repository intentionally does not redistribute harmful-prompt datasets.
Prepare one JSONL file from the original, license-compatible sources. Every line
must have:

    {"id":"...", "text":"...", "label":"harmful|benign",
     "family":"...", "split":"direction|fit|test_id|collateral_id",
     "source":"..."}

Use behavior/template identity for `family`; paraphrases of one behavior must
share a family. IDs must be unique. Construction, fitting, and locked-test
families and normalized prompt texts must be disjoint. For XSTest, the upstream
safe-prompt `type` is the family. A type is assigned to only one split and one
execution profile. When an exact prompt target is reached partway through a
25-prompt XSTest type, the remaining rows in that type are discarded rather
than reassigned. For HarmBench, normalized behavior text defines the leakage
family, so distinct BehaviorIDs with identical text stay in the same split.

Recommended primary sources are the official HarmBench behavior CSV for harmful
direct requests and the official XSTest prompt CSV plus ordinary benign
instructions for collateral measurement. The currently pinned source files are:

- HarmBench commit `8e1604d1171fe8a48d8febecd22f600e462bdcdd`:
  `data/behavior_datasets/harmbench_behaviors_text_all.csv`
- XSTest commit `d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d`:
  `xstest_prompts.csv`

Preserve the upstream commit, raw-file SHA-256, filtering script, and final
JSONL SHA-256 in the run artifact. Do not silently replace or extend the locked
test file after measuring it.

Use `mechtomo prepare-data --profile pilot` for the reserved pilot and
`--profile full` for the locked full run. The pilot reserves 16 harmful
and 16 benign construction prompts, 8 harmful fitting prompts, 8 harmful test
prompts, and 25 benign collateral prompts. The full profile excludes every
pilot family, then selects 32+32 construction prompts, 112 harmful fitting
prompts, 224 harmful test prompts, and 150 benign collateral prompts. Together
the profiles use all 400 canonical HarmBench behaviors without cross-split text
reuse. XSTest types are assigned atomically; unused rows are recorded as
exclusions, not silently moved to another split.

For the execution smoke, a tiny locally reviewed JSONL with at least one harmful
and one benign construction prompt, several fit prompts, several locked target
prompts, and several benign collateral prompts is sufficient. Smoke results are
never paper evidence.
