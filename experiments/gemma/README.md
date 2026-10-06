# Gemma measurement design and intervention selection

This completed, single-scale Gemma-2-9B-it comparison contains all 2,618 original
checkpoint archives, frozen fits and choices, analysis code, CPU tests, and
precision qualification reports. It compares coordinate probing, aggregate
ridge, aggregate OMP, direct search, random choice, and no intervention.

## CPU reproduction

From the repository root, with the review dependencies installed:

```sh
python experiments/gemma/test_release.py
python experiments/gemma/reproduce.py
```

These commands need only NumPy and the standard library. They do not download
weights, read credentials, or allocate a GPU. The replay checks every checkpoint
hash, commitment-before-held-out ordering, all 96 saved fits and choices, the
complete frozen analysis including its bootstrap, and all 48 post-hoc R-squared
rows. Outputs are printed; retained files are never overwritten.

The primary comparison is at 16 fitting plus eight validation measurements:
aggregate ridge reduces mean-response MSE by 21.6% relative to coordinate ridge.
The paired interval is conditional on three fixed designs and resamples prompt
families, not designs. Both ridge predictions are weak at this budget; OMP has
mean R-squared 0.62 but selects a weaker intervention than aggregate ridge.

At the secondary budget of 32 fitting plus eight validation measurements, mean
R-squared is 0.962 for aggregate ridge, 0.994 for OMP, and 0.154 for coordinate
ridge, whose positive/negative probes cover 16 of the 32 directions. R-squared
was added after the primary analysis as descriptive context; both measurement
budgets were predeclared. Each map selects the largest predicted target response
from 64 fixed interventions plus no intervention. At this budget OMP
selects the best menu item in all three designs and aggregate ridge in two.
`analysis/results.json` retains all methods, budgets, per-design values,
selection outcomes, and collateral measurements. Stronger target responses
generally accompany larger benign responses at the primary budget; all actions
use the same coefficient norm, and benign results cover only two families.

## Protocol and provenance

`PROTOCOL.md` governs acquisition at scale 0.5 and supersedes the scale/budget
details in `protocol_history/`. The readiness amendment requires inclusion of
a complete comparison regardless of outcome. `analysis/integrity.json` retains
aggregate occupancy and measurement costs: 7.31 runtime-hours, including
preflights, versus a 14-hour allowance. It is an elapsed-time ledger, not billing.

`packet/analysis.py`, numerical worker code, designs, and checkpoint arrays retain
their original computations. This public copy includes first-party author notices.
`COPY_PROVENANCE.json` records original and release hashes; the original freeze
is historical evidence, not a claim that omitted private files are distributed.
Account/lease identifiers, authorization records, screenshots, and host launch
tools are excluded. Do not use this release to resume a historical live run.

Third-party prompt text is not redistributed. `packet/data/prompt_metadata.jsonl`
preserves row order, source IDs, split labels, and family labels, which suffice
for replay. The release test substitutes this metadata for the original prompt
file's hash check; the numerical tests are unchanged. After checking upstream
dataset terms, `packet/prepare.py` can retrieve pinned public sources and
regenerate the original prompt file for independent work. Its expected SHA-256
is `4c505354d60341f0d24103b75d88c29731fee4ec24d68cb9931402148b893dbf`.
The scientific collection source is included for inspection; recreating an
orchestrated GPU collection also requires a new, locally managed checkpoint
acknowledgment supervisor. No live controller or credentials are distributed.

Gemma weights require acceptance of their upstream license and authorized access;
they are not included. The adapter uses float32 with TF32 disabled after the
bfloat16 preflight failed its displacement tolerance. See `precision/` for both
reports. This is a cross-family measurement comparison, not a model-size scaling
law, generated-refusal-rate study, or safety evaluation.
