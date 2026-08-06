# IOI Stage 1: Self-Repair Feasibility Test

This package runs the smallest IOI experiment needed before building a full ObserverBench task.
It tests whether published GPT-2-small IOI Name Mover heads and Backup Name Mover heads show a non-additive self-repair term under group ablation.

The experiment is deliberately narrow. It does **not** rediscover the IOI circuit. It uses IOI as a known-answer test instrument: primary Name Mover heads and Backup Name Mover heads are published circuit groups, and we ask whether singleton-additive ablation misses their conditional redundancy.

## Effect sign convention

All effects are positive drops in IOI logit difference:

```text
drop(G) = LD_clean - LD_ablate(G)
```

The self-repair interaction is:

```text
interaction = drop(primary + backup) - drop(primary) - drop(backup)
relative_interaction = interaction / abs(drop(primary))
```

A positive interaction means that ablating both groups hurts more than the sum of ablating each group alone.

## Pre-registered go/no-go rule

```text
relative_interaction >= 0.30:
    GO: build the Stage-2 IOI ObserverBench task

0.10 <= relative_interaction < 0.30:
    DIAGNOSE: signal exists but is weak; inspect head sets, prompt templates, ablation convention

relative_interaction < 0.10:
    STOP/REDESIGN: do not build full IOI task yet
```

## Default head groups

Defaults follow the IOI circuit in GPT-2-small:

```text
primary Name Movers:
    9.9, 9.6, 10.0

Backup Name Movers:
    9.0, 9.7, 10.1, 10.2, 10.6, 10.10, 11.2, 11.9
```

These can be overridden with `--primary-heads` and `--backup-heads`.

## Install

```bash
cd ioi_stage1_v0
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Run mean ablation, END position only

This is the primary Stage-1 run.

```bash
PYTHONPATH=. python scripts/ioi_stage1_self_repair.py \
  --outdir runs/ioi_stage1_mean_end \
  --device auto \
  --n-prompts 256 \
  --n-reference 512 \
  --batch-size 32 \
  --ablation mean \
  --positions end
```

## Robustness: zero ablation

Run zero ablation second, not first. It can be off-manifold.

```bash
PYTHONPATH=. python scripts/ioi_stage1_self_repair.py \
  --outdir runs/ioi_stage1_zero_end \
  --device auto \
  --n-prompts 256 \
  --batch-size 32 \
  --ablation zero \
  --positions end
```

## Run both conventions

```bash
PYTHONPATH=. python scripts/ioi_stage1_self_repair.py \
  --outdir runs/ioi_stage1_both_end \
  --device auto \
  --n-prompts 256 \
  --n-reference 512 \
  --batch-size 32 \
  --ablation both \
  --positions end
```

## Outputs

Each run writes:

```text
ioi_stage1_condition_results.csv
ioi_stage1_summary.csv
ioi_stage1_metadata.json
ioi_stage1_prompts.csv
ioi_stage1_bar.png
```

The most important file is `ioi_stage1_summary.csv`. Send this file back first.

## What to send back

Please send:

```text
runs/ioi_stage1_mean_end/ioi_stage1_summary.csv
runs/ioi_stage1_mean_end/ioi_stage1_condition_results.csv
runs/ioi_stage1_mean_end/ioi_stage1_bar.png
```

If you run zero ablation too, send the same files for that run.

## Notes

- The primary ablation convention is template-conditioned mean replacement of `hook_z` for the selected heads, at the final prediction position.
- The hook used is `blocks.{layer}.attn.hook_z`, with shape `[batch, pos, head, d_head]` in TransformerLens.
- The script uses a fixed IOI template by default to keep token length constant for per-position mean replacement.
- The task is a feasibility check. Stage 2 should only be built if the five-row table clears the go/no-go threshold.
