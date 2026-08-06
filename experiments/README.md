# Experiment and artifact map

The repository contains every experiment reported in version 2 of the paper.
The harnesses remain separate because they use different dependency stacks and
access regimes. Frozen outputs sit beside the code that produced them.

| Paper section | Experiment | Source and frozen outputs | Hardware for a full rerun |
|---|---|---|---|
| Sec. 4 | Belief-state observer and control | [`hmm/`](hmm/) | CPU, MPS, or GPU |
| Secs. 5.1--5.2 | Aggregate recovery and finite AtP calibration | [`hmm/`](hmm/) and [`revision_checks/`](revision_checks/) | CPU, MPS, or GPU |
| Sec. 5.3 | Planted pair recovery | [`planted_interactions/`](planted_interactions/) | CPU |
| Sec. 5.3 | Designed HVP recovery | [`hvp_interactions/`](hvp_interactions/) | CPU |
| Sec. 5.4 | Tracr basis and detector writeback | [`tracr/`](tracr/) | CPU; pinned Tracr environment |
| Sec. 5.5 | GPT-2-small IOI | [`ioi/`](ioi/) and [`revision_checks/`](revision_checks/) | GPU/MPS and TransformerLens |
| Sec. 5.6 | Qwen-2.5-7B finite-response surface | [`qwen/`](qwen/) | CPU for frozen analysis; A100/H100-class GPU for full measurement |

## Recommended order

1. Run `python ../scripts/verify_release.py` from this directory, or
   `python scripts/verify_release.py` from the repository root.
2. Reproduce the Qwen result from its public notebook or install the Qwen
   subproject and run its tests.
3. Use the frozen CSV/JSON/NPZ files for paper-table and figure analysis.
4. Run an expensive model experiment only when fresh measurements are needed.

## HMM, aggregate recovery, and AtP calibration

The HMM harness trains the two-HMM transformer and produces the checkpoint used
by the aggregate and finite-scale experiments. The relevant entry points are:

```bash
cd experiments/hmm
python hmm_observer_control.py --quick --device cpu --outdir runs/smoke
python nt_mi_correspondence.py --run-dir frozen --device cpu --outdir runs/aggregate
python attribution_vs_finite_step0.py --run-dir frozen --device cpu --quick \
  --outdir runs/calibration
```

The paper's crossed context-by-design calibration analysis is in
`revision_checks/calibration_seed_factorial.py`; its official outputs are under
`revision_checks/frozen/calibration_seed_factorial_disjoint/`.

## Interaction experiments

Both interaction harnesses are standalone CPU studies:

```bash
cd experiments/planted_interactions
python claim3_planted_reach.py --quick --outdir runs/smoke

cd ../hvp_interactions
python claim3_hvp_baseline.py --quick --outdir runs/smoke
```

Their frozen budget, noise, support, and response-prediction tables are included.

## Tracr

The Tracr directory contains the label-basis analysis and final-residual detector
writeback used in the paper. Its README records the upstream Tracr dependency and
commands. We do not vendor the archived upstream repository.

## IOI

The four IOI stages retain their original command-line interfaces. Stage 1
checks complete group effects, Stage 2b measures broad random subsets, Stage 2c
uses the primary-stratified design, and Stage 2d performs the capacity-matched
pair decomposition. The later lexical/order, ablation, fold, and bootstrap
checks are in `revision_checks/`.

The frozen IOI outputs are sufficient for the post-processing stages; a full
fresh measurement requires GPT-2-small and TransformerLens.

## Qwen-2.5-7B

The Qwen subproject is the maintained public path. It includes a CPU-only frozen
analysis, a tested Python package, a resumable GPU runner, immutable model and
dataset revisions, the complete 401-action measurements, and a Colab notebook.
See [`qwen/README.md`](qwen/README.md) and
[`qwen/docs/QWEN_RESULT.md`](qwen/docs/QWEN_RESULT.md).

## Authorship notice

Every Python experiment file and the public notebook include:

> Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
