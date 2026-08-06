# Repository Boundary

This repository owns the experiments reported in *Mechanistic Tomography*,
including the Qwen-2.5-7B weak-ground-truth extension added in paper v2. Its
purpose is to make each reported result inspectable and reproducible without
turning the artifact into a second benchmark platform.

## What belongs here

| Study | Repository ownership |
|---|---|
| HMM observer/control | Experiment-local observer, fixed controller, posterior references, error, and collateral-effect evaluation |
| HMM forward sparse recovery | Masks, finite interventions, sparse recovery, budgets, and evaluation |
| AtP calibration | Perturbation-scale calibration, held-out effects, and forward-recovery comparisons |
| Interactions | Planted main/pair effects, lifted masks, HVP and forward measurements, and confound checks |
| Tracr | Residual-label matrix, native/restricted/lifted bases, group effects, and final residual writeback |
| IOI | Group definitions, immutable measurements, folds/seeds, deterministic postprocessing, statistics, figures, and tables |
| Qwen extension | Frozen behavioral response surface, additive/lifted fits, held-out comparison, data contract, and Colab notebook |

The Qwen study remains a deliberately narrower claim than the ground-truthed
experiments. It tests finite-effect prediction on a declared intervention
surface under weak ground truth. It does not establish a represented variable,
sparse circuit support, or causal head group in Qwen.

## What belongs in ObserverBench

ObserverBench is the reusable benchmark and evaluation system. It owns general
observer/task/controller interfaces, ObserverCards, task registries, reusable
controller suites, adapters, plugins, and leaderboard infrastructure. Those
systems are not copied into this paper-reproduction repository.

## Reproduction levels

1. **Frozen reproduction:** verify checksums and recompute reported metrics from
   the saved measurements without downloading model weights.
2. **Native smoke runs:** exercise the smaller HMM and planted-interaction
   pipelines locally.
3. **Full reruns:** follow the experiment-specific instructions when the needed
   model, accelerator, and source datasets are available.

The Qwen directory preserves its pre-run source fingerprint. That is why a few
historical metadata strings remain unchanged inside the fingerprinted package;
the public documentation here records the completed scientific status.
