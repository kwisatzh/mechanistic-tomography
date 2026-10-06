# Planted interaction recovery

This CPU testbed separates three reasons for poor intervention prediction:
missing interaction features, too few measurements, and a correlated feature
with no causal effect. The response has 64 candidate components and four
nonzero main-and-pair terms among 2,080 candidates. The code and frozen results
are retained unchanged.

## Correlated distractor

`frozen/claim3_bad_design_confound/` records the distractor test. With probability
0.85 on each training row, one noncausal coordinate copies a causal coordinate.
The two columns are strongly correlated, not identical. Test masks are drawn
independently, with no forced copying.

At intervention scale 8, ridge assigns the distractor nonzero coefficients in
all three seeds, with absolute sizes 0.062 to 0.219. First-order and lifted OMP
assign it zero weight. Lifted OMP predicts the independent masks with R-squared
above 0.998 in every seed.

In `claim3_planted_reach.py`, `fit_omp` chooses sparsity using a validation split
from the training measurements, then refits on all training measurements. The
independent test masks score the resulting map without changing its coefficients
or selected features. Thus sparse fitting excludes the distractor, and test
masks verify prediction after the training correlation is removed. This is not
a recovery test for exactly aliased columns or aliased pair features.

## Noise and measurement access

`frozen/claim3_noise_combined/` summarizes the noise sweep with 96 forward
measurements. At scale 5, prediction and pair recall remain high through noise
standard deviation 0.01, while pair recall remains high through 0.05 after
prediction degrades. At scale 8, the corresponding levels are 0.05 and 0.2.
Finding the interacting pairs can therefore remain reliable after coefficient
errors make their predicted effects inaccurate.

The separate [HVP study](../hvp_interactions/) measures local pairs using
Hessian-vector products. In the quadratic testbed, combining those pairs with
gradient main effects meets the prediction and recall thresholds with 12 HVP
queries at scale 5 and 24 at scale 8, plus a backward pass. Pair-only HVP maps
locate the pairs but predict poorly without the main effects (R-squared 0.36
and 0.60). The forward-measurement noise sweep is not an HVP noise experiment.

## Retained records

The `frozen/` directories contain per-seed results, coefficients, configurations,
and combined budget and noise summaries. Reading these records requires no
new experiment. `claim3_planted_reach.py --help` documents the original runner
for fresh CPU measurements; those measurements are separate from reproducing
the retained results.
