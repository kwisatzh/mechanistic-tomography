# Mechanistic Tomography

Paper, experiments, and frozen results for:

> Vijay Erramilli. *Mechanistic Tomography: Designed Measurement for Control-Oriented Interpretability*. Version 2.0, 2026.

- [Project page](https://kwisatzh.github.io/mechanistic-tomography/)
- [Version 2 PDF](assets/mechanistic-tomography-v2.pdf)
- [Version 2 source](paper/source/nt_mi_control_position_v21-v2.tex)
- [Version 1 archival DOI](https://doi.org/10.5281/zenodo.21797578)
- [Experiment guide](experiments/README.md)

Version 2 includes the held-out Qwen-2.5-7B experiment, the Gemma-2-9B-it
measurement comparison, and a prospective audit of Qwen edit precision.
The October 8 update adds reproduction settings, corrects the HMM measurement
accounting, and retains the recovered specificity records and CPU verification.
The V2 filename and URL are unchanged; Version 1 remains frozen at Zenodo.

The October 9 consistency update keeps the MI-oriented framing and full
appendices while making the theory easier to follow. It adds consolidated
Gemma prediction-and-choice and IOI interaction tables, clarifies Qwen's
length-normalized fixed-continuation score and squared-weight features, and
corrects the HVP figure's description of the synthetic pair terms. The paper
remains 35 pages. Numerical results, experimental code, and retained data are
unchanged; no model experiments were run for this update.

## HMM control and independent testing

The HMM study evaluates one estimate-and-edit step per token position. Observer
estimates from an unedited pass set edit sizes; a second pass applies the edits
and scores the outputs. Edited outputs do not feed another observer update.
The [HMM guide](experiments/hmm/README.md) lists all ten observers and distinguishes
the fixed-direction comparison from the mixed observer-and-actuator test.

The specificity result concerns output-probability movement, which increases
from 0.037 to 0.079; implied-log-odds movement instead decreases slightly.
The original and CPU records agree on that distinction. The trained readout and
exact posterior control about equally well, and their small ordering reverses
between records. The HMM recovery's 12 measurements count fitting only: 64 more
were used for validation. Gemma supplies the equal-total-budget comparison.

The [interaction guide](experiments/planted_interactions/README.md) explains the
correlated-distractor result: sparse fitting assigns the false feature zero
weight, while independent test masks evaluate the map without changing its
coefficients. The forward-measurement noise sweep also shows that locating
interactions can remain reliable after their coefficients become too noisy
for accurate prediction.

## Gemma measurement design

On Gemma-2-9B-it, 32 fitting and eight validation measurements give mean
held-out-response R-squared of 0.962 for aggregate ridge and 0.994 for OMP,
versus 0.154 for coordinate ridge, whose signed probes cover half the 32 directions.
Both measurement budgets were predeclared; the R-squared summaries are
post-hoc descriptive context. At the predeclared smaller fitting
budget of 16, aggregate ridge reduces MSE by 21.6%, with a paired interval
conditional on three fixed designs; both ridge maps predict weakly there.
OMP predicts better in all three designs but selects a weaker intervention in
two, and on average, at that smaller budget.
The choice task uses 64 fixed interventions plus no intervention. At the larger
budget, OMP selects the best menu item in all three designs. Selection and benign
collateral effects are reported alongside prediction, now in Section 5.1 of V2.

The [Gemma package](experiments/gemma/README.md) retains all 2,618 checkpoints,
fits, selections, analysis code, and CPU tests. Reproduce it without model weights:

```sh
python -m pip install numpy==2.3.5
python experiments/gemma/test_release.py
python experiments/gemma/reproduce.py
python experiments/qwen_precision/reproduce.py
```

## Qwen-2.5-7B result

The Qwen study measures a finite refusal-response surface for 401 designed
actions. On 128 held-out actions and 224 held-out prompts, the calibrated
additive map gives MAE 0.003790 and R2 0.9829. The lifted pairwise map gives
MAE 0.003801 and R2 0.9835. The relative lifted MAE improvement is -0.29%, with
a paired two-way-bootstrap 95% interval of [-3.56%, 5.65%].

The interval includes zero and extends slightly above the predeclared 5%
practical-improvement threshold. A gain just above that threshold is not ruled
out, but no MAE benefit is detected. The procedure stops at the calibrated
additive map for this basis, prompt distribution, and intervention range,
as implemented.

**Precision disclosure (October 4):** the historical pipeline added edits in
bfloat16 activation arithmetic. A prospective audit on 29 fixed actions and
16 benign sequences found median relative displacement error of 8.24% and maximum
29.80%, above the predeclared 20% tolerance. Historical activations were not
retained; this does not reconstruct their errors. The original results and
stopping decision describe the implemented bfloat16 response surface.
The [audit package](experiments/qwen_precision/README.md) contains the unchanged
before/after arrays and a CPU verifier. Gemma used float32 after its own bfloat16
preflight failed the same tolerance. No historical Qwen predictions were rerun.

Start with the
[public Colab notebook](experiments/qwen/notebooks/mechanistic_tomography_qwen_colab.ipynb).
Its default CPU path reproduces the reported table from frozen measurements
without loading Qwen. The optional GPU path reruns the measurements using the
pinned model revision.

## Reproducing the paper

The experiments are intentionally kept as small, self-contained harnesses.
Each directory contains the source, its own dependencies and commands, and the
frozen outputs used by the paper. See [experiments/README.md](experiments/README.md)
for the paper-section-to-code map.

The fastest verification path is:

```bash
python scripts/verify_release.py
python -m venv .venv-qwen
source .venv-qwen/bin/activate
pip install -e 'experiments/qwen[test]'
pytest experiments/qwen/tests
```

Full transformer reruns are opt-in. They require the hardware and external
model dependencies documented by the corresponding experiment. Frozen tables
and raw-enough measurements are included so analysis does not require another
model run.

## Integrity

```text
c94a297cac5988fc519c9d12dfb3c92c968b4fb221de9d84c0eff153b18de375  assets/mechanistic-tomography-v1.pdf
fae8a2faa4547bd2247eb23874d833fb65104f7b9cd096c28eb815604ffc9ee7  assets/mechanistic-tomography-v2.pdf
aca53bf0c108a0de1812edbbbf98ece0612a304a151f50cf3f13e109ac01544e  experiments/qwen/artifacts/frozen/qwen2_5_7b_a100_full_results.zip
```

## License

Code is licensed under the [Apache License 2.0](LICENSE). Paper text, PDFs, and
figures are released under [CC BY 4.0](LICENSE-PAPER.md).
