# Mechanistic Tomography

Paper, experiments, and frozen results for:

> Vijay Erramilli. *Mechanistic Tomography: Designed Measurement for Control-Oriented Interpretability*. Version 2.0, 2026.

- [Project page](https://kwisatzh.github.io/mechanistic-tomography/)
- [Version 2 PDF](assets/mechanistic-tomography-v2.pdf)
- [Version 2 source](paper/source/nt_mi_control_position_v21-v2.tex)
- [Version 1 archival DOI](https://doi.org/10.5281/zenodo.21797578)
- [Experiment guide](experiments/README.md)

Version 2 adds a held-out Qwen-2.5-7B experiment and releases the complete
reproducibility package. Version 1 remains frozen at Zenodo.

## Qwen-2.5-7B result

The Qwen study measures a finite refusal-response surface for 401 designed
actions. On 128 held-out actions and 224 held-out prompts, the calibrated
additive map reaches MAE 0.003790 and R2 0.9829. The lifted pairwise map reaches
MAE 0.003801 and R2 0.9835. The relative lifted MAE improvement is -0.29%, with
a paired two-way-bootstrap 95% interval of [-3.56%, 5.65%].

This is a stopping result for the measurement procedure: finite calibration is
adequate on the declared surface, so the held-out residual does not justify the
larger pairwise family. It is not a claim that interactions disappear in larger
models, and it does not establish mechanistic ground truth.

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
70cc633c69e89425de74696bbd04c60127ff2f32f0bbea0010e85e49c9906599  assets/mechanistic-tomography-v2.pdf
aca53bf0c108a0de1812edbbbf98ece0612a304a151f50cf3f13e109ac01544e  experiments/qwen/artifacts/frozen/qwen2_5_7b_a100_full_results.zip
```

## License

Code is licensed under the [Apache License 2.0](LICENSE). Paper text, PDFs, and
figures are released under [CC BY 4.0](LICENSE-PAPER.md).
