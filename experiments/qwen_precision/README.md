# Prospective Qwen edit-accuracy audit

This audit measures how closely the original bfloat16 edit hook realizes its
intended displacement. It does not rerun or alter the historical Qwen response
surface, fits, or stopping comparison.

From the repository root:

```sh
python experiments/qwen_precision/reproduce.py
```

NumPy and the standard library suffice. This verifies all 29 retained archives
and recomputes errors for 1,216 nonzero layer-sequence edits from their before,
after, and intended arrays. Median relative displacement error is 8.24%, and the
maximum is 29.80%, above the predeclared 20% tolerance. Reproduction passes; the
edit-fidelity qualification fails. The saved report retains per-action errors,
norm ratios, cosines, and all zero/inactive checks.

With PyTorch installed, the three CPU hook tests can also be run:

```sh
python experiments/qwen_precision/test_audit.py
```

The original numerical audit and edit hook are supplied for inspection. The
private lease controller and account records are excluded. Historical activations
were not retained, so these engineering fixtures cannot reconstruct historical
rounding errors. The original scientific result describes the implemented
bfloat16 response surface. Full off-position tensors were not saved; those
position and hook-cleanup checks remain worker receipts rather than independent
array replays. `verification.json` records the original CPU check and total
elapsed runtime of 8.66 minutes, including the failed setup attempt.

`COPY_PROVENANCE.json` distinguishes byte-identical retained data from anonymized
author comments and documentation. No model weights, secrets, or account records
are included, and this package is not a live-runtime restart bundle.
