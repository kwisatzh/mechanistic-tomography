# Result Provenance

Every reported claim must be traceable through this chain:

`claim -> figure/table -> derived output -> frozen input -> command -> code/config/environment -> checksum`

Generated figures are never the source of truth. Frozen artifacts must be granular enough to recompute the reported statistics; metric summaries alone are insufficient.

## Required frozen contents

Each result directory under `artifacts/frozen` must contain:

- a short README describing the experiment and data units;
- the resolved configuration and captured environment;
- raw or minimally processed measurements;
- all splits, folds, masks, and random seeds;
- machine-readable reported metrics;
- a SHA-256 checksum list and a manifest conforming to the schema below.

Result-specific minimums are:

| Result | Required frozen measurements and metadata |
|---|---|
| R1 | Per-seed/per-observer RMSE, target error, nuisance collateral effect, controller/actuator configuration, observer definitions, represented and posterior references, and trajectories/checkpoints or an exact retraining configuration |
| R2 | Intervention masks/designs, responses, train/test split, budgets, seeds, predictions, sparse support, and ground truth |
| R3 | AtP maps, finite probes, scalar gains, held-out effects, seeds, perturbation scales, and comparison inputs |
| R4 | Masks or HVP query matrices, responses, main/pair ground truth and support, noise/confound settings, grids, splits, and seeds |
| R5 | Sequences, residual-label matrix, group/basis definitions, targets, splits, ablation outputs, and writeback outputs |
| R6 | Whole-group and subset masks/responses, P/B/E definitions, primary-stratified design, folds/seeds, prompt IDs or hashes, model and revision metadata, and regression/bootstrap inputs |

## Frozen-artifact manifest schema

Each result has one `manifest.json`. The following is the normative shape; optional values may be `null`, but keys must not be silently omitted when their absence affects reproduction.

```json
{
  "schema_version": "1.0",
  "paper": {
    "title": "string",
    "version": "string",
    "commit": "40-character git SHA",
    "release_tag": "string or null",
    "archive_doi": "string or null"
  },
  "result": {
    "result_id": "R1",
    "title": "string",
    "paper_locations": {
      "sections": ["string"],
      "figures": ["string"],
      "tables": ["string"]
    },
    "producer": {
      "repository": "URL or repository identifier",
      "commit": "40-character git SHA",
      "release_tag": "string or null",
      "archive_doi": "string or null",
      "entrypoint": "repository-relative path",
      "command": "exact non-interactive command",
      "source_files": [
        {"path": "repository-relative path", "sha256": "64 lowercase hex characters"}
      ]
    },
    "config": {
      "path": "repository-relative path",
      "sha256": "64 lowercase hex characters"
    },
    "environment": {
      "path": "repository-relative path",
      "sha256": "64 lowercase hex characters",
      "python": "version string",
      "lockfile": {
        "path": "repository-relative path",
        "sha256": "64 lowercase hex characters"
      },
      "extras": ["string"],
      "hardware": "string",
      "device": "cpu, cuda, mps, or other"
    },
    "external_models": [
      {"id": "string", "revision": "immutable revision", "framework": "string", "license": "string"}
    ],
    "seeds": [0],
    "inputs": [
      {
        "path": "repository-relative path",
        "media_type": "MIME type",
        "sha256": "64 lowercase hex characters",
        "role": "raw, minimally-processed, split, fold, mask, or reference",
        "upstream": {
          "repository": "string or null",
          "commit": "string or null",
          "release_tag": "string or null",
          "archive_doi": "string or null"
        }
      }
    ],
    "outputs": [
      {
        "path": "repository-relative path",
        "media_type": "MIME type",
        "sha256": "64 lowercase hex characters",
        "role": "metric, figure, table, or diagnostic",
        "canonical": true
      }
    ],
    "claims": [
      {
        "claim_id": "stable identifier",
        "statement": "machine-readable short statement",
        "test": "test name or repository-relative test path",
        "tolerance": "explicit numerical threshold or interval"
      }
    ],
    "runtime": {
      "mode": "frozen, smoke, or full",
      "expected": "human-readable duration",
      "hardware": "reference hardware"
    },
    "created_at": "RFC 3339 UTC timestamp",
    "notes": "string or null"
  }
}
```

## Validation rules

The provenance check must fail if:

- a path is absolute, escapes the repository, is missing, or has the wrong SHA-256 digest;
- an external model or upstream dataset lacks an immutable revision;
- a seed, split, fold, intervention mask, or configuration needed by an analysis is absent;
- a figure/table identifier is duplicated or is not mapped to a claim and producing command;
- a reported statistic cannot be recomputed from frozen measurements;
- an IOI artifact lacks its pinned ObserverBench commit plus release tag or archive DOI.

Claim tests should assert scientific tolerances, not exact plot pixels. They should cover the paper's actual comparisons: R1 observer rank agreement and nuisance collateral effects; R2 recovery-versus-budget behavior; R3 calibrated AtP held-out performance; R4 main and pair recovery separately; R5 native/restricted/lifted recovery and writeback; and R6 the predeclared IOI contrasts, including `I_PE > I_PB` and held-out P×E dominance among evaluated pairs.

## Release policy

Use repository-relative manifests, immutable commits and model revisions, and lowercase SHA-256 digests. Keep ordinary Git history compact. Large frozen artifacts may live in the archival release rather than Git when the manifest records their stable retrieval location and checksum; reproduction must not depend on Git LFS. Do not archive model weights when an immutable public model identifier is sufficient.
