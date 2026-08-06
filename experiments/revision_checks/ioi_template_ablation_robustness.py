#!/usr/bin/env python3
# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
"""Run the direct IOI group-interaction check across templates and ablations.

This wrapper composes the measurement primitives in the existing Stage 2c
script.  It loads GPT-2-small once, reuses the published head groups, and runs
the 21 anchor masks (clean, 13 single heads, and 7 non-empty group masks) for
four published IOI lexical frames in both ABBA and BABA order.  It repeats each
condition with template-conditioned mean ablation and zero ablation.

The primary robustness gate is pre-specified: the direct P-E interaction must
exceed the direct P-B interaction under both ablation conventions, and the
hierarchical 95% interval for P-E minus P-B must remain above zero.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


ATTRIBUTION = (
    "Experiments designed/concieved by Vijay Erramilli. "
    "Code written by Vijay Erramilli and Codex"
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE2C_SCRIPT = (
    REPOSITORY_ROOT
    / "experiments"
    / "ioi"
    / "stage2c"
    / "ioi_stage2c_primary_stratified.py"
)

# The first four lexical frames are from Redwood Research's published IOI
# dataset implementation.  We fix one published place/object pair per frame so
# that each template has a constant token length, then pair every frame with
# both of the dataset's ABBA/BABA name orders.
LEXICAL_FRAMES: Sequence[Tuple[str, str, str]] = (
    (
        "t0",
        "Then, {first} and {second} went to the store. {s} gave a ring to",
        "went/store/ring",
    ),
    (
        "t1",
        "Then, {first} and {second} had a lot of fun at the garden. "
        "{s} gave a drink to",
        "fun/garden/drink",
    ),
    (
        "t2",
        "Then, {first} and {second} were working at the office. "
        "{s} decided to give a computer to",
        "working/office/computer",
    ),
    (
        "t3",
        "Then, {first} and {second} were thinking about going to the station. "
        "{s} wanted to give a necklace to",
        "thinking/station/necklace",
    ),
)


def published_templates() -> List[Dict[str, str]]:
    """Return four lexical frames in each published name-order convention."""
    rows: List[Dict[str, str]] = []
    for frame_id, frame, lexical_description in LEXICAL_FRAMES:
        rows.append(
            {
                "template_id": f"abba_{frame_id}",
                "order": "ABBA",
                "frame_id": frame_id,
                "lexical_description": lexical_description,
                "template": frame.format(first="{io}", second="{s}", s="{s}"),
            }
        )
        rows.append(
            {
                "template_id": f"baba_{frame_id}",
                "order": "BABA",
                "frame_id": frame_id,
                "lexical_description": lexical_description,
                "template": frame.format(first="{s}", second="{io}", s="{s}"),
            }
        )
    return rows


def load_stage2c(path: Path):
    """Import the existing Stage 2c script without copying its primitives."""
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("ntmi_ioi_stage2c", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Stage 2c script from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_anchor_design(stage2c):
    primary = stage2c.parse_heads(stage2c.DEFAULT_PRIMARY_HEADS)
    backup = stage2c.parse_heads(stage2c.DEFAULT_BACKUP_HEADS)
    extra = stage2c.parse_heads(stage2c.DEFAULT_EXTRA_HEADS)
    heads = stage2c.build_head_records(primary, backup, extra)
    masks = stage2c.sample_primary_stratified_masks(
        heads,
        n_subsets=21,
        seed=333,
        include_all_singletons=True,
        include_group_subsets=True,
    )
    if masks.shape != (21, 13):
        raise RuntimeError(f"Expected a 21 x 13 anchor design, got {masks.shape}")
    return heads, masks


def group_code(stage2c, mask: np.ndarray, heads) -> str:
    counts = stage2c.group_counts(mask, heads)
    totals = stage2c.group_counts(np.ones_like(mask), heads)
    active = []
    for group in ("P", "B", "E"):
        count = counts[group]
        if count == totals[group]:
            active.append(group)
        elif count != 0:
            return "singleton" if int(mask.sum()) == 1 else "partial"
    return "".join(active) or "clean"


def anchor_rows(stage2c, masks: np.ndarray, heads) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for subset_idx, mask in enumerate(masks):
        counts = stage2c.group_counts(mask, heads)
        rows.append(
            {
                "subset_idx": subset_idx,
                "subset_name": stage2c.mask_name(mask, heads),
                "group_code": group_code(stage2c, mask, heads),
                "mask_bits": "".join(str(int(bit)) for bit in mask),
                "n_heads": int(mask.sum()),
                "n_P": counts["P"],
                "n_B": counts["B"],
                "n_E": counts["E"],
            }
        )
    codes = [row["group_code"] for row in rows]
    for required in ("clean", "P", "B", "E", "PB", "PE", "BE", "PBE"):
        if codes.count(required) != 1:
            raise RuntimeError(f"Anchor design must contain exactly one {required} mask")
    return rows


def measure_condition(
    stage2c,
    model,
    device: str,
    template_row: Mapping[str, str],
    heads,
    masks: np.ndarray,
    design_rows: Sequence[Mapping[str, object]],
    n_prompts: int,
    n_reference: int,
    batch_size: int,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Measure one template under both ablation conventions."""
    records = stage2c.make_prompts(
        model, n_prompts, seed, template_row["template"], names=None
    )
    ref_records = stage2c.make_prompts(
        model, n_reference, seed + 10_000, template_row["template"], names=None
    )
    tokens = stage2c.tokens_for_records(model, records, device)
    ref_tokens = stage2c.tokens_for_records(model, ref_records, device)
    if tokens.shape[1] != ref_tokens.shape[1]:
        raise RuntimeError(
            f"{template_row['template_id']}: eval length {tokens.shape[1]} "
            f"!= reference length {ref_tokens.shape[1]}"
        )
    io_tokens, s_tokens = stage2c.answer_token_tensors(records, device)
    clean = stage2c.compute_logit_diffs(
        model,
        tokens,
        io_tokens,
        s_tokens,
        batch_size,
        heads=None,
        ablation="none",
        mean_z=None,
        position_mode="end",
    )
    prompt_rows = [
        {
            "template_id": template_row["template_id"],
            "order": template_row["order"],
            "frame_id": template_row["frame_id"],
            "prompt_idx": prompt_idx,
            **asdict(record),
        }
        for prompt_idx, record in enumerate(records)
    ]

    long_rows: List[Dict[str, object]] = []
    for ablation in ("mean", "zero"):
        mean_z = (
            stage2c.compute_mean_z(model, ref_tokens, batch_size)
            if ablation == "mean"
            else None
        )
        for design_row, mask in zip(design_rows, masks):
            selected_heads = stage2c.mask_to_heads(mask, heads)
            if selected_heads:
                logit_diff = stage2c.compute_logit_diffs(
                    model,
                    tokens,
                    io_tokens,
                    s_tokens,
                    batch_size,
                    heads=selected_heads,
                    ablation=ablation,
                    mean_z=mean_z,
                    position_mode="end",
                )
            else:
                logit_diff = clean
            drops = clean - logit_diff
            for prompt_idx, (ld, drop) in enumerate(zip(logit_diff, drops)):
                long_rows.append(
                    {
                        "template_id": template_row["template_id"],
                        "order": template_row["order"],
                        "frame_id": template_row["frame_id"],
                        "ablation": ablation,
                        "prompt_idx": prompt_idx,
                        "subset_idx": int(design_row["subset_idx"]),
                        "subset_name": design_row["subset_name"],
                        "group_code": design_row["group_code"],
                        "logit_diff": float(ld),
                        "drop_from_clean": float(drop),
                    }
                )
    return prompt_rows, long_rows


def direct_metric_rows(measurements: pd.DataFrame) -> pd.DataFrame:
    """Convert the eight group anchors into per-prompt direct interactions."""
    group = measurements[measurements["group_code"].isin(
        ["clean", "P", "B", "E", "PB", "PE", "BE", "PBE"]
    )]
    pivot = group.pivot_table(
        index=[
            "template_id",
            "order",
            "frame_id",
            "ablation",
            "prompt_idx",
            "io_name",
            "s_name",
            "io_token",
            "s_token",
        ],
        columns="group_code",
        values="drop_from_clean",
        aggfunc="first",
    ).reset_index()
    required = ["P", "B", "E", "PB", "PE", "BE", "PBE"]
    missing = [column for column in required if column not in pivot]
    if missing:
        raise RuntimeError(f"Missing group-mask measurements: {missing}")

    for code in required:
        pivot[f"D_{code}"] = pivot[code]
    pivot["I_PB"] = pivot["PB"] - pivot["P"] - pivot["B"]
    pivot["I_PE"] = pivot["PE"] - pivot["P"] - pivot["E"]
    pivot["I_BE"] = pivot["BE"] - pivot["B"] - pivot["E"]
    pivot["I_PB_per_pair"] = pivot["I_PB"] / 24.0
    pivot["I_PE_per_pair"] = pivot["I_PE"] / 6.0
    pivot["I_BE_per_pair"] = pivot["I_BE"] / 16.0
    pivot["I_PE_minus_PB"] = pivot["I_PE"] - pivot["I_PB"]
    pivot["I_PB_minus_BE"] = pivot["I_PB"] - pivot["I_BE"]
    pivot["I_PE_minus_BE"] = pivot["I_PE"] - pivot["I_BE"]
    pivot["I_PBE_third_order"] = (
        pivot["PBE"]
        - pivot["PB"]
        - pivot["PE"]
        - pivot["BE"]
        + pivot["P"]
        + pivot["B"]
        + pivot["E"]
    )
    return pivot


METRICS: Sequence[str] = (
    "D_P",
    "D_B",
    "D_E",
    "D_PB",
    "D_PE",
    "D_BE",
    "D_PBE",
    "I_PB",
    "I_PE",
    "I_BE",
    "I_PB_per_pair",
    "I_PE_per_pair",
    "I_BE_per_pair",
    "I_PE_minus_PB",
    "I_PB_minus_BE",
    "I_PE_minus_BE",
    "I_PBE_third_order",
)


def summarize_metrics(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    template_rows: List[Dict[str, object]] = []
    for keys, frame in metrics.groupby(
        ["template_id", "order", "frame_id", "ablation"], sort=True
    ):
        base = dict(zip(["template_id", "order", "frame_id", "ablation"], keys))
        values = {metric: float(frame[metric].mean()) for metric in METRICS}
        interactions = {pair: values[f"I_{pair}"] for pair in ("PB", "PE", "BE")}
        ranked = sorted(interactions, key=interactions.get, reverse=True)
        template_rows.append(
            {
                **base,
                "n_prompts": len(frame),
                **values,
                "interaction_rank": ">".join(ranked),
                "pe_largest": ranked[0] == "PE",
                "pe_pb_be_order": ranked == ["PE", "PB", "BE"],
            }
        )
    per_template = pd.DataFrame(template_rows)

    overall_rows: List[Dict[str, object]] = []
    for ablation, frame in metrics.groupby("ablation", sort=True):
        values = {metric: float(frame[metric].mean()) for metric in METRICS}
        template_slice = per_template[per_template["ablation"] == ablation]
        overall_rows.append(
            {
                "ablation": ablation,
                "n_templates": int(frame["template_id"].nunique()),
                "n_prompts_per_template": int(frame.groupby("template_id").size().iloc[0]),
                **values,
                "templates_pe_largest": int(template_slice["pe_largest"].sum()),
                "templates_pe_pb_be_order": int(template_slice["pe_pb_be_order"].sum()),
            }
        )
    return per_template, pd.DataFrame(overall_rows)


def hierarchical_bootstrap(
    metrics: pd.DataFrame, repeats: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Resample templates, then paired name rows within sampled templates."""
    rng = np.random.default_rng(seed)
    template_ids = sorted(metrics["template_id"].unique())
    arrays: Dict[Tuple[str, str, str], np.ndarray] = {}
    for template_id in template_ids:
        for ablation in ("mean", "zero"):
            frame = metrics[
                (metrics["template_id"] == template_id)
                & (metrics["ablation"] == ablation)
            ].sort_values("prompt_idx")
            for metric in METRICS:
                arrays[(template_id, ablation, metric)] = frame[metric].to_numpy(float)

    rows: List[Dict[str, object]] = []
    for repeat in range(repeats):
        sampled_template_positions = rng.integers(0, len(template_ids), len(template_ids))
        values: Dict[Tuple[str, str], List[np.ndarray]] = {
            (ablation, metric): []
            for ablation in ("mean", "zero")
            for metric in METRICS
        }
        for template_position in sampled_template_positions:
            template_id = template_ids[int(template_position)]
            n_rows = len(arrays[(template_id, "mean", METRICS[0])])
            sampled_names = rng.integers(0, n_rows, n_rows)
            for ablation in ("mean", "zero"):
                for metric in METRICS:
                    values[(ablation, metric)].append(
                        arrays[(template_id, ablation, metric)][sampled_names]
                    )
        for ablation in ("mean", "zero"):
            for metric in METRICS:
                rows.append(
                    {
                        "bootstrap": repeat,
                        "ablation": ablation,
                        "metric": metric,
                        "value": float(np.concatenate(values[(ablation, metric)]).mean()),
                    }
                )
    bootstrap = pd.DataFrame(rows)
    summary_rows: List[Dict[str, object]] = []
    for (ablation, metric), frame in bootstrap.groupby(["ablation", "metric"]):
        vals = frame["value"].to_numpy(float)
        summary_rows.append(
            {
                "ablation": ablation,
                "metric": metric,
                "n_bootstrap": len(vals),
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "q025": float(np.quantile(vals, 0.025)),
                "q975": float(np.quantile(vals, 0.975)),
                "p_gt_zero": float(np.mean(vals > 0.0)),
            }
        )
    return bootstrap, pd.DataFrame(summary_rows)


def leave_one_template_out(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for heldout in sorted(metrics["template_id"].unique()):
        for ablation in ("mean", "zero"):
            heldout_frame = metrics[
                (metrics["template_id"] == heldout)
                & (metrics["ablation"] == ablation)
            ]
            remaining = metrics[
                (metrics["template_id"] != heldout)
                & (metrics["ablation"] == ablation)
            ]
            for metric in ("I_PB", "I_PE", "I_BE", "I_PE_minus_PB"):
                rows.append(
                    {
                        "heldout_template": heldout,
                        "ablation": ablation,
                        "metric": metric,
                        "heldout_mean": float(heldout_frame[metric].mean()),
                        "remaining_templates_mean": float(remaining[metric].mean()),
                    }
                )
    return pd.DataFrame(rows)


def robustness_gate(bootstrap_summary: pd.DataFrame, overall: pd.DataFrame) -> Dict[str, object]:
    gate: Dict[str, object] = {
        "definition": (
            "I_PE > I_PB for mean and zero ablation, with the hierarchical "
            "95% interval for I_PE_minus_PB above zero under both conventions"
        )
    }
    passes: List[bool] = []
    for ablation in ("mean", "zero"):
        point = overall.loc[overall["ablation"] == ablation].iloc[0]
        interval = bootstrap_summary[
            (bootstrap_summary["ablation"] == ablation)
            & (bootstrap_summary["metric"] == "I_PE_minus_PB")
        ].iloc[0]
        point_pass = bool(point["I_PE"] > point["I_PB"])
        interval_pass = bool(interval["q025"] > 0.0)
        gate[f"{ablation}_point_PE_gt_PB"] = point_pass
        gate[f"{ablation}_I_PE_minus_PB_q025"] = float(interval["q025"])
        gate[f"{ablation}_interval_above_zero"] = interval_pass
        passes.extend([point_pass, interval_pass])
    gate["pass"] = bool(all(passes))
    return gate


def plot_template_interactions(per_template: pd.DataFrame, outpath: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharey=True)
    colors = {"I_PB": "#4C78A8", "I_PE": "#E45756", "I_BE": "#72B7B2"}
    for ax, ablation in zip(axes, ("mean", "zero")):
        frame = per_template[per_template["ablation"] == ablation].sort_values(
            ["frame_id", "order"]
        )
        x = np.arange(len(frame))
        for metric in ("I_PB", "I_PE", "I_BE"):
            ax.plot(
                x,
                frame[metric],
                marker="o",
                linewidth=1.5,
                color=colors[metric],
                label=metric.replace("I_", ""),
            )
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(frame["template_id"], rotation=45, ha="right")
        ax.set_title(f"{ablation.capitalize()} ablation")
        ax.set_xlabel("template")
    axes[0].set_ylabel("direct group interaction (logit-difference drop)")
    axes[1].legend(title="pair", frameon=False)
    fig.suptitle("IOI direct interactions across published templates")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2c-script", type=Path, default=DEFAULT_STAGE2C_SCRIPT)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("runs/ioi_template_ablation_robustness"),
    )
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-prompts", type=int, default=128)
    parser.add_argument("--n-reference", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate imports, templates, heads, and the 21-mask design without loading a model.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    stage2c = load_stage2c(args.stage2c_script)
    heads, masks = build_anchor_design(stage2c)
    design_rows = anchor_rows(stage2c, masks, heads)
    templates = published_templates()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "attribution": ATTRIBUTION,
                    "stage2c_script": str(args.stage2c_script),
                    "n_templates": len(templates),
                    "orders": pd.Series([row["order"] for row in templates]).value_counts().to_dict(),
                    "n_heads": len(heads),
                    "n_masks": len(masks),
                    "group_anchors": [row["group_code"] for row in design_rows],
                    "planned_intervened_examples": (
                        len(templates) * 2 * (len(masks) - 1) * args.n_prompts
                    ),
                },
                indent=2,
            )
        )
        return 0

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = outdir / ".cache"
    (cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_dir / "fontconfig").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    device = stage2c.choose_device(args.device)
    torch.set_grad_enabled(False)
    print(f"Loading {args.model} once on {device}...")
    model = stage2c.load_model(args.model, device)

    all_prompts: List[Dict[str, object]] = []
    all_measurements: List[Dict[str, object]] = []
    for template_index, template_row in enumerate(templates, start=1):
        print(f"\nTemplate {template_index}/{len(templates)}: {template_row['template_id']}")
        prompt_rows, measurement_rows = measure_condition(
            stage2c=stage2c,
            model=model,
            device=device,
            template_row=template_row,
            heads=heads,
            masks=masks,
            design_rows=design_rows,
            n_prompts=args.n_prompts,
            n_reference=args.n_reference,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        all_prompts.extend(prompt_rows)
        all_measurements.extend(measurement_rows)

    prompts = pd.DataFrame(all_prompts)
    measurements = pd.DataFrame(all_measurements)
    prompt_keys = prompts[
        ["template_id", "prompt_idx", "io_name", "s_name", "io_token", "s_token"]
    ]
    measurements = measurements.merge(
        prompt_keys,
        on=["template_id", "prompt_idx"],
        how="left",
        validate="many_to_one",
    )
    metrics = direct_metric_rows(measurements)
    per_template, overall = summarize_metrics(metrics)
    bootstrap, bootstrap_summary = hierarchical_bootstrap(
        metrics, args.bootstrap_repeats, args.seed + 2027
    )
    loto = leave_one_template_out(metrics)
    gate = robustness_gate(bootstrap_summary, overall)

    pd.DataFrame([asdict(head) for head in heads]).to_csv(
        outdir / "ioi_robustness_head_records.csv", index=False
    )
    pd.DataFrame(design_rows).to_csv(outdir / "ioi_robustness_anchor_design.csv", index=False)
    pd.DataFrame(templates).to_csv(outdir / "ioi_robustness_templates.csv", index=False)
    prompts.to_csv(outdir / "ioi_robustness_prompts.csv", index=False)
    measurements.to_csv(outdir / "ioi_robustness_per_prompt_measurements.csv", index=False)
    metrics.to_csv(outdir / "ioi_robustness_per_prompt_metrics.csv", index=False)
    per_template.to_csv(outdir / "ioi_robustness_per_template.csv", index=False)
    overall.to_csv(outdir / "ioi_robustness_overall.csv", index=False)
    bootstrap.to_csv(outdir / "ioi_robustness_hierarchical_bootstrap.csv", index=False)
    bootstrap_summary.to_csv(
        outdir / "ioi_robustness_hierarchical_bootstrap_summary.csv", index=False
    )
    loto.to_csv(outdir / "ioi_robustness_leave_one_template_out.csv", index=False)

    metadata = {
        "attribution": ATTRIBUTION,
        "args": {
            **vars(args),
            "stage2c_script": str(args.stage2c_script),
            "outdir": str(args.outdir),
        },
        "device": device,
        "template_source": (
            "Redwood Research Easy-Transformer ioi_dataset.py; first four "
            "BABA_TEMPLATES and their generated ABBA counterparts"
        ),
        "template_source_url": (
            "https://github.com/redwoodresearch/Easy-Transformer/blob/main/"
            "easy_transformer/ioi_dataset.py"
        ),
        "sign_convention": "positive effect = clean logit difference - ablated logit difference",
        "pair_counts": {"PB": 24, "PE": 6, "BE": 16},
        "robustness_gate": gate,
    }
    (outdir / "ioi_robustness_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    plot_template_interactions(per_template, outdir / "ioi_robustness_interactions.png")
    print("\nOverall direct effects and interactions:")
    print(overall.to_string(index=False))
    print("\nRobustness gate:")
    print(json.dumps(gate, indent=2))
    print(f"\nWrote outputs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
