# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import AnalysisConfig, SurfaceMeasurements, analyze_surface, load_surface
from .design import make_action_design
from .runner import run_experiment
from .toy import simulate_surface
from .prepare_prompts import add_arguments as add_prepare_arguments, run_from_args as prepare_from_args


def _toy(args: argparse.Namespace) -> None:
    design = make_action_design(
        n_sites=args.sites,
        split_sizes={"calibration": 64, "validation": 32, "test": 64},
        seed=args.seed,
    )
    simulated = simulate_surface(design, n_prompts=160, interaction_strength=args.interaction, seed=args.seed + 1)
    surface = SurfaceMeasurements(
        design=design,
        fit_effects=simulated.effects[:80],
        fit_groups=simulated.groups[:80],
        test_effects=simulated.effects[80:],
        test_groups=simulated.groups[80:],
        test_collateral=np.abs(simulated.effects[80:]) * 0.02,
    )
    summary = analyze_surface(
        surface,
        args.outdir,
        AnalysisConfig(bootstrap_repeats=args.bootstrap, bootstrap_seed=args.seed + 3),
    )
    print(json.dumps(summary["primary"], indent=2, sort_keys=True))


def _analyze(args: argparse.Namespace) -> None:
    surface = load_surface(args.surface)
    summary = analyze_surface(
        surface,
        args.outdir,
        AnalysisConfig(bootstrap_repeats=args.bootstrap, bootstrap_seed=args.seed),
    )
    print(json.dumps(summary["primary"], indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mechtomo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    toy = subparsers.add_parser("toy", help="run the CPU-only end-to-end falsifier")
    toy.add_argument("--outdir", type=Path, required=True)
    toy.add_argument("--sites", type=int, default=8)
    toy.add_argument("--interaction", type=float, default=0.8)
    toy.add_argument("--bootstrap", type=int, default=300)
    toy.add_argument("--seed", type=int, default=7)
    toy.set_defaults(func=_toy)

    qwen = subparsers.add_parser("qwen", help="run a resumable Qwen experiment stage")
    qwen.add_argument("--config", type=Path, required=True)
    qwen.add_argument("--outdir", type=Path, required=True)
    qwen.add_argument("--stage", choices=("directions", "measure", "analyze", "all"), default="all")
    qwen.set_defaults(func=lambda args: run_experiment(args.config, args.outdir, args.stage))

    analyze = subparsers.add_parser("analyze", help="analyze frozen surface measurements")
    analyze.add_argument("--surface", type=Path, required=True)
    analyze.add_argument("--outdir", type=Path, required=True)
    analyze.add_argument("--bootstrap", type=int, default=2000)
    analyze.add_argument("--seed", type=int, default=29)
    analyze.set_defaults(func=_analyze)

    prepare = subparsers.add_parser("prepare-data", help="prepare a locked prompt JSONL from official CSVs")
    add_prepare_arguments(prepare)
    prepare.set_defaults(func=prepare_from_args)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
