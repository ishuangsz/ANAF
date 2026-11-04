"""Command-line interface for ANAF training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .trainer import evaluate_dataset, resolve_device, train_dataset


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="ANAF YAML configuration")
    parser.add_argument("--dataset", required=True, help="SMD, SMAP, JD1, or JD2")
    parser.add_argument("--data-root", type=Path, required=True, help="processed dataset root")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=None, help="override the configured seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anaf", description="Train and evaluate ANAF anomaly detectors"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train all entities in a dataset")
    _common_arguments(train)
    train.add_argument("--output-dir", type=Path, required=True, help="checkpoint output root")

    evaluate = subparsers.add_parser("evaluate", help="evaluate all entity checkpoints")
    _common_arguments(evaluate)
    evaluate.add_argument(
        "--checkpoint-dir", type=Path, required=True, help="checkpoint root created by train"
    )
    return parser


def main(arguments: list[str] | None = None) -> None:
    args = build_parser().parse_args(arguments)
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    project = load_config(args.config)
    dataset = project.dataset(args.dataset)
    seed = project.training.seed if args.seed is None else args.seed
    if seed < 0:
        raise SystemExit("--seed must be non-negative")
    device = resolve_device(args.device)

    if args.command == "train":
        train_dataset(
            project,
            dataset.name,
            args.data_root,
            args.output_dir,
            device=device,
            seed=seed,
            num_workers=args.num_workers,
        )
    else:
        evaluate_dataset(
            project,
            dataset.name,
            args.data_root,
            args.checkpoint_dir,
            device=device,
            seed=seed,
            num_workers=args.num_workers,
        )
