"""Command line interface for the clean CSV Kronos pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from .backtest import run_walk_forward_backtest
from .config import load_pipeline_config
from .data import clean_data_dir
from .download import download_symbols, load_symbols_from_file
from .predict import predict_symbols


def _parse_symbols(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None
    symbols: list[str] = []
    for value in values:
        symbols.extend([part.strip().upper() for part in value.split(",") if part.strip()])
    return symbols or None


def _resolve_symbols(args: argparse.Namespace) -> list[str] | None:
    symbols = _parse_symbols(args.symbols)
    if args.symbols_file:
        file_symbols = load_symbols_from_file(args.symbols_file)
        symbols = [*(symbols or []), *file_symbols]
    if symbols:
        return list(dict.fromkeys([s.upper() for s in symbols]))
    return None


def cmd_download(args: argparse.Namespace) -> None:
    cfg = load_pipeline_config(args.config)
    symbols = _resolve_symbols(args)
    if not symbols:
        raise ValueError("download requires --symbols or --symbols-file")
    report = download_symbols(
        symbols=symbols,
        output_dir=args.raw_data_dir or cfg.paths.raw_data_dir,
        config=cfg.download,
    )
    print(report.to_string(index=False))


def cmd_clean(args: argparse.Namespace) -> None:
    cfg = load_pipeline_config(args.config)
    symbols = _resolve_symbols(args)
    reports = clean_data_dir(
        input_dir=args.raw_data_dir or cfg.paths.raw_data_dir,
        output_dir=args.clean_data_dir or cfg.paths.clean_data_dir,
        symbols=symbols,
        min_rows=args.min_rows,
    )
    rows = [r.__dict__ for r in reports]
    report_df = pd.DataFrame(rows)
    out_dir = Path(args.clean_data_dir or cfg.paths.clean_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(out_dir / "clean_report.csv", index=False)
    print(report_df.to_string(index=False))


def cmd_predict(args: argparse.Namespace) -> None:
    cfg = load_pipeline_config(args.config)
    symbols = _resolve_symbols(args)
    ranking = predict_symbols(
        clean_dir=args.clean_data_dir or cfg.paths.clean_data_dir,
        output_dir=args.predictions_dir or cfg.paths.predictions_dir,
        config=cfg.prediction,
        symbols=symbols,
    )
    print(ranking.to_string(index=False))


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = load_pipeline_config(args.config)
    symbols = _resolve_symbols(args)
    summary = run_walk_forward_backtest(
        clean_dir=args.clean_data_dir or cfg.paths.clean_data_dir,
        output_dir=args.backtest_dir or cfg.paths.backtest_dir,
        config=cfg.to_backtest_config(),
        symbols=symbols,
    )
    print(summary.to_string(index=False))


def cmd_run_all(args: argparse.Namespace) -> None:
    cfg = load_pipeline_config(args.config)
    symbols = _resolve_symbols(args)

    if args.download_first:
        if not symbols:
            raise ValueError("run-all --download-first requires --symbols or --symbols-file")
        download_report = download_symbols(
            symbols=symbols,
            output_dir=args.raw_data_dir or cfg.paths.raw_data_dir,
            config=cfg.download,
        )
        print("\n=== Download report ===")
        print(download_report.to_string(index=False))

    reports = clean_data_dir(
        input_dir=args.raw_data_dir or cfg.paths.raw_data_dir,
        output_dir=args.clean_data_dir or cfg.paths.clean_data_dir,
        symbols=symbols,
        min_rows=args.min_rows,
    )
    report_df = pd.DataFrame([r.__dict__ for r in reports])
    print("\n=== Clean report ===")
    print(report_df.to_string(index=False))

    ranking = predict_symbols(
        clean_dir=args.clean_data_dir or cfg.paths.clean_data_dir,
        output_dir=args.predictions_dir or cfg.paths.predictions_dir,
        config=cfg.prediction,
        symbols=symbols,
    )
    print("\n=== Latest prediction ranking ===")
    print(ranking.to_string(index=False))

    summary = run_walk_forward_backtest(
        clean_dir=args.clean_data_dir or cfg.paths.clean_data_dir,
        output_dir=args.backtest_dir or cfg.paths.backtest_dir,
        config=cfg.to_backtest_config(),
        symbols=symbols,
    )
    print("\n=== Walk-forward summary ===")
    print(summary.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean CSV -> Kronos prediction -> walk-forward backtest")
    parser.add_argument("command", choices=["download", "clean", "predict", "backtest", "run-all"])
    parser.add_argument("--config", default="configs/kronos_csv_pipeline.yaml", help="Pipeline YAML config")
    parser.add_argument("--symbols", nargs="*", help="Symbols, space-separated or comma-separated")
    parser.add_argument("--symbols-file", default=None, help="Text file with symbols, one per line or comma-separated")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--clean-data-dir", default=None)
    parser.add_argument("--predictions-dir", default=None)
    parser.add_argument("--backtest-dir", default=None)
    parser.add_argument("--min-rows", type=int, default=128)
    parser.add_argument("--download-first", action="store_true", help="For run-all: download data before cleaning")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        cmd_download(args)
    elif args.command == "clean":
        cmd_clean(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "run-all":
        cmd_run_all(args)
    else:
        parser.error(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
