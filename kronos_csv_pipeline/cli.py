"""Command line interface for the clean CSV Kronos pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from .backtest import run_walk_forward_backtest
from .baseline import run_baseline_backtest
from .compare import compare_results
from .config import load_pipeline_config
from .data import clean_data_dir
from .download import download_symbols, load_symbols_from_file
from .predict import predict_symbols
from .risk import apply_risk_filter
from .scanner import scan_opportunities
from .validate import validate_pipeline


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


def cmd_validate(args: argparse.Namespace) -> None:
    symbols = _resolve_symbols(args)
    report = validate_pipeline(
        config_path=args.config,
        symbols=symbols,
        check_raw=not args.skip_raw_check,
        check_clean=args.check_clean,
        min_rows=args.min_rows,
    )
    print(report.to_string(index=False))
    if args.validation_report:
        Path(args.validation_report).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.validation_report, index=False)


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


def cmd_baseline(args: argparse.Namespace) -> None:
    cfg = load_pipeline_config(args.config)
    symbols = _resolve_symbols(args)
    summary = run_baseline_backtest(
        clean_dir=args.clean_data_dir or cfg.paths.clean_data_dir,
        output_dir=args.baseline_dir or cfg.paths.baseline_dir,
        config=cfg.baseline,
        symbols=symbols,
    )
    print(summary.to_string(index=False))


def cmd_compare(args: argparse.Namespace) -> None:
    cfg = load_pipeline_config(args.config)
    report = compare_results(
        kronos_summary_path=args.kronos_summary or Path(cfg.paths.backtest_dir) / "walk_forward_summary.csv",
        baseline_summary_path=args.baseline_summary or Path(cfg.paths.baseline_dir) / f"baseline_{cfg.baseline.strategy}_summary.csv",
        output_dir=args.compare_dir or cfg.paths.compare_dir,
        config=cfg.compare,
    )
    print(report.to_string(index=False))


def cmd_scan(args: argparse.Namespace) -> None:
    cfg = load_pipeline_config(args.config)
    result = scan_opportunities(
        predictions_path=args.prediction_ranking or Path(cfg.paths.predictions_dir) / "prediction_ranking.csv",
        backtest_path=args.backtest_summary or Path(cfg.paths.backtest_dir) / "walk_forward_summary.csv",
        output_dir=args.scanner_dir or cfg.paths.scanner_dir,
        config=cfg.scanner,
    )
    print(result.to_string(index=False))


def cmd_risk(args: argparse.Namespace) -> None:
    cfg = load_pipeline_config(args.config)
    approved = apply_risk_filter(
        scanner_path=args.scanner_input or Path(cfg.paths.scanner_dir) / "portfolio_scan.csv",
        output_dir=args.risk_dir or cfg.paths.risk_dir,
        config=cfg.risk,
        liquidity_path=args.liquidity_file,
    )
    print(approved.to_string(index=False))


def cmd_run_all(args: argparse.Namespace) -> None:
    cfg = load_pipeline_config(args.config)
    symbols = _resolve_symbols(args)

    if args.validate_first:
        validation = validate_pipeline(
            config_path=args.config,
            symbols=symbols,
            check_raw=not args.download_first,
            check_clean=False,
            min_rows=args.min_rows,
        )
        print("\n=== Validation report ===")
        print(validation.to_string(index=False))

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

    baseline_ran = False
    if args.baseline_only:
        baseline = run_baseline_backtest(
            clean_dir=args.clean_data_dir or cfg.paths.clean_data_dir,
            output_dir=args.baseline_dir or cfg.paths.baseline_dir,
            config=cfg.baseline,
            symbols=symbols,
        )
        print("\n=== Baseline summary ===")
        print(baseline.to_string(index=False))
        return

    if args.run_baseline:
        baseline = run_baseline_backtest(
            clean_dir=args.clean_data_dir or cfg.paths.clean_data_dir,
            output_dir=args.baseline_dir or cfg.paths.baseline_dir,
            config=cfg.baseline,
            symbols=symbols,
        )
        baseline_ran = True
        print("\n=== Baseline summary ===")
        print(baseline.to_string(index=False))

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

    if args.compare or baseline_ran:
        comparison = compare_results(
            kronos_summary_path=Path(args.backtest_dir or cfg.paths.backtest_dir) / "walk_forward_summary.csv",
            baseline_summary_path=Path(args.baseline_dir or cfg.paths.baseline_dir) / f"baseline_{cfg.baseline.strategy}_summary.csv",
            output_dir=args.compare_dir or cfg.paths.compare_dir,
            config=cfg.compare,
        )
        print("\n=== Kronos vs baseline comparison ===")
        print(comparison.to_string(index=False))

    scan = scan_opportunities(
        predictions_path=Path(args.predictions_dir or cfg.paths.predictions_dir) / "prediction_ranking.csv",
        backtest_path=Path(args.backtest_dir or cfg.paths.backtest_dir) / "walk_forward_summary.csv",
        output_dir=args.scanner_dir or cfg.paths.scanner_dir,
        config=cfg.scanner,
    )
    print("\n=== Portfolio scan ===")
    print(scan.to_string(index=False))

    if args.apply_risk:
        approved = apply_risk_filter(
            scanner_path=Path(args.scanner_dir or cfg.paths.scanner_dir) / "portfolio_scan.csv",
            output_dir=args.risk_dir or cfg.paths.risk_dir,
            config=cfg.risk,
            liquidity_path=args.liquidity_file,
        )
        print("\n=== Risk-approved candidates ===")
        print(approved.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean CSV -> Kronos prediction -> walk-forward backtest -> scanner")
    parser.add_argument("command", choices=["validate", "download", "clean", "predict", "backtest", "baseline", "compare", "scan", "risk", "run-all"])
    parser.add_argument("--config", default="configs/kronos_csv_pipeline.yaml", help="Pipeline YAML config")
    parser.add_argument("--symbols", nargs="*", help="Symbols, space-separated or comma-separated")
    parser.add_argument("--symbols-file", default=None, help="Text file with symbols, one per line or comma-separated")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--clean-data-dir", default=None)
    parser.add_argument("--predictions-dir", default=None)
    parser.add_argument("--backtest-dir", default=None)
    parser.add_argument("--scanner-dir", default=None)
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--compare-dir", default=None)
    parser.add_argument("--risk-dir", default=None)
    parser.add_argument("--scanner-input", default=None)
    parser.add_argument("--liquidity-file", default=None)
    parser.add_argument("--prediction-ranking", default=None)
    parser.add_argument("--backtest-summary", default=None)
    parser.add_argument("--kronos-summary", default=None)
    parser.add_argument("--baseline-summary", default=None)
    parser.add_argument("--validation-report", default=None)
    parser.add_argument("--min-rows", type=int, default=128)
    parser.add_argument("--download-first", action="store_true", help="For run-all: download data before cleaning")
    parser.add_argument("--validate-first", action="store_true", help="For run-all: validate config and environment before running")
    parser.add_argument("--check-clean", action="store_true", help="For validate: also validate clean_data_dir")
    parser.add_argument("--skip-raw-check", action="store_true", help="For validate: skip raw CSV validation")
    parser.add_argument("--run-baseline", action="store_true", help="For run-all: also run fast baseline before Kronos")
    parser.add_argument("--baseline-only", action="store_true", help="For run-all: clean and run baseline only, without Kronos")
    parser.add_argument("--compare", action="store_true", help="For run-all: compare Kronos backtest against baseline output")
    parser.add_argument("--apply-risk", action="store_true", help="For run-all: apply risk filter after scanner")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "clean":
        cmd_clean(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "baseline":
        cmd_baseline(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "scan":
        cmd_scan(args)
    elif args.command == "risk":
        cmd_risk(args)
    elif args.command == "run-all":
        cmd_run_all(args)
    else:
        parser.error(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
