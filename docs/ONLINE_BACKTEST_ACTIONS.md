# Online Baseline Backtest

This project includes a manual GitHub Actions workflow for downloading market data with `yfinance` and running the CSV baseline pipeline on a GitHub runner.

Workflow file:

```text
.github/workflows/online-baseline-backtest.yml
```

## Run from GitHub

Open the repository page, then use:

```text
Actions -> Online baseline backtest -> Run workflow
```

Default inputs:

```text
symbols_file: configs/nasdaq_watchlist.txt
config: configs/kronos_csv_pipeline.yaml
min_rows: 260
run_kronos: false
```

## Fast baseline run

Keep `run_kronos` set to `false` for a quick run. The workflow downloads data, cleans CSV files, and writes baseline results.

Main output artifact:

```text
online-baseline-backtest-outputs
```

Important files inside the artifact:

```text
outputs/baseline/baseline_momentum_summary.csv
data/raw_csv/download_report.csv
data/clean_csv/clean_report.csv
```

## Kronos run

Set `run_kronos` to `true` to also run the Kronos path after baseline. This requires the runner to access model weights, or local model paths configured in:

```text
configs/kronos_csv_pipeline.yaml
```

## Result summary

The GitHub Actions run summary prints:

```text
row count
average win_rate
median win_rate
average profit_factor
top rows sorted by score
```

Use the artifact CSV files for detailed analysis.

## Recommended sequence

1. Run baseline-only first.
2. Download the artifact.
3. Review `baseline_momentum_summary.csv`.
4. Run with `run_kronos: true` only after the baseline works.
5. Compare Kronos metrics against the baseline metrics.
