# Baseline Benchmark Results

This document records the first offline baseline benchmark on the uploaded `bb(1).7z` stock CSV universe.

The baseline does **not** use Kronos weights. It is a fast momentum benchmark used to validate the data universe and establish a minimum bar for Kronos.

## Dataset summary

```text
Raw CSV files: 1077
Cleaned usable CSV files: 988
Prediction-eligible files, lookback >= 512: 808
Walk-forward-eligible files, rows >= 1029: 762
```

A Top50 high-liquidity subset was selected by recent average traded amount for the first benchmark.

## Baseline strategy

Default config:

```yaml
baseline:
  strategy: momentum
  lookback: 120
  pred_len: 5
  step: 5
  signal_window: 20
  buy_threshold: 0.01
  fee_rate: 0.0005
  slippage_rate: 0.0005
```

The strategy uses only historical prices before each forecast origin. It opens a long trade when the recent momentum signal is above the threshold and exits after `pred_len` bars.

## Top50 baseline summary

```text
Universe: Top50 high-liquidity symbols
BUY: 7
WATCH: 24
SKIP: 19
Average win_rate: ~50.20%
Median win_rate: ~49.98%
Average profit_factor: ~1.08
```

Interpretation:

- The baseline is only slightly above break-even by profit factor.
- The average win rate is close to 50%, so the edge is weak.
- Several high-return names also show very large drawdowns, so raw return alone is not enough.
- Kronos should be judged against this baseline. If Kronos cannot improve profit factor, drawdown, and stability, it does not add useful trading edge.

## Example top candidates from baseline

| Symbol | Action | Win Rate | Profit Factor | Total Return | Max Drawdown | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| CRDO | BUY | 58.76% | 1.33 | 60.55% | -60.00% | 1.17 |
| CLS | WATCH | 49.33% | 1.12 | 21.68% | -94.32% | 0.46 |
| BE | BUY | 51.93% | 1.46 | 489.13% | -62.73% | 1.52 |
| CVNA | BUY | 54.94% | 1.48 | 779.07% | -65.96% | 1.56 |
| COHR | BUY | 50.85% | 1.29 | 2846.93% | -82.57% | 0.90 |

## Local commands

Run baseline only:

```bash
python -m kronos_csv_pipeline.cli baseline \
  --config configs/kronos_csv_pipeline.yaml \
  --symbols-file reports/top50_symbols.txt
```

Or clean and run baseline only:

```bash
python -m kronos_csv_pipeline.cli run-all \
  --config configs/kronos_csv_pipeline.yaml \
  --symbols-file reports/top50_symbols.txt \
  --baseline-only
```

Output:

```text
outputs/baseline/baseline_momentum_summary.csv
outputs/baseline/baseline_momentum_trades/*.csv
```

## Next step: Kronos comparison

The next benchmark should run true Kronos walk-forward on the same Top50 subset, then compare:

```text
win_rate
profit_factor
total_return
max_drawdown
sharpe
trade_count
excess_return
```

Kronos should be considered useful only if it improves the baseline on risk-adjusted metrics, not just raw return.
