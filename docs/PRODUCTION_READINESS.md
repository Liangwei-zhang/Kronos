# Production Readiness Checklist

This project is currently a research / scanner pipeline. It is **not ready for fully automated live trading** until the gates below are completed and verified.

## Current status

| Area | Status | Notes |
|---|---:|---|
| CSV ingestion and cleaning | Good for research | Handles Yahoo/yfinance style OHLCV CSVs and standardizes fields. |
| Batch universe processing | Good for research | Tested on the uploaded 1077-file universe. |
| Baseline backtesting | Available | Momentum baseline Top50 result: win rate ~50.20%, profit factor ~1.08. |
| Kronos walk-forward validation | Blocking | True Kronos Top50 / Top200 / full-universe comparison is still required. |
| Scanner | Research-ready | Outputs BUY / WATCH / SKIP, JSON report, risk flags, suggested position sizing. |
| Risk controls | Basic | Has per-trade position sizing, stop loss suggestion, drawdown/profit-factor blocks. |
| Paper trading | Missing | Required before live trading. |
| Broker execution | Missing | No order routing or execution reconciliation yet. |
| Monitoring / alerts | Missing | Required for production. |

## Minimum gates before paper trading

Paper trading should not start until all of these pass:

```text
[ ] True Kronos walk-forward on Top50 high-liquidity symbols
[ ] True Kronos walk-forward on Top200 symbols
[ ] True Kronos walk-forward on full eligible universe
[ ] Kronos vs baseline comparison report
[ ] Profit factor consistently above baseline
[ ] Max drawdown acceptable versus baseline
[ ] Sharpe improves versus baseline
[ ] No single-symbol outlier dominates results
[ ] Data validation report generated for every run
[ ] Scanner JSON output validated by tests
```

Suggested minimum Kronos edge over current baseline:

```text
baseline win_rate:      ~50.20%
baseline profit_factor: ~1.08

minimum target win_rate:      > 52%
minimum target profit_factor: > 1.25
minimum target Sharpe:        > 0.80
```

These thresholds are not guarantees of profitability. They are minimum research gates before paper trading.

## Minimum gates before small live capital

Small live capital should not start until all paper-trading gates pass:

```text
[ ] 4-8 weeks of paper trading results
[ ] Paper trading slippage measured against backtest assumptions
[ ] Daily signal log stored
[ ] Daily PnL log stored
[ ] Position exposure report generated
[ ] Manual review before order submission
[ ] Max daily loss stop rule implemented
[ ] Max portfolio drawdown stop rule implemented
[ ] Consecutive loss stop rule implemented
[ ] Earnings-date blackout rule implemented
[ ] Low-liquidity hard filter implemented
[ ] Broker order status reconciliation implemented
```

## Minimum gates before automated live trading

Fully automated live trading should not start until:

```text
[ ] Broker integration supports paper and live modes separately
[ ] Production config cannot accidentally switch to live mode
[ ] Every order has pre-trade risk checks
[ ] Every order has post-trade reconciliation
[ ] Monitoring alerts for data failure, model failure, and broker failure
[ ] Emergency kill switch exists
[ ] Strategy runs are reproducible from logs
[ ] Model version and config version are logged for every signal
[ ] Live trading starts with strict notional caps
```

## Required reports

Each production-like run should produce:

```text
outputs/validation_report.csv
outputs/predictions/prediction_ranking.csv
outputs/backtest/walk_forward_summary.csv
outputs/scanner/portfolio_scan.csv
outputs/scanner/portfolio_scan.json
outputs/risk/portfolio_exposure.csv
outputs/logs/run_metadata.json
```

## Current blocker

The current blocker is the open benchmark task:

```text
Issue #2: Benchmark Kronos against Top50 baseline results
```

Do not treat the scanner as production-ready until Issue #2 is completed and Kronos demonstrates a stable edge over the baseline.

## Recommended next implementation order

1. Complete Kronos-vs-baseline benchmark.
2. Add comparison report generator.
3. Add portfolio-level risk report.
4. Add paper-trading order simulator.
5. Add run metadata logging.
6. Add monitoring / alert hooks.
7. Add broker integration in paper mode only.
8. Only then consider small live capital with manual approval.
