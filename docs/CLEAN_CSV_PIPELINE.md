# Clean CSV Kronos Pipeline

This pipeline keeps the workflow simple and testable:

```text
raw per-symbol CSV
  -> canonical clean CSV
  -> Kronos batch prediction
  -> strict walk-forward backtest
  -> opportunity ranking
```

## 1. CSV format

Put one CSV per symbol in `data/raw_csv/`:

```text
data/raw_csv/AAPL.csv
data/raw_csv/MSFT.csv
data/raw_csv/NVDA.csv
```

Required columns:

```text
timestamps,open,high,low,close
```

Optional columns:

```text
volume,amount
```

If `volume` is missing it is filled with `0`. If `amount` is missing it is calculated as `close * volume`.

Accepted timestamp aliases include `timestamp`, `datetime`, `date`, and `time`. The cleaner also supports several Chinese column aliases used by A-share data exports.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Edit config

Default config:

```bash
configs/kronos_csv_pipeline.yaml
```

Important fields:

```yaml
prediction:
  model_key: kronos-small
  model_path: null          # set to local fine-tuned model path when available
  tokenizer_path: null      # set to local fine-tuned tokenizer path when available
  lookback: 512
  pred_len: 5

walk_forward:
  lookback: 512
  pred_len: 5
  step: 5
  buy_threshold: 0.01
  fee_rate: 0.0005
  slippage_rate: 0.0005
```

## 4. Clean data

```bash
python -m kronos_csv_pipeline.cli clean \
  --config configs/kronos_csv_pipeline.yaml \
  --symbols AAPL MSFT NVDA
```

Outputs:

```text
data/clean_csv/AAPL.csv
data/clean_csv/MSFT.csv
data/clean_csv/NVDA.csv
data/clean_csv/clean_report.csv
```

## 5. Batch prediction

```bash
python -m kronos_csv_pipeline.cli predict \
  --config configs/kronos_csv_pipeline.yaml \
  --symbols AAPL MSFT NVDA
```

Outputs:

```text
outputs/predictions/AAPL_prediction.csv
outputs/predictions/MSFT_prediction.csv
outputs/predictions/NVDA_prediction.csv
outputs/predictions/prediction_ranking.csv
```

## 6. Strict walk-forward backtest

```bash
python -m kronos_csv_pipeline.cli backtest \
  --config configs/kronos_csv_pipeline.yaml \
  --symbols AAPL MSFT NVDA
```

Outputs:

```text
outputs/backtest/walk_forward_summary.csv
outputs/backtest/trades/AAPL_walk_forward_trades.csv
```

The backtester uses only the previous `lookback` rows at each forecast origin. It compares the forecast against the next `pred_len` rows. Future prices are never passed into the model input.

Known future timestamps are used for time embeddings. This is allowed because market calendars are known before a trade and do not leak future prices.

## 7. Run all steps

```bash
python -m kronos_csv_pipeline.cli run-all \
  --config configs/kronos_csv_pipeline.yaml \
  --symbols AAPL MSFT NVDA
```

## 8. Use a fine-tuned model

After fine-tuning with `finetune_csv/train_sequential.py`, update the config:

```yaml
prediction:
  model_path: finetune_csv/finetuned/my_experiment/basemodel/best_model
  tokenizer_path: finetune_csv/finetuned/my_experiment/tokenizer/best_model
```

Then rerun prediction/backtest.

## 9. Output metrics

The walk-forward summary includes:

```text
total_return
benchmark_return
excess_return
max_drawdown
sharpe
win_rate
avg_trade_return
profit_factor
trade_count
score
```

The `score` is only a ranking helper. Do not treat it as a guarantee of future profit.

## 10. Notes

This is research infrastructure, not a production trading system. Before using signals with real money, add:

- robust survivorship-bias-free data
- corporate action handling
- liquidity filters
- earnings/news blackout rules
- realistic order execution
- out-of-sample validation
- transaction-cost sensitivity tests
