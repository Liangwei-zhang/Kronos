"""Strict walk-forward backtesting for Kronos predictions.

The backtester only uses data available before each forecast origin. It can use
pretrained Kronos models or locally fine-tuned model/tokenizer directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS, infer_future_timestamps, load_cleaned_symbols
from .predict import PredictionConfig, load_predictor


@dataclass(frozen=True)
class WalkForwardConfig:
    lookback: int = 512
    pred_len: int = 5
    step: int = 5
    min_train_rows: int = 512
    initial_capital: float = 100_000.0
    buy_threshold: float = 0.01
    sell_threshold: float = -0.01
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    max_position_pct: float = 1.0
    allow_short: bool = False


@dataclass(frozen=True)
class BacktestConfig:
    prediction: PredictionConfig
    walk_forward: WalkForwardConfig


def _safe_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _profit_factor(trade_returns: list[float]) -> float:
    gains = sum(r for r in trade_returns if r > 0)
    losses = abs(sum(r for r in trade_returns if r < 0))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _score_row(row: dict) -> float:
    return (
        row["total_return"] * 0.35
        + row["win_rate"] * 0.20
        + row["sharpe"] * 0.20
        + row["profit_factor_capped"] * 0.10
        + row["excess_return"] * 0.15
        + row["max_drawdown"] * 0.20
    )


def walk_forward_symbol(
    symbol: str,
    df: pd.DataFrame,
    predictor,
    pred_cfg: PredictionConfig,
    wf_cfg: WalkForwardConfig,
) -> tuple[pd.DataFrame, dict]:
    """Run walk-forward backtest for one symbol.

    Every forecast origin uses exactly the preceding lookback rows and compares
    against the next pred_len rows. No future rows are passed to Kronos inputs.
    """

    if len(df) < wf_cfg.lookback + wf_cfg.pred_len + wf_cfg.min_train_rows:
        raise ValueError(
            f"{symbol}: insufficient rows {len(df)} for lookback={wf_cfg.lookback}, pred_len={wf_cfg.pred_len}"
        )

    cash = wf_cfg.initial_capital
    shares = 0.0
    equity_rows: list[dict] = []
    trade_returns: list[float] = []
    entry_price: float | None = None

    start_origin = max(wf_cfg.lookback, wf_cfg.min_train_rows)
    last_origin = len(df) - wf_cfg.pred_len

    for origin in range(start_origin, last_origin + 1, wf_cfg.step):
        hist = df.iloc[origin - wf_cfg.lookback : origin].copy()
        future = df.iloc[origin : origin + wf_cfg.pred_len].copy()

        x_df = hist[FEATURE_COLUMNS].reset_index(drop=True)
        x_ts = pd.Series(pd.to_datetime(hist["timestamps"]), name="timestamps").reset_index(drop=True)
        # Use actual future timestamps only for timestamp embeddings. This is not
        # price leakage because timestamps are known before the trade.
        y_ts = pd.Series(pd.to_datetime(future["timestamps"]), name="timestamps").reset_index(drop=True)
        if len(y_ts) != wf_cfg.pred_len:
            y_ts = infer_future_timestamps(x_ts, wf_cfg.pred_len)

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=wf_cfg.pred_len,
            T=pred_cfg.temperature,
            top_k=pred_cfg.top_k,
            top_p=pred_cfg.top_p,
            sample_count=pred_cfg.sample_count,
            verbose=False,
        )

        last_close = float(hist["close"].iloc[-1])
        pred_final_close = float(pred_df["close"].iloc[-1])
        expected_return = pred_final_close / last_close - 1.0

        entry_exec = float(future["open"].iloc[0]) * (1.0 + wf_cfg.slippage_rate)
        exit_exec = float(future["close"].iloc[-1]) * (1.0 - wf_cfg.slippage_rate)
        actual_return = exit_exec / entry_exec - 1.0

        signal = 0
        if expected_return >= wf_cfg.buy_threshold:
            signal = 1
        elif wf_cfg.allow_short and expected_return <= wf_cfg.sell_threshold:
            signal = -1

        trade_return = 0.0
        if signal == 1:
            deploy_cash = cash * wf_cfg.max_position_pct
            buy_fee = deploy_cash * wf_cfg.fee_rate
            qty = max(0.0, (deploy_cash - buy_fee) / entry_exec)
            sell_value = qty * exit_exec
            sell_fee = sell_value * wf_cfg.fee_rate
            pnl = sell_value - sell_fee - deploy_cash
            cash += pnl
            trade_return = pnl / deploy_cash if deploy_cash > 0 else 0.0
            trade_returns.append(trade_return)
            entry_price = entry_exec
        elif signal == -1:
            deploy_cash = cash * wf_cfg.max_position_pct
            short_entry = float(future["open"].iloc[0]) * (1.0 - wf_cfg.slippage_rate)
            short_exit = float(future["close"].iloc[-1]) * (1.0 + wf_cfg.slippage_rate)
            gross_return = short_entry / short_exit - 1.0
            fees = 2 * wf_cfg.fee_rate
            pnl = deploy_cash * (gross_return - fees)
            cash += pnl
            trade_return = pnl / deploy_cash if deploy_cash > 0 else 0.0
            trade_returns.append(trade_return)
            entry_price = short_entry

        equity_rows.append(
            {
                "symbol": symbol,
                "origin_index": origin,
                "origin_time": pd.to_datetime(hist["timestamps"].iloc[-1]),
                "exit_time": pd.to_datetime(future["timestamps"].iloc[-1]),
                "last_close": last_close,
                "pred_final_close": pred_final_close,
                "expected_return": expected_return,
                "actual_return": actual_return,
                "signal": signal,
                "trade_return": trade_return,
                "equity": cash,
            }
        )

    trades = pd.DataFrame(equity_rows)
    if trades.empty:
        raise ValueError(f"{symbol}: no walk-forward trades generated")

    equity = trades["equity"].astype(float)
    returns = equity.pct_change().fillna(0.0)
    buy_hold_return = float(df["close"].iloc[last_origin] / df["close"].iloc[start_origin] - 1.0)
    completed_trade_returns = [float(x) for x in trade_returns]
    win_rate = float(np.mean([r > 0 for r in completed_trade_returns])) if completed_trade_returns else 0.0
    pf = _profit_factor(completed_trade_returns)

    summary = {
        "symbol": symbol,
        "rows": len(df),
        "walks": len(trades),
        "trade_count": len(completed_trade_returns),
        "initial_capital": wf_cfg.initial_capital,
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / wf_cfg.initial_capital - 1.0),
        "benchmark_return": buy_hold_return,
        "excess_return": float(equity.iloc[-1] / wf_cfg.initial_capital - 1.0 - buy_hold_return),
        "max_drawdown": _max_drawdown(equity),
        "sharpe": _safe_sharpe(returns),
        "win_rate": win_rate,
        "avg_trade_return": float(np.mean(completed_trade_returns)) if completed_trade_returns else 0.0,
        "profit_factor": pf,
        "profit_factor_capped": min(pf if np.isfinite(pf) else 5.0, 5.0) / 5.0,
    }
    summary["score"] = _score_row(summary)
    return trades, summary


def run_walk_forward_backtest(
    clean_dir: str | Path,
    output_dir: str | Path,
    config: BacktestConfig,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run strict walk-forward backtest over many symbols and write reports."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_dir = output_dir / "trades"
    trades_dir.mkdir(exist_ok=True)

    data = load_cleaned_symbols(clean_dir, symbols)
    predictor = load_predictor(config.prediction)

    summaries: list[dict] = []
    for symbol, df in sorted(data.items()):
        try:
            trades, summary = walk_forward_symbol(symbol, df, predictor, config.prediction, config.walk_forward)
            trades.to_csv(trades_dir / f"{symbol}_walk_forward_trades.csv", index=False)
            summaries.append(summary)
        except Exception as exc:
            summaries.append({"symbol": symbol, "error": str(exc), "score": -999.0})

    summary_df = pd.DataFrame(summaries)
    if "score" in summary_df.columns:
        summary_df = summary_df.sort_values("score", ascending=False).reset_index(drop=True)
    summary_df.to_csv(output_dir / "walk_forward_summary.csv", index=False)
    return summary_df
