"""Offline baseline models for fast walk-forward evaluation without Kronos weights.

These baselines do not use future prices when generating signals. They are meant
as sanity checks and benchmarks for Kronos, not replacements for Kronos.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .backtest import _max_drawdown, _profit_factor, _safe_sharpe, _score_row
from .data import load_cleaned_symbols


BASELINE_STRATEGIES = ["momentum", "mean_reversion", "ma_cross"]


@dataclass(frozen=True)
class BaselineConfig:
    strategy: str = "momentum"  # momentum, mean_reversion, ma_cross
    lookback: int = 120
    pred_len: int = 5
    step: int = 5
    signal_window: int = 20
    fast_ma: int = 20
    slow_ma: int = 60
    initial_capital: float = 100_000.0
    buy_threshold: float = 0.01
    sell_threshold: float = -0.01
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    max_position_pct: float = 1.0
    allow_short: bool = False


def _expected_return(hist: pd.DataFrame, cfg: BaselineConfig) -> float:
    close = pd.to_numeric(hist["close"], errors="coerce").dropna()
    if len(close) < max(2, cfg.signal_window):
        return 0.0

    strategy = cfg.strategy.lower()
    if strategy == "momentum":
        return float(close.iloc[-1] / close.iloc[-cfg.signal_window] - 1.0)

    if strategy == "mean_reversion":
        momentum = float(close.iloc[-1] / close.iloc[-cfg.signal_window] - 1.0)
        return -momentum

    if strategy == "ma_cross":
        if len(close) < max(cfg.fast_ma, cfg.slow_ma):
            return 0.0
        fast = float(close.tail(cfg.fast_ma).mean())
        slow = float(close.tail(cfg.slow_ma).mean())
        if slow == 0:
            return 0.0
        return fast / slow - 1.0

    raise ValueError(f"Unsupported baseline strategy: {cfg.strategy}")


def baseline_walk_forward_symbol(symbol: str, df: pd.DataFrame, cfg: BaselineConfig) -> tuple[pd.DataFrame, dict]:
    if len(df) < cfg.lookback + cfg.pred_len:
        raise ValueError(f"{symbol}: insufficient rows {len(df)} for baseline lookback={cfg.lookback}")

    cash = cfg.initial_capital
    rows: list[dict] = []
    trade_returns: list[float] = []

    start_origin = cfg.lookback
    last_origin = len(df) - cfg.pred_len

    for origin in range(start_origin, last_origin + 1, cfg.step):
        hist = df.iloc[origin - cfg.lookback : origin].copy()
        future = df.iloc[origin : origin + cfg.pred_len].copy()

        expected = _expected_return(hist, cfg)
        last_close = float(hist["close"].iloc[-1])
        entry_exec = float(future["open"].iloc[0]) * (1.0 + cfg.slippage_rate)
        exit_exec = float(future["close"].iloc[-1]) * (1.0 - cfg.slippage_rate)
        actual_return = exit_exec / entry_exec - 1.0

        signal = 0
        if expected >= cfg.buy_threshold:
            signal = 1
        elif cfg.allow_short and expected <= cfg.sell_threshold:
            signal = -1

        trade_return = 0.0
        if signal == 1:
            deploy_cash = cash * cfg.max_position_pct
            buy_fee = deploy_cash * cfg.fee_rate
            qty = max(0.0, (deploy_cash - buy_fee) / entry_exec)
            sell_value = qty * exit_exec
            sell_fee = sell_value * cfg.fee_rate
            pnl = sell_value - sell_fee - deploy_cash
            cash += pnl
            trade_return = pnl / deploy_cash if deploy_cash > 0 else 0.0
            trade_returns.append(trade_return)
        elif signal == -1:
            deploy_cash = cash * cfg.max_position_pct
            short_entry = float(future["open"].iloc[0]) * (1.0 - cfg.slippage_rate)
            short_exit = float(future["close"].iloc[-1]) * (1.0 + cfg.slippage_rate)
            gross_return = short_entry / short_exit - 1.0
            fees = 2 * cfg.fee_rate
            pnl = deploy_cash * (gross_return - fees)
            cash += pnl
            trade_return = pnl / deploy_cash if deploy_cash > 0 else 0.0
            trade_returns.append(trade_return)

        rows.append(
            {
                "symbol": symbol,
                "origin_index": origin,
                "origin_time": pd.to_datetime(hist["timestamps"].iloc[-1]),
                "exit_time": pd.to_datetime(future["timestamps"].iloc[-1]),
                "last_close": last_close,
                "expected_return": expected,
                "actual_return": actual_return,
                "signal": signal,
                "trade_return": trade_return,
                "equity": cash,
            }
        )

    trades = pd.DataFrame(rows)
    if trades.empty:
        raise ValueError(f"{symbol}: no baseline walks generated")

    equity = trades["equity"].astype(float)
    returns = equity.pct_change().fillna(0.0)
    buy_hold_return = float(df["close"].iloc[last_origin] / df["close"].iloc[start_origin] - 1.0)
    completed = [float(x) for x in trade_returns]
    win_rate = float(np.mean([r > 0 for r in completed])) if completed else 0.0
    pf = _profit_factor(completed)

    summary = {
        "symbol": symbol,
        "strategy": cfg.strategy,
        "rows": len(df),
        "walks": len(trades),
        "trade_count": len(completed),
        "initial_capital": cfg.initial_capital,
        "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / cfg.initial_capital - 1.0),
        "benchmark_return": buy_hold_return,
        "excess_return": float(equity.iloc[-1] / cfg.initial_capital - 1.0 - buy_hold_return),
        "max_drawdown": _max_drawdown(equity),
        "sharpe": _safe_sharpe(returns),
        "win_rate": win_rate,
        "avg_trade_return": float(np.mean(completed)) if completed else 0.0,
        "profit_factor": pf,
        "profit_factor_capped": min(pf if np.isfinite(pf) else 5.0, 5.0) / 5.0,
    }
    summary["score"] = _score_row(summary)
    return trades, summary


def run_baseline_backtest(
    clean_dir: str | Path,
    output_dir: str | Path,
    config: BaselineConfig,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run a fast baseline backtest over cleaned CSV files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_dir = output_dir / f"baseline_{config.strategy}_trades"
    trades_dir.mkdir(exist_ok=True)

    data = load_cleaned_symbols(clean_dir, symbols)
    summaries: list[dict] = []
    for symbol, df in sorted(data.items()):
        try:
            trades, summary = baseline_walk_forward_symbol(symbol, df, config)
            trades.to_csv(trades_dir / f"{symbol}_{config.strategy}_trades.csv", index=False)
            summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            summaries.append({"symbol": symbol, "strategy": config.strategy, "error": str(exc), "score": -999.0})

    summary_df = pd.DataFrame(summaries)
    if "score" in summary_df.columns:
        summary_df = summary_df.sort_values("score", ascending=False).reset_index(drop=True)
    summary_df.to_csv(output_dir / f"baseline_{config.strategy}_summary.csv", index=False)
    return summary_df


def _aggregate_strategy_result(strategy: str, df: pd.DataFrame) -> dict:
    valid = df[df.get("score", -999.0) > -999.0].copy()
    if valid.empty:
        return {
            "strategy": strategy,
            "symbols": 0,
            "avg_win_rate": 0.0,
            "median_win_rate": 0.0,
            "avg_profit_factor": 0.0,
            "median_profit_factor": 0.0,
            "avg_total_return": 0.0,
            "avg_max_drawdown": 0.0,
            "avg_sharpe": 0.0,
            "avg_score": -999.0,
        }
    pf = pd.to_numeric(valid["profit_factor"].replace(np.inf, 5.0), errors="coerce").fillna(0.0)
    return {
        "strategy": strategy,
        "symbols": len(valid),
        "avg_win_rate": float(valid["win_rate"].mean()),
        "median_win_rate": float(valid["win_rate"].median()),
        "avg_profit_factor": float(pf.mean()),
        "median_profit_factor": float(pf.median()),
        "avg_total_return": float(valid["total_return"].mean()),
        "avg_max_drawdown": float(valid["max_drawdown"].mean()),
        "avg_sharpe": float(valid["sharpe"].mean()),
        "avg_score": float(valid["score"].mean()),
    }


def run_baseline_suite(
    clean_dir: str | Path,
    output_dir: str | Path,
    config: BaselineConfig,
    symbols: Sequence[str] | None = None,
    strategies: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run multiple baseline strategies and write a comparison table.

    This is useful before running Kronos: it establishes a no-model benchmark for
    win rate, profit factor, drawdown, and score on the same universe.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = list(strategies or BASELINE_STRATEGIES)

    aggregate_rows: list[dict] = []
    all_rows: list[pd.DataFrame] = []
    for strategy in strategies:
        strategy_cfg = replace(config, strategy=strategy)
        result = run_baseline_backtest(clean_dir, output_dir, strategy_cfg, symbols=symbols)
        aggregate_rows.append(_aggregate_strategy_result(strategy, result))
        all_rows.append(result.assign(strategy=strategy))

    comparison = pd.DataFrame(aggregate_rows).sort_values("avg_score", ascending=False).reset_index(drop=True)
    comparison.to_csv(output_dir / "baseline_strategy_comparison.csv", index=False)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(output_dir / "baseline_all_strategy_symbols.csv", index=False)
        best_by_symbol = combined.sort_values("score", ascending=False).drop_duplicates("symbol", keep="first")
        best_by_symbol = best_by_symbol.sort_values("score", ascending=False).reset_index(drop=True)
        best_by_symbol.to_csv(output_dir / "baseline_best_strategy_by_symbol.csv", index=False)

    return comparison
