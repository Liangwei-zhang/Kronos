"""Portfolio scanner for combining predictions and walk-forward backtest metrics.

The scanner is intentionally rule-based. It combines the latest Kronos prediction
ranking with historical walk-forward performance, then outputs actionable buckets:
BUY / WATCH / SKIP.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScannerConfig:
    min_predicted_return: float = 0.01
    min_win_rate: float = 0.50
    min_sharpe: float = 0.0
    max_drawdown_floor: float = -0.30
    min_trade_count: int = 5
    buy_score: float = 0.60
    watch_score: float = 0.35
    top_n: int = 20


def _safe_read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required scanner input not found: {path}")
    return pd.read_csv(path)


def _norm_positive(series: pd.Series, cap: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if cap <= 0:
        return pd.Series(0.0, index=values.index)
    return (values.clip(lower=0.0, upper=cap) / cap).fillna(0.0)


def _norm_drawdown(series: pd.Series, floor: float) -> pd.Series:
    """Map drawdown to 0..1 where 1 is no drawdown and 0 is floor-or-worse."""

    dd = pd.to_numeric(series, errors="coerce").fillna(floor)
    floor = min(floor, -1e-9)
    score = 1.0 - (dd.clip(lower=floor, upper=0.0).abs() / abs(floor))
    return score.clip(0.0, 1.0)


def _action(row: pd.Series, cfg: ScannerConfig) -> str:
    if row.get("error", "") not in ["", np.nan] and isinstance(row.get("error", ""), str):
        if row.get("error", ""):
            return "SKIP"

    hard_pass = (
        row["predicted_return"] >= cfg.min_predicted_return
        and row["win_rate"] >= cfg.min_win_rate
        and row["sharpe"] >= cfg.min_sharpe
        and row["max_drawdown"] >= cfg.max_drawdown_floor
        and row["trade_count"] >= cfg.min_trade_count
    )
    if hard_pass and row["final_score"] >= cfg.buy_score:
        return "BUY"
    if row["final_score"] >= cfg.watch_score:
        return "WATCH"
    return "SKIP"


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    clean = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)
    return clean.to_dict(orient="records")


def write_scan_json_report(result: pd.DataFrame, output_dir: str | Path, config: ScannerConfig) -> Path:
    """Write a compact machine-readable scanner report."""

    output_dir = Path(output_dir)
    by_action = {
        action: _json_safe_records(result[result["action"] == action])
        for action in ["BUY", "WATCH", "SKIP"]
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "counts": {action: len(rows) for action, rows in by_action.items()},
        "top": _json_safe_records(result.head(10)),
        "buy": by_action["BUY"],
        "watch": by_action["WATCH"],
        "skip": by_action["SKIP"],
    }
    out_path = output_dir / "portfolio_scan.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def scan_opportunities(
    predictions_path: str | Path,
    backtest_path: str | Path,
    output_dir: str | Path,
    config: ScannerConfig | None = None,
) -> pd.DataFrame:
    """Merge latest predictions and backtest performance into a ranked action list."""

    cfg = config or ScannerConfig()
    pred = _safe_read_csv(predictions_path)
    bt = _safe_read_csv(backtest_path)

    if "symbol" not in pred.columns or "symbol" not in bt.columns:
        raise ValueError("Both prediction and backtest CSVs must contain a symbol column")

    pred["symbol"] = pred["symbol"].astype(str).str.upper()
    bt["symbol"] = bt["symbol"].astype(str).str.upper()

    merged = pred.merge(bt, on="symbol", how="left", suffixes=("_pred", "_bt"))

    numeric_defaults = {
        "predicted_return": 0.0,
        "predicted_upside": 0.0,
        "predicted_drawdown": 0.0,
        "total_return": 0.0,
        "excess_return": 0.0,
        "max_drawdown": cfg.max_drawdown_floor,
        "sharpe": 0.0,
        "win_rate": 0.0,
        "avg_trade_return": 0.0,
        "profit_factor": 0.0,
        "trade_count": 0,
    }
    for col, default in numeric_defaults.items():
        if col not in merged.columns:
            merged[col] = default
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(default)

    # Component scores, each roughly 0..1.
    merged["pred_return_score"] = _norm_positive(merged["predicted_return"], cap=0.08)
    merged["upside_score"] = _norm_positive(merged["predicted_upside"], cap=0.12)
    merged["bt_return_score"] = _norm_positive(merged["total_return"], cap=1.0)
    merged["excess_score"] = _norm_positive(merged["excess_return"], cap=0.75)
    merged["sharpe_score"] = _norm_positive(merged["sharpe"], cap=3.0)
    merged["win_score"] = pd.to_numeric(merged["win_rate"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    merged["drawdown_score"] = _norm_drawdown(merged["max_drawdown"], floor=cfg.max_drawdown_floor)
    merged["profit_factor_score"] = _norm_positive(merged["profit_factor"].replace(np.inf, 5.0), cap=3.0)
    merged["trade_count_score"] = _norm_positive(merged["trade_count"], cap=max(cfg.min_trade_count * 3, 1))

    merged["final_score"] = (
        merged["pred_return_score"] * 0.22
        + merged["upside_score"] * 0.08
        + merged["bt_return_score"] * 0.14
        + merged["excess_score"] * 0.12
        + merged["sharpe_score"] * 0.12
        + merged["win_score"] * 0.12
        + merged["drawdown_score"] * 0.10
        + merged["profit_factor_score"] * 0.05
        + merged["trade_count_score"] * 0.05
    )

    if "error" not in merged.columns:
        merged["error"] = ""
    merged["action"] = merged.apply(lambda row: _action(row, cfg), axis=1)

    ordered_cols = [
        "symbol",
        "action",
        "final_score",
        "predicted_return",
        "predicted_upside",
        "predicted_drawdown",
        "total_return",
        "benchmark_return",
        "excess_return",
        "max_drawdown",
        "sharpe",
        "win_rate",
        "avg_trade_return",
        "profit_factor",
        "trade_count",
        "last_close",
        "pred_final_close",
        "error",
    ]
    for col in ordered_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    result = merged[ordered_cols].sort_values("final_score", ascending=False).reset_index(drop=True)
    if cfg.top_n > 0:
        result = result.head(cfg.top_n)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "portfolio_scan.csv", index=False)
    result[result["action"] == "BUY"].to_csv(output_dir / "buy_candidates.csv", index=False)
    result[result["action"] == "WATCH"].to_csv(output_dir / "watch_candidates.csv", index=False)
    result[result["action"] == "SKIP"].to_csv(output_dir / "skip_candidates.csv", index=False)
    write_scan_json_report(result, output_dir, cfg)
    return result
