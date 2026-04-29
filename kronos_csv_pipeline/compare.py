"""Compare Kronos walk-forward results against a lightweight baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CompareConfig:
    min_profit_factor_delta: float = 0.10
    min_sharpe_delta: float = 0.10
    max_drawdown_delta: float = 0.05
    min_win_rate_delta: float = 0.00
    min_symbols: int = 5


def _read(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Comparison input not found: {path}")
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"{path} must contain a symbol column")
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.mean()) if len(s) else 0.0


def compare_results(
    kronos_summary_path: str | Path,
    baseline_summary_path: str | Path,
    output_dir: str | Path,
    config: CompareConfig | None = None,
) -> pd.DataFrame:
    """Create symbol-level and aggregate Kronos-vs-baseline comparison reports."""

    cfg = config or CompareConfig()
    kronos = _read(kronos_summary_path)
    baseline = _read(baseline_summary_path)

    merged = kronos.merge(baseline, on="symbol", how="inner", suffixes=("_kronos", "_baseline"))
    if merged.empty:
        raise ValueError("No overlapping symbols between Kronos and baseline summaries")

    metrics = ["win_rate", "profit_factor", "total_return", "excess_return", "max_drawdown", "sharpe", "trade_count"]
    for metric in metrics:
        merged[f"{metric}_kronos"] = _num(merged, f"{metric}_kronos")
        merged[f"{metric}_baseline"] = _num(merged, f"{metric}_baseline")
        merged[f"{metric}_delta"] = merged[f"{metric}_kronos"] - merged[f"{metric}_baseline"]

    # For max_drawdown, a positive delta means the drawdown is less negative / better.
    merged["kronos_better"] = (
        (merged["profit_factor_delta"] >= cfg.min_profit_factor_delta)
        & (merged["sharpe_delta"] >= cfg.min_sharpe_delta)
        & (merged["max_drawdown_delta"] >= -cfg.max_drawdown_delta)
        & (merged["win_rate_delta"] >= cfg.min_win_rate_delta)
    )

    merged["comparison_score"] = (
        merged["profit_factor_delta"].clip(-2, 2) * 0.30
        + merged["sharpe_delta"].clip(-3, 3) * 0.25
        + merged["win_rate_delta"].clip(-1, 1) * 0.20
        + merged["excess_return_delta"].clip(-2, 2) * 0.15
        + merged["max_drawdown_delta"].clip(-1, 1) * 0.10
    )

    merged = merged.sort_values("comparison_score", ascending=False).reset_index(drop=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "kronos_vs_baseline.csv"
    merged.to_csv(csv_path, index=False)

    aggregate = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol_count": int(len(merged)),
        "kronos_better_count": int(merged["kronos_better"].sum()),
        "kronos_better_rate": float(merged["kronos_better"].mean()),
        "kronos_avg_win_rate": _safe_mean(merged["win_rate_kronos"]),
        "baseline_avg_win_rate": _safe_mean(merged["win_rate_baseline"]),
        "avg_win_rate_delta": _safe_mean(merged["win_rate_delta"]),
        "kronos_avg_profit_factor": _safe_mean(merged["profit_factor_kronos"]),
        "baseline_avg_profit_factor": _safe_mean(merged["profit_factor_baseline"]),
        "avg_profit_factor_delta": _safe_mean(merged["profit_factor_delta"]),
        "kronos_avg_sharpe": _safe_mean(merged["sharpe_kronos"]),
        "baseline_avg_sharpe": _safe_mean(merged["sharpe_baseline"]),
        "avg_sharpe_delta": _safe_mean(merged["sharpe_delta"]),
        "kronos_avg_max_drawdown": _safe_mean(merged["max_drawdown_kronos"]),
        "baseline_avg_max_drawdown": _safe_mean(merged["max_drawdown_baseline"]),
        "avg_max_drawdown_delta": _safe_mean(merged["max_drawdown_delta"]),
        "verdict": "PASS" if len(merged) >= cfg.min_symbols and merged["kronos_better"].mean() >= 0.50 else "FAIL",
        "criteria": cfg.__dict__,
        "csv_report": str(csv_path),
    }
    (output_dir / "kronos_vs_baseline_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return merged
