"""Risk filter for turning scanner signals into safer candidate orders.

This module does not place orders. It applies portfolio-level guardrails to
scanner output and produces a risk-approved candidate list for paper trading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    account_equity: float = 100_000.0
    max_position_pct: float = 0.05
    max_total_exposure_pct: float = 0.50
    max_positions: int = 10
    min_final_score: float = 0.35
    min_predicted_return: float = 0.01
    min_win_rate: float = 0.50
    min_profit_factor: float = 1.05
    max_symbol_drawdown: float = -0.50
    min_avg_dollar_volume: float = 5_000_000.0
    allow_watch: bool = False


def _load_optional_liquidity(liquidity_path: str | Path | None) -> pd.DataFrame | None:
    if not liquidity_path:
        return None
    path = Path(liquidity_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        return None
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def apply_risk_filter(
    scanner_path: str | Path,
    output_dir: str | Path,
    config: RiskConfig | None = None,
    liquidity_path: str | Path | None = None,
) -> pd.DataFrame:
    """Filter scanner output and create a risk-approved allocation plan."""

    cfg = config or RiskConfig()
    scanner_path = Path(scanner_path)
    if not scanner_path.exists():
        raise FileNotFoundError(f"Scanner file not found: {scanner_path}")

    df = pd.read_csv(scanner_path)
    if "symbol" not in df.columns or "action" not in df.columns:
        raise ValueError("Scanner CSV must contain symbol and action columns")
    df["symbol"] = df["symbol"].astype(str).str.upper()

    liquidity = _load_optional_liquidity(liquidity_path)
    if liquidity is not None:
        liq_cols = [c for c in ["symbol", "avg_amount_60", "avg_dollar_volume", "avg_volume_60"] if c in liquidity.columns]
        df = df.merge(liquidity[liq_cols], on="symbol", how="left")

    numeric_defaults = {
        "final_score": 0.0,
        "predicted_return": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": -1.0,
        "last_close": np.nan,
        "pred_final_close": np.nan,
        "avg_amount_60": np.nan,
        "avg_dollar_volume": np.nan,
    }
    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    allowed_actions = ["BUY", "WATCH"] if cfg.allow_watch else ["BUY"]
    df["risk_reasons"] = ""

    def add_reason(mask: pd.Series, reason: str) -> None:
        df.loc[mask, "risk_reasons"] = df.loc[mask, "risk_reasons"].apply(
            lambda x: reason if not x else f"{x};{reason}"
        )

    add_reason(~df["action"].isin(allowed_actions), "action_not_allowed")
    add_reason(df["final_score"] < cfg.min_final_score, "low_final_score")
    add_reason(df["predicted_return"] < cfg.min_predicted_return, "low_predicted_return")
    add_reason(df["win_rate"] < cfg.min_win_rate, "low_win_rate")
    add_reason(df["profit_factor"].replace(np.inf, 999.0) < cfg.min_profit_factor, "low_profit_factor")
    add_reason(df["max_drawdown"] < cfg.max_symbol_drawdown, "symbol_drawdown_too_large")

    liquidity_col = "avg_amount_60" if "avg_amount_60" in df.columns else "avg_dollar_volume"
    if liquidity_col in df.columns and df[liquidity_col].notna().any():
        add_reason(df[liquidity_col].fillna(0.0) < cfg.min_avg_dollar_volume, "low_liquidity")

    approved = df[df["risk_reasons"] == ""].copy()
    approved = approved.sort_values("final_score", ascending=False).head(cfg.max_positions).reset_index(drop=True)

    max_position_value = cfg.account_equity * cfg.max_position_pct
    max_total_value = cfg.account_equity * cfg.max_total_exposure_pct
    if len(approved) > 0:
        equal_weight_value = min(max_position_value, max_total_value / len(approved))
        approved["target_position_value"] = equal_weight_value
        approved["target_weight"] = equal_weight_value / cfg.account_equity
        approved["estimated_shares"] = np.floor(approved["target_position_value"] / approved["last_close"].replace(0, np.nan)).fillna(0).astype(int)
    else:
        approved["target_position_value"] = []
        approved["target_weight"] = []
        approved["estimated_shares"] = []

    rejected = df[df["risk_reasons"] != ""].copy().sort_values("final_score", ascending=False)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    approved.to_csv(output_dir / "risk_approved_candidates.csv", index=False)
    rejected.to_csv(output_dir / "risk_rejected_candidates.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                **asdict(cfg),
                "input_rows": len(df),
                "approved_rows": len(approved),
                "rejected_rows": len(rejected),
                "planned_total_exposure": float(approved["target_position_value"].sum()) if len(approved) else 0.0,
            }
        ]
    )
    summary.to_csv(output_dir / "risk_summary.csv", index=False)
    return approved
