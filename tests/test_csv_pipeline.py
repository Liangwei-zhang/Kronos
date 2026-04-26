from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kronos_csv_pipeline.data import clean_data_dir, load_clean_csv
from kronos_csv_pipeline.scanner import ScannerConfig, scan_opportunities


def test_load_clean_csv_repairs_and_standardizes(tmp_path: Path):
    raw = tmp_path / "AAPL.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02", "2024-01-02", "bad"],
            "open": [10.0, 9.0, 9.5, 1.0],
            "high": [9.0, 9.2, 9.7, 1.0],
            "low": [11.0, 8.8, 9.1, 1.0],
            "close": [10.5, 9.1, 9.6, 1.0],
            "volume": [100, 90, 95, 1],
        }
    ).to_csv(raw, index=False)

    df = load_clean_csv(raw, min_rows=2)

    assert list(df.columns) == ["symbol", "timestamps", "open", "high", "low", "close", "volume", "amount"]
    assert df["symbol"].unique().tolist() == ["AAPL"]
    assert len(df) == 2
    assert df["timestamps"].is_monotonic_increasing
    assert (df["high"] >= df[["open", "close", "low"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close", "high"]].min(axis=1)).all()
    assert "amount" in df.columns


def test_clean_data_dir_writes_report(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    clean_dir = tmp_path / "clean"
    raw_dir.mkdir()
    pd.DataFrame(
        {
            "timestamps": pd.date_range("2024-01-01", periods=3),
            "open": [1, 2, 3],
            "high": [1.2, 2.2, 3.2],
            "low": [0.8, 1.8, 2.8],
            "close": [1.1, 2.1, 3.1],
            "volume": [10, 20, 30],
        }
    ).to_csv(raw_dir / "MSFT.csv", index=False)

    reports = clean_data_dir(raw_dir, clean_dir, min_rows=3)

    assert len(reports) == 1
    assert (clean_dir / "MSFT.csv").exists()
    assert reports[0].rows_after == 3


def test_scan_opportunities_outputs_csv_and_json(tmp_path: Path):
    pred_path = tmp_path / "prediction_ranking.csv"
    bt_path = tmp_path / "walk_forward_summary.csv"
    out_dir = tmp_path / "scanner"

    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "last_close": [100.0, 50.0],
            "pred_final_close": [105.0, 49.0],
            "predicted_return": [0.05, -0.02],
            "predicted_upside": [0.07, 0.01],
            "predicted_drawdown": [-0.01, -0.04],
        }
    ).to_csv(pred_path, index=False)

    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "total_return": [0.40, -0.10],
            "benchmark_return": [0.20, 0.05],
            "excess_return": [0.20, -0.15],
            "max_drawdown": [-0.12, -0.35],
            "sharpe": [1.2, -0.2],
            "win_rate": [0.6, 0.4],
            "avg_trade_return": [0.02, -0.01],
            "profit_factor": [1.8, 0.7],
            "trade_count": [12, 12],
        }
    ).to_csv(bt_path, index=False)

    result = scan_opportunities(
        pred_path,
        bt_path,
        out_dir,
        ScannerConfig(buy_score=0.35, watch_score=0.20, min_trade_count=5),
    )

    assert (out_dir / "portfolio_scan.csv").exists()
    assert (out_dir / "portfolio_scan.json").exists()
    assert result.iloc[0]["symbol"] == "AAA"
    assert result.iloc[0]["action"] in {"BUY", "WATCH"}

    payload = json.loads((out_dir / "portfolio_scan.json").read_text(encoding="utf-8"))
    assert "generated_at" in payload
    assert "counts" in payload
    assert "top" in payload
