"""Utilities for loading and cleaning OHLCV CSV files for Kronos.

Expected canonical columns:
    symbol, timestamps, open, high, low, close, volume, amount

The cleaning code is deliberately conservative: it sorts by time, removes duplicate
bars, coerces numeric fields, drops invalid rows, and derives missing optional
fields without using future data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close"]
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
CANONICAL_COLUMNS = ["symbol", "timestamps", *FEATURE_COLUMNS]

COLUMN_ALIASES = {
    "timestamp": "timestamps",
    "time": "timestamps",
    "datetime": "timestamps",
    "date": "timestamps",
    "日期": "timestamps",
    "开盘": "open",
    "开盘价": "open",
    "最高": "high",
    "最高价": "high",
    "最低": "low",
    "最低价": "low",
    "收盘": "close",
    "收盘价": "close",
    "成交量": "volume",
    "成交额": "amount",
    "ticker": "symbol",
    "stock_code": "symbol",
    "code": "symbol",
}


@dataclass(frozen=True)
class CleanReport:
    """Summary of one cleaned CSV file."""

    symbol: str
    input_path: str
    output_path: str | None
    rows_before: int
    rows_after: int
    start: str | None
    end: str | None
    dropped_rows: int


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = str(col).strip()
        lower_key = key.lower()
        renamed[col] = COLUMN_ALIASES.get(key, COLUMN_ALIASES.get(lower_key, lower_key))
    return df.rename(columns=renamed)


def _infer_symbol(path: str | Path, df: pd.DataFrame, symbol: str | None) -> str:
    if symbol:
        return symbol.upper()
    if "symbol" in df.columns and df["symbol"].notna().any():
        return str(df["symbol"].dropna().iloc[0]).upper()
    return Path(path).stem.upper()


def load_clean_csv(
    path: str | Path,
    symbol: str | None = None,
    min_rows: int = 0,
    fix_ohlc: bool = True,
) -> pd.DataFrame:
    """Load one CSV and return canonical Kronos-ready data.

    Parameters
    ----------
    path:
        CSV file path.
    symbol:
        Optional symbol override. When omitted, the function uses the CSV symbol
        column or the file stem.
    min_rows:
        Raise an error when the cleaned file has fewer rows than this.
    fix_ohlc:
        If true, high/low are repaired from open/high/low/close when only high-low
        consistency is violated. This avoids throwing away otherwise usable bars.
    """

    path = Path(path)
    df = pd.read_csv(path)
    df = _normalise_columns(df)
    rows_before = len(df)

    missing = [c for c in ["timestamps", *PRICE_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    sym = _infer_symbol(path, df, symbol)
    df["symbol"] = sym
    df["timestamps"] = pd.to_datetime(df["timestamps"], errors="coerce", utc=False)

    for col in PRICE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    if "amount" not in df.columns:
        df["amount"] = df["close"] * df["volume"]
    else:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["amount"] = df["amount"].fillna(df["close"] * df["volume"])

    df = df.dropna(subset=["timestamps", *PRICE_COLUMNS])
    df = df.sort_values("timestamps").drop_duplicates("timestamps", keep="last").reset_index(drop=True)

    # Remove non-positive price rows. Volume can be zero for indices or illiquid bars.
    price_positive = (df[PRICE_COLUMNS] > 0).all(axis=1)
    df = df.loc[price_positive].copy()

    if fix_ohlc:
        ohlc_max = df[PRICE_COLUMNS].max(axis=1)
        ohlc_min = df[PRICE_COLUMNS].min(axis=1)
        df["high"] = np.maximum(df["high"].to_numpy(), ohlc_max.to_numpy())
        df["low"] = np.minimum(df["low"].to_numpy(), ohlc_min.to_numpy())
    else:
        valid_high = df["high"] >= df[["open", "close", "low"]].max(axis=1)
        valid_low = df["low"] <= df[["open", "close", "high"]].min(axis=1)
        df = df.loc[valid_high & valid_low].copy()

    df = df[CANONICAL_COLUMNS].reset_index(drop=True)

    if len(df) < min_rows:
        raise ValueError(f"{path} has {len(df)} clean rows, below min_rows={min_rows}")

    return df


def list_csv_files(data_dir: str | Path, symbols: Sequence[str] | None = None) -> list[Path]:
    """Return CSV files for a data directory, optionally filtered by symbols."""

    data_dir = Path(data_dir)
    if symbols:
        wanted = {s.upper() for s in symbols}
        files = [p for p in data_dir.glob("*.csv") if p.stem.upper() in wanted]
    else:
        files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return sorted(files)


def clean_data_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    symbols: Sequence[str] | None = None,
    min_rows: int = 0,
) -> list[CleanReport]:
    """Clean every CSV in a directory and write canonical CSV files."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[CleanReport] = []
    for csv_path in list_csv_files(input_dir, symbols):
        raw_rows = len(pd.read_csv(csv_path, usecols=lambda _: True))
        df = load_clean_csv(csv_path, min_rows=min_rows)
        symbol = str(df["symbol"].iloc[0])
        out_path = output_dir / f"{symbol}.csv"
        df.to_csv(out_path, index=False)
        reports.append(
            CleanReport(
                symbol=symbol,
                input_path=str(csv_path),
                output_path=str(out_path),
                rows_before=raw_rows,
                rows_after=len(df),
                start=df["timestamps"].iloc[0].isoformat() if len(df) else None,
                end=df["timestamps"].iloc[-1].isoformat() if len(df) else None,
                dropped_rows=raw_rows - len(df),
            )
        )
    return reports


def infer_future_timestamps(x_timestamp: pd.Series | pd.DatetimeIndex, pred_len: int) -> pd.Series:
    """Infer future timestamps using the median historical interval.

    For daily bars with weekday-only history, this returns business days. For
    intraday bars, it uses the median time delta directly.
    """

    ts = pd.Series(pd.to_datetime(x_timestamp)).sort_values().reset_index(drop=True)
    if len(ts) < 2:
        raise ValueError("Need at least two timestamps to infer future timestamps")

    diffs = ts.diff().dropna()
    median_diff = diffs.median()
    last_ts = ts.iloc[-1]

    if median_diff >= pd.Timedelta(hours=20):
        # Daily bars: keep weekends out by default.
        future = pd.bdate_range(start=last_ts + pd.Timedelta(days=1), periods=pred_len)
    else:
        future = pd.date_range(start=last_ts + median_diff, periods=pred_len, freq=median_diff)
    return pd.Series(future, name="timestamps")


def load_cleaned_symbols(clean_dir: str | Path, symbols: Iterable[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load already-cleaned canonical CSV files into a symbol -> DataFrame map."""

    data: dict[str, pd.DataFrame] = {}
    for path in list_csv_files(clean_dir, list(symbols) if symbols else None):
        df = load_clean_csv(path)
        sym = str(df["symbol"].iloc[0]).upper()
        data[sym] = df
    return data
