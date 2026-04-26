"""Market data downloader for the clean CSV Kronos pipeline.

The implementation uses yfinance as an optional runtime dependency. It writes one
raw canonical CSV per symbol so the rest of the pipeline stays data-source agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class DownloadConfig:
    provider: str = "yfinance"
    period: str = "5y"
    interval: str = "1d"
    auto_adjust: bool = True
    start: str | None = None
    end: str | None = None


def load_symbols_from_file(path: str | Path) -> list[str]:
    """Load symbols from a text file.

    Supports one symbol per line, comma-separated lines, and comments beginning
    with '#'.
    """

    symbols: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            for part in line.replace(";", ",").split(","):
                sym = part.strip().upper()
                if sym:
                    symbols.append(sym)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(symbols))


def _import_yfinance():
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError("Install yfinance first: pip install yfinance") from exc
    return yf


def _normalize_yfinance_frame(symbol: str, raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError(f"{symbol}: downloaded data is empty")

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return a multi-index when multiple tickers are requested.
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.reset_index()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    timestamp_col = None
    for candidate in ["date", "datetime", "timestamp"]:
        if candidate in df.columns:
            timestamp_col = candidate
            break
    if timestamp_col is None:
        raise ValueError(f"{symbol}: timestamp column not found in downloaded data")

    rename_map = {
        timestamp_col: "timestamps",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    df = df.rename(columns=rename_map)

    required = ["timestamps", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns after download: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["amount"] = df["close"] * df["volume"].fillna(0.0)
    df.insert(0, "symbol", symbol.upper())
    return df[["symbol", "timestamps", "open", "high", "low", "close", "volume", "amount"]]


def download_yfinance_symbol(symbol: str, config: DownloadConfig) -> pd.DataFrame:
    yf = _import_yfinance()
    kwargs = {
        "interval": config.interval,
        "auto_adjust": config.auto_adjust,
        "progress": False,
        "threads": False,
    }
    if config.start or config.end:
        if config.start:
            kwargs["start"] = config.start
        if config.end:
            kwargs["end"] = config.end
    else:
        kwargs["period"] = config.period

    raw = yf.download(symbol, **kwargs)
    return _normalize_yfinance_frame(symbol, raw)


def download_symbols(
    symbols: Sequence[str] | Iterable[str],
    output_dir: str | Path,
    config: DownloadConfig | None = None,
) -> pd.DataFrame:
    """Download market data and write one CSV per symbol.

    Returns a report DataFrame with status and row counts.
    """

    config = config or DownloadConfig()
    provider = config.provider.lower()
    if provider != "yfinance":
        raise ValueError(f"Unsupported provider: {config.provider}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    for symbol in [s.upper() for s in symbols]:
        try:
            df = download_yfinance_symbol(symbol, config)
            out_path = output_dir / f"{symbol}.csv"
            df.to_csv(out_path, index=False)
            reports.append(
                {
                    "symbol": symbol,
                    "status": "ok",
                    "rows": len(df),
                    "start": pd.to_datetime(df["timestamps"]).min(),
                    "end": pd.to_datetime(df["timestamps"]).max(),
                    "path": str(out_path),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            reports.append(
                {
                    "symbol": symbol,
                    "status": "error",
                    "rows": 0,
                    "start": "",
                    "end": "",
                    "path": "",
                    "error": str(exc),
                }
            )

    report_df = pd.DataFrame(reports)
    report_df.to_csv(output_dir / "download_report.csv", index=False)
    return report_df
