"""Validation checks for the clean CSV Kronos pipeline."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import load_pipeline_config
from .data import list_csv_files, load_clean_csv


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    area: str
    message: str


def _module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _add(issues: list[ValidationIssue], severity: str, area: str, message: str) -> None:
    issues.append(ValidationIssue(severity=severity, area=area, message=message))


def validate_pipeline(
    config_path: str | Path,
    symbols: Sequence[str] | None = None,
    check_raw: bool = True,
    check_clean: bool = False,
    min_rows: int = 128,
) -> pd.DataFrame:
    """Validate runtime environment, config, and available CSV data."""

    issues: list[ValidationIssue] = []

    if sys.version_info < (3, 10):
        _add(issues, "error", "python", f"Python 3.10+ required, current={sys.version.split()[0]}")
    else:
        _add(issues, "ok", "python", f"Python version {sys.version.split()[0]}")

    required_modules = ["numpy", "pandas", "torch", "yaml", "huggingface_hub", "safetensors"]
    optional_modules = ["yfinance", "flask", "plotly"]
    for module in required_modules:
        _add(issues, "ok" if _module_exists(module) else "error", "dependency", module)
    for module in optional_modules:
        _add(issues, "ok" if _module_exists(module) else "warning", "optional_dependency", module)

    try:
        cfg = load_pipeline_config(config_path)
        _add(issues, "ok", "config", f"Loaded config: {config_path}")
    except Exception as exc:  # noqa: BLE001
        _add(issues, "error", "config", str(exc))
        return pd.DataFrame([i.__dict__ for i in issues])

    if cfg.prediction.lookback > (cfg.prediction.max_context or cfg.prediction.lookback):
        _add(
            issues,
            "warning",
            "prediction",
            f"lookback={cfg.prediction.lookback} exceeds max_context={cfg.prediction.max_context}; predictor will truncate context",
        )

    if cfg.walk_forward.lookback != cfg.prediction.lookback:
        _add(issues, "warning", "config", "prediction.lookback and walk_forward.lookback differ")
    if cfg.walk_forward.pred_len != cfg.prediction.pred_len:
        _add(issues, "warning", "config", "prediction.pred_len and walk_forward.pred_len differ")

    for path_name, path_value in [
        ("raw_data_dir", cfg.paths.raw_data_dir),
        ("clean_data_dir", cfg.paths.clean_data_dir),
        ("predictions_dir", cfg.paths.predictions_dir),
        ("backtest_dir", cfg.paths.backtest_dir),
        ("scanner_dir", cfg.paths.scanner_dir),
    ]:
        path = Path(path_value)
        if path.exists():
            _add(issues, "ok", "path", f"{path_name}: {path}")
        else:
            severity = "warning" if path_name != "raw_data_dir" else "warning"
            _add(issues, severity, "path", f"{path_name} does not exist yet: {path}")

    selected_symbols = [s.upper() for s in symbols] if symbols else None

    def validate_dir(dir_path: str, area: str) -> None:
        try:
            files = list_csv_files(dir_path, selected_symbols)
            _add(issues, "ok", area, f"Found {len(files)} CSV files in {dir_path}")
            for csv_path in files[:50]:
                try:
                    df = load_clean_csv(csv_path, min_rows=min_rows)
                    sym = str(df["symbol"].iloc[0]) if len(df) else csv_path.stem.upper()
                    _add(issues, "ok", area, f"{sym}: {len(df)} clean rows, {df['timestamps'].iloc[0]} -> {df['timestamps'].iloc[-1]}")
                except Exception as exc:  # noqa: BLE001
                    _add(issues, "error", area, f"{csv_path.name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            _add(issues, "warning", area, str(exc))

    if check_raw:
        validate_dir(cfg.paths.raw_data_dir, "raw_data")
    if check_clean:
        validate_dir(cfg.paths.clean_data_dir, "clean_data")

    return pd.DataFrame([i.__dict__ for i in issues])
