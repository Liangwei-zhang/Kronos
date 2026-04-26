"""Configuration helpers for the clean CSV Kronos pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .backtest import BacktestConfig, WalkForwardConfig
from .download import DownloadConfig
from .predict import PredictionConfig
from .scanner import ScannerConfig


@dataclass(frozen=True)
class PipelinePaths:
    raw_data_dir: str = "data/raw_csv"
    clean_data_dir: str = "data/clean_csv"
    predictions_dir: str = "outputs/predictions"
    backtest_dir: str = "outputs/backtest"
    scanner_dir: str = "outputs/scanner"


@dataclass(frozen=True)
class PipelineConfig:
    paths: PipelinePaths
    download: DownloadConfig
    prediction: PredictionConfig
    walk_forward: WalkForwardConfig
    scanner: ScannerConfig

    def to_backtest_config(self) -> BacktestConfig:
        return BacktestConfig(prediction=self.prediction, walk_forward=self.walk_forward)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping")
    return value


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    paths = PipelinePaths(**_section(raw, "paths"))
    download = DownloadConfig(**_section(raw, "download"))
    prediction = PredictionConfig(**_section(raw, "prediction"))
    walk_forward = WalkForwardConfig(**_section(raw, "walk_forward"))
    scanner = ScannerConfig(**_section(raw, "scanner"))
    return PipelineConfig(paths=paths, download=download, prediction=prediction, walk_forward=walk_forward, scanner=scanner)
