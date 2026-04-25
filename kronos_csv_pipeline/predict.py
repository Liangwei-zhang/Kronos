"""Batch prediction helpers for Kronos over cleaned CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from model import Kronos, KronosPredictor, KronosTokenizer

from .data import FEATURE_COLUMNS, infer_future_timestamps, load_cleaned_symbols


MODEL_REGISTRY = {
    "kronos-mini": {
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context": 2048,
    },
    "kronos-small": {
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
    "kronos-base": {
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
}


@dataclass(frozen=True)
class PredictionConfig:
    model_key: str = "kronos-small"
    model_path: str | None = None
    tokenizer_path: str | None = None
    max_context: int | None = None
    lookback: int = 512
    pred_len: int = 5
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.9
    sample_count: int = 1
    device: str | None = None


def auto_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_predictor(config: PredictionConfig) -> KronosPredictor:
    """Load Kronos tokenizer/model and return a predictor."""

    registry = MODEL_REGISTRY.get(config.model_key, MODEL_REGISTRY["kronos-small"])
    tokenizer_path = config.tokenizer_path or registry["tokenizer_id"]
    model_path = config.model_path or registry["model_id"]
    max_context = config.max_context or int(registry["max_context"])

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    model = Kronos.from_pretrained(model_path)
    return KronosPredictor(model, tokenizer, device=auto_device(config.device), max_context=max_context)


def _prepare_batch_inputs(
    symbol_data: dict[str, pd.DataFrame],
    lookback: int,
    pred_len: int,
) -> tuple[list[str], list[pd.DataFrame], list[pd.Series], list[pd.Series]]:
    symbols: list[str] = []
    df_list: list[pd.DataFrame] = []
    x_ts_list: list[pd.Series] = []
    y_ts_list: list[pd.Series] = []

    for symbol, df in sorted(symbol_data.items()):
        if len(df) < lookback:
            continue
        tail = df.tail(lookback).copy()
        x_df = tail[FEATURE_COLUMNS]
        x_ts = pd.Series(pd.to_datetime(tail["timestamps"]), name="timestamps").reset_index(drop=True)
        y_ts = infer_future_timestamps(x_ts, pred_len)
        symbols.append(symbol)
        df_list.append(x_df.reset_index(drop=True))
        x_ts_list.append(x_ts)
        y_ts_list.append(y_ts)

    if not symbols:
        raise ValueError("No symbols have enough rows for the requested lookback")
    return symbols, df_list, x_ts_list, y_ts_list


def predict_symbols(
    clean_dir: str | Path,
    output_dir: str | Path,
    config: PredictionConfig,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Predict future OHLCV bars for many symbols and write per-symbol CSVs.

    Returns a compact ranking table with the latest close, predicted final close,
    and predicted return.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_cleaned_symbols(clean_dir, symbols)
    batch_symbols, df_list, x_ts_list, y_ts_list = _prepare_batch_inputs(data, config.lookback, config.pred_len)
    predictor = load_predictor(config)

    preds = predictor.predict_batch(
        df_list=df_list,
        x_timestamp_list=x_ts_list,
        y_timestamp_list=y_ts_list,
        pred_len=config.pred_len,
        T=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        sample_count=config.sample_count,
        verbose=False,
    )

    rows = []
    for symbol, pred_df, source_df in zip(batch_symbols, preds, df_list):
        pred_out = pred_df.reset_index().rename(columns={"index": "timestamps"})
        pred_out.insert(0, "symbol", symbol)
        pred_path = output_dir / f"{symbol}_prediction.csv"
        pred_out.to_csv(pred_path, index=False)

        last_close = float(source_df["close"].iloc[-1])
        pred_final_close = float(pred_df["close"].iloc[-1])
        pred_max_close = float(pred_df["close"].max())
        pred_min_close = float(pred_df["close"].min())
        predicted_return = pred_final_close / last_close - 1.0
        predicted_upside = pred_max_close / last_close - 1.0
        predicted_drawdown = pred_min_close / last_close - 1.0

        rows.append(
            {
                "symbol": symbol,
                "last_close": last_close,
                "pred_final_close": pred_final_close,
                "predicted_return": predicted_return,
                "predicted_upside": predicted_upside,
                "predicted_drawdown": predicted_drawdown,
                "prediction_file": str(pred_path),
            }
        )

    ranking = pd.DataFrame(rows).sort_values("predicted_return", ascending=False).reset_index(drop=True)
    ranking.to_csv(output_dir / "prediction_ranking.csv", index=False)
    return ranking
