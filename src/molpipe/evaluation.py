"""Evaluation nodes: predictions + the metrics dicts the gate consumes.

Metrics contract (what gate_decision receives):
    {"mae": float, "abs_errors": np.ndarray}   # abs_errors feeds the bootstrap
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _metrics(predictions: np.ndarray, test_frame: pd.DataFrame, target_column: str) -> dict:
    abs_errors = np.abs(predictions - test_frame[target_column].to_numpy())
    return {"mae": float(abs_errors.mean()), "abs_errors": abs_errors}


def candidate_predictions(candidate_model: Any, test_features: np.ndarray) -> np.ndarray:
    return np.asarray(candidate_model.predict(test_features)).ravel()


def champion_predictions(
    champion_model: Any | None, test_features: np.ndarray
) -> np.ndarray | None:
    """None on cold start — no champion to benchmark against."""
    if champion_model is None:
        return None
    return np.asarray(champion_model.predict(test_features)).ravel()


def candidate_metrics(
    candidate_predictions: np.ndarray, test_frame: pd.DataFrame, target_column: str
) -> dict:
    return _metrics(candidate_predictions, test_frame, target_column)


def champion_metrics(
    champion_predictions: np.ndarray | None, test_frame: pd.DataFrame, target_column: str
) -> dict | None:
    if champion_predictions is None:
        return None
    return _metrics(champion_predictions, test_frame, target_column)
