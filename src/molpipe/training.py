"""Training node: fit the candidate. Estimator choice is config, not code."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

_ESTIMATORS = {
    "dummy": DummyRegressor,
    "hist_gbr": HistGradientBoostingRegressor,
    "ridge": Ridge,
}


def candidate_model(
    train_features: np.ndarray,
    train_frame: pd.DataFrame,
    target_column: str,
    model_spec: dict[str, Any],
) -> Any:
    """Fit model_spec["kind"] with model_spec["params"] on the train split."""
    kind = model_spec["kind"]
    if kind not in _ESTIMATORS:
        raise ValueError(f"unknown model kind {kind!r}; choose from {sorted(_ESTIMATORS)}")
    estimator = _ESTIMATORS[kind](**model_spec.get("params", {}))
    return estimator.fit(train_features, train_frame[target_column].to_numpy())
