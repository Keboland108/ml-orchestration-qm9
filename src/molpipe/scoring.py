"""Scoring nodes: predict with the champion, rank, cut a shortlist.

Hamilton module for pipeline #2. Shares ingestion, validation, features
and champion with the training pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def scored_frame(
    validated_frame: pd.DataFrame,
    scoring_features: np.ndarray,
    champion_model: Any | None,
    target_column: str,
) -> pd.DataFrame:
    """Champion predictions joined onto the frame. No champion = hard error."""
    if champion_model is None:
        raise ValueError("no champion model in the registry - run the training pipeline first")
    frame = validated_frame.copy()
    frame[f"predicted_{target_column}"] = champion_model.predict(scoring_features)
    return frame


def shortlist(
    scored_frame: pd.DataFrame,
    target_column: str,
    shortlist_size: int,
    rank_ascending: bool,
) -> pd.DataFrame:
    """Top shortlist_size rows, sorted by predicted target per rank_ascending."""
    ranked = scored_frame.sort_values(f"predicted_{target_column}", ascending=rank_ascending)
    return ranked.head(shortlist_size).reset_index(drop=True)
