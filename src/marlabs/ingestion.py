"""Ingestion nodes: read one raw chunk, enforce the EDA cleaning spec.

Hamilton module — function name = node, param names = upstream nodes or
driver config. Knows nothing about QM9: column names arrive as config.
"""

from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)


def raw_frame(raw_path: str) -> pd.DataFrame:
    """Read one raw CSV (a landed chunk or the full dataset)."""
    return pd.read_csv(raw_path)


def validated_frame(
    raw_frame: pd.DataFrame, smiles_column: str, target_column: str
) -> pd.DataFrame:
    """Cleaning + validation per the EDA decisions.

    Schema-level problems RAISE (a malformed chunk must fail loudly):
    missing required columns, non-numeric target.
    Row-level dirt is DROPPED AND COUNTED (a few bad rows must not kill
    a run): null smiles/target, duplicate SMILES (keep-first),
    RDKit-unparseable SMILES.
    """
    missing = {smiles_column, target_column} - set(raw_frame.columns)
    if missing:
        raise ValueError(f"raw frame missing required columns: {sorted(missing)}")
    if not pd.api.types.is_numeric_dtype(raw_frame[target_column]):
        raise ValueError(f"target column {target_column!r} is not numeric")

    n_start = len(raw_frame)
    frame = raw_frame.dropna(subset=[smiles_column, target_column])
    frame = frame.drop_duplicates(subset=smiles_column, keep="first")
    parseable = frame[smiles_column].map(lambda s: Chem.MolFromSmiles(s) is not None)
    frame = frame[parseable].reset_index(drop=True)

    n_dropped = n_start - len(frame)
    if n_dropped:
        logger.warning(
            "validated_frame dropped %d/%d rows (nulls, duplicate or unparseable SMILES)",
            n_dropped,
            n_start,
        )
    return frame
