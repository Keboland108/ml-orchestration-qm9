"""Ingestion nodes: read one raw chunk, enforce the EDA cleaning spec.

Hamilton module — function name = node, param names = upstream nodes or
driver config. Knows nothing about QM9: column names arrive as config.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)


def raw_frame(raw_path: str) -> pd.DataFrame:
    """Read one raw CSV (a landed chunk or the full dataset)."""
    return pd.read_csv(raw_path)


def raw_data_hash(raw_path: str) -> str:
    """ARRIVAL identity: sha256 of the raw file bytes, born at ingest.

    Job: idempotency ("seen this exact file?"). Repackaged chunks with the
    same rows get a different hash — by design; content identity is the
    other hash's job.
    """
    h = hashlib.sha256()
    with Path(raw_path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dataset_content_hash(validated_frame: pd.DataFrame, smiles_column: str) -> str:
    """DATASET identity: sha256 over the sorted row identities (canonical
    SMILES) of the deduped frame.

    Packaging-invariant: the same rows arriving in differently-shaped chunks
    produce the same hash. Answers "what data trained this model."
    """
    keys = "\n".join(sorted(validated_frame[smiles_column]))
    return hashlib.sha256(keys.encode()).hexdigest()


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
    frame = pd.DataFrame(frame[parseable].reset_index(drop=True))

    n_dropped = n_start - len(frame)
    if n_dropped:
        logger.warning(
            "validated_frame dropped %d/%d rows (nulls, duplicate or unparseable SMILES)",
            n_dropped,
            n_start,
        )
    return frame
