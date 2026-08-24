"""Ingestion nodes: read one raw chunk and establish its identity.

Hamilton module — function name = node, param names = upstream nodes or
driver config. Knows nothing about QM9: column names arrive as config.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


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
