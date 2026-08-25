"""Validation node: enforce the EDA cleaning spec on an ingested frame.

Hamilton module — one graded logic, one module, one test file
(mirrors gate.py / test_gate.py).
"""

from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)


def _canonical_smiles(value: object) -> str | None:
    """RDKit's canonical form, or None if the value will not parse.

    Canonicalizing here — the one stage that already parses every row —
    makes every downstream identity per-molecule instead of per-string.
    The content hash, the hash split, and duplicate detection all stop
    depending on how the supplier happened to write the SMILES.
    """
    if not isinstance(value, str):
        return None
    mol = Chem.MolFromSmiles(value)
    return None if mol is None else Chem.MolToSmiles(mol)


def validated_frame(
    raw_frame: pd.DataFrame, smiles_column: str, target_column: str
) -> pd.DataFrame:
    """Cleaning + validation per the EDA decisions.

    Schema-level problems RAISE (a malformed chunk must fail loudly):
    missing required columns, non-numeric target.
    Row-level dirt is DROPPED AND COUNTED (a few bad rows must not kill
    a run): null smiles/target, duplicate SMILES (keep-first),
    RDKit-unparseable SMILES.
    A chunk with NO valid rows left is a malformed chunk, not row dirt —
    it raises here, at the module that decided, not two stages later.

    SMILES are canonicalized before deduplication, so two spellings of the
    same molecule collapse to one row instead of surviving as two.
    """
    missing = {smiles_column, target_column} - set(raw_frame.columns)
    if missing:
        raise ValueError(f"raw frame missing required columns: {sorted(missing)}")
    if not pd.api.types.is_numeric_dtype(raw_frame[target_column]):
        raise ValueError(f"target column {target_column!r} is not numeric")

    n_start = len(raw_frame)
    frame = raw_frame.dropna(subset=[smiles_column, target_column])
    # Canonicalize first: unparseable rows become NaN and drop, and dedup
    # then compares molecules rather than spellings.
    frame = frame.assign(**{smiles_column: frame[smiles_column].map(_canonical_smiles)})
    frame = frame.dropna(subset=[smiles_column])
    frame = frame.drop_duplicates(subset=smiles_column, keep="first")
    frame = pd.DataFrame(frame.reset_index(drop=True))

    n_dropped = n_start - len(frame)
    if n_dropped:
        logger.warning(
            "validated_frame dropped %d/%d rows (nulls, duplicate or unparseable SMILES)",
            n_dropped,
            n_start,
        )
    if frame.empty:
        raise ValueError(f"no valid rows remain after validation ({n_start} raw rows, all dropped)")
    return frame
