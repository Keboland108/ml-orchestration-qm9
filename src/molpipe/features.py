"""Split + featurization nodes. Shared verbatim by every pipeline.

The split is hash-based: a molecule's membership is decided by hashing
its SMILES (salted with split_seed). Deterministic, zero stored state,
and stable under data arrival — a molecule that was ever in test stays
in test forever, so a champion's training rows can never drift into a
later run's benchmark set.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def _test_mask(smiles: pd.Series, test_fraction: float, split_seed: int) -> pd.Series:
    def in_test(s: str) -> bool:
        digest = hashlib.md5(f"{split_seed}:{s}".encode()).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF < test_fraction

    return smiles.map(in_test)


def train_frame(
    validated_frame: pd.DataFrame, smiles_column: str, test_fraction: float, split_seed: int
) -> pd.DataFrame:
    mask = _test_mask(validated_frame[smiles_column], test_fraction, split_seed)
    return validated_frame[~mask].reset_index(drop=True)


def test_frame(
    validated_frame: pd.DataFrame, smiles_column: str, test_fraction: float, split_seed: int
) -> pd.DataFrame:
    """The held-out benchmark set — the rows the gate judges on."""
    mask = _test_mask(validated_frame[smiles_column], test_fraction, split_seed)
    return validated_frame[mask].reset_index(drop=True)


def _fingerprints(smiles: pd.Series, fp_radius: int, fp_bits: int) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_bits)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return np.stack([gen.GetFingerprintAsNumPy(m) for m in mols]).astype(np.uint8)


def train_features(
    train_frame: pd.DataFrame, smiles_column: str, fp_radius: int, fp_bits: int
) -> np.ndarray:
    return _fingerprints(train_frame[smiles_column], fp_radius, fp_bits)


def test_features(
    test_frame: pd.DataFrame, smiles_column: str, fp_radius: int, fp_bits: int
) -> np.ndarray:
    return _fingerprints(test_frame[smiles_column], fp_radius, fp_bits)
