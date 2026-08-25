"""Tests for the validation logic

Frames are built in memory. No test reads the live dataset.
"""

import pandas as pd
import pytest

from molpipe.validation import validated_frame

SMILES = "smiles"
TARGET = "u0_atom"

# Valid SMILES: "C", "CC", "CCO", "c1ccccc1". Unparseable: "C((C".


def _frame(rows: list[tuple]) -> pd.DataFrame:
    """Make a raw frame from (smiles, target) rows."""
    return pd.DataFrame(rows, columns=[SMILES, TARGET])


def test_clean_frame_passes_unchanged():
    rows = [("CCCO", 1), ("CC", 2), ("CCO", 1.5), ("c1ccccc1", 1.25)]

    df = _frame(rows)
    vdf = validated_frame(df, SMILES, TARGET)

    assert len(vdf) == len(df)


def test_missing_column_raises():

    with pytest.raises(ValueError):
        rows = [("c", 1), ("CC", 2), ("CCO", 1.5), ("c1ccccc1", 1.25)]

        df = _frame(rows)

        df = df.drop(columns=[SMILES])

        validated_frame(df, SMILES, TARGET)


def test_non_numeric_target_raises():
    with pytest.raises(ValueError):
        rows = [("c", "1"), ("CC", "2"), ("CCO", "1.5"), ("c1ccccc1", "1.25")]

        df = _frame(rows)
        validated_frame(df, SMILES, TARGET)


def test_null_rows_dropped():
    rows = [(None, 1), ("CC", 2), ("CCO", None), ("c1ccccc1", 1.25)]

    df = _frame(rows)
    vdf = validated_frame(df, SMILES, TARGET)

    assert not vdf.isnull().any(axis=1).any()


def test_duplicate_smiles_keeps_first():
    rows = [("CC", 1), ("CC", 2), ("CCO", 1.15), ("c1ccccc1", 1.2)]

    df = _frame(rows)
    vdf = validated_frame(df, SMILES, TARGET)

    assert (vdf[SMILES] == "CC").sum() == 1
    assert vdf.loc[vdf[SMILES] == "CC", TARGET].iloc[0] == 1


def test_smiles_are_canonicalized():
    """Non-canonical input comes out in RDKit's canonical form."""
    rows = [("OCC", 1.0), ("C1=CC=CC=C1", 2.0)]

    vdf = validated_frame(_frame(rows), SMILES, TARGET)

    assert list(vdf[SMILES]) == ["CCO", "c1ccccc1"]


def test_duplicate_molecules_collapse_across_spellings():
    """Same molecule, two spellings: one row, not two.

    This is what makes the content hash and the hash split per-molecule
    rather than per-string.
    """
    rows = [("CCO", 1.0), ("OCC", 2.0), ("C(C)O", 3.0)]

    vdf = validated_frame(_frame(rows), SMILES, TARGET)

    assert len(vdf) == 1
    assert vdf[TARGET].iloc[0] == 1.0


def test_non_string_smiles_dropped():
    rows = [("CC", 1.0), (3.5, 2.0)]

    vdf = validated_frame(_frame(rows), SMILES, TARGET)

    assert list(vdf[SMILES]) == ["CC"]


def test_all_rows_dirty_raises():
    rows = [(None, 1), ("C((C", 2.0)]

    with pytest.raises(ValueError):
        validated_frame(_frame(rows), SMILES, TARGET)


def test_unparseable_smiles_dropped():
    rows = [(None, 1), ("CC", 2), ("CCO", None), ("c1ccccc1", 1.25), ("C((C", 0.75)]

    df = _frame(rows)
    vdf = validated_frame(df, SMILES, TARGET)

    exists = vdf.isin(["C((C"]).any().any()

    assert not exists
