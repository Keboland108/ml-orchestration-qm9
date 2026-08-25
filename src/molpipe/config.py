"""Config: the QM9 defaults, and the schema every config is checked against.

NOT a Hamilton module. The engine modules receive config values by parameter
name, so a typo'd key silently keeps the default and an out-of-range value
surfaces as a KeyError halfway through a DAG run. Everything here runs before
a driver is built, so a bad config fails in milliseconds instead of after
featurizing 130k molecules.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from molpipe.training import ESTIMATORS

QM9_CONFIG: dict = {
    "smiles_column": "smiles",
    "target_column": "u0_atom",
    "test_fraction": 0.2,
    "split_seed": 7,
    "fp_radius": 2,
    "fp_bits": 2048,
    "model_spec": {"kind": "dummy"},
    "model_name": "qm9-property-model",
    "registry_uri": "sqlite:///mlflow.db",
    "gate_config": {"min_samples": 1000, "margin": 0.5, "n_boot": 1000, "seed": 7},
    "data_path": "data/raw/qm9.csv",
    "shortlist_size": 100,
    "rank_ascending": True,
}


class ConfigError(ValueError):
    """A config that would fail inside the DAG, rejected before the DAG runs."""


# key -> (accepted types, extra predicate, what the predicate requires)
_Rule = tuple[Any, Callable[[Any], bool] | None, str]

_RULES: dict[str, _Rule] = {
    "smiles_column": (str, None, ""),
    "target_column": (str, None, ""),
    "test_fraction": ((int, float), lambda v: 0 < v < 1, "must be between 0 and 1 exclusive"),
    "split_seed": (int, None, ""),
    "fp_radius": (int, lambda v: v >= 0, "must be 0 or greater"),
    "fp_bits": (int, lambda v: v > 0, "must be greater than 0"),
    "model_spec": (dict, None, ""),
    "model_name": (str, lambda v: bool(v.strip()), "must not be empty"),
    "registry_uri": (str, lambda v: bool(v.strip()), "must not be empty"),
    "gate_config": (dict, None, ""),
    "data_path": (str, None, ""),
    "shortlist_size": (int, lambda v: v > 0, "must be greater than 0"),
    "rank_ascending": (bool, None, ""),
}

_GATE_RULES: dict[str, _Rule] = {
    "min_samples": (int, lambda v: v >= 0, "must be 0 or greater"),
    "margin": ((int, float), lambda v: v >= 0, "must be 0 or greater"),
    "n_boot": (int, lambda v: v > 0, "must be greater than 0"),
    "seed": (int, None, ""),
}

_SPEC_RULES: dict[str, _Rule] = {
    "kind": (str, lambda v: v in ESTIMATORS, f"must be one of {sorted(ESTIMATORS)}"),
    "params": (dict, None, ""),
}

# Sections that merge key-by-key instead of being replaced wholesale.
_NESTED = {"gate_config": _GATE_RULES, "model_spec": _SPEC_RULES}


def _type_name(expected: Any) -> str:
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    return expected.__name__


def _check_value(label: str, value: Any, rule: _Rule) -> None:
    expected, predicate, requirement = rule
    # bool subclasses int, so `"split_seed": true` would otherwise type-check.
    mistyped = not isinstance(value, expected) or (isinstance(value, bool) and expected is not bool)
    if mistyped:
        raise ConfigError(
            f"{label}: expected {_type_name(expected)}, got {type(value).__name__} ({value!r})"
        )
    if predicate is not None and not predicate(value):
        raise ConfigError(f"{label}: {requirement} (got {value!r})")


def _check_section(label: str, section: dict, rules: dict[str, _Rule], required: set[str]) -> None:
    for key in section:
        if key in rules:
            continue
        near = difflib.get_close_matches(key, rules, n=1)
        hint = f" - did you mean {near[0]!r}?" if near else f" (known keys: {sorted(rules)})"
        raise ConfigError(f"unknown key {label}{key!r}{hint}")

    missing = sorted(required - set(section))
    if missing:
        raise ConfigError(f"missing required config keys: {missing}")

    for key, value in section.items():
        _check_value(f"{label}{key}", value, rules[key])


def validate_config(config: dict) -> dict:
    """Raise ConfigError unless every key is known, typed, and in range.

    Returns the config unchanged so callers can validate inline.
    """
    if not isinstance(config, dict):
        raise ConfigError(f"config must be a JSON object, got {type(config).__name__}")

    _check_section("", config, _RULES, set(_RULES))
    for section, rules in _NESTED.items():
        required = {"kind"} if section == "model_spec" else set(rules)
        _check_section(f"{section}.", config[section], rules, required)
    return config


def merge_config(overrides: dict, base: dict | None = None) -> dict:
    """Overrides layered on the defaults; the nested sections merge key-by-key.

    Deep-merging is what makes `{"gate_config": {"margin": 1.0}}` mean "raise
    the margin" rather than "delete min_samples, n_boot and seed". The copy
    also keeps callers from mutating QM9_CONFIG's nested dicts in place.
    """
    merged = {k: dict(v) if isinstance(v, dict) else v for k, v in (base or QM9_CONFIG).items()}
    for key, value in overrides.items():
        if key in _NESTED and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def resolve_config(overrides: dict | None) -> dict:
    """The entry-point path: merge over the defaults, then validate."""
    return validate_config(merge_config(overrides or {}))


def load_config(path: Path | str | None) -> dict:
    """Read a JSON config file and resolve it against the QM9 defaults."""
    if path is None:
        return resolve_config(None)
    try:
        overrides = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: not valid JSON ({exc})") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: cannot be read ({exc})") from exc
    if not isinstance(overrides, dict):
        raise ConfigError(f"{path}: expected a JSON object, got {type(overrides).__name__}")
    return resolve_config(overrides)
