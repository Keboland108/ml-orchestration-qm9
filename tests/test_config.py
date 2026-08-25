"""Config validation: a bad config must fail before the DAG is built."""

from __future__ import annotations

import json

import pytest

from molpipe.config import (
    QM9_CONFIG,
    ConfigError,
    load_config,
    merge_config,
    resolve_config,
    validate_config,
)


def test_defaults_are_valid():
    assert validate_config(dict(QM9_CONFIG)) is not None


def test_unknown_key_is_rejected_with_a_suggestion():
    with pytest.raises(ConfigError, match="test_fraction"):
        resolve_config({"test_fracton": 0.3})


def test_unknown_nested_key_is_rejected():
    with pytest.raises(ConfigError, match="gate_config.n_bootstrap|n_boot"):
        resolve_config({"gate_config": {"n_bootstrap": 500}})


def test_missing_key_is_rejected():
    partial = {k: v for k, v in QM9_CONFIG.items() if k != "registry_uri"}
    with pytest.raises(ConfigError, match="registry_uri"):
        validate_config(partial)


def test_nested_override_merges_instead_of_replacing():
    """The bug this guards: a partial gate_config used to delete the other keys."""
    cfg = resolve_config({"gate_config": {"margin": 1.0}})
    assert cfg["gate_config"]["margin"] == 1.0
    assert cfg["gate_config"]["min_samples"] == QM9_CONFIG["gate_config"]["min_samples"]
    assert cfg["gate_config"]["n_boot"] == QM9_CONFIG["gate_config"]["n_boot"]
    assert cfg["gate_config"]["seed"] == QM9_CONFIG["gate_config"]["seed"]


def test_merge_does_not_mutate_the_defaults():
    cfg = merge_config({"gate_config": {"margin": 99.0}})
    cfg["gate_config"]["min_samples"] = 1
    assert QM9_CONFIG["gate_config"]["margin"] == 0.5
    assert QM9_CONFIG["gate_config"]["min_samples"] == 1000


def test_wrong_type_is_rejected():
    with pytest.raises(ConfigError, match="expected int or float"):
        resolve_config({"test_fraction": "0.2"})


def test_bool_does_not_pass_as_an_int():
    with pytest.raises(ConfigError, match="expected int"):
        resolve_config({"split_seed": True})


@pytest.mark.parametrize(
    "overrides",
    [
        {"test_fraction": 1.5},
        {"test_fraction": 0},
        {"fp_bits": 0},
        {"shortlist_size": -1},
        {"gate_config": {"margin": -0.5}},
        {"gate_config": {"n_boot": 0}},
        {"model_name": "  "},
    ],
)
def test_out_of_range_values_are_rejected(overrides):
    with pytest.raises(ConfigError):
        resolve_config(overrides)


def test_unknown_model_kind_is_rejected():
    with pytest.raises(ConfigError, match="hist_gbr"):
        resolve_config({"model_spec": {"kind": "xgboost"}})


def test_model_params_pass_through():
    cfg = resolve_config({"model_spec": {"kind": "ridge", "params": {"alpha": 2.0}}})
    assert cfg["model_spec"] == {"kind": "ridge", "params": {"alpha": 2.0}}


def test_load_config_reads_and_resolves_a_file(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"target_column": "homo", "gate_config": {"margin": 0.25}}))

    cfg = load_config(path)

    assert cfg["target_column"] == "homo"
    assert cfg["gate_config"]["margin"] == 0.25
    assert cfg["gate_config"]["n_boot"] == 1000
    assert cfg["smiles_column"] == "smiles"


def test_malformed_json_is_a_config_error(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(path)


def test_json_that_is_not_an_object_is_rejected(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(ConfigError, match="expected a JSON object"):
        load_config(path)


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot be read"):
        load_config(tmp_path / "absent.json")
