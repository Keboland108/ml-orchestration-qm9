"""Pipeline assembly: a driver = engine modules + config + requested outputs.

The engine modules know nothing about QM9. Everything QM9-specific lives
in the config dicts HERE, and only here. Split of responsibility:
with_config() carries what the pipeline IS (column names, model spec —
stable across runs); execute(inputs=) carries what this run PROCESSES
(the raw_path of an arriving chunk).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime

import mlflow
from hamilton import driver
from mlflow.tracking import MlflowClient

from molpipe import evaluation, features, gate, ingestion, models, training

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
}


def build_training_driver(config: dict | None = None) -> driver.Driver:
    """Pipeline #1: ingest -> train -> benchmark -> gate."""
    return (
        driver.Builder()
        .with_modules(ingestion, features, training, evaluation, models, gate)
        .with_config(dict(config or QM9_CONFIG))
        .build()
    )


def _config_hash(config: dict) -> str:
    """Identity of what the pipeline IS: sha256 of the canonicalized config."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _enact(result: dict, config: dict) -> None:
    """The edge: act on the gate's decision. The ONLY place that writes
    to the registry. Every run logs its audit record, win or lose."""
    decision = result["gate_decision"]
    mlflow.set_tracking_uri(config["registry_uri"])

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_kind": config["model_spec"]["kind"],
                "raw_data_hash": result["raw_data_hash"],
                "dataset_content_hash": result["dataset_content_hash"],
                "config_hash": _config_hash(config),
                "git_sha": _git_sha(),
            }
        )
        mlflow.log_metric("candidate_mae", result["candidate_metrics"]["mae"])
        if decision.delta_observed is not None:
            mlflow.log_metric("delta_observed", decision.delta_observed)
            mlflow.log_metric("ci_low", decision.ci_low)
            mlflow.log_metric("ci_high", decision.ci_high)
        mlflow.log_dict(asdict(decision), "gate_decision.json")
        mlflow.log_dict(dict(config), "config.json")

        if not decision.promote:
            return  # non-promotion is a recorded success; nothing to enact

        model_info = mlflow.sklearn.log_model(
            sk_model=result["candidate_model"], registered_model_name=config["model_name"]
        )

        version = model_info.registered_model_version

        client = MlflowClient(registry_uri=config["registry_uri"])
        client.set_registered_model_alias(
            name=config["model_name"], alias="champion", version=str(version)
        )

        client.set_model_version_tag(
            name=config["model_name"],
            key="promoted_at",
            value=datetime.now(UTC).isoformat(),
            version=str(version),
        )


def run_training(raw_path: str, config: dict | None = None) -> dict:
    """On-demand entry point — the CLI half of requirement 6."""
    cfg = dict(config or QM9_CONFIG)
    dr = build_training_driver(cfg)
    result = dr.execute(
        [
            "gate_decision",
            "candidate_metrics",
            "champion_metrics",
            "candidate_model",
            "raw_data_hash",
            "dataset_content_hash",
        ],
        inputs={"raw_path": raw_path},
    )
    _enact(result, cfg)
    return result


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/qm9.csv"
    result = run_training(path)
    decision = result["gate_decision"]
    print(f"candidate mae: {result['candidate_metrics']['mae']:.3f}")
    print(f"promote={decision.promote}  reason: {decision.reason}")
    for check in decision.checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.reason}")
