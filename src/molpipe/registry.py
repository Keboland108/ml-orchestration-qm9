"""Registry writes: audit record, promotion, rollback.

NOT a Hamilton module. Never pass this file to a driver. The DAG reads
the registry through champion.py; this file is the only writer.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from molpipe.ingestion import raw_data_hash

CHAMPION_ALIAS = "champion"


def _config_hash(config: dict) -> str:
    """sha256 of the canonicalized config."""
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


def apply_gate_decision(result: dict, config: dict) -> None:
    """Log the audit record for every run. Write to the registry only on promote."""
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
            return  # non-promotion is a recorded success; nothing to write

        model_info = mlflow.sklearn.log_model(
            sk_model=result["candidate_model"], registered_model_name=config["model_name"]
        )

        version = model_info.registered_model_version

        client = MlflowClient(registry_uri=config["registry_uri"])
        client.set_registered_model_alias(
            name=config["model_name"], alias=CHAMPION_ALIAS, version=str(version)
        )

        client.set_model_version_tag(
            name=config["model_name"],
            key="promoted_at",
            value=datetime.now(UTC).isoformat(),
            version=str(version),
        )


def record_failure(config: dict, raw_path: str, error: BaseException) -> None:
    """Audit record for a crashed training run: one FAILED MLflow run.

    Called at the driver boundary when the DAG raises. The run carries the
    chunk's arrival identity, the config/code identity, and the error as a
    failure.json artifact. Re-raising stays the caller's job — the process
    fails loudly, but the run ledger keeps no gaps.
    """
    mlflow.set_tracking_uri(config["registry_uri"])
    try:
        arrival_hash = raw_data_hash(raw_path)
    except OSError:
        arrival_hash = "unreadable"

    mlflow.start_run()
    try:
        mlflow.log_params(
            {
                "model_kind": config["model_spec"]["kind"],
                "raw_data_hash": arrival_hash,
                "config_hash": _config_hash(config),
                "git_sha": _git_sha(),
            }
        )
        mlflow.log_dict(
            {"error_type": type(error).__name__, "error": str(error), "raw_path": raw_path},
            "failure.json",
        )
    finally:
        mlflow.end_run(status="FAILED")


def _eligible(version: ModelVersion) -> bool:
    """Promoted at some point, and not rolled back since that promotion."""
    promoted = version.tags.get("promoted_at")
    if promoted is None:
        return False
    rolled_back = version.tags.get("rolled_back_at")
    # ISO-8601 UTC strings: lexicographic order equals chronological order.
    return rolled_back is None or rolled_back < promoted


def _previous_champion(client: MlflowClient, model_name: str, current: int) -> ModelVersion:
    """Newest version older than `current` that is eligible to hold the alias."""
    versions = client.search_model_versions(f"name = '{model_name}'")
    eligible = [v for v in versions if int(v.version) < current and _eligible(v)]
    if not eligible:
        raise ValueError(f"no eligible previous champion for {model_name!r}")
    return max(eligible, key=lambda v: int(v.version))


def rollback(config: dict, reason: str) -> None:
    """Point the champion alias back at the previous champion version."""
    client = MlflowClient(registry_uri=config["registry_uri"])
    name = config["model_name"]

    demoted = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    target = _previous_champion(client, name, int(demoted.version))

    client.set_registered_model_alias(name, CHAMPION_ALIAS, target.version)

    stamp = datetime.now(UTC).isoformat()
    client.set_model_version_tag(name, demoted.version, "rolled_back_at", stamp)
    client.set_model_version_tag(name, demoted.version, "rollback_reason", reason)
