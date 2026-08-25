"""Pipeline assembly: a driver = engine modules + config + requested outputs.

The engine modules know nothing about QM9. Everything QM9-specific lives
in the config dicts HERE, and only here. Split of responsibility:
with_config() carries what the pipeline IS (column names, model spec —
stable across runs); execute(inputs=) carries what this run PROCESSES
(the raw_path of an arriving chunk).
"""

from __future__ import annotations

import logging
from pathlib import Path

from hamilton import driver

from molpipe import (
    champion,
    evaluation,
    features,
    gate,
    ingestion,
    scoring,
    training,
    validation,
)
from molpipe.config import QM9_CONFIG, resolve_config, validate_config
from molpipe.registry import apply_gate_decision, record_failure

logger = logging.getLogger(__name__)

__all__ = [
    "QM9_CONFIG",
    "build_scoring_driver",
    "build_training_driver",
    "run_scoring",
    "run_training",
    "score_pipeline",
    "train_pipeline",
]


def build_training_driver(config: dict | None = None) -> driver.Driver:
    """Pipeline #1: ingest -> train -> benchmark -> gate."""
    return (
        driver.Builder()
        .with_modules(ingestion, validation, features, training, evaluation, champion, gate)
        .with_config(validate_config(dict(config or QM9_CONFIG)))
        .build()
    )


def build_scoring_driver(config: dict | None = None) -> driver.Driver:
    """Pipeline #2: ingest -> validate -> featurize -> score -> shortlist."""
    return (
        driver.Builder()
        .with_modules(ingestion, validation, features, champion, scoring)
        .with_config(validate_config(dict(config or QM9_CONFIG)))
        .build()
    )


def run_training(raw_path: str, config: dict | None = None) -> dict:
    """On-demand entry point — the CLI half of requirement 6.

    A crash anywhere in the DAG still leaves an audit record: the boundary
    catches it, writes a FAILED run via record_failure, and re-raises. The
    process fails loudly; the ledger keeps no gaps.
    """
    cfg = resolve_config(config)
    dr = build_training_driver(cfg)
    try:
        result = dr.execute(
            [
                "gate_decision",
                "candidate_metrics",
                "champion_metrics",
                "candidate_model",
                "raw_data_hash",
                "dataset_content_hash",
                "test_features",
            ],
            inputs={"raw_path": raw_path},
        )
    except Exception as exc:
        try:
            record_failure(cfg, raw_path, exc)
        except Exception:
            # The audit write must never mask the original failure.
            logger.exception("failure record could not be written")
        raise
    apply_gate_decision(result, cfg)
    return result


def run_scoring(raw_path: str, config: dict | None = None) -> dict:
    """On-demand scoring entry point."""
    cfg = resolve_config(config)
    dr = build_scoring_driver(cfg)
    return dr.execute(["shortlist", "raw_data_hash"], inputs={"raw_path": raw_path})


def score_pipeline(path: Path, config: dict | None = None) -> None:
    cfg = resolve_config(config)
    result = run_scoring(str(path), cfg)
    ranked = result["shortlist"]
    out = Path(path).with_name("shortlist.csv")
    ranked.to_csv(out, index=False)
    predicted = f"predicted_{cfg['target_column']}"
    print(f"shortlist ({len(ranked)} rows) -> {out}")
    print(ranked[[cfg["smiles_column"], predicted]].head(10).to_string(index=False))


def train_pipeline(path: Path, config: dict | None = None) -> None:
    result = run_training(str(path), config)
    decision = result["gate_decision"]
    print(f"candidate mae: {result['candidate_metrics']['mae']:.3f}")
    print(f"promote={decision.promote}  reason: {decision.reason}")
    for check in decision.checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.reason}")
