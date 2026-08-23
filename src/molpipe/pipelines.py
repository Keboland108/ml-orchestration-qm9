"""Pipeline assembly: a driver = engine modules + config + requested outputs.

The engine modules know nothing about QM9. Everything QM9-specific lives
in the config dicts HERE, and only here. Split of responsibility:
with_config() carries what the pipeline IS (column names, model spec —
stable across runs); execute(inputs=) carries what this run PROCESSES
(the raw_path of an arriving chunk).
"""

from __future__ import annotations

from hamilton import driver

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


def run_training(raw_path: str, config: dict | None = None) -> dict:
    """On-demand entry point — the CLI half of requirement 6."""
    dr = build_training_driver(config)
    return dr.execute(
        ["gate_decision", "candidate_metrics", "champion_metrics"],
        inputs={"raw_path": raw_path},
    )


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/qm9.csv"
    result = run_training(path)
    decision = result["gate_decision"]
    print(f"candidate mae: {result['candidate_metrics']['mae']:.3f}")
    print(f"promote={decision.promote}  reason: {decision.reason}")
    for check in decision.checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.reason}")
