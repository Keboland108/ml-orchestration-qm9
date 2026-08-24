"""File watcher: the automatic half of the trigger layer.

Polls a directory for new molecule files. Each new file is assessed by
the retrain advisor; training runs only on a positive recommendation.
The advisor recommends, this loop triggers, the gate decides promotion.

The run ledger doubles as the idempotency manifest: a file whose
raw_data_hash already appears on a run was trained on and is skipped.
Advise-skips are remembered only for this process's lifetime.
"""

from __future__ import annotations

import time
from pathlib import Path

from mlflow import MlflowClient

from molpipe.agents import retrain_recommendation
from molpipe.ingestion import raw_data_hash
from molpipe.pipelines import run_training


def _already_trained(config: dict, file_hash: str) -> bool:
    uri = config["registry_uri"]
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment_ids = [e.experiment_id for e in client.search_experiments()]
    runs = client.search_runs(
        experiment_ids,
        filter_string=f"params.raw_data_hash = '{file_hash}'",
        max_results=1,
    )
    return bool(runs)


def scan_once(config: dict, directory: Path, seen: set[str]) -> None:
    for path in sorted(directory.glob("*.csv")):
        digest = raw_data_hash(str(path))
        if digest in seen:
            continue
        seen.add(digest)

        if _already_trained(config, digest):
            print(f"{path.name}: already trained on this exact file - skip", flush=True)
            continue

        print(f"{path.name}: new arrival, consulting the retrain advisor", flush=True)
        result = retrain_recommendation(config, str(path))
        rec = result["recommendation"]
        print(f"{path.name}: retrain={rec['retrain']} - {rec['reason']}", flush=True)

        if rec["retrain"]:
            run_training(str(path), config)


def watch_directory(config: dict, directory: str, interval: int = 10) -> None:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    print(f"watching {target} every {interval}s - ctrl-c to stop", flush=True)
    while True:
        scan_once(config, target, seen)
        time.sleep(interval)
