"""File watcher: the automatic half of the trigger layer.

Polls a directory for new molecule files. Each new file is assessed by
the retrain advisor; training runs only on a positive recommendation.
The advisor recommends, this loop triggers, the gate decides promotion.

The run ledger doubles as the idempotency manifest: a file whose
raw_data_hash appears on a FINISHED run was trained on and is skipped.
FAILED runs do not count, so a crashed chunk is eligible again on the
next watcher session. One attempt per file per session: a file that
errors is parked and announced, and the loop survives to the next scan.
Advise-skips are likewise remembered only for this process's lifetime.
"""

from __future__ import annotations

import time
from pathlib import Path

from mlflow import MlflowClient

from molpipe.agents import retrain_recommendation
from molpipe.ingestion import raw_data_hash
from molpipe.pipelines import run_training


def _already_trained(config: dict, file_hash: str) -> bool:
    """FINISHED runs only: a FAILED run is an attempt, not a result."""
    uri = config["registry_uri"]
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment_ids = [e.experiment_id for e in client.search_experiments()]
    runs = client.search_runs(
        experiment_ids,
        filter_string=(f"params.raw_data_hash = '{file_hash}' and attributes.status = 'FINISHED'"),
        max_results=1,
    )
    return bool(runs)


def _consider(config: dict, path: Path, digest: str) -> None:
    if _already_trained(config, digest):
        print(f"{path.name}: already trained on this exact file - skip", flush=True)
        return

    print(f"{path.name}: new arrival, consulting the retrain advisor", flush=True)
    result = retrain_recommendation(config, str(path))
    rec = result["recommendation"]
    print(f"{path.name}: retrain={rec['retrain']} - {rec['reason']}", flush=True)

    if rec["retrain"]:
        run_training(str(path), config)


def scan_once(config: dict, directory: Path, seen: set[str]) -> None:
    for path in sorted(directory.glob("*.csv")):
        digest = raw_data_hash(str(path))
        if digest in seen:
            continue
        seen.add(digest)

        try:
            _consider(config, path, digest)
        except Exception as exc:
            # One bad file (API blip, crash mid-DAG) must not kill the loop.
            # A training crash already wrote its FAILED run at the driver
            # boundary; here the file is parked and the watcher moves on.
            print(f"{path.name}: error ({exc}) - parked for this session", flush=True)


def watch_directory(config: dict, directory: str, interval: int = 10) -> None:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    print(f"watching {target} every {interval}s - ctrl-c to stop", flush=True)
    while True:
        scan_once(config, target, seen)
        time.sleep(interval)
