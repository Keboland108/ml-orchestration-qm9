"""A crash mid-DAG must still leave an audit record.

The failing input is real, not mocked: a chunk whose rows all fail
validation reaches featurization as an empty frame, and zero
fingerprints cannot be stacked. The boundary in run_training catches
the crash, writes one FAILED run, and re-raises.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from molpipe.pipelines import QM9_CONFIG, run_training


def test_crash_writes_failed_run(tmp_path):
    chunk = tmp_path / "bad_chunk.csv"
    pd.DataFrame({"smiles": ["C((C", "also(not(smiles"], "u0_atom": [1.0, 2.0]}).to_csv(
        chunk, index=False
    )

    uri = f"sqlite:///{tmp_path}/registry.db"
    cfg = {**QM9_CONFIG, "registry_uri": uri}

    with pytest.raises(ValueError):
        run_training(str(chunk), cfg)

    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment_ids = [e.experiment_id for e in client.search_experiments()]
    runs = client.search_runs(experiment_ids)
    assert len(runs) == 1, "exactly one audit run for the crashed attempt"

    run = runs[0]
    assert run.info.status == "FAILED"
    assert "raw_data_hash" in run.data.params, "chunk identity recorded"

    failure_path = client.download_artifacts(run.info.run_id, "failure.json")
    failure = json.loads(Path(failure_path).read_text())
    assert failure["error_type"] == "ValueError"
    assert failure["raw_path"] == str(chunk)
