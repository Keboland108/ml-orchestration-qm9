"""Champion access: the read-side DAG source node. Writes live in registry.py.

Vocabulary: the registered model version holding the `champion` alias is THE
champion. Pipelines compare a candidate against it. No other terms.
"""

from typing import Any

import mlflow
from mlflow.exceptions import MlflowException


def champion_model(model_name: str, registry_uri: str) -> Any | None:
    """Hamilton node: resolve models:/<model_name>@champion.

    None = cold start (no champion yet) — an expected state, not a failure.
    Only the not-found case maps to None; infrastructure errors raise.
    Read-only I/O — a source node, same category as raw_frame. The WRITE
    side (register/alias/tags) lives in registry.py, never in the DAG.
    """
    mlflow.set_tracking_uri(registry_uri)
    try:
        return mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
    except MlflowException as exc:
        if exc.get_http_status_code() == 404 or "not found" in str(exc).lower():
            return None
        raise
