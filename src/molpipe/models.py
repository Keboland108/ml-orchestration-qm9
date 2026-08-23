"""Registry access. Impure by design — this is the edge, not the engine."""

import mlflow
from mlflow.exceptions import MlflowException


def get_champion_model(model_name: str, registry_uri: str):
    """Resolve the current champion. None = cold start (no champion yet).

    Only the not-found case maps to None. Infrastructure failures raise.
    """
    mlflow.set_tracking_uri(registry_uri)
    try:
        return mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
    except MlflowException as exc:
        if exc.get_http_status_code() == 404 or "not found" in str(exc).lower():
            return None
        raise


# NOTE(Kyle): there is no get_candidate_model — the candidate never lives in
# the registry before the gate. Training produces it in memory and hands it
# to evaluation directly. Registration happens only after a promote decision.
