"""Test rollback eligibility against a real registry backend.

Versions are created directly via the client. Rollback never loads
artifacts, so fake sources are enough.
"""

from mlflow.tracking import MlflowClient

from molpipe.registry import CHAMPION_ALIAS, rollback

NAME = "test-model"


def test_rollback_skips_version_rolled_back_since_promotion(tmp_path):
    uri = f"sqlite:///{tmp_path}/registry.db"
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_registered_model(NAME)

    # v1: promoted once, never rolled back — eligible.
    client.create_model_version(
        NAME, source="fake", tags={"promoted_at": "2026-08-24T01:00:00+00:00"}
    )
    # v2: promoted, then rolled back AFTER that promotion — not eligible.
    client.create_model_version(
        NAME,
        source="fake",
        tags={
            "promoted_at": "2026-08-24T02:00:00+00:00",
            "rolled_back_at": "2026-08-24T03:00:00+00:00",
        },
    )
    # v3: current champion.
    v3 = client.create_model_version(
        NAME, source="fake", tags={"promoted_at": "2026-08-24T04:00:00+00:00"}
    )
    client.set_registered_model_alias(NAME, CHAMPION_ALIAS, v3.version)

    rollback({"model_name": NAME, "registry_uri": uri}, reason="test")

    champ = client.get_model_version_by_alias(NAME, CHAMPION_ALIAS)
    assert int(champ.version) == 1, "v2 must be skipped: rolled back after promotion"

    demoted = client.get_model_version(NAME, "3")
    assert "rolled_back_at" in demoted.tags
    assert demoted.tags["rollback_reason"] == "test"
