"""The agentic edge: every LLM call in this repo lives here.

NOT a Hamilton module and not part of any decision path. Agents read
state and write annotations or recommendations. They never move an
alias, never train, never gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient

from molpipe.registry import CHAMPION_ALIAS

SYSTEM = (
    "You write short promotion narratives for an ML model audit record. "
    "A deterministic gate already made the decision from the metrics. "
    "Never question or soften the decision. Explain in plain language: "
    "what the candidate was, what it was compared against, which checks "
    "ran, and why the decision followed. Under 200 words, markdown."
)


def explain_run(config: dict, run_id: str | None = None) -> str:
    """Fetch a run's audit record, ask Claude for a narrative, log it back."""
    load_dotenv()
    uri = config["registry_uri"]
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)

    if run_id is None:
        experiment_ids = [e.experiment_id for e in client.search_experiments()]
        runs = client.search_runs(
            experiment_ids, order_by=["attributes.start_time DESC"], max_results=1
        )
        if not runs:
            raise ValueError("no runs in the tracking store - run a pipeline first")
        run = runs[0]
        run_id = run.info.run_id
    else:
        run = client.get_run(run_id)

    decision_path = client.download_artifacts(run_id, "gate_decision.json")
    decision = json.loads(Path(decision_path).read_text())

    payload = {
        "gate_decision": decision,
        "params": run.data.params,
        "metrics": run.data.metrics,
    }

    api = anthropic.Anthropic()
    response = api.messages.create(
        model="claude-opus-5",
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("model declined to write the narrative")
    narrative = "".join(b.text for b in response.content if b.type == "text")

    client.log_text(run_id, narrative, "rationale.md")
    return narrative


RETRAIN_SYSTEM = (
    "You advise an ML team on retraining cadence. You receive "
    "deterministic facts about an incoming data chunk and the current "
    "champion model. Recommend whether retraining is warranted. You "
    "cannot trigger anything; a human or scheduler reads this. Be "
    "direct when the evidence is weak. Keep the reason under 80 words "
    "of plain prose - no JSON, no quotes, no escape sequences."
)

RETRAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "retrain": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["retrain", "reason"],
    "additionalProperties": False,
}


def _chunk_facts(config: dict, incoming_path: str) -> dict:
    """Deterministic evidence. The agent judges these numbers, never invents them."""
    smiles = config["smiles_column"]
    reference = pd.read_csv(config["data_path"])[smiles]
    incoming = pd.read_csv(incoming_path)[smiles]
    known = set(reference)
    novel = sum(1 for s in incoming if s not in known)

    uri = config["registry_uri"]
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    champ = client.get_model_version_by_alias(config["model_name"], CHAMPION_ALIAS)
    run = client.get_run(champ.run_id)

    return {
        "incoming_rows": int(len(incoming)),
        "novel_smiles": int(novel),
        "novel_fraction": round(novel / max(len(incoming), 1), 4),
        "reference_rows": int(len(reference)),
        "champion_version": str(champ.version),
        "champion_promoted_at": champ.tags.get("promoted_at"),
        "champion_model_kind": run.data.params.get("model_kind"),
        "champion_mae": run.data.metrics.get("candidate_mae"),
    }


def retrain_recommendation(config: dict, incoming_path: str) -> dict:
    """Facts in, structured recommendation out. Nothing is triggered here."""
    load_dotenv()
    facts = _chunk_facts(config, incoming_path)

    api = anthropic.Anthropic()
    response = api.messages.create(
        model="claude-opus-5",
        max_tokens=2000,
        system=RETRAIN_SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": RETRAIN_SCHEMA}},
        messages=[{"role": "user", "content": json.dumps(facts, indent=2)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("model declined to write the recommendation")
    text = next(b.text for b in response.content if b.type == "text")

    return {"facts": facts, "recommendation": json.loads(text)}
