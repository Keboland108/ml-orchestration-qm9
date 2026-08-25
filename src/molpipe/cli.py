from pathlib import Path
from typing import Annotated

import typer

from molpipe.pipelines import QM9_CONFIG, score_pipeline, train_pipeline

app = typer.Typer()


@app.command()
def train(
    path: Annotated[Path, typer.Argument(help="raw CSV chunk to process")] = Path(
        QM9_CONFIG["data_path"]
    ),
    model: Annotated[
        str | None, typer.Option(help="estimator kind override (dummy | ridge)")
    ] = None,
) -> None:
    config = QM9_CONFIG.copy()
    if model is not None:
        config["model_spec"] = {"kind": model}
    train_pipeline(path=path, config=config)


@app.command()
def score(
    path: Annotated[Path, typer.Argument(help="CSV of molecules to score")] = Path(
        "data/raw/incoming.csv"
    ),
) -> None:
    config = QM9_CONFIG.copy()
    score_pipeline(path=path, config=config)


@app.command()
def explain(
    run_id: Annotated[
        str | None, typer.Argument(help="MLflow run id; defaults to the latest run")
    ] = None,
) -> None:
    from molpipe.agents import explain_run

    narrative = explain_run(QM9_CONFIG.copy(), run_id)
    print(narrative)


@app.command()
def advise(
    path: Annotated[Path, typer.Argument(help="incoming CSV to assess")] = Path(
        "data/raw/incoming.csv"
    ),
    reference: Annotated[
        Path | None,
        typer.Option(help="reference dataset override (default: config data_path)"),
    ] = None,
) -> None:
    import json

    from molpipe.agents import retrain_recommendation

    config = QM9_CONFIG.copy()
    if reference is not None:
        config["data_path"] = str(reference)
    result = retrain_recommendation(config, str(path))
    print(json.dumps(result["facts"], indent=2))
    rec = result["recommendation"]
    print(f"retrain={rec['retrain']}  reason: {rec['reason']}")


@app.command()
def watch(
    directory: Annotated[Path, typer.Argument(help="landing directory to poll")] = Path(
        "data/raw/landing"
    ),
    interval: Annotated[int, typer.Option(help="poll interval in seconds")] = 10,
) -> None:
    from molpipe.watch import watch_directory

    watch_directory(QM9_CONFIG.copy(), str(directory), interval)


@app.command()
def rollback(
    reason: Annotated[
        str, typer.Option(help="reason recorded on the demoted version")
    ] = "manual rollback",
) -> None:
    from mlflow import MlflowClient

    from molpipe import registry

    config = QM9_CONFIG.copy()
    client = MlflowClient(registry_uri=config["registry_uri"])
    name = config["model_name"]
    before = client.get_model_version_by_alias(name, registry.CHAMPION_ALIAS)
    registry.rollback(config, reason)
    after = client.get_model_version_by_alias(name, registry.CHAMPION_ALIAS)
    print(f"champion: version {before.version} -> version {after.version}")
