import json
from pathlib import Path
from typing import Annotated

import typer

from molpipe.config import ConfigError, load_config, validate_config
from molpipe.pipelines import score_pipeline, train_pipeline

app = typer.Typer(
    help="Model-lifecycle pipelines over QM9: train, gate, promote, score, roll back.",
    no_args_is_help=True,
)

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", help="JSON config file; keys merge over the QM9 defaults"),
]


def _load_config(config_path: Path | None) -> dict:
    """QM9 defaults, with the JSON file's keys merged over them and validated."""
    try:
        return load_config(config_path)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc), param_hint="--config") from exc


@app.command()
def train(
    path: Annotated[
        Path | None, typer.Argument(help="raw CSV chunk to process [default: config data_path]")
    ] = None,
    model: Annotated[
        str | None, typer.Option(help="estimator kind override (dummy | ridge | hist_gbr)")
    ] = None,
    config: ConfigOpt = None,
) -> None:
    """Run the training pipeline on a chunk. The gate decides promotion."""
    cfg = _load_config(config)
    if model is not None:
        cfg["model_spec"] = {"kind": model}
        try:
            validate_config(cfg)
        except ConfigError as exc:
            raise typer.BadParameter(str(exc), param_hint="--model") from exc
    target = path if path is not None else Path(cfg["data_path"])
    train_pipeline(path=target, config=cfg)


@app.command()
def score(
    path: Annotated[Path, typer.Argument(help="CSV of molecules to score")] = Path(
        "data/raw/incoming.csv"
    ),
    config: ConfigOpt = None,
) -> None:
    """Predict with the champion and write a ranked shortlist.csv."""
    score_pipeline(path=path, config=_load_config(config))


@app.command()
def explain(
    run_id: Annotated[
        str | None, typer.Argument(help="MLflow run id; defaults to the latest run")
    ] = None,
    config: ConfigOpt = None,
) -> None:
    """LLM narrative for a run's gate decision. Logged back onto the run."""
    from molpipe.agents import explain_run

    narrative = explain_run(_load_config(config), run_id)
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
    config: ConfigOpt = None,
) -> None:
    """LLM retrain recommendation from deterministic data facts."""
    from molpipe.agents import retrain_recommendation

    cfg = _load_config(config)
    if reference is not None:
        cfg["data_path"] = str(reference)
    result = retrain_recommendation(cfg, str(path))
    print(json.dumps(result["facts"], indent=2))
    rec = result["recommendation"]
    print(f"retrain={rec['retrain']}  reason: {rec['reason']}")


@app.command()
def watch(
    directory: Annotated[Path, typer.Argument(help="landing directory to poll")] = Path(
        "data/landing"
    ),
    interval: Annotated[int, typer.Option(help="poll interval in seconds")] = 10,
    config: ConfigOpt = None,
) -> None:
    """Poll a landing directory. The advisor screens each new file; training runs on a yes."""
    from molpipe.watch import watch_directory

    watch_directory(_load_config(config), str(directory), interval)


@app.command()
def rollback(
    reason: Annotated[
        str, typer.Option(help="reason recorded on the demoted version")
    ] = "manual rollback",
    config: ConfigOpt = None,
) -> None:
    """Repoint the champion alias to the previous eligible version."""
    from mlflow import MlflowClient

    from molpipe import registry

    cfg = _load_config(config)
    client = MlflowClient(registry_uri=cfg["registry_uri"])
    name = cfg["model_name"]
    before = client.get_model_version_by_alias(name, registry.CHAMPION_ALIAS)
    registry.rollback(cfg, reason)
    after = client.get_model_version_by_alias(name, registry.CHAMPION_ALIAS)
    print(f"champion: version {before.version} -> version {after.version}")
