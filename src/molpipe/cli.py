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
) -> None:
    config = QM9_CONFIG.copy()
    train_pipeline(path=path, config=config)


@app.command()
def score(
    path: Annotated[Path, typer.Argument(help="CSV of molecules to score")] = Path(
        "data/raw/incoming.csv"
    ),
) -> None:
    config = QM9_CONFIG.copy()
    score_pipeline(path=path, config=config)
