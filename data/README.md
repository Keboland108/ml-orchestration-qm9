# Data

Everything in this directory except this file is gitignored.
The pipelines read from `data/raw/`. Run all commands from the repo root.

Install the environment first: `uv sync`. Then:

```bash
uv run invoke data     # download QM9 -> data/raw/qm9.csv
uv run invoke chunk    # sample the demo chunks (see below)
uv run invoke paper    # download the QM9 paper -> docs/ (gitignored)
```

## Layout after download

| Path | What it is |
|---|---|
| `raw/qm9.csv` | full QM9: 133,885 molecules, SMILES + computed properties |
| `raw/incoming.csv` | 5,000-row sample — the "arriving chunk" for the train and score demos |
| `raw/reference_half.csv`, `raw/incoming_novel.csv` | disjoint halves of a 20,000-row sample — the retrain-advisor demo (novel chemistry against a reference) |
| `chunks/chunk_01..03.csv` | the first 22,500 rows in file order, three sequential chunks of 7,500 — data arriving over time, sized so each held-out split clears the gate's evidence floor |
| `landing/` | drop directory the watcher polls; created on first `molpipe watch` |

## Simulate arrival over time

QM9 file order is roughly molecule-size order, so later chunks drift in chemistry.
Start `uv run molpipe watch` in a second terminal, then:

```bash
uv run molpipe train data/chunks/chunk_01.csv --model ridge   # first arrival: cold start
uv run molpipe train data/chunks/chunk_02.csv --model ridge   # second arrival: challenges the champion
cp data/chunks/chunk_03.csv data/landing/                     # third arrival: the watcher and advisor take it
```

The training target is `u0_atom` (atomization energy at 0 K).
Dataset: Ramakrishnan et al., *Quantum chemistry structures and properties of 134 kilo molecules*, Scientific Data (2014).
