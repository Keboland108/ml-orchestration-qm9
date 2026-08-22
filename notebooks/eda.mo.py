import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    from rdkit import Chem

    return Chem, mo, pd, plt


@app.cell
def _(mo):
    DATA_PATH = mo.notebook_dir().parent / "data"
    RAW_DATA_PATH = DATA_PATH / "raw"

    return (RAW_DATA_PATH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What does the data look like? Does it look like what the paper is describing?
    """)
    return


@app.cell
def _(RAW_DATA_PATH, pd):
    df = pd.read_csv(RAW_DATA_PATH / "qm9.csv")
    return (df,)


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### We have smiles strings, are they all unique?
    """)
    return


@app.cell
def _(df):
    print(len(df["smiles"].unique()))
    return


@app.cell
def _(df):
    dup_smiles = df[df["smiles"].duplicated(keep=False)].sort_values("smiles")
    return (dup_smiles,)


@app.cell
def _(dup_smiles):
    dup_smiles
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### within an identical string, are the features the same?
    """)
    return


@app.cell
def _(dup_smiles):
    num_cols = dup_smiles.select_dtypes("number").columns
    dup_spread = dup_smiles.groupby("smiles")[list(num_cols)].agg(lambda s: s.max() - s.min())
    dup_spread.max().sort_values(ascending=False).to_frame("max_within_pair_spread")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Can we parse all the strings using RDKit?
    """)
    return


@app.cell
def _(Chem, df):
    m = Chem.MolFromSmiles(df["smiles"].iloc[0])
    return


@app.cell
def _(Chem, df):
    parseable = df["smiles"].map(lambda s: Chem.MolFromSmiles(s) is not None)
    f"unparseable SMILES: {(~parseable).sum()} of {len(df)}"
    return


@app.cell
def _(df, pd):
    pd.DataFrame({"dtype": df.dtypes.astype(str), "nulls": df.isna().sum()})
    return


@app.cell
def _(df):
    df["u0_atom"].describe().to_frame("u0_atom")
    return


@app.cell
def _(df, plt):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["u0_atom"], bins=80, color="#31688E")
    ax.set_xlabel("u0_atom — atomization energy at 0 K (kcal/mol)")
    ax.set_ylabel("molecules")
    ax.set_title("Target distribution")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Does temperature change the atomization energy? Expectation: larger magnitude at 0 K than at 298 K.
    """)
    return


@app.cell
def _(df):
    atom_cols = ["u0_atom", "u298_atom", "h298_atom", "g298_atom"]
    df[atom_cols].abs().mean().sort_values(ascending=False).to_frame("mean |atomization energy| (kcal/mol)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decisions from this EDA

    - **Features: the SMILES string only** (Morgan fingerprints). All 19 numeric columns are
      outputs of the DFT simulation — using them as features would defeat the point
      (predict what the simulation would say, *before* paying for it).
    - **Target: `u0_atom`** — atomization energy at 0 K, kcal/mol. Canonical in the literature
      (Rupp et al. 2012); already in the units of the chemical-accuracy anchor (1 kcal/mol).
      **Metric: MAE.** The anchor gives meaning; the gate judges candidate-vs-incumbent ΔMAE.
    - **Cleaning spec for the validate node:** RDKit parse check (0 failures in this file, still
      a guardrail — the paper itself reports 3,054 SMILES-consistency failures in the original
      release); dedup by SMILES keep-first (83 duplicate pairs, max `u0_atom` disagreement
      0.014 kcal/mol — immaterial, but removes train/test leakage); null + dtype + range checks
      (0 nulls found).
    - **Split: random, after dedup.** Scaffold-based splitting noted as a known limitation.
    """)
    return


if __name__ == "__main__":
    app.run()
