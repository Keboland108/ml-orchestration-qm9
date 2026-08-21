"""Invoke tasks — the single source of truth for lint/test commands.

Run manually as `uv run inv <task>` (e.g. `uv run inv lint`); CI runs the same
tasks, so local and CI never drift.
"""

from invoke import task


@task
def lint(c):
    """Lint and format-check (no changes)."""
    c.run("ruff check .")
    c.run("ruff format --check .")


@task
def fmt(c):
    """Auto-format and apply safe lint fixes."""
    c.run("ruff format .")
    c.run("ruff check --fix .")


@task
def test(c):
    """Run the test suite."""
    c.run("pytest")


@task
def data(c):
    """Download the QM9 dataset into data/raw/ (gitignored)."""
    c.run("python scripts/get_data.py data")


@task
def paper(c):
    """Download the QM9 paper (Ramakrishnan et al. 2014) into docs/ (gitignored)."""
    c.run("python scripts/get_data.py paper")
