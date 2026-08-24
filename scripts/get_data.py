"""Download QM9 assets into gitignored dirs: the dataset CSV and the source paper.

Idempotent: skips a download if the file already exists. Prints the SHA-256
of the file either way — raw-data provenance starts here.

Source: Ramakrishnan et al. (2014), Scientific Data 1:140022 — see data/README.md.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QM9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
PAPER_URL = "https://www.nature.com/articles/sdata201422.pdf"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"already present: {dest}")
    else:
        print(f"downloading {url} -> {dest}")
        # UA header: nature.com rejects urllib's default
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(req) as resp, tmp.open("wb") as f:  # noqa: S310
            while block := resp.read(1 << 20):
                f.write(block)
        tmp.rename(dest)  # atomic: never leave a half-written file
    print(f"size:   {dest.stat().st_size:,} bytes")
    print(f"sha256: {sha256_of(dest)}")


def get_data() -> None:
    _download(QM9_URL, REPO_ROOT / "data" / "raw" / "qm9.csv")


def get_paper() -> None:
    _download(PAPER_URL, REPO_ROOT / "docs" / "ramakrishnan_2014_qm9.pdf")


def get_chunk() -> None:
    """Sample rows from qm9.csv into incoming.csv - a newly-arrived chunk to score."""
    import pandas as pd

    src = REPO_ROOT / "data" / "raw" / "qm9.csv"
    dest = REPO_ROOT / "data" / "raw" / "incoming.csv"
    frame = pd.read_csv(src).sample(n=5000, random_state=7)
    frame.to_csv(dest, index=False)
    print(f"wrote {len(frame):,} rows -> {dest}")
    print(f"sha256: {sha256_of(dest)}")


def main() -> int:
    targets = {"data": get_data, "paper": get_paper, "chunk": get_chunk}
    name = sys.argv[1] if len(sys.argv) > 1 else "data"
    if name not in targets:
        print(f"usage: get_data.py [{'|'.join(targets)}]", file=sys.stderr)
        return 2
    targets[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
