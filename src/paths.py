"""Repo-relative locations, independent of the working directory.

`DATA` is `$AT_DATA` or the `data/` folder next to the repo (the wrapper directory); `REPOS` is `$AT_REPOS` or the
`repos/` folder there (read-only reference repositories).
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AT_DATA", REPO.parent / "data"))
REPOS = Path(os.environ.get("AT_REPOS", REPO.parent / "repos"))
OUTPUTS = REPO / "outputs"
CHECKPOINTS = REPO / "checkpoints"
