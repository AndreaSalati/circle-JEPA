"""Machine-agnostic data root and conda environment resolution.

Add candidate roots / env names here in priority order; first existing one wins.
"""

import subprocess
from pathlib import Path

_CANDIDATE_DATA_ROOTS = [
    Path("/home/maxine/Documents/andrea/scCircadianMeta/data"),
    Path("/Users/salati/Documents/CODE/github/scCircadianMeta/data"),
]

_CANDIDATE_REPO_ROOTS = [
    Path("/home/maxine/Documents/andrea/circle-JEPA"),
    Path("/Users/salati/Documents/CODE/github/circle-JEPA"),
]


_CANDIDATE_ENVS = ["torch", "ML-gpu"]


def get_conda_env() -> str:
    """Return the name of the first conda environment that exists on this machine."""
    try:
        out = subprocess.check_output(
            ["conda", "env", "list"], text=True, stderr=subprocess.DEVNULL
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError("conda not found; cannot detect environment")
    existing = {
        line.split()[0]
        for line in out.splitlines()
        if line and not line.startswith("#")
    }
    for env in _CANDIDATE_ENVS:
        if env in existing:
            return env
    raise RuntimeError(
        f"None of the expected conda envs found: {_CANDIDATE_ENVS}\n"
        f"Available: {sorted(existing)}"
    )


def get_data_root() -> Path:
    for candidate in _CANDIDATE_DATA_ROOTS:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the scCircadianMeta data directory. "
        f"Tried:\n" + "\n".join(f"  {p}" for p in _CANDIDATE_DATA_ROOTS)
    )


def get_repo_root() -> Path:
    for candidate in _CANDIDATE_REPO_ROOTS:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the circle-JEPA repo directory. "
        f"Tried:\n" + "\n".join(f"  {p}" for p in _CANDIDATE_REPO_ROOTS)
    )
