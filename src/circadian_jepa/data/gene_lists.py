from pathlib import Path

import scritmo as sr

_RESOURCES_DIR = Path(__file__).parent / "resources"


def get_default_beta_path() -> Path:
    return _RESOURCES_DIR / "ccg_zhang_context.csv"


def get_default_beta() -> sr.Beta:
    return sr.Beta(str(get_default_beta_path()))


def get_circadian_genes(species: str = "mouse") -> list[str]:
    """Return gene names from the bundled ccg_zhang_context.csv.

    species='mouse' returns capitalized names (Bmal1, Per2, ...).
    species='human' returns all-uppercase (BMAL1, PER2, ...).
    """
    genes = get_default_beta().index.tolist()
    if species == "human":
        return [g.upper() for g in genes]
    return genes
