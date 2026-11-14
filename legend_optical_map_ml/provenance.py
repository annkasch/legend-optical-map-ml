"""Which code wrote an output of this repository: the ``code`` block of its JSON files.

The block is RESUM_FLEX's ``code_provenance`` for this repository's checkout (commit, branch,
dirty flag, Python, installed ``PACKAGES``, command line) plus ``resum_flex``: the git state of
the toolkit checkout. Written to ``code.json`` of each prepared view, each photon store shard
summary (and so the manifest's ``shards``) and ``benchmark.json`` (``docs/outputs.md``).
"""

from pathlib import Path

from core.provenance import SOURCE as RESUM_FLEX_SOURCE
from core.provenance import code_provenance

SOURCE = Path(__file__).resolve().parent
PACKAGES = (
    "legend-optical-map-ml",
    "resum-flex",
    "numpy",
    "scipy",
    "h5py",
    "pydantic",
    "pyyaml",
    "matplotlib",
)
GIT_FIELDS = ("source", "commit", "branch", "dirty", "git_error")


def code_block() -> dict:
    """Git state of this repository and RESUM_FLEX, Python and package versions, command."""
    record = code_provenance(SOURCE, PACKAGES)
    toolkit = code_provenance(RESUM_FLEX_SOURCE)
    record["resum_flex"] = {k: toolkit[k] for k in GIT_FIELDS if k in toolkit}
    return record
