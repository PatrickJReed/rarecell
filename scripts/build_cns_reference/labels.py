"""Resolve native BICCN obs columns by trying known candidate names."""

from __future__ import annotations

import pandas as pd
from rarecell.errors import ReferenceBuildError

# Candidate column names in CELLxGENE Discover H5ADs for the Siletti WHB atlas.
# Ordered most-specific first; the first present wins.
SUPERCLUSTER_CANDIDATES = ["supercluster_term", "Supercluster", "supercluster"]
# Prefer the ABC-annotated "cluster_name" (e.g. "MGE_259") when present, so
# cluster models train on lineage-grounded labels rather than bare numbers.
CLUSTER_CANDIDATES = ["cluster_name", "cluster_id", "Cluster", "cluster"]
SUBCLUSTER_CANDIDATES = ["subcluster_id", "Subcluster", "subcluster"]
DONOR_CANDIDATES = ["donor_id", "donor", "DonorID"]


def resolve_label_column(obs: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in obs.columns:
            return c
    raise ReferenceBuildError(
        f"None of the candidate columns {candidates} are present in obs "
        f"(have: {list(obs.columns)[:20]})"
    )
