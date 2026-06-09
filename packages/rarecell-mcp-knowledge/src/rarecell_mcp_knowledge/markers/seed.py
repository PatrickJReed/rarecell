"""Seed the markers catalog from CellMarker + PanglaoDB TSVs."""

from __future__ import annotations

import csv
from pathlib import Path

from rarecell_mcp_knowledge.markers.catalog import MarkersCatalog


def seed_catalog_from_tsv(
    catalog: MarkersCatalog,
    *,
    cellmarker_tsv: Path | None = None,
    panglaodb_tsv: Path | None = None,
) -> dict[str, int]:
    """Seed the catalog from TSV files. Returns counts per source."""
    counts = {"cellmarker": 0, "panglaodb": 0}

    if cellmarker_tsv:
        with open(cellmarker_tsv, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("species") != "Human":
                    continue
                catalog.insert(
                    source="cellmarker",
                    cell_type=row["cell_name"],
                    tissue=row.get("tissue_class"),
                    gene=row["marker"],
                    citation_id=f"pmid:{row['pmid']}" if row.get("pmid") else None,
                )
                counts["cellmarker"] += 1

    if panglaodb_tsv:
        with open(panglaodb_tsv, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("species") != "Hs":
                    continue
                catalog.insert(
                    source="panglaodb",
                    cell_type=row["cell_type"],
                    tissue=None,
                    gene=row["official_symbol"],
                    citation_id=None,
                )
                counts["panglaodb"] += 1

    return counts
