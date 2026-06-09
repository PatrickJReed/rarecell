"""SQLite-backed marker catalog. Aggregates CellMarker + PanglaoDB."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rarecell_mcp_knowledge.citation import Citation, RetrievalHit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS markers (
    source TEXT NOT NULL,
    cell_type TEXT NOT NULL,
    tissue TEXT,
    gene TEXT NOT NULL,
    citation_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_cell_tissue ON markers (cell_type, tissue);
"""


class MarkersCatalog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(_SCHEMA)

    def insert(
        self,
        source: str,
        cell_type: str,
        tissue: str | None,
        gene: str,
        citation_id: str | None,
    ) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO markers "
                "(source, cell_type, tissue, gene, citation_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (source, cell_type.lower(), (tissue or "").lower(), gene, citation_id),
            )

    def _ilike(self, s: str) -> str:
        return f"%{s.lower()}%"

    def search_markers(
        self,
        cell_type: str,
        tissue: str | None = None,
    ) -> list[RetrievalHit]:
        with sqlite3.connect(self.path) as conn:
            if tissue:
                rows = conn.execute(
                    "SELECT source, cell_type, tissue, gene, citation_id "
                    "FROM markers "
                    "WHERE cell_type LIKE ? AND tissue LIKE ?",
                    (self._ilike(cell_type), self._ilike(tissue)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT source, cell_type, tissue, gene, citation_id "
                    "FROM markers WHERE cell_type LIKE ?",
                    (self._ilike(cell_type),),
                ).fetchall()
        return self._rows_to_hits(rows, cell_type)

    def _rows_to_hits(self, rows: list, query: str) -> list[RetrievalHit]:
        grouped: dict[tuple, dict] = {}
        for source, ct, tis, gene, cit_id in rows:
            key = (source, ct, tis)
            g = grouped.setdefault(key, {"genes": set(), "citation_ids": set()})
            g["genes"].add(gene)
            if cit_id:
                g["citation_ids"].add(cit_id)

        hits: list[RetrievalHit] = []
        for (source, ct, tis), agg in grouped.items():
            genes = sorted(agg["genes"])
            cit_ids = sorted(agg["citation_ids"])
            citation = Citation(
                source_id=f"{source}:{ct}:{tis}",
                source=source,
                title=f"{ct} markers ({source})",
            )
            hits.append(
                RetrievalHit(
                    citation=citation,
                    title=f"{ct} markers in {tis or 'any tissue'}",
                    snippet=", ".join(genes[:10]) + ("..." if len(genes) > 10 else ""),
                    payload={
                        "genes": genes,
                        "citation_ids": cit_ids,
                        "cell_type": ct,
                        "tissue": tis,
                    },
                    source=source,
                )
            )
        return hits

    def get_canonical_panel(self, cell_type: str) -> RetrievalHit:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT gene FROM markers WHERE cell_type LIKE ?",
                (self._ilike(cell_type),),
            ).fetchall()
        genes = sorted(g for (g,) in rows)
        citation = Citation(
            source_id=f"canonical:{cell_type.lower()}",
            source="manual",
            title=f"Canonical {cell_type} panel (rarecell aggregate)",
        )
        return RetrievalHit(
            citation=citation,
            title=f"Canonical {cell_type} panel",
            snippet=", ".join(genes[:10]) + ("..." if len(genes) > 10 else ""),
            payload={"genes": genes, "cell_type": cell_type},
            source="manual",
        )
