import pytest
import respx
from httpx import Response
from rarecell_mcp_knowledge.errors import BackendUnreachableError, InvalidQueryError
from rarecell_mcp_knowledge.msigdb import fetch_msigdb_gene_set

MSIGDB_CARD = {
    "standardName": "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "description": "Genes up-regulated in response to IFNG.",
    "members": ["IFIT1", "IFIT2", "IFIT3", "STAT1", "OAS1"],
    "pmid": "26771021",
}


@respx.mock
def test_fetch_known_gene_set():
    respx.get(
        "https://www.gsea-msigdb.org/gsea/msigdb/cards/HALLMARK_INTERFERON_GAMMA_RESPONSE.json"
    ).mock(return_value=Response(200, json=MSIGDB_CARD))

    hit = fetch_msigdb_gene_set("HALLMARK_INTERFERON_GAMMA_RESPONSE")
    assert hit.citation.source == "msigdb"
    assert hit.payload["genes"] == ["IFIT1", "IFIT2", "IFIT3", "STAT1", "OAS1"]
    assert "IFNG" in hit.snippet


@respx.mock
def test_fetch_unknown_gene_set_raises():
    respx.get("https://www.gsea-msigdb.org/gsea/msigdb/cards/UNKNOWN.json").mock(
        return_value=Response(404)
    )

    with pytest.raises(InvalidQueryError):
        fetch_msigdb_gene_set("UNKNOWN")


@respx.mock
def test_fetch_msigdb_unreachable():
    respx.get("https://www.gsea-msigdb.org/gsea/msigdb/cards/X.json").mock(
        return_value=Response(503)
    )

    with pytest.raises(BackendUnreachableError):
        fetch_msigdb_gene_set("X")
