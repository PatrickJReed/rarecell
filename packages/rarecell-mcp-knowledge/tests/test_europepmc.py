import respx
from httpx import Response
from rarecell_mcp_knowledge.literature.europepmc import EuropePMCClient

SEARCH_RESPONSE = {
    "hitCount": 2,
    "resultList": {
        "result": [
            {
                "id": "12345",
                "pmid": "12345",
                "title": "T cell markers in brain",
                "abstractText": "We identify pan-T markers...",
                "doi": "10.1234/abc",
                "pubYear": "2023",
                "authorString": "Smith J, Lee K",
            },
            {
                "id": "67890",
                "pmid": "67890",
                "title": "Microglia in ALS",
                "abstractText": "Microglia release...",
                "doi": "10.5678/def",
                "pubYear": "2024",
                "authorString": "Garcia M",
            },
        ]
    },
}


@respx.mock
def test_search_literature_returns_hits():
    respx.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search").mock(
        return_value=Response(200, json=SEARCH_RESPONSE)
    )

    client = EuropePMCClient()
    hits = client.search("T cell brain", page_size=2)
    assert len(hits) == 2
    assert hits[0].citation.source == "europepmc"
    assert hits[0].citation.source_id == "pmid:12345"
    assert "T cell markers" in hits[0].title


FETCH_RESPONSE = {
    "resultList": {
        "result": [
            {
                "id": "12345",
                "pmid": "12345",
                "title": "T cell markers in brain",
                "abstractText": "Full abstract here.",
                "doi": "10.1234/abc",
                "pubYear": "2023",
                "authorString": "Smith J",
            },
        ]
    },
}


@respx.mock
def test_fetch_abstract_returns_full_text():
    respx.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search").mock(
        return_value=Response(200, json=FETCH_RESPONSE)
    )

    client = EuropePMCClient()
    record = client.fetch_abstract("12345")
    assert record.citation.source_id == "pmid:12345"
    assert "Full abstract here." in record.snippet


@respx.mock
def test_search_handles_http_error():
    import pytest
    from rarecell_mcp_knowledge.errors import BackendUnreachableError

    respx.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search").mock(
        return_value=Response(503)
    )

    client = EuropePMCClient()
    with pytest.raises(BackendUnreachableError):
        client.search("foo")
