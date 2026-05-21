import respx
from httpx import Response
from rarecell.rag.knowledge import build_knowledge_session
from rarecell.rag.literature import LiteratureRetriever


@respx.mock
def test_literature_retriever_returns_hits(tmp_path):
    respx.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search").mock(
        return_value=Response(
            200,
            json={
                "hitCount": 1,
                "resultList": {
                    "result": [
                        {
                            "id": "12345",
                            "pmid": "12345",
                            "title": "T cell markers in brain",
                            "abstractText": "Pan-T markers...",
                            "doi": "10.1234/abc",
                            "pubYear": "2024",
                            "authorString": "X",
                        }
                    ]
                },
            },
        )
    )

    session = build_knowledge_session(
        catalog_path=tmp_path / "m.sqlite",
        cache_path=tmp_path / "c.sqlite",
    )
    retriever = LiteratureRetriever(session=session)
    hits = retriever.search("T cell brain")
    assert len(hits) == 1
    assert hits[0].citation.source_id == "pmid:12345"
    assert hits[0].source == "europepmc"


@respx.mock
def test_literature_retriever_fetch_abstract(tmp_path):
    respx.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search").mock(
        return_value=Response(
            200,
            json={
                "resultList": {
                    "result": [
                        {
                            "id": "111",
                            "pmid": "111",
                            "title": "T cell",
                            "abstractText": "abstract",
                            "doi": "10.1/x",
                            "pubYear": "2024",
                            "authorString": "X",
                        }
                    ]
                },
            },
        )
    )
    session = build_knowledge_session(
        catalog_path=tmp_path / "m.sqlite",
        cache_path=tmp_path / "c.sqlite",
    )
    hit = LiteratureRetriever(session=session).fetch_abstract("111")
    assert hit.citation.source_id == "pmid:111"
