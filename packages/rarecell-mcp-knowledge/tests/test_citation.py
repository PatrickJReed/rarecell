from rarecell_mcp_knowledge.citation import Citation, RetrievalHit


def test_citation_minimum():
    c = Citation(source_id="pmid:12345", source="europepmc")
    assert c.source == "europepmc"
    assert c.title is None


def test_retrieval_hit_minimum():
    c = Citation(source_id="cellmarker:T_cell:blood", source="cellmarker")
    h = RetrievalHit(
        citation=c,
        title="T cell markers",
        snippet="CD3D, CD3E",
        payload={"genes": ["CD3D", "CD3E"]},
        source="cellmarker",
    )
    assert h.payload["genes"] == ["CD3D", "CD3E"]
    assert h.retrieved_at  # datetime auto-populated
