from rarecell.rag import Citation, RetrievalHit
from rarecell.rag.base import Retriever


def test_protocol_importable():
    assert Retriever is not None


def test_citation_reexported_from_knowledge_pkg():
    c = Citation(source_id="pmid:1", source="europepmc")
    assert c.source == "europepmc"


def test_retrieval_hit_reexported():
    c = Citation(source_id="pmid:1", source="europepmc")
    h = RetrievalHit(citation=c, title="t", snippet="s", payload={}, source="europepmc")
    assert h.title == "t"
