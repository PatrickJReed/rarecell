import respx
from httpx import Response
from rarecell_mcp_knowledge.enrichr import enrichr_enrich


@respx.mock
def test_enrichr_two_step_call():
    respx.post("https://maayanlab.cloud/Enrichr/addList").mock(
        return_value=Response(200, json={"userListId": 12345})
    )
    respx.get("https://maayanlab.cloud/Enrichr/enrich").mock(
        return_value=Response(
            200,
            json={
                "GO_Biological_Process_2023": [
                    [
                        1,
                        "T cell activation (GO:0042110)",
                        1e-10,
                        5.2,
                        100.0,
                        ["CD3D", "CD3E"],
                        0.001,
                        0,
                        0,
                    ],
                    [2, "B cell activation (GO:0042113)", 1e-8, 3.1, 50.0, ["MS4A1"], 0.01, 0, 0],
                ],
            },
        )
    )

    results = enrichr_enrich(
        genes=["CD3D", "CD3E", "MS4A1"],
        library="GO_Biological_Process_2023",
    )
    assert len(results) == 2
    assert results[0].title.startswith("T cell activation")
    assert results[0].payload["overlap_genes"] == ["CD3D", "CD3E"]
    assert results[0].source == "enrichr"


@respx.mock
def test_enrichr_empty_response():
    respx.post("https://maayanlab.cloud/Enrichr/addList").mock(
        return_value=Response(200, json={"userListId": 99})
    )
    respx.get("https://maayanlab.cloud/Enrichr/enrich").mock(
        return_value=Response(200, json={"GO_Biological_Process_2023": []})
    )

    results = enrichr_enrich(genes=["RANDOM"], library="GO_Biological_Process_2023")
    assert results == []
