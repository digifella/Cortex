from cortex_engine.claim_mapper import extract_claims
from cortex_engine.citation_formatter import annotate_text_with_citations


def test_extract_claims_filters_short_lines():
    text = (
        "Short sentence. "
        "This is a sufficiently long sentence that should be treated as a claim because it has enough words. "
        "Tiny."
    )
    claims = extract_claims(text)
    assert len(claims) == 1
    assert "sufficiently long sentence" in claims[0].text


def test_annotate_text_with_citations_inserts_numeric_refs():
    draft = "Team performance improved significantly after introducing psychological safety practices."
    result = {
        "claims": [
            {
                "claim_id": "c1",
                "claim_text": draft,
                "status": "supported",
                "evidence": [
                    {"source_file": "paper_a.pdf", "doc_id": "d1"},
                    {"source_file": "paper_b.pdf", "doc_id": "d2"},
                ],
            }
        ]
    }
    cited, refs = annotate_text_with_citations(draft, result, include_statuses=("supported",), max_refs_per_claim=2)
    assert "[1]" in cited and "[2]" in cited
    assert refs[0].startswith("[1] ")
    assert refs[1].startswith("[2] ")
