import pytest

from worker import intel_public_draft as draft


def test_pdf_summary_draft_links_source_without_copying_full_text():
    article = {"title": "Digital health-enabled genomics", "date": "2022-07-07",
               "date_source": "pdf_first_page_footer",
               "doi": "10.1016/j.ajhg.2022.05.001", "attachment_sha256": "a" * 64}
    generated = {"summary": "The paper surveys how digital decision support, chatbots and online services could expand access to genomic care while requiring stronger evidence about outcomes and equity.",
                 "key_points": ["Digital decision tools may help people understand testing choices.",
                                "The authors discuss chatbots as a way to scale genetic counselling."],
                 "caveat": "The authors say comparative effectiveness, cost and equity evidence is still needed."}
    text = draft.build_draft(article, generated)
    assert 'source_date: "2022-07-07"' in text
    assert 'source_date_source: "pdf_first_page_footer"' in text
    assert "https://doi.org/10.1016/j.ajhg.2022.05.001" in text
    assert "wiki-ready: false" in text
    assert "full extracted PDF text" not in text
    assert 'credibility_tier_value: 5' in text
    assert 'source_article_type: "perspective"' in text
    assert 'journal_match_method: "none"' in text


def test_unverified_doi_does_not_inherit_peer_review_status():
    quality = draft.source_quality("10.1234/example")
    assert quality["credibility_tier_value"] == 0
    assert quality["journal_match_method"] == "none"


def test_private_contact_in_qwen_output_blocks_public_draft():
    article = {"title": "Paper", "date": "2022-07-07", "date_source": "pdf_first_page_footer",
               "doi": "10.1234/example", "attachment_sha256": "a" * 64}
    generated = {"summary": "A " * 50 + "paul@example.com", "key_points": ["A useful point about access.",
                                                                        "A useful point about evidence."],
                 "caveat": "The evidence has limits that need further study."}
    with pytest.raises(ValueError, match="private address"):
        draft.build_draft(article, generated)
