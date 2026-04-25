from worker.handlers import market_radar as mr
from worker.handlers import weekly_report as wr
from unittest.mock import Mock


def test_build_geography_policy_defaults_to_australia():
    policy = mr._build_geography_policy(focus={}, target={}, source_sites={})

    assert policy["primary"] == "Australia"
    assert policy["strict"] is True
    assert "Australia" in policy["query_terms"]


def test_filter_signals_for_geography_drops_uk_namesake_and_keeps_australian_signal():
    policy = mr._build_geography_policy(
        focus={"industry": "water utilities"},
        target={"name": "South East Water", "notes": "Victorian water utility"},
        source_sites={},
    )
    signals = [
        {
            "headline": "Ofwat fines South East Water after supply failures in the UK",
            "url": "https://www.ofwat.gov.uk/news/south-east-water-fine/",
            "date": "2026-03-18",
            "snippet": "The UK regulator found failures across England.",
            "source": "ofwat.gov.uk",
        },
        {
            "headline": "South East Water launches Melbourne water-saving campaign",
            "url": "https://southeastwater.com.au/news/melbourne-water-saving-campaign",
            "date": "2026-03-19",
            "snippet": "The Victorian utility expanded its customer drought response.",
            "source": "southeastwater.com.au",
        },
    ]

    filtered = mr._filter_signals_for_geography(signals, policy, target_name="South East Water")

    assert len(filtered) == 1
    assert filtered[0]["source"] == "southeastwater.com.au"


def test_filter_signals_for_geography_requires_local_match_for_broad_scan():
    policy = mr._build_geography_policy(focus={"industry": "water utilities"}, source_sites={})
    signals = [
        {
            "headline": "Water utilities face drought adaptation pressures in Victoria",
            "url": "https://water.vic.gov.au/news/drought-update",
            "date": "2026-03-20",
            "snippet": "Melbourne storages continue to decline.",
            "source": "water.vic.gov.au",
        },
        {
            "headline": "UK water market review published",
            "url": "https://example.co.uk/water-market-review",
            "date": "2026-03-20",
            "snippet": "Ofwat published a new sector update.",
            "source": "example.co.uk",
        },
        {
            "headline": "Global utility digitisation outlook",
            "url": "https://example.com/global-utility-digitisation",
            "date": "2026-03-20",
            "snippet": "A broad international article with no Australian angle.",
            "source": "example.com",
        },
    ]

    filtered = mr._filter_signals_for_geography(signals, policy, require_geo_match=True)

    assert [item["source"] for item in filtered] == ["water.vic.gov.au"]


def test_weekly_report_geography_defaults_to_australia():
    geography = wr._resolve_report_geography(report_scope={}, strategic_context={})

    assert geography == "Australia"


def test_weekly_report_geography_uses_explicit_scope():
    geography = wr._resolve_report_geography(
        report_scope={"country": "New Zealand"},
        strategic_context={},
    )

    assert geography == "New Zealand"


def test_weekly_report_formats_submitted_intel_with_submitter_attribution():
    section = wr._format_submitted_intel_section(
        [
            {
                "primary_entity_name": "Barwon Water",
                "title": "Board org chart shared",
                "intel_date": "2026-03-23",
                "submitted_by": "alice@example.com",
                "text_note": "Latest board and executive structure.",
                "content": "https://barwonwater.vic.gov.au/org-chart",
                "source_type": "mailbox_document",
            }
        ]
    )

    assert "Provenance: Submitted by alice@example.com; 2026-03-23" in section
    assert "Reference: https://barwonwater.vic.gov.au/org-chart" in section
    assert "**Barwon Water: Board org chart shared**" in section


def test_weekly_report_does_not_treat_submitter_as_document_subject():
    section = wr._format_submitted_intel_section(
        [
            {
                "stakeholder_name": "Paul Cooper / Longboardfella Consulting",
                "primary_entity_name": "Barwon Water",
                "title": "Org chart",
                "intel_date": "2026-03-23",
                "submitted_by": "paul@longboardfella.com.au",
                "text_note": "Board and executive organisation chart for Barwon Water.",
                "source_type": "mailbox_document",
            }
        ]
    )

    assert "**Barwon Water: Org chart**" in section
    assert "Entity: Barwon Water" in section
    assert "Entity: Paul Cooper / Longboardfella Consulting" not in section


def test_weekly_report_suppresses_known_submitter_identity_in_provenance():
    section = wr._format_submitted_intel_section(
        [
            {
                "primary_entity_name": "Goulburn Valley Water",
                "title": "Strategy 2035",
                "intel_date": "2026-03-24",
                "submitted_by_name": "Paul Cooper",
                "submitted_by": "paul@longboardfella.com.au",
                "text_note": "Long-term strategy document for GVW.",
                "source_type": "mailbox_document",
            }
        ]
    )

    assert "Paul Cooper" not in section
    assert "Longboardfella" not in section
    assert "Provenance: Submitted by External contributor; 2026-03-24" in section


def test_weekly_report_formats_web_intel_with_source_reference():
    section = wr._format_web_intel_section(
        [
            {
                "headline": "Barwon Water updates drought response",
                "date": "2026-03-22",
                "source": "barwonwater.vic.gov.au",
                "url": "https://barwonwater.vic.gov.au/news/drought-response",
                "snippet": "The utility expanded regional water-saving measures.",
            }
        ]
    )

    assert "Source: barwonwater.vic.gov.au" in section
    assert "Reference: https://barwonwater.vic.gov.au/news/drought-response" in section
    assert "**Barwon Water updates drought response**" in section


def test_weekly_report_formats_sector_sweep_as_separate_subsection():
    section = wr._format_web_intel_section(
        [
            {
                "headline": "Barwon Water updates drought response",
                "date": "2026-03-22",
                "source": "barwonwater.vic.gov.au",
                "url": "https://barwonwater.vic.gov.au/news/drought-response",
                "snippet": "The utility expanded regional water-saving measures.",
            },
            {
                "headline": "Water sector sweep",
                "type": "sector_sweep",
                "source": "Anthropic web search",
                "summary": "Regulatory and capital signals across Australian water utilities.",
            },
        ]
    )

    assert "### Targeted Web Research" in section
    assert "### Final Sector Sweep" in section
    assert "**Water sector sweep**" in section


def test_weekly_report_source_separation_instruction_mentions_both_streams():
    instruction = wr._source_separation_instruction()

    assert "Submitted Intelligence" in instruction
    assert "Internet-Sourced Intelligence" in instruction


def test_weekly_report_evidence_priority_instruction_prefers_submitted_intel():
    instruction = wr._evidence_priority_instruction(
        [{"title": "GVW RFP intel"}],
        [{"headline": "Water sector sweep", "type": "sector_sweep"}],
    )

    assert "Submitted intelligence is the primary evidence stream" in instruction
    assert "secondary enrichment" in instruction


def test_weekly_report_required_output_template_enforces_explicit_sections():
    template = wr._required_output_template()

    assert "SUBMITTED INTELLIGENCE" in template
    assert "INTERNET-SOURCED INTELLIGENCE" in template
    assert "STAKEHOLDER HIGHLIGHTS" in template


def test_weekly_report_organisation_coverage_instruction_lists_submitted_entities():
    instruction = wr._organisation_coverage_instruction(
        {"organisations": [{"name": "Goulburn Valley Water"}, {"name": "Barwon Water"}]},
        [
            {"primary_entity_name": "Goulburn Valley Water", "title": "IT strategy RFP"},
            {"primary_entity_name": "Barwon Water", "title": "Annual report"},
        ],
    )

    assert "Goulburn Valley Water" in instruction
    assert "Barwon Water" in instruction
    assert "Do not collapse" in instruction


def test_weekly_report_stakeholder_guidance_mentions_submitter_account_signals():
    instruction = wr._stakeholder_guidance_instruction([{"submitted_by": "paul@example.com"}])

    assert "relationship or account signals" in instruction
    assert "submitters as provenance" in instruction
    assert "Do not surface suppressed submitter identities by name" in instruction


def test_weekly_report_selects_installed_ollama_model():
    selected = wr._select_installed_ollama_model(
        ["qwen2.5:14b", "mistral-small3.2:latest", "mistral:latest"],
        ["llava:7b", "mistral-small3.2:latest"],
    )

    assert selected == "mistral-small3.2:latest"


def test_weekly_report_resolve_ollama_model_uses_installed_local_model(monkeypatch):
    monkeypatch.delenv("CORTEX_WEEKLY_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("CORTEX_WATCH_OLLAMA_MODEL", raising=False)
    monkeypatch.setenv("LOCAL_LLM_SYNTHESIS_MODEL", "qwen2.5:72b-instruct-q4_K_M")

    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "models": [
            {"name": "mistral-small3.2:latest"},
            {"name": "qwen3-vl:8b"},
        ]
    }
    monkeypatch.setattr(wr.requests, "get", lambda *args, **kwargs: response)

    assert wr._resolve_ollama_model() == "mistral-small3.2:latest"


def test_weekly_report_prefers_anthropic_for_internet_enriched_mode():
    assert wr._preferred_synthesis_provider("submitted_and_internet", []) == "anthropic"
    assert wr._preferred_synthesis_provider("submitted_only", [{"headline": "Web item"}]) == "anthropic"
    assert wr._preferred_synthesis_provider("submitted_only", []) == "ollama"


def test_weekly_report_merges_sector_sweep_with_deferred_web_intel():
    merged = wr._merge_web_intel(
        {"deferred": True, "target_count": 7, "targets": ["Barwon Water", "Seqwater"]},
        [{"headline": "Water sector sweep", "summary": "Regulatory and capital signals.", "type": "sector_sweep"}],
    )

    assert isinstance(merged, list)
    assert merged[0]["headline"] == "Water sector sweep"
    assert merged[1]["headline"] == "Deferred target web research"
    assert "Deferred target count: 7" in merged[1]["summary"]


def test_weekly_report_builds_sector_sweep_prompt_with_industries_and_geography():
    prompt = wr._build_sector_sweep_prompt(
        org_name="Escient",
        geography="Australia",
        report_scope={
            "industries": ["Water"],
            "organisations": [{"name": "Barwon Water"}],
            "stakeholders": [{"name": "Sarah Cumming"}],
        },
        date_range={"start": "2026-03-19", "end": "2026-03-26"},
    )

    assert "Australia" in prompt
    assert "Water" in prompt
    assert "Barwon Water" in prompt
    assert "Sarah Cumming" in prompt
