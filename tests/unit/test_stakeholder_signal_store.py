from __future__ import annotations

import json
import pickle

from cortex_engine.stakeholder_signal_store import StakeholderSignalStore, _ollama_watch_timeout_seconds


def test_profile_sync_and_signal_matching(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    sync_result = store.upsert_profiles(
        org_name="Silverchain",
        profiles=[
            {
                "canonical_name": "Steve Hodgkinson",
                "current_employer": "Silverchain",
                "current_role": "Chief Information Officer",
                "linkedin_url": "https://www.linkedin.com/in/steve-hodgkinson",
                "aliases": ["Stephen Hodgkinson"],
            }
        ],
        source="market_radar",
        trace_id="trace-abc",
    )
    assert sync_result["added"] == 1

    signal = store.ingest_signal(
        {
            "org_name": "Silverchain",
            "subject": "Steve Hodgkinson started a new role",
            "raw_text": "Steve Hodgkinson has started a new role as Chief Information Officer at Silverchain.",
            "parsed_candidate_name": "Steve Hodgkinson",
            "parsed_candidate_employer": "Silverchain",
            "primary_url": "https://example.com/post/1",
            "message_id": "<msg-1>",
        }
    )

    assert signal["signal_id"].startswith("sig_")
    assert len(signal["matches"]) == 1
    assert signal["matches"][0]["canonical_name"] == "Steve Hodgkinson"
    assert signal["matches"][0]["score"] >= 0.8
    assert tmp_path.joinpath("raw_signals", f"{signal['signal_id']}.md").exists()


def test_signal_deduplicates_on_hash(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    payload = {
        "org_name": "Acme",
        "subject": "LinkedIn notification",
        "raw_text": "Jane Doe mentioned Acme in a post.",
        "message_id": "<msg-2>",
    }

    first = store.ingest_signal(payload)
    second = store.ingest_signal(payload)

    assert first["signal_id"] == second["signal_id"]
    state = store.get_state()
    assert len(state["signals"]) == 1


def test_ingest_watch_signals_batch(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Jane Smith",
                "target_type": "person",
                "current_employer": "BigBank",
            }
        ],
        source="market_radar",
    )

    result = store.ingest_watch_signals(
        {
            "org_name": "Longboardfella",
            "source_system": "market_radar_watch",
            "source_job": "321",
            "signals": [
                {
                    "target_name": "Jane Smith",
                    "target_type": "person",
                    "current_employer": "BigBank",
                    "headline": "Jane Smith joins BigBank digital team",
                    "url": "https://example.com/jane",
                    "date": "2026-03-16T00:00:00Z",
                    "snippet": "Appointment confirmed in trade media.",
                    "source_type": "news",
                }
            ],
        }
    )

    assert result["signal_count"] == 1
    assert result["ingested_count"] == 1
    assert result["matched_signal_count"] == 1
    assert len(result["profiles_touched"]) == 1


def test_profile_sync_and_signal_ingest_build_stakeholder_graph(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Jane Smith",
                "target_type": "person",
                "current_employer": "BigBank",
                "affiliations": [{"org_name_text": "BigBank", "role": "VP Digital", "is_primary": 1}],
                "alumni": ["McKinsey"],
                "linkedin_connections": [{"member": "paul@example.com", "degree": "1st"}],
            }
        ],
        source="market_radar",
    )
    profile = store.list_profiles(org_name="Longboardfella")[0]
    signal = store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "subject": "Jane Smith joins BigBank digital team",
            "raw_text": "Jane Smith has joined BigBank's digital team.",
            "parsed_candidate_name": "Jane Smith",
            "parsed_candidate_employer": "BigBank",
            "source_name": "Trade Press",
            "message_id": "<graph-signal>",
        }
    )

    graph_path = tmp_path.parent / "knowledge_cortex.gpickle"
    assert graph_path.exists()
    with open(graph_path, "rb") as handle:
        graph = pickle.load(handle)

    profile_node = next(
        node for node, attrs in graph.nodes(data=True)
        if attrs.get("profile_key") == profile["profile_key"]
    )
    signal_node = next(node for node, attrs in graph.nodes(data=True) if attrs.get("signal_id") == signal["signal_id"])
    assert any(attrs.get("relationship") == "tracked_by" for _, _, attrs in graph.edges(profile_node, data=True))
    assert any(attrs.get("relationship") == "mentions_profile" for _, _, attrs in graph.edges(signal_node, data=True))


def test_rebuild_stakeholder_graph_backfills_existing_state(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Jane Smith",
                "target_type": "person",
                "current_employer": "BigBank",
                "alumni": ["McKinsey"],
            }
        ],
        org_alumni=["McKinsey"],
        source="market_radar",
    )
    store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "subject": "Jane Smith joins BigBank digital team",
            "raw_text": "Jane Smith has joined BigBank's digital team.",
            "parsed_candidate_name": "Jane Smith",
            "parsed_candidate_employer": "BigBank",
            "message_id": "<rebuild-signal>",
        }
    )

    result = store.rebuild_stakeholder_graph()

    assert result["profiles_processed"] == 1
    assert result["signals_processed"] == 1
    assert result["org_contexts_processed"] == 1
    assert result["graph_nodes"] > 0
    assert result["graph_edges"] > 0


def test_build_graph_view_returns_scoped_network_payload(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Jane Smith",
                "target_type": "person",
                "current_employer": "BigBank",
                "current_role": "VP Digital",
                "watch_status": "watch",
                "affiliations": [{"org_name_text": "BigBank", "role": "VP Digital", "is_primary": 1}],
                "alumni": ["McKinsey"],
                "linkedin_connections": [{"member": "paul@example.com", "degree": "1st"}],
            },
            {
                "canonical_name": "John Roe",
                "target_type": "person",
                "current_employer": "Medibank",
                "current_role": "Chief Strategy Officer",
                "watch_status": "watch",
                "affiliations": [{"org_name_text": "Medibank", "role": "Chief Strategy Officer", "is_primary": 1}],
                "alumni": ["McKinsey"],
            },
        ],
        org_alumni=["McKinsey"],
        source="market_radar",
    )
    store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "subject": "Jane Smith joins BigBank digital team",
            "raw_text": "Jane Smith joins BigBank digital team.",
            "parsed_candidate_name": "Jane Smith",
            "parsed_candidate_employer": "BigBank",
            "source_name": "Trade Press",
            "message_id": "<graph-view-1>",
        }
    )

    graph_view = store.build_graph_view(
        org_name="Longboardfella",
        view_mode="watch_network",
        include_signals=True,
        include_sources=True,
        include_lab_members=True,
        include_alumni=True,
        max_hops=2,
        max_nodes=50,
        max_edges=50,
        top_k_paths=5,
    )

    assert graph_view["node_count"] > 0
    assert graph_view["edge_count"] > 0
    assert graph_view["has_paths"] is True
    output_path = tmp_path / "graph_views" / f"{graph_view['graph_id']}.json"
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    node_types = {node["type"] for node in payload["nodes"]}
    edge_types = {edge["type"] for edge in payload["edges"]}
    path_types = {path["type"] for path in payload["paths"]}

    assert {"person", "organization", "alumni_group", "lab_member", "subscriber_org"}.issubset(node_types)
    assert {"works_at", "alumni_of", "linkedin_connection"}.issubset(edge_types)
    assert {"warm_intro", "shared_alumni"}.intersection(path_types)
    assert payload["summary"]["warm_intro_paths"] >= 1


def test_build_graph_view_supports_industry_network(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "external_profile_id": "ind-1",
                "canonical_name": "Digital Health",
                "target_type": "industry",
                "watch_status": "watch",
                "key_themes": ["telehealth", "privacy"],
                "description": "Digital health sector",
            },
            {
                "external_profile_id": "org-1",
                "canonical_name": "Healthdirect Australia",
                "target_type": "organisation",
                "watch_status": "watch",
                "industry_affiliations": [{"industry_profile_key": "ind-1", "industry_name": "Digital Health", "role": "key player"}],
            },
            {
                "external_profile_id": "person-1",
                "canonical_name": "Jane Smith",
                "target_type": "person",
                "watch_status": "watch",
                "current_employer": "Healthdirect Australia",
                "affiliations": [{"org_name_text": "Healthdirect Australia", "role": "Executive", "is_primary": 1}],
                "industry_affiliations": [{"industry_profile_key": "ind-1", "industry_name": "Digital Health", "role": "speaker"}],
            },
        ],
        source="market_radar",
    )
    profiles = store.list_profiles(org_name="Longboardfella")
    industry_key = next(item["profile_key"] for item in profiles if item["target_type"] == "industry")
    org_key = next(item["profile_key"] for item in profiles if item["canonical_name"] == "Healthdirect Australia")

    graph_view = store.build_graph_view(
        org_name="Longboardfella",
        view_mode="industry_network",
        focus_profile_key=industry_key,
        child_profile_keys=[org_key],
        include_signals=False,
        include_alumni=True,
        max_hops=2,
        max_nodes=50,
        max_edges=50,
    )

    payload = json.loads((tmp_path / "graph_views" / f"{graph_view['graph_id']}.json").read_text(encoding="utf-8"))
    node_types = {node["type"] for node in payload["nodes"]}
    edge_types = {edge["type"] for edge in payload["edges"]}
    assert "industry" in node_types
    assert "belongs_to_industry" in edge_types
    assert payload["summary"]["watched_industries"] >= 1


def test_organisation_target_matches_without_person_fields(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Digital Health",
        profiles=[
            {
                "canonical_name": "Silverchain",
                "target_type": "organisation",
                "tags": ["aged care", "care services"],
            }
        ],
        source="market_radar",
    )

    signal = store.ingest_signal(
        {
            "org_name": "Digital Health",
            "target_type": "organisation",
            "subject": "Silverchain expands aged care services",
            "raw_text": "Silverchain announced a new aged care services initiative across WA.",
            "parsed_candidate_name": "Silverchain",
        }
    )

    assert len(signal["matches"]) == 1
    assert signal["matches"][0]["target_type"] == "organisation"


def test_generate_digest_writes_markdown(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Silverchain",
        profiles=[{"canonical_name": "Steve Hodgkinson", "current_employer": "Silverchain"}],
        source="market_radar",
    )
    store.ingest_signal(
        {
            "org_name": "Silverchain",
            "subject": "Steve Hodgkinson started a new role",
            "raw_text": "Steve Hodgkinson has started a new role at Silverchain.",
            "parsed_candidate_name": "Steve Hodgkinson",
            "parsed_candidate_employer": "Silverchain",
        }
    )

    digest = store.generate_digest(org_name="Silverchain")

    assert digest["signal_count"] == 1
    digest_path = tmp_path / "digests" / f"{digest['digest_id']}.md"
    assert digest_path.exists()
    assert "Stakeholder Intelligence Digest" in digest_path.read_text(encoding="utf-8")
    assert digest["llm_synthesised"] is False
    assert digest["profiles_covered"] == 1


def test_generate_digest_filters_exact_name_only_noise_domains(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Terry Symonds",
                "current_employer": "Berry Street Yooralla",
                "target_type": "person",
            }
        ],
        source="market_radar",
    )

    store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "subject": "Terry Symonds | Actor",
            "raw_text": "IMDb actor page for Terry Symonds.",
            "parsed_candidate_name": "Terry Symonds",
            "primary_url": "https://www.imdb.com/name/nm4588154/",
            "message_id": "<terry-imdb>",
        }
    )

    digest = store.generate_digest(org_name="Longboardfella", report_depth="detailed")

    assert digest["signal_count"] == 0
    digest_text = (tmp_path / "digests" / f"{digest['digest_id']}.md").read_text(encoding="utf-8")
    assert "No matching signals for this window." in digest_text


def test_generate_digest_keeps_linkedin_exact_identity_match(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Dr Louise Schaper",
                "current_employer": "Pulse-IT",
                "target_type": "person",
                "linkedin_url": "https://www.linkedin.com/in/louiseschaper/",
            }
        ],
        source="market_radar",
    )

    store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "subject": "Dr Louise Schaper - Pulse+IT | LinkedIn",
            "raw_text": "Professional profile for Dr Louise Schaper at Pulse+IT.",
            "parsed_candidate_name": "Dr Louise Schaper",
            "primary_url": "https://www.linkedin.com/in/louiseschaper/",
            "message_id": "<louise-linkedin>",
        }
    )

    digest = store.generate_digest(org_name="Longboardfella", report_depth="detailed")

    assert digest["signal_count"] == 1


def test_generate_digest_filters_zoominfo_even_when_employer_anchored(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Bettina McMahon",
                "current_employer": "Healthdirect Australia",
                "target_type": "person",
            }
        ],
        source="market_radar",
    )

    store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "subject": "Bettina McMahon - Healthdirect Australia",
            "raw_text": "Bettina McMahon is listed at Healthdirect Australia.",
            "parsed_candidate_name": "Bettina McMahon",
            "parsed_candidate_employer": "Healthdirect Australia",
            "primary_url": "https://www.zoominfo.com/p/Bettina-Mcmahon/1634368121",
            "message_id": "<bettina-zoominfo>",
        }
    )

    digest = store.generate_digest(org_name="Longboardfella", report_depth="detailed")

    assert digest["signal_count"] == 0
    digest_text = (tmp_path / "digests" / f"{digest['digest_id']}.md").read_text(encoding="utf-8")
    assert "No matching signals for this window." in digest_text


def test_industry_target_matches_from_tags_and_name(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Health Watch",
        profiles=[
            {
                "canonical_name": "Digital Health",
                "target_type": "industry",
                "tags": ["telehealth", "virtual care", "health technology"],
            }
        ],
        source="market_radar",
    )

    signal = store.ingest_signal(
        {
            "org_name": "Health Watch",
            "target_type": "industry",
            "subject": "Digital health investment rises",
            "raw_text": "Telehealth and virtual care platforms are attracting new investment.",
            "parsed_candidate_name": "Digital Health",
            "tags": ["telehealth"],
        }
    )

    assert len(signal["matches"]) == 1
    assert signal["matches"][0]["target_type"] == "industry"


def test_theme_target_matches_from_theme_name(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Gov Intel",
        profiles=[
            {
                "canonical_name": "Cyber Security",
                "target_type": "theme",
                "tags": ["cyber", "security", "resilience"],
            }
        ],
        source="market_radar",
    )

    signal = store.ingest_signal(
        {
            "org_name": "Gov Intel",
            "target_type": "theme",
            "subject": "Cyber security uplift program announced",
            "raw_text": "A new cyber resilience and security uplift is being funded.",
            "parsed_candidate_name": "Cyber Security",
        }
    )

    assert len(signal["matches"]) == 1
    assert signal["matches"][0]["target_type"] == "theme"


def test_signal_matches_when_org_name_is_variant(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Carolyn Bell",
                "target_type": "person",
                "current_employer": "Silverchain",
            }
        ],
        source="market_radar",
    )

    signal = store.ingest_signal(
        {
            "org_name": "Longboardfella Consulting Pty Ltd",
            "target_type": "person",
            "subject": "Carolyn Bell joined a new Silverchain initiative",
            "raw_text": "Carolyn Bell has been mentioned in connection with a new Silverchain initiative.",
            "parsed_candidate_name": "Carolyn Bell",
            "parsed_candidate_employer": "Silverchain",
        }
    )

    assert len(signal["matches"]) == 1
    assert signal["matches"][0]["canonical_name"] == "Carolyn Bell"


def test_person_change_signal_creates_update_suggestion(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Steve Hodgkinson",
                "target_type": "person",
                "current_employer": "Vic Police",
                "current_role": "Chief Digital Officer",
            }
        ],
        source="market_radar",
    )

    signal = store.ingest_signal(
        {
            "org_name": "Longboardfella Consulting Pty Ltd",
            "target_type": "person",
            "subject": "Steve Hodgkinson started a new role as Chief Information Officer at Silverchain",
            "raw_text": "Steve Hodgkinson started a new role as Chief Information Officer at Silverchain.",
            "parsed_candidate_name": "Steve Hodgkinson",
            "parsed_candidate_employer": "Silverchain",
        }
    )

    assert len(signal["matches"]) == 1
    assert len(signal["update_suggestion_ids"]) >= 1

    suggestions = store.list_update_suggestions(org_name="Longboardfella")
    assert any(item["field_name"] == "current_employer" and item["proposed_value"] == "Silverchain" for item in suggestions)
    assert any(item["field_name"] == "current_role" and "Chief Information Officer" in item["proposed_value"] for item in suggestions)


def test_profile_sync_preserves_external_profile_identity_and_rich_org_fields(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    first = store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "external_profile_id": "42",
                "canonical_name": "Silverchain",
                "target_type": "organisation",
                "email": "contact@silverchain.org.au",
                "industry": "Aged Care",
                "status": "active",
                "watch_status": "watch",
                "last_verified_at": "2026-03-08",
                "website_url": "https://www.silverchain.org.au",
                "parent_entity": "Silverchain Group",
                "acn_abn": "12 345 678 901",
                "phone": "+61 8 9200 0000",
                "address": {
                    "street": "44 Brown St",
                    "city": "Perth",
                    "state": "WA",
                    "postcode": "6000",
                    "country": "Australia",
                },
                "aliases": ["SCG"],
            }
        ],
        source="market_radar",
    )
    assert first["added"] == 1

    initial_profile = store.list_profiles(org_name="Longboardfella")[0]
    initial_key = initial_profile["profile_key"]

    second = store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "external_profile_id": "42",
                "canonical_name": "Silverchain Group",
                "target_type": "organisation",
                "email": "hello@silverchain.org.au",
                "industry": "Community Care",
                "status": "watch",
                "watch_status": "off",
                "last_verified_at": "2026-03-09",
                "website_url": "https://www.silverchain.org.au",
                "parent_entity": "National Care Group",
                "acn_abn": "12 345 678 901",
                "phone": "+61 8 9200 9999",
                "address": {
                    "street": "1 New St",
                    "city": "Perth",
                    "state": "WA",
                    "postcode": "6001",
                    "country": "Australia",
                },
                "aliases": ["SCG"],
            }
        ],
        source="market_radar",
    )

    assert second["updated"] == 1
    profiles = store.list_profiles(org_name="Longboardfella")
    assert len(profiles) == 1
    assert profiles[0]["profile_key"] == initial_key
    assert profiles[0]["external_profile_id"] == "42"
    assert profiles[0]["canonical_name"] == "Silverchain Group"
    assert profiles[0]["email"] == "hello@silverchain.org.au"
    assert profiles[0]["industry"] == "Community Care"
    assert profiles[0]["status"] == "watch"
    assert profiles[0]["watch_status"] == "off"
    assert profiles[0]["last_verified_at"] == "2026-03-09"
    assert profiles[0]["website_url"] == "https://www.silverchain.org.au"
    assert profiles[0]["parent_entity"] == "National Care Group"
    assert profiles[0]["phone"] == "+61 8 9200 9999"
    assert profiles[0]["address"]["postcode"] == "6001"


def test_profile_sync_without_external_id_does_not_duplicate_on_employer_change(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Steve Hodgkinson",
                "target_type": "person",
                "current_employer": "Vic Police",
                "linkedin_url": "https://www.linkedin.com/in/steve-hodgkinson",
            }
        ],
        source="market_radar",
    )
    initial_profile = store.list_profiles(org_name="Longboardfella")[0]

    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Steve Hodgkinson",
                "target_type": "person",
                "current_employer": "Silverchain",
                "linkedin_url": "https://www.linkedin.com/in/steve-hodgkinson",
            }
        ],
        source="market_radar",
    )

    profiles = store.list_profiles(org_name="Longboardfella")
    assert len(profiles) == 1
    assert profiles[0]["profile_key"] == initial_profile["profile_key"]
    assert profiles[0]["current_employer"] == "Silverchain"


def test_profile_sync_preserves_affiliations_and_matches_non_primary_employer(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        org_alumni=["SMS", "Escient"],
        profiles=[
            {
                "canonical_name": "Jane Smith",
                "target_type": "person",
                "affiliations": [
                    {
                        "org_name_text": "Acme Corp",
                        "role": "Chief Strategy Officer",
                        "affiliation_type": "current",
                        "confidence": "confirmed",
                        "is_primary": 1,
                    },
                    {
                        "org_name_text": "Board Co",
                        "role": "Non-Executive Director",
                        "affiliation_type": "board",
                        "confidence": "probable",
                        "is_primary": 0,
                    },
                ],
            }
        ],
        source="market_radar",
    )

    profiles = store.list_profiles(org_name="Longboardfella")
    assert len(profiles) == 1
    assert profiles[0]["current_employer"] == "Acme Corp"
    assert profiles[0]["current_role"] == "Chief Strategy Officer"
    assert len(profiles[0]["affiliations"]) == 2
    assert profiles[0]["known_employers"] == ["Acme Corp", "Board Co"]
    assert store.get_org_context("Longboardfella")["org_alumni"] == ["SMS", "Escient"]

    signal = store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "target_type": "person",
            "subject": "Jane Smith joined the Board Co advisory council",
            "raw_text": "Jane Smith joined the Board Co advisory council as a Non-Executive Director.",
            "parsed_candidate_name": "Jane Smith",
            "parsed_candidate_employer": "Board Co",
        }
    )

    assert len(signal["matches"]) == 1
    assert signal["matches"][0]["canonical_name"] == "Jane Smith"
    assert signal["matches"][0]["affiliations"][1]["org_name_text"] == "Board Co"


def test_profile_sync_persists_org_strategic_profile_context(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    result = store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Digital Health",
                "target_type": "industry",
            }
        ],
        org_strategic_profile={
            "description": "Primary sector strategy",
            "industries": ["Digital Health", "Healthcare"],
            "key_themes": ["transformation"],
        },
        source="market_radar",
    )

    context = store.get_org_context("Longboardfella")
    assert result["org_strategic_industry_count"] == 2
    assert context["org_strategic_profile"]["industries"] == ["Digital Health", "Healthcare"]
    assert context["org_strategic_profile"]["key_themes"] == ["transformation"]


def test_reconcile_intel_note_delivery_records_note_and_graph_links(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Mic Cavazzini",
                "target_type": "person",
                "watch_status": "watch",
            },
            {
                "canonical_name": "RACP",
                "target_type": "organisation",
            },
        ],
        source="market_radar",
    )

    result = store.reconcile_intel_note_delivery(
        org_name="Longboardfella",
        trace_id="trace-note-1",
        payload={
            "note": {
                "title": "Meeting with Mic Cavazzini",
                "note_date": "2026-03-19",
                "submitted_by": "paul@example.com",
                "content": "Discussed strategic transformation.",
            },
            "primary_entity": {
                "name": "Mic Cavazzini",
                "target_type": "person",
            },
            "referenced_entities": [
                {
                    "name": "RACP",
                    "target_type": "organisation",
                    "reference_type": "meeting",
                    "confidence": "confirmed",
                }
            ],
        },
        response={"intel_id": "note_abc123"},
    )

    assert result["intel_id"] == "note_abc123"
    assert result["linked_entities"] >= 2
    notes = store.list_intel_notes(org_name="Longboardfella")
    assert notes[0]["intel_id"] == "note_abc123"


def test_profile_sync_upgrades_legacy_profile_to_external_profile_id(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Silverchain Group",
                "target_type": "organisation",
                "website_url": "https://www.silverchain.org.au",
            }
        ],
        source="legacy_sync",
    )
    legacy = store.list_profiles(org_name="Longboardfella")[0]

    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "external_profile_id": "77",
                "canonical_name": "Silverchain Group",
                "target_type": "organisation",
                "website_url": "https://www.silverchain.org.au",
                "parent_entity": "National Care Group",
            }
        ],
        source="market_radar",
    )

    profiles = store.list_profiles(org_name="Longboardfella")
    assert len(profiles) == 1
    assert profiles[0]["profile_key"] == legacy["profile_key"]
    assert profiles[0]["external_profile_id"] == "77"
    assert profiles[0]["parent_entity"] == "National Care Group"


def test_profile_sync_replace_scope_removes_stale_market_radar_profiles(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {"external_profile_id": "1", "canonical_name": "Carolyn Bell", "target_type": "person"},
            {"external_profile_id": "2", "canonical_name": "Steve Hodgkinson", "target_type": "person"},
        ],
        source="market_radar",
        replace_org_scope=True,
    )

    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {"external_profile_id": "1", "canonical_name": "Carolyn Bell", "target_type": "person"},
        ],
        source="market_radar",
        replace_org_scope=True,
    )

    profiles = store.list_profiles(org_name="Longboardfella")
    assert len(profiles) == 1
    assert profiles[0]["canonical_name"] == "Carolyn Bell"


def test_watch_status_defaults_to_off_and_accepts_watch(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {"canonical_name": "Carolyn Bell", "target_type": "person", "watch_status": "watch"},
            {"canonical_name": "Ian Oppermann", "target_type": "person", "watch_status": "invalid"},
        ],
        source="market_radar",
    )

    profiles = {profile["canonical_name"]: profile for profile in store.list_profiles(org_name="Longboardfella")}
    assert profiles["Carolyn Bell"]["watch_status"] == "watch"
    assert profiles["Ian Oppermann"]["watch_status"] == "off"


def test_generate_digest_can_use_llm_synthesis_and_profile_filter(tmp_path, monkeypatch):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {"external_profile_id": "1", "canonical_name": "Carolyn Bell", "target_type": "person", "watch_status": "watch"},
            {"external_profile_id": "2", "canonical_name": "Silverchain Group", "target_type": "organisation"},
        ],
        source="market_radar",
    )
    profiles = store.list_profiles(org_name="Longboardfella")
    carolyn_key = next(profile["profile_key"] for profile in profiles if profile["canonical_name"] == "Carolyn Bell")

    store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "subject": "Carolyn Bell started a new role",
            "raw_text": "Carolyn Bell started a new role as ED Aged Care at Silverchain.",
            "parsed_candidate_name": "Carolyn Bell",
            "parsed_candidate_employer": "Silverchain",
            "message_id": "<carolyn-watch>",
        }
    )
    store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "target_type": "organisation",
            "subject": "Silverchain expands community care services",
            "raw_text": "Silverchain expanded community care services in WA.",
            "parsed_candidate_name": "Silverchain Group",
            "message_id": "<silverchain-org>",
        }
    )

    def fake_synth(self, raw_data, org_name, provider, model, report_depth):
        assert org_name == "Longboardfella"
        assert provider == "ollama"
        assert model == "qwen2.5:14b"
        assert report_depth == "detailed"
        assert "Carolyn Bell" in raw_data
        assert "Silverchain expands community care services" not in raw_data
        assert '"relationship_context"' in raw_data
        assert '"priority_signals"' in raw_data
        return "# WATCH Report\n\n## Individual Updates\n- Carolyn Bell moved roles.", "ollama", "qwen2.5:14b"

    monkeypatch.setattr(StakeholderSignalStore, "_llm_synthesise", fake_synth)

    digest = store.generate_digest(
        org_name="Longboardfella",
        profile_keys=[carolyn_key],
        llm_synthesis=True,
        llm_provider="ollama",
        llm_model="qwen2.5:14b",
    )

    digest_text = (tmp_path / "digests" / f"{digest['digest_id']}.md").read_text(encoding="utf-8")
    assert digest["signal_count"] == 1
    assert digest["llm_synthesised"] is True
    assert digest["profiles_covered"] == 1
    assert digest["period_end"]
    assert digest["llm_provider"] == "ollama"
    assert digest["llm_model"] == "qwen2.5:14b"
    assert "# WATCH Report" in digest_text
    assert "Carolyn Bell moved roles." in digest_text


def test_generate_digest_strips_empty_llm_filler_sections(tmp_path, monkeypatch):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Carolyn Bell",
                "target_type": "person",
                "current_employer": "Silverchain",
            }
        ],
        source="market_radar",
    )
    store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "subject": "Carolyn Bell started a new role",
            "raw_text": "Carolyn Bell started a new role as ED Aged Care at Silverchain.",
            "parsed_candidate_name": "Carolyn Bell",
            "parsed_candidate_employer": "Silverchain",
            "message_id": "<carolyn-filler-test>",
        }
    )

    def fake_synth(self, raw_data, org_name, provider, model, report_depth):
        return (
            "# Stakeholder Intelligence Digest\n\n"
            "## Stakeholder Digest for Longboardfella\n\n"
            "### Carolyn Bell\n"
            "- Update retained.\n\n"
            "### Alumni Context\n"
            "- No relevant alumni context provided for the organization Longboardfella.\n\n"
            "### Weak Linkage Context\n"
            "- No alumni or weak linkage context available for the provided signals.\n",
            "ollama",
            "qwen2.5:14b",
        )

    monkeypatch.setattr(StakeholderSignalStore, "_llm_synthesise", fake_synth)

    digest = store.generate_digest(
        org_name="Longboardfella",
        llm_synthesis=True,
        llm_provider="ollama",
        llm_model="qwen2.5:14b",
    )

    digest_text = (tmp_path / "digests" / f"{digest['digest_id']}.md").read_text(encoding="utf-8")
    assert "Update retained." in digest_text
    assert "Alumni Context" not in digest_text
    assert "Weak Linkage Context" not in digest_text
    assert "No relevant alumni context" not in digest_text
    assert "No alumni or weak linkage context" not in digest_text


def test_llm_synthesise_falls_back_to_anthropic_when_ollama_fails(tmp_path, monkeypatch):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "canonical_name": "Carolyn Bell",
                "target_type": "person",
                "current_employer": "Silverchain",
            }
        ],
        source="market_radar",
    )
    store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "subject": "Carolyn Bell started a new role",
            "raw_text": "Carolyn Bell started a new role as ED Aged Care at Silverchain.",
            "parsed_candidate_name": "Carolyn Bell",
            "parsed_candidate_employer": "Silverchain",
            "message_id": "<carolyn-fallback>",
        }
    )

    def fake_ollama(self, system, user, model):
        return None, "qwen2.5:14b"

    def fake_anthropic(self, system, user, model):
        assert "Carolyn Bell" in user
        assert model == "claude-haiku-4-5-20251001"
        return "# Claude Digest\n\n- Synthesised fallback output.", model

    monkeypatch.setattr(StakeholderSignalStore, "_call_ollama", fake_ollama)
    monkeypatch.setattr(StakeholderSignalStore, "_call_anthropic", fake_anthropic)

    digest = store.generate_digest(
        org_name="Longboardfella",
        llm_synthesis=True,
        llm_provider="ollama",
        llm_model="qwen2.5:14b",
    )

    digest_text = (tmp_path / "digests" / f"{digest['digest_id']}.md").read_text(encoding="utf-8")
    assert digest["llm_synthesised"] is True
    assert digest["llm_provider"] == "anthropic"
    assert digest["llm_model"] == "claude-haiku-4-5-20251001"
    assert "Synthesised fallback output." in digest_text


def test_generate_digest_summary_sets_escalation_metadata(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "external_profile_id": "1",
                "canonical_name": "Steve Hodgkinson",
                "target_type": "person",
                "current_employer": "Vic Police",
                "current_role": "Chief Digital Officer",
            }
        ],
        source="market_radar",
    )
    profile = store.list_profiles(org_name="Longboardfella")[0]

    store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "subject": "Steve Hodgkinson started a new role as Chief Information Officer at Silverchain",
            "raw_text": "Steve Hodgkinson started a new role as Chief Information Officer at Silverchain.",
            "parsed_candidate_name": "Steve Hodgkinson",
            "parsed_candidate_employer": "Silverchain",
            "message_id": "<digest-summary-escalate>",
        }
    )

    digest = store.generate_digest(
        org_name="Longboardfella",
        profile_keys=[profile["profile_key"]],
        report_depth="summary",
        digest_tier="priority",
        priority_profile_keys=[profile["profile_key"]],
        deep_analysis=True,
    )

    digest_text = (tmp_path / "digests" / f"{digest['digest_id']}.md").read_text(encoding="utf-8")
    assert digest["report_depth"] == "summary"
    assert digest["digest_tier"] == "priority"
    assert digest["deep_analysis"] is True
    assert digest["escalate"] is True
    assert digest["escalate_profiles"] == [profile["profile_key"]]
    assert "leadership change" in digest["escalate_reason"]
    assert "## Key Signals" in digest_text


def test_generate_digest_includes_alumni_context_signals(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[{"external_profile_id": "1", "canonical_name": "Carolyn Bell", "target_type": "person"}],
        org_alumni=["SMS", "Deloitte"],
        source="market_radar",
    )

    store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "subject": "Deloitte appoints new health sector lead",
            "raw_text": "Deloitte has appointed a new health sector lead in Australia.",
            "message_id": "<alumni-only-signal>",
        }
    )

    digest = store.generate_digest(
        org_name="Longboardfella",
        report_depth="summary",
        matched_only=True,
    )

    digest_text = (tmp_path / "digests" / f"{digest['digest_id']}.md").read_text(encoding="utf-8")
    assert digest["signal_count"] == 1
    assert digest["org_alumni"] == ["SMS", "Deloitte"]
    assert digest["signals"][0]["alumni_hits"] == ["Deloitte"]
    assert "Alumni context: SMS, Deloitte" in digest_text
    assert "[alumni: Deloitte]" in digest_text


def test_generate_digest_supports_industry_scope_and_shared_visibility(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "external_profile_id": "ind-1",
                "canonical_name": "Digital Health",
                "target_type": "industry",
                "watch_status": "watch",
                "description": "Sector profile",
                "key_themes": ["telehealth", "privacy"],
                "regulatory_context": "Privacy reform",
                "market_size": "$10B",
            },
            {
                "external_profile_id": "org-1",
                "canonical_name": "Healthdirect Australia",
                "target_type": "organisation",
                "watch_status": "watch",
                "industry_affiliations": [{"industry_profile_key": "ind-1", "industry_name": "Digital Health", "role": "key player"}],
            },
            {
                "external_profile_id": "person-1",
                "canonical_name": "Jane Smith",
                "target_type": "person",
                "watch_status": "watch",
                "current_employer": "Healthdirect Australia",
                "affiliations": [{"org_name_text": "Healthdirect Australia", "role": "Executive", "is_primary": 1}],
                "industry_affiliations": [{"industry_profile_key": "ind-1", "industry_name": "Digital Health", "role": "speaker"}],
            },
        ],
        source="market_radar",
    )
    profiles = store.list_profiles(org_name="Longboardfella")
    industry_key = next(item["profile_key"] for item in profiles if item["target_type"] == "industry")
    org_key = next(item["profile_key"] for item in profiles if item["canonical_name"] == "Healthdirect Australia")
    person_key = next(item["profile_key"] for item in profiles if item["canonical_name"] == "Jane Smith")

    store.ingest_signal(
        {
            "org_name": "Escient",
            "source_org_name": "Escient",
            "visible_to_orgs": ["Longboardfella", "Escient"],
            "shared_with_orgs": ["Longboardfella"],
            "scope_profile_key": industry_key,
            "child_profile_keys": [org_key, person_key],
            "child_org_names": ["Healthdirect Australia"],
            "key_themes": ["telehealth"],
            "subject": "Healthdirect expands telehealth advisory program",
            "raw_text": "Healthdirect Australia expands a telehealth advisory program led by Jane Smith.",
            "parsed_candidate_name": "Jane Smith",
            "parsed_candidate_employer": "Healthdirect Australia",
            "message_id": "<industry-shared-1>",
        }
    )

    digest = store.generate_digest(
        org_name="Longboardfella",
        scope_type="industry",
        scope_profile_key=industry_key,
        child_profile_keys=[org_key, person_key],
        child_org_names=["Healthdirect Australia"],
        key_themes=["telehealth"],
        regulatory_context="Privacy reform",
        market_size="$10B",
        report_depth="detailed",
    )

    digest_text = (tmp_path / "digests" / f"{digest['digest_id']}.md").read_text(encoding="utf-8")
    assert digest["scope_type"] == "industry"
    assert digest["signal_count"] == 1
    assert "Industry Intelligence Digest" in digest_text
    assert "Healthdirect Australia" in digest_text
    assert "telehealth" in digest_text.lower()


def test_generate_digest_adds_relationship_temporal_and_scoring_metadata(tmp_path):
    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Longboardfella",
        profiles=[
            {
                "external_profile_id": "1",
                "canonical_name": "Jane Smith",
                "target_type": "person",
                "current_employer": "BigBank",
                "known_employers": ["BigBank", "McKinsey"],
                "alumni": ["McKinsey"],
                "linkedin_connections": [{"member": "paul@example.com", "degree": "1st"}],
                "watch_status": "watch",
            },
            {
                "external_profile_id": "2",
                "canonical_name": "Bob Jones",
                "target_type": "person",
                "current_employer": "OtherBank",
                "known_employers": ["OtherBank", "McKinsey"],
                "alumni": ["McKinsey"],
                "watch_status": "watch",
            },
        ],
        org_alumni=["McKinsey"],
        source="market_radar",
    )
    profiles = store.list_profiles(org_name="Longboardfella")
    jane_key = next(item["profile_key"] for item in profiles if item["canonical_name"] == "Jane Smith")
    bob_key = next(item["profile_key"] for item in profiles if item["canonical_name"] == "Bob Jones")

    store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "subject": "Jane Smith joins BigBank digital team",
            "raw_text": "Official announcement confirms Jane Smith joined BigBank.",
            "parsed_candidate_name": "Jane Smith",
            "parsed_candidate_employer": "BigBank",
            "primary_url": "https://www.bigbank.com.au/media/jane",
            "source_type": "official",
            "received_at": "2026-03-16T00:00:00Z",
            "message_id": "<jane-1>",
        }
    )
    store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "subject": "Jane Smith joins BigBank digital team",
            "raw_text": "Trade press also reports Jane Smith moved to BigBank.",
            "parsed_candidate_name": "Jane Smith",
            "parsed_candidate_employer": "BigBank",
            "primary_url": "https://news.example.com/jane",
            "source_type": "news",
            "received_at": "2026-03-15T00:00:00Z",
            "message_id": "<jane-2>",
        }
    )
    store.ingest_signal(
        {
            "org_name": "Longboardfella",
            "subject": "Bob Jones shares BigBank strategy panel",
            "raw_text": "Bob Jones and Jane Smith both appeared on a BigBank strategy panel.",
            "parsed_candidate_name": "Bob Jones",
            "parsed_candidate_employer": "OtherBank",
            "received_at": "2026-03-16T01:00:00Z",
            "message_id": "<bob-1>",
        }
    )

    digest = store.generate_digest(
        org_name="Longboardfella",
        profile_keys=[jane_key, bob_key],
        since_ts="2026-03-14T00:00:00Z",
        report_depth="detailed",
    )

    digest_text = (tmp_path / "digests" / f"{digest['digest_id']}.md").read_text(encoding="utf-8")
    assert digest["relationship_intelligence_count"] > 0
    assert digest["top_signal_scores"]
    assert digest["signals"][0]["confidence_band"] in {"High", "Medium", "Low", "User-confirmed"}
    assert "## Relationship Intelligence" in digest_text
    assert "graph shows a 1st-degree LinkedIn path via paul@example.com" in digest_text
    assert "shares graph alumni bridges with the org: McKinsey" in digest_text
    assert "linked in the graph via alumni groups: McKinsey" in digest_text
    assert "## Highest-Scoring Signals" in digest_text


def test_resolve_ollama_model_prefers_qwen35_and_falls_back_to_installed(tmp_path, monkeypatch):
    store = StakeholderSignalStore(base_path=tmp_path)

    monkeypatch.setattr(
        StakeholderSignalStore,
        "_get_ollama_installed_models",
        lambda self: ["qwen2.5:14b-instruct-q4_K_M", "mistral:latest"],
    )
    assert store._resolve_ollama_model("") == "qwen2.5:14b-instruct-q4_K_M"

    monkeypatch.setattr(
        StakeholderSignalStore,
        "_get_ollama_installed_models",
        lambda self: ["qwen3.5:9b", "qwen2.5:14b-instruct-q4_K_M"],
    )
    assert store._resolve_ollama_model("") == "qwen3.5:9b"


def test_ollama_watch_timeout_env_override(monkeypatch):
    monkeypatch.delenv("CORTEX_WATCH_OLLAMA_TIMEOUT", raising=False)
    assert _ollama_watch_timeout_seconds() == 300

    monkeypatch.setenv("CORTEX_WATCH_OLLAMA_TIMEOUT", "450")
    assert _ollama_watch_timeout_seconds() == 450

    monkeypatch.setenv("CORTEX_WATCH_OLLAMA_TIMEOUT", "bad")
    assert _ollama_watch_timeout_seconds() == 300
