from __future__ import annotations

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

    def fake_synth(self, raw_data, org_name, provider, model):
        assert org_name == "Longboardfella"
        assert provider == "ollama"
        assert model == "qwen2.5:14b"
        assert "Carolyn Bell" in raw_data
        assert "Silverchain expands community care services" not in raw_data
        return "# WATCH Report\n\n## Individual Updates\n- Carolyn Bell moved roles.", "qwen2.5:14b"

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
    assert digest["llm_model"] == "qwen2.5:14b"
    assert "# WATCH Report" in digest_text
    assert "Carolyn Bell moved roles." in digest_text


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
