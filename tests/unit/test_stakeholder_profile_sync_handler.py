from worker.handlers import stakeholder_profile_sync


def test_stakeholder_profile_sync_preserves_existing_org_strategic_profile_when_omitted(tmp_path, monkeypatch):
    from cortex_engine.stakeholder_signal_store import StakeholderSignalStore

    store = StakeholderSignalStore(base_path=tmp_path)
    store.upsert_profiles(
        org_name="Escient",
        profiles=[
            {
                "canonical_name": "Escient",
                "target_type": "organisation",
            }
        ],
        org_strategic_profile={
            "description": "Digital transformation consultancy",
            "priority_industries": ["Water Utilities"],
            "key_themes": ["Technology Strategy & Enablement"],
            "strategic_objectives": ["Win technology strategy work"],
        },
        source="seed",
    )

    monkeypatch.setattr(stakeholder_profile_sync, "StakeholderSignalStore", lambda: store)

    result = stakeholder_profile_sync.handle(
        input_path=None,
        input_data={
            "org_name": "Escient",
            "profiles": [
                {
                    "canonical_name": "Jane Example",
                    "target_type": "person",
                    "current_role": "Partner",
                }
            ],
            "source_system": "market_radar",
        },
        job={},
    )

    context = store.get_org_context("Escient")

    assert result["output_data"]["status"] == "synced"
    assert result["output_data"]["org_strategic_industry_count"] == 0
    assert context["org_strategic_profile"]["priority_industries"] == ["Water Utilities"]
    assert context["org_strategic_profile"]["key_themes"] == ["Technology Strategy & Enablement"]

