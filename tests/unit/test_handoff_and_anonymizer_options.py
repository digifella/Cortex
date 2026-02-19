from __future__ import annotations

from cortex_engine.anonymizer import (
    AnonymizationMapping,
    AnonymizationOptions,
    DocumentAnonymizer,
)
from cortex_engine.entity_extractor import ExtractedEntity
from cortex_engine.handoff_contract import (
    HANDOFF_CONTRACT_VERSION,
    normalize_handoff_metadata,
)


class _DummyEntityExtractor:
    def extract_entities_and_relationships(self, text, metadata):
        return [
            ExtractedEntity(name="Alice Jones", entity_type="person", confidence=0.9),
            ExtractedEntity(name="Acme Corp", entity_type="organization", confidence=0.9),
        ], []


def _build_anonymizer(monkeypatch):
    monkeypatch.setattr("cortex_engine.anonymizer.EntityExtractor", _DummyEntityExtractor)
    return DocumentAnonymizer()


def test_pronoun_redaction_toggle(monkeypatch):
    anonymizer = _build_anonymizer(monkeypatch)
    mapping = AnonymizationMapping()
    text = "He said she helped them."

    redacted = anonymizer.anonymize_text(
        text=text,
        entities=[],
        mapping=mapping,
        options=AnonymizationOptions(redact_personal_pronouns=True),
    )
    assert redacted.count("[PRONOUN]") == 3

    unchanged = anonymizer.anonymize_text(
        text=text,
        entities=[],
        mapping=AnonymizationMapping(),
        options=AnonymizationOptions(redact_personal_pronouns=False),
    )
    assert "[PRONOUN]" not in unchanged
    assert "He said she helped them." in unchanged


def test_custom_company_name_redaction(monkeypatch):
    anonymizer = _build_anonymizer(monkeypatch)
    mapping = AnonymizationMapping()
    text = "Longboardfella Consulting Pty Ltd delivered the milestone."

    output = anonymizer.anonymize_text(
        text=text,
        entities=[],
        mapping=mapping,
        options=AnonymizationOptions(
            redact_company_names=True,
            custom_company_names=["Longboardfella Consulting Pty Ltd"],
        ),
    )
    assert "Company 1" in output
    assert "Longboardfella Consulting Pty Ltd" not in output


def test_entity_type_toggle_filters_organizations(monkeypatch):
    anonymizer = _build_anonymizer(monkeypatch)
    monkeypatch.setattr(anonymizer, "extract_names_with_comprehensive_patterns", lambda text: [])
    monkeypatch.setattr(anonymizer, "llm_powered_entity_detection", lambda text: [])
    monkeypatch.setattr(anonymizer, "post_process_missed_entities", lambda text, existing: [])
    anonymizer.nlp = None

    entities = anonymizer.identify_entities_for_anonymization(
        text="Alice Jones worked with Acme Corp.",
        filename="sample.txt",
        confidence_threshold=0.1,
        options=AnonymizationOptions(redact_organizations=False),
    )

    entity_types = {e.entity_type for e in entities}
    assert "person" in entity_types
    assert "organization" not in entity_types


def test_handoff_metadata_normalization():
    metadata = normalize_handoff_metadata(
        job={
            "source_system": "website",
            "idempotency_key": "idem-1",
            "tenant_id": "tenant-a",
            "project_id": "project-alpha",
        },
        input_data={"trace_id": "trace-explicit"},
    )
    assert metadata["contract_version"] == HANDOFF_CONTRACT_VERSION
    assert metadata["trace_id"] == "trace-explicit"
    assert metadata["idempotency_key"] == "idem-1"
    assert metadata["source_system"] == "website"
    assert metadata["tenant_id"] == "tenant-a"
    assert metadata["project_id"] == "project-alpha"

    generated = normalize_handoff_metadata(job={}, input_data={})
    assert generated["trace_id"].startswith("trace-")
    assert generated["tenant_id"] == "default"
    assert generated["project_id"] == "default"


def test_preserve_source_formatting_normalization(monkeypatch):
    anonymizer = _build_anonymizer(monkeypatch)
    raw = (
        "This is a wrapped line<CR>\n"
        "that should flow into paragraph<CRLF>\n"
        "\n"
        "- bullet item one\n"
        "- bullet item two\n"
    )
    cleaned = anonymizer._normalize_extracted_text(raw)
    assert "<CR>" not in cleaned
    assert "<CRLF>" not in cleaned
    assert "This is a wrapped line that should flow into paragraph" in cleaned
    assert "- bullet item one" in cleaned
    assert "- bullet item two" in cleaned
