"""Unit tests for cortex_engine.included_study_filter."""

from __future__ import annotations

from cortex_engine.included_study_filter import (
    DEFAULT_PAPER_FILTERS,
    apply_paper_filters,
    build_prompt_filter_hint,
    classify_row,
    normalize_filters,
)


def _row(**overrides):
    base = {
        "keep": True,
        "table_number": "2",
        "table_title": "Included studies in DLBCL patients",
        "group_label": "",
        "trial_label": "",
        "combined_group": "",
        "title": "",
        "citation_display": "",
        "journal": "",
        "notes": "",
        "bibliography_entry_text": "",
        "study_design": "",
        "outcome_measure": "",
        "outcome_result": "",
    }
    base.update(overrides)
    return base


def test_defaults_drop_economic_rows_and_keep_clinical_rows():
    rows = [
        _row(title="Greil 1999", study_design="RCT"),
        _row(
            table_title="Cost-effectiveness analyses of rituximab",
            title="Ferrara 2010",
            notes="ICER reported per QALY",
        ),
    ]
    filtered = apply_paper_filters(rows, DEFAULT_PAPER_FILTERS)
    assert filtered[0]["keep"] is True
    assert filtered[0]["drop_reasons"] == ""
    assert filtered[1]["keep"] is False
    assert "economic_excluded" in filtered[1]["drop_reasons"]


def test_include_economic_keeps_economic_rows():
    rows = [
        _row(
            table_title="Economic studies",
            title="Ferrara 2010",
            notes="Cost-utility",
        ),
    ]
    filters = {**DEFAULT_PAPER_FILTERS, "include_economic": True}
    filtered = apply_paper_filters(rows, filters)
    assert filtered[0]["keep"] is True


def test_rct_only_drops_observational():
    rows = [
        _row(title="Smith 2014", study_design="Retrospective cohort"),
        _row(title="Jones 2018", study_design="Randomized controlled trial"),
    ]
    filters = {**DEFAULT_PAPER_FILTERS, "rct_only": True}
    filtered = apply_paper_filters(rows, filters)
    assert filtered[0]["keep"] is False
    assert "not_rct" in filtered[0]["drop_reasons"]
    assert filtered[1]["keep"] is True


def test_rct_only_keeps_phase3_trial():
    rows = [_row(title="X", study_design="Phase III trial")]
    filters = {**DEFAULT_PAPER_FILTERS, "rct_only": True}
    filtered = apply_paper_filters(rows, filters)
    assert filtered[0]["keep"] is True


def test_leukemia_only_drops_non_leukemia_diseases():
    rows = [
        _row(table_title="Included studies in acute myeloid leukaemia", title="Row A"),
        _row(table_title="Included studies in diffuse large B-cell lymphoma", title="Row B"),
        _row(table_title="Included studies in breast cancer", title="Row C"),
    ]
    filters = {**DEFAULT_PAPER_FILTERS, "leukemia_only": True}
    filtered = apply_paper_filters(rows, filters)
    assert filtered[0]["keep"] is True
    assert filtered[1]["keep"] is False
    assert filtered[2]["keep"] is False
    assert "not_leukemia" in filtered[1]["drop_reasons"]


def test_cll_only_implies_leukemia_and_drops_other_leukemias():
    rows = [
        _row(
            table_title="Included studies in chronic lymphocytic leukaemia",
            title="Row A",
        ),
        _row(table_title="Included studies in acute myeloid leukaemia", title="Row B"),
        _row(table_title="Included studies in DLBCL", title="Row C"),
    ]
    filters = {**DEFAULT_PAPER_FILTERS, "cll_only": True}
    filtered = apply_paper_filters(rows, filters)
    assert filtered[0]["keep"] is True
    assert filtered[1]["keep"] is False
    assert "not_cll" in filtered[1]["drop_reasons"]
    assert filtered[2]["keep"] is False


def test_apply_paper_filters_respects_user_existing_drop():
    rows = [
        _row(keep=False, title="Already dropped", study_design="RCT"),
    ]
    filtered = apply_paper_filters(rows, DEFAULT_PAPER_FILTERS)
    assert filtered[0]["keep"] is False


def test_normalize_filters_cll_implies_leukemia():
    normalized = normalize_filters({"cll_only": True})
    assert normalized["leukemia_only"] is True


def test_build_prompt_filter_hint_empty_when_default_no_gates():
    hint = build_prompt_filter_hint(
        {**DEFAULT_PAPER_FILTERS, "include_economic": True}
    )
    assert hint == ""


def test_build_prompt_filter_hint_rct_and_leukemia():
    hint = build_prompt_filter_hint(
        {**DEFAULT_PAPER_FILTERS, "rct_only": True, "leukemia_only": True}
    )
    assert "randomized" in hint.lower()
    assert "leukemia" in hint.lower()


def test_classify_row_signals_reflect_matches():
    row = _row(
        table_title="Cost-effectiveness in chronic lymphocytic leukaemia",
        study_design="Randomized controlled trial",
    )
    verdict = classify_row(row, {**DEFAULT_PAPER_FILTERS, "include_economic": True})
    signals = verdict["signals"]
    assert signals["is_rct"] is True
    assert signals["is_cll"] is True
    assert signals["is_leukemia"] is True
    assert signals["is_economic"] is True


def test_economic_row_with_cll_signal_survives_default_filters():
    # Rationale: a CLL trial appearing in an econ table should not be blindly dropped.
    rows = [
        _row(
            table_title="Economic studies in chronic lymphocytic leukaemia",
            title="Greil 1999",
            study_design="RCT",
        ),
    ]
    filtered = apply_paper_filters(rows, DEFAULT_PAPER_FILTERS)
    assert filtered[0]["keep"] is True
