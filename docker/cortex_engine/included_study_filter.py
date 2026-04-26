"""Paper-type filter for included-study extraction results.

Supports five UI checkboxes (see pages/7_Document_Extract.py):
    all_clinical       - keep every clinical study (default on; acts as 'no filter')
    rct_only           - keep only randomized/controlled trials
    leukemia_only      - keep only leukemia (any subtype)
    cll_only           - keep only CLL (implies leukemia_only)
    include_economic   - include rows from economic/HTA tables (default off)

The filter produces (keep, drop_reasons) per row. Rows marked `keep=False` remain
in the editor grid with a populated `drop_reasons` field so the user can review and
manually re-enable if the heuristic is wrong.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

DEFAULT_PAPER_FILTERS: Dict[str, bool] = {
    "all_clinical": True,
    "rct_only": False,
    "leukemia_only": False,
    "cll_only": False,
    "include_economic": False,
}

_RCT_PATTERNS = (
    re.compile(r"\brct\b"),
    re.compile(r"\brcts\b"),
    re.compile(r"\brandomis(?:ed|ation)\b"),
    re.compile(r"\brandomiz(?:ed|ation)\b"),
    re.compile(r"\bcontrolled\s+trial\b"),
    re.compile(r"\bclinical\s+trial\b"),
    re.compile(r"\bphase\s+(?:i|ii|iii|iv|1|2|3|4)\b"),
    re.compile(r"\bcrossover\b"),
    re.compile(r"\bcross[- ]over\b"),
)

_NON_RCT_DESIGN_PATTERNS = (
    re.compile(r"\bretrospective\b"),
    re.compile(r"\bobservational\b"),
    re.compile(r"\bcohort\b"),
    re.compile(r"\bcase[- ]series\b"),
    re.compile(r"\bcase[- ]report\b"),
    re.compile(r"\bcase[- ]control\b"),
    re.compile(r"\bcross[- ]section\w*"),
    re.compile(r"\breview\b"),
    re.compile(r"\bregistry\b"),
    re.compile(r"\bsurvey\b"),
    re.compile(r"\bqualitative\b"),
)

_LEUKEMIA_PATTERNS = (
    re.compile(r"\bleuk[ae]mi(?:a|as|c)\b"),
    re.compile(r"\bleuk[ae]mi\w+"),
    re.compile(r"\baml\b"),
    re.compile(r"\bcml\b"),
    re.compile(r"\bcll\b"),
    re.compile(r"\bsll\b"),
    re.compile(r"\bhcl\b"),
    re.compile(r"\bmds(?:/aml)?\b"),
    re.compile(r"\bacute\s+myel\w+"),
    re.compile(r"\bacute\s+lymph\w+"),
    re.compile(r"\bchronic\s+myel\w+"),
    re.compile(r"\bchronic\s+lymph\w+"),
    re.compile(r"\bhairy\s+cell\b"),
    re.compile(r"\bt[- ]cell\s+leuk\w+"),
    re.compile(r"\bb[- ]cell\s+leuk\w+"),
    re.compile(r"\bapl\b"),
)

_CLL_PATTERNS = (
    re.compile(r"\bcll\b"),
    re.compile(r"\bcll/sll\b"),
    re.compile(r"\bchronic\s+lymphocyt\w+"),
    re.compile(r"\bchronic\s+lymphoid\s+leuk\w+"),
    re.compile(r"\bsll\b"),
    re.compile(r"\bsmall\s+lymphocyt\w+"),
)

_ECONOMIC_PATTERNS = (
    re.compile(r"\beconomic\b"),
    re.compile(r"\bcost[- ]effectiv\w+"),
    re.compile(r"\bcost[- ]utilit\w+"),
    re.compile(r"\bcost[- ]benefit\b"),
    re.compile(r"\bhealth\s+economic\b"),
    re.compile(r"\bhta\b"),
    re.compile(r"\bicer\b"),
    re.compile(r"\bqaly\b"),
    re.compile(r"\bmarkov\s+model\b"),
    re.compile(r"\bbudget\s+impact\b"),
)


def _row_text_blob(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    for field in (
        "table_title",
        "grouping_basis",
        "group_label",
        "trial_label",
        "combined_group",
        "title",
        "citation_display",
        "journal",
        "notes",
        "bibliography_entry_text",
        "study_design",
        "outcome_measure",
        "outcome_result",
    ):
        value = row.get(field)
        if value:
            parts.append(str(value))
    return " ".join(parts).lower()


def _row_design_blob(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    for field in ("study_design", "trial_label", "notes"):
        value = row.get(field)
        if value:
            parts.append(str(value))
    return " ".join(parts).lower()


def _matches_any(patterns: Iterable[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def normalize_filters(filters: Dict[str, Any] | None) -> Dict[str, bool]:
    merged = dict(DEFAULT_PAPER_FILTERS)
    for key, value in dict(filters or {}).items():
        if key in merged:
            merged[key] = bool(value)
    if merged["cll_only"]:
        merged["leukemia_only"] = True
    return merged


def classify_row(row: Dict[str, Any], filters: Dict[str, Any] | None) -> Dict[str, Any]:
    """Evaluate a single row against the filters.

    Returns a dict: {keep: bool, drop_reasons: list[str], signals: dict[str, bool]}.
    Does not mutate the input row.
    """
    normalized = normalize_filters(filters)
    blob = _row_text_blob(row)
    design_blob = _row_design_blob(row)

    is_rct = _matches_any(_RCT_PATTERNS, design_blob) or _matches_any(_RCT_PATTERNS, blob)
    is_non_rct_design = _matches_any(_NON_RCT_DESIGN_PATTERNS, design_blob)
    is_leukemia = _matches_any(_LEUKEMIA_PATTERNS, blob)
    is_cll = _matches_any(_CLL_PATTERNS, blob)
    is_economic = _matches_any(_ECONOMIC_PATTERNS, blob)

    drop_reasons: List[str] = []

    if normalized["rct_only"]:
        if not is_rct or (is_non_rct_design and not is_rct):
            drop_reasons.append("not_rct")

    if normalized["cll_only"]:
        if not is_cll:
            drop_reasons.append("not_cll")
    elif normalized["leukemia_only"]:
        if not is_leukemia:
            drop_reasons.append("not_leukemia")

    if not normalized["include_economic"] and is_economic and not is_leukemia and not is_cll:
        # Only drop pure economic rows if no clinical disease signal is also present.
        drop_reasons.append("economic_excluded")

    return {
        "keep": not drop_reasons,
        "drop_reasons": drop_reasons,
        "signals": {
            "is_rct": is_rct,
            "is_non_rct_design": is_non_rct_design,
            "is_leukemia": is_leukemia,
            "is_cll": is_cll,
            "is_economic": is_economic,
        },
    }


def apply_paper_filters(
    rows: List[Dict[str, Any]],
    filters: Dict[str, Any] | None,
    *,
    respect_existing_drop: bool = True,
) -> List[Dict[str, Any]]:
    """Return a new list of rows with `keep` and `drop_reasons` updated.

    When `respect_existing_drop` is True, rows the user has already unchecked in the editor
    stay unchecked (we only ever *add* filter-driven drops; we never re-enable a manual drop).
    """
    normalized = normalize_filters(filters)
    updated: List[Dict[str, Any]] = []
    for row in rows or []:
        new_row = dict(row)
        verdict = classify_row(new_row, normalized)
        filter_reasons = list(verdict["drop_reasons"])
        filter_keep = bool(verdict["keep"])

        user_keep = bool(new_row.get("keep", True)) if respect_existing_drop else True
        final_keep = user_keep and filter_keep
        new_row["keep"] = final_keep
        new_row["drop_reasons"] = ", ".join(filter_reasons)
        updated.append(new_row)
    return updated


def build_prompt_filter_hint(filters: Dict[str, Any] | None) -> str:
    """Compose a short instruction fragment to append to the extractor LLM prompt."""
    normalized = normalize_filters(filters)
    clauses: List[str] = []
    if normalized["rct_only"]:
        clauses.append(
            "Include only randomized or controlled clinical trials; treat observational, cohort, "
            "case series, registry, or narrative-only studies as non-trial and flag them "
            "with needs_review=true and notes explaining the design mismatch."
        )
    if normalized["cll_only"]:
        clauses.append(
            "Only return rows where the population is chronic lymphocytic leukemia (CLL) or "
            "small lymphocytic lymphoma (SLL); flag non-CLL rows with needs_review=true and "
            "note the disease."
        )
    elif normalized["leukemia_only"]:
        clauses.append(
            "Only return rows where the population is any form of leukemia (AML, ALL, CML, CLL, "
            "HCL, MDS/AML, acute myeloid, acute lymphoblastic, chronic myeloid, chronic "
            "lymphocytic, hairy cell); flag non-leukemia rows with needs_review=true and note the disease."
        )
    if not normalized["include_economic"]:
        clauses.append(
            "Do not return rows from purely economic or health-technology-assessment (HTA) "
            "tables (cost-effectiveness, cost-utility, QALY, Markov, budget impact) unless the "
            "same row is also explicitly tied to a clinical trial."
        )
    if not clauses:
        return ""
    return "Additional filtering constraints: " + " ".join(clauses)
