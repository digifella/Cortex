from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import unquote

from cortex_engine.stakeholder_signal_matcher import normalize_lookup

_THEME_MARKERS = ("priority", "strategic", "pillar", "focus area", "objective", "goal", "direction")
_INITIATIVE_MARKERS = ("initiative", "program", "programme", "project", "roadmap", "transformation", "reform")
_NOISY_THEME_MARKERS = (
    "2026 to 2030 strategic direction",
    "te ahunga rautaki",
    "our strategic focus",
    "four strategic focus areas",
    "our strategic priorities",
    "strategic direction |",
)
_STRATEGY_HEADING_STOPWORDS = {
    "acknowledgement of country",
    "acknowledgement of traditional custodians",
    "message from the managing director",
    "our drivers for change",
    "strategic and operating context",
    "our 2030 strategy",
    "our values",
    "our purpose",
    "language statement",
    "contents",
    "creating a brighter future",
    "yarra valley water 2030 strategy",
}
_INLINE_STRATEGY_SENTENCE_TOKENS = {
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "can",
    "cannot",
    "could",
    "will",
    "would",
    "should",
    "since",
    "depends",
    "told",
}
_GENERIC_DOC_TOKENS = {
    "attachment",
    "compressed",
    "copy",
    "direction",
    "doc",
    "document",
    "draft",
    "file",
    "final",
    "image",
    "img",
    "outlook",
    "pdf",
    "picture",
    "plan",
    "report",
    "scan",
    "screenshot",
    "screen",
    "shot",
    "strategic",
    "v",
    "version",
    "website",
}
_GENERATED_ATTACHMENT_RE = re.compile(
    r"^(?:screenshot|screen shot|img|image|photo|picture|scan|attachment|document|file|outlook)(?:[\s_-]+\d.*)?$",
    re.IGNORECASE,
)
_STRATEGIC_SIGNAL_SPECS = (
    {
        "headline": "Membership base is changing",
        "category": "member_base",
        "markers": (
            "our members are changing",
            "women now outnumber men",
            "current trainees are women",
            "cultural diversity continues to grow",
        ),
    },
    {
        "headline": "Member trust and value-for-money pressure",
        "category": "member_pressure",
        "markers": (
            "member satisfaction has been declining since 2016",
            "value for money",
            "complexity, communication and responsiveness",
            "worth their investment of time",
            "frustration with complexity",
        ),
    },
    {
        "headline": "Digital-first service transformation",
        "category": "digital_transformation",
        "markers": (
            "digital-first is now the baseline",
            "digital capability is foundational",
            "digitally enabled environments",
            "unified digital ecosystem",
            "data and technology",
            "digital services are intuitive",
        ),
    },
    {
        "headline": "Operating model simplification",
        "category": "operating_model",
        "markers": (
            "delivery has become too complex",
            "simplifying how the college works",
            "simplifying and connecting the college",
            "one connected organisation",
            "simplify our operations",
            "clearer in its priorities",
        ),
    },
    {
        "headline": "Revenue diversification and financial sustainability",
        "category": "financial_sustainability",
        "markers": (
            "membership fees cannot increase without limit",
            "diversified revenue",
            "financial sustainability",
            "new revenue",
            "careful steward of member funds",
        ),
    },
    {
        "headline": "Workforce sustainability and workplace pressure",
        "category": "workforce",
        "markers": (
            "workforce sustainability",
            "workforce shortages",
            "unsafe workplaces",
            "wellbeing of physicians",
            "safe, inclusive workplaces",
            "workforce planning",
        ),
    },
    {
        "headline": "First Nations and cultural commitments",
        "category": "community_commitment",
        "markers": (
            "core business of the racp",
            "advancing aboriginal, torres strait islander and maori health and education as core business",
            "cultural safety",
            "indigenous strategic framework",
            "self-determination",
            "indigenous leadership",
        ),
    },
    {
        "headline": "Partnership and policy influence expansion",
        "category": "partnerships",
        "markers": (
            "strategic partnerships",
            "governments, health services and communities",
            "policy, advocacy, and system leadership",
            "voice of physicians",
            "grow influence, reach, and resilience",
        ),
    },
    {
        "headline": "Execution discipline and data-led governance",
        "category": "execution",
        "markers": (
            "clear accountability",
            "regular review of priorities",
            "objectives and key results",
            "disciplined execution",
            "evidence driven action",
            "test and learn before scaling",
        ),
    },
    {
        "headline": "Operational transformation and resilience",
        "category": "operational_transformation",
        "markers": (
            "operational transformation",
            "operational resilience",
            "flood operations centre",
            "dam operations",
            "water security",
            "continuity of water supply",
        ),
    },
    {
        "headline": "Capital delivery and asset transformation",
        "category": "capital_delivery",
        "markers": (
            "capital delivery transformation",
            "capital program",
            "asset information and management",
            "dam improvement",
            "invested over",
            "asset maintenance services",
        ),
    },
    {
        "headline": "Stakeholder, customer and community engagement",
        "category": "stakeholder_engagement",
        "markers": (
            "engaging with stakeholders",
            "working with customers",
            "engaging with the community",
            "shareholder expectations and social licence",
            "retailer customers",
            "community recreation facilities",
        ),
    },
    {
        "headline": "Technology and cyber enablement",
        "category": "technology_enablement",
        "markers": (
            "use of technology as an enabler",
            "process improvement and use of technology as an enabler",
            "information systems, record keeping and cyber security",
            "cyber security",
            "digital capability",
            "data and technology",
        ),
    },
)
_PERSON_PREFIXES = ("dr", "professor", "associate professor", "mr", "mrs", "ms", "miss", "sir", "dame")
_PERSON_CREDENTIAL_TOKENS = {
    "frcp",
    "fracp",
    "fracs",
    "phd",
    "ao",
    "am",
    "mbbs",
    "md",
    "msc",
    "gaicd",
    "faidh",
    "chia",
    "afhea",
}
_NON_PERSON_NAME_TOKENS = {
    "attendance",
    "board",
    "capabilities",
    "centre",
    "center",
    "committee",
    "council",
    "cpd",
    "current",
    "directors",
    "education",
    "framework",
    "governance",
    "initiative",
    "initiatives",
    "message",
    "opceo",
    "policy",
    "key",
    "management",
    "personnel",
    "program",
    "programme",
    "reform",
    "review",
    "strategy",
    "support",
    "term",
    "transformation",
}
_ROLE_KEYWORDS = (
    "president",
    "chair",
    "chief executive officer",
    "chief executive",
    "ceo",
    "director",
    "board",
    "executive",
    "dean",
    "lead",
)
_ROLE_LINE_EXCLUSION_MARKERS = (
    "attendance",
    "current directors",
    "term ceased",
    "the following table",
    "our board approved",
    "approved a new education",
)
_ORG_NAME_STOPWORDS = {"a", "an", "and", "at", "for", "in", "of", "on", "or", "the", "to", "with"}
_PERFORMANCE_INDICATOR_SPECS = (
    {
        "label": "Annual revenue",
        "category": "financial_performance",
        "patterns": (
            r"Revenue for the year increased to \$(\d+(?:\.\d+)?)\s+million,\s+up from \$(\d+(?:\.\d+)?)\s+million",
        ),
        "value_template": "${0} million (from ${1} million)",
    },
    {
        "label": "Net result",
        "category": "financial_performance",
        "patterns": (
            r"contributing towards a \$(\d+(?:\.\d+)?)\s+million\s+net surplus",
        ),
        "value_template": "${0} million net surplus",
    },
    {
        "label": "Capital works program",
        "category": "capital_investment",
        "patterns": (
            r"delivered a \$(\d+(?:\.\d+)?)\s+million capital works program",
            r"We delivered \$(\d+(?:\.\d+)?)\s+million of capital and related infrastructure works",
        ),
        "value_template": "${0} million",
    },
    {
        "label": "Total assets",
        "category": "financial_capacity",
        "patterns": (
            r"total assets reached \$(\d+(?:\.\d+)?)\s+billion",
        ),
        "value_template": "${0} billion",
    },
    {
        "label": "Total debt",
        "category": "financial_capacity",
        "patterns": (
            r"Total debt increased by \$(\d+(?:\.\d+)?)\s+million to \$(\d+(?:\.\d+)?)\s+million",
        ),
        "value_template": "${1} million total debt",
    },
    {
        "label": "Cash position",
        "category": "financial_capacity",
        "patterns": (
            r"cash and cash equivalents increasing by \$(\d+(?:\.\d+)?)\s+million to \$(\d+(?:\.\d+)?)\s+million",
        ),
        "value_template": "${1} million cash",
    },
    {
        "label": "Water recycling rate",
        "category": "operational_performance",
        "patterns": (
            r"We achieved (\d+(?:\.\d+)?)%\s+water recycling",
            r"recycling rate rose to more than (\d+(?:\.\d+)?)%",
        ),
        "value_template": "{0}%",
    },
    {
        "label": "Membership scale",
        "category": "member_scale",
        "patterns": (
            r"(\d{1,3}(?:,\d{3})+)\s+RACP Members Overview",
            r"(\d{1,3}(?:,\d{3})+)\s+All Members",
            r"(\d{1,3}(?:,\d{3})+)[-\s]member College",
        ),
        "value_template": "{0} members",
    },
    {
        "label": "Annual member growth",
        "category": "member_growth",
        "patterns": (
            r"Annual Growth\s+[▲+]?\s*[\d,]+\s*\((\d+(?:\.\d+)?)%\)",
            r"(\d+(?:\.\d+)?)%\s+year on year increase in members engagement",
        ),
        "value_template": "{0}%",
    },
    {
        "label": "Member engagement uplift",
        "category": "engagement",
        "patterns": (
            r"(\d+(?:\.\d+)?)%\s+year on year increase in members engagement with CPD content",
            r"(\d+(?:\.\d+)?)%\s+increase in member feedback on the 2025 CPD Framework changes",
        ),
        "value_template": "{0}%",
    },
    {
        "label": "Workforce pressure on members",
        "category": "workforce",
        "patterns": (
            r"(\d+(?:\.\d+)?)%\s+members say in survey that workforce pressures affect personal lives",
            r"(\d+)\s+per cent of members told us workforce pressures severely impact their personal and family lives",
        ),
        "value_template": "{0}%",
    },
    {
        "label": "Policy and advocacy reach",
        "category": "advocacy",
        "patterns": (
            r"(\d+)\s+RACP and member policy statements,\s*submissions,\s*endorsements",
        ),
        "value_template": "{0} policy actions",
    },
    {
        "label": "Stakeholder engagement",
        "category": "engagement",
        "patterns": (
            r"(\d+)\s+meetings with MPs and other key stakeholders",
            r"(\d+)\s+member consultation workshops",
        ),
        "value_template": "{0} engagements",
    },
    {
        "label": "Operating revenue growth",
        "category": "financial_performance",
        "patterns": (
            r"provision of services increased from \$(\d+(?:\.\d+)?)m to\s+\$(\d+(?:\.\d+)?)m",
        ),
        "value_template": "${0}m to ${1}m",
    },
    {
        "label": "Total revenue growth",
        "category": "financial_performance",
        "patterns": (
            r"total revenue and other income(?: for the year 2024)? increased from \$(\d+(?:\.\d+)?)m to\s+\$(\d+(?:\.\d+)?)m",
        ),
        "value_template": "${0}m to ${1}m",
    },
    {
        "label": "Operating result",
        "category": "financial_performance",
        "patterns": (
            r"The deficit of \$(\d+(?:\.\d+)?)m \(\$(\d+(?:\.\d+)?)m 2023\)",
        ),
        "value_template": "Deficit ${0}m",
    },
    {
        "label": "Restricted cash position",
        "category": "financial_capacity",
        "patterns": (
            r"Cash and cash equivalents include \$(\d[\d,]+)",
        ),
        "value_template": "${0}",
    },
    {
        "label": "CPD Home accreditation",
        "category": "accreditation",
        "patterns": (
            r"achieved a ([A-Za-z-]+) accreditation as a CPD Home",
            r"re-accredited as a training provider and as a CPD Home",
        ),
        "value_template": "{0}",
    },
    {
        "label": "CPD compliance",
        "category": "compliance",
        "patterns": (
            r"(\d+)\s+per cent of our members completing their CPD requirements",
        ),
        "value_template": "{0}%",
    },
    {
        "label": "Population served",
        "category": "service_scale",
        "patterns": (
            r"(\d+(?:\.\d+)?)m\s+Population served",
            r"servicing\s+(\d+(?:\.\d+)?)\s*million\s+Victorians",
            r"more than\s+(\d+(?:\.\d+)?)\s+million(?:\s+[A-Za-z]+)?\s+residents",
        ),
        "value_template": "{0}m",
    },
    {
        "label": "Infrastructure asset base",
        "category": "asset_base",
        "patterns": (
            r"\$(\d+(?:\.\d+)?)b\s+Infrastructure asset base",
            r"\$(\d+(?:\.\d+)?)\s*billion worth of assets",
        ),
        "value_template": "${0}b",
    },
    {
        "label": "Business customers",
        "category": "service_scale",
        "patterns": (
            r"(\d+)k\s+Business customers",
            r"(\d{1,3}(?:,\d{3})+)\s+business customers",
        ),
        "value_template": "{0}k",
    },
    {
        "label": "Water delivered",
        "category": "service_delivery",
        "patterns": (
            r"delivered\s+(\d+)\s+billion litres of water",
        ),
        "value_template": "{0} billion litres",
    },
    {
        "label": "Bulk water supplied",
        "category": "service_delivery",
        "patterns": (
            r"supplied approximately\s+(\d{1,3}(?:,\d{3})*)\s+megalitres",
        ),
        "value_template": "{0} ML",
    },
    {
        "label": "Drinking water compliance",
        "category": "service_quality",
        "patterns": (
            r"all\s+(\d+(?:\.\d+)?)%\s+compliant with Victorian Safe Drinking Water Regulations",
        ),
        "value_template": "{0}%",
    },
    {
        "label": "Financial hardship customers",
        "category": "customer_support",
        "patterns": (
            r"managing\s+(\d{1,3}(?:,\d{3})+)\s+customers experiencing financial hardship",
        ),
        "value_template": "{0}",
    },
    {
        "label": "Customer satisfaction",
        "category": "customer_experience",
        "patterns": (
            r"(\d+(?:\.\d+)?)%\s+of customers were satisfied with their most recent interaction",
        ),
        "value_template": "{0}%",
    },
    {
        "label": "Service restoration within 4 hours",
        "category": "service_reliability",
        "patterns": (
            r"restored services within 4 hours for\s+(\d+(?:\.\d+)?)%\s+of customers",
        ),
        "value_template": "{0}%",
    },
    {
        "label": "Service restoration within 12 hours",
        "category": "service_reliability",
        "patterns": (
            r"within 12 hours for\s+(\d+(?:\.\d+)?)%\s+of customers",
        ),
        "value_template": "{0}%",
    },
    {
        "label": "Fault handling satisfaction",
        "category": "customer_experience",
        "patterns": (
            r"achieved\s+(\d+(?:\.\d+)?)%\s+customer satisfaction with fault call handling",
        ),
        "value_template": "{0}%",
    },
    {
        "label": "Field response satisfaction",
        "category": "customer_experience",
        "patterns": (
            r"and\s+(\d+(?:\.\d+)?)%\s+satisfaction with our field response",
        ),
        "value_template": "{0}%",
    },
    {
        "label": "Community rebate recommended",
        "category": "customer_value",
        "patterns": (
            r"return\s+\$(\d+(?:\.\d+)?)\s+million to customers and community",
        ),
        "value_template": "${0} million",
    },
    {
        "label": "Renewable electricity usage",
        "category": "sustainability",
        "patterns": (
            r"renewable electricity powered\s+(\d+(?:\.\d+)?)%\s+of our operations",
            r"will use\s+(\d+(?:\.\d+)?)%\s+renewable electricity",
        ),
        "value_template": "{0}%",
    },
    {
        "label": "Infrastructure investment",
        "category": "capital_investment",
        "patterns": (
            r"invested over\s+\$(\d+(?:\.\d+)?)\s+million in infrastructure",
            r"invested\s+\$(\d+(?:\.\d+)?)\s+million in infrastructure",
        ),
        "value_template": "${0} million",
    },
    {
        "label": "Water storage level",
        "category": "water_security",
        "patterns": (
            r"combined storage level of\s+(\d+(?:\.\d+)?)%",
            r"storage levels remained above\s+(\d+(?:\.\d+)?)%",
        ),
        "value_template": "{0}%",
    },
    {
        "label": "Recreational visitors",
        "category": "community_engagement",
        "patterns": (
            r"welcomed\s+(\d+(?:\.\d+)?)\s+million recreational visitors",
        ),
        "value_template": "{0} million",
    },
    {
        "label": "Flood event activations",
        "category": "operational_resilience",
        "patterns": (
            r"Flood Operations Centre was activated\s+(\d+)\s+times",
        ),
        "value_template": "{0} activations",
    },
)
_PROJECT_AMOUNT_RE = re.compile(
    r"(?:and\s+the\s+|and\s+|the\s+)?([A-Z][A-Za-z0-9&'’()./\- ]{2,90}?)\s*\(\$(\d+(?:\.\d+)?)\s*(million|billion|m|bn)\)",
    re.IGNORECASE,
)


def _read_document_text(attachment: Dict[str, Any]) -> str:
    path_text = str(attachment.get("stored_path") or "").strip()
    excerpt = str(attachment.get("excerpt") or "").strip()
    if not path_text:
        return excerpt
    path_obj = Path(path_text)
    if not path_obj.exists():
        return excerpt
    suffix = path_obj.suffix.lower()
    if suffix == ".pdf" and shutil.which("pdftotext"):
        try:
            result = subprocess.run(
                ["pdftotext", str(path_obj), "-"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0 and str(result.stdout or "").strip():
                return str(result.stdout).strip()
        except Exception:
            return excerpt
    return excerpt


def _compact_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"[\x00-\x1f]+", " ", str(raw_line or ""))
        line = (
            line.replace("Ƽ", "fi")
            .replace("ƽ", "fl")
            .replace("ƾ", "ffi")
            .replace("ﬀ", "ff")
            .replace("ﬁ", "fi")
            .replace("ﬂ", "fl")
            .replace("–", "-")
        )
        line = re.sub(r"\s+", " ", line.strip())
        if not line:
            continue
        lines.append(line)
    repaired: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            index + 1 < len(lines)
            and re.match(r"^[a-z'’]", line)
            and re.fullmatch(r"[A-Z]", lines[index + 1])
        ):
            line = lines[index + 1] + line
            index += 1
        repaired.append(line)
        index += 1
    return repaired


def _extract_label_value(lines: List[str], label: str) -> str:
    lowered_label = normalize_lookup(label)
    for index, line in enumerate(lines):
        lowered = normalize_lookup(line)
        if lowered == lowered_label and index + 1 < len(lines):
            return lines[index + 1]
        if lowered.startswith(lowered_label + " "):
            return line[len(label):].strip(" :-")
        if lowered.startswith(lowered_label + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def _dedupe_keep_order(items: List[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for item in items:
        key = normalize_lookup(item)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _looks_like_noisy_theme_line(value: str) -> bool:
    normalized = normalize_lookup(value)
    if not normalized:
        return True
    if any(marker in normalized for marker in _NOISY_THEME_MARKERS):
        return True
    if len(value) < 4:
        return True
    return False


def _looks_like_generated_attachment_name(value: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(value or "").strip(" -_"))
    if not cleaned:
        return True
    if _GENERATED_ATTACHMENT_RE.match(cleaned):
        return True
    tokens = [token for token in re.split(r"[\s_-]+", cleaned) if token]
    if not tokens:
        return True
    alpha_tokens = [token for token in tokens if re.search(r"[A-Za-z]", token)]
    if not alpha_tokens:
        return True
    if all(token.lower() in _GENERIC_DOC_TOKENS for token in alpha_tokens):
        return True
    return len(alpha_tokens) == 1 and alpha_tokens[0].lower() in {
        "screenshot",
        "image",
        "img",
        "photo",
        "picture",
        "scan",
        "attachment",
        "document",
        "file",
        "outlook",
    }


def _org_name_rank(value: str) -> tuple[int, int]:
    cleaned = re.sub(r"\s+", " ", str(value or "").strip(" -_"))
    tokens = [token for token in cleaned.split(" ") if token]
    alpha_tokens = [token for token in tokens if re.search(r"[A-Za-z]", token)]
    score = 0
    if len(alpha_tokens) >= 3:
        score += 4
    elif len(alpha_tokens) == 2:
        score += 2
    if any(token.lower() in {"college", "university", "department", "institute", "association", "hospital", "physicians"} for token in alpha_tokens):
        score += 2
    if len(alpha_tokens) == 1 and alpha_tokens[0].isupper() and len(alpha_tokens[0]) <= 8:
        score -= 3
    return (score, len(cleaned))


def _clean_org_candidate(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "").strip(" -"))
    if not cleaned:
        return ""
    sentence_parts = [part.strip(" -") for part in re.split(r"[.!?]+", cleaned) if part.strip(" -")]
    if sentence_parts:
        cleaned = sentence_parts[-1]
    cleaned = re.sub(
        r"\b(?:Chief Executive Officer|CEO|President|Chair|Director|Council|Committee)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" -")
    cleaned = re.sub(r"^(?:and|the|of)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" -")


def _looks_like_noisy_org_candidate(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    lowered = normalize_lookup(text)
    if not lowered:
        return True
    if any(
        marker in lowered
        for marker in (
            "dear ministers",
            "i am pleased to submit",
            "presentation to the parliament",
            "board committees",
            "responsible ministers",
            "independent auditor",
        )
    ):
        return True
    if lowered.startswith(
        (
            "official",
            "delivery of ",
            "development of ",
            "replacement of ",
            "embedding of ",
            "establishment of ",
            "engagement with ",
            "undertaking ",
            "refreshing of ",
        )
    ):
        return True
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text) if word]
    if len(words) > 7:
        return True
    if any(keyword in lowered for keyword in ("chairperson", "chief executive officer", "board")) and len(words) > 3:
        return True
    return False


def _looks_like_generic_org_placeholder(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    lowered = normalize_lookup(text)
    if not lowered:
        return True
    words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text)
        if normalize_lookup(word) not in _ORG_NAME_STOPWORDS
    ]
    if not words:
        return True
    generic_tokens = {"our", "corporate", "strategy", "plan", "report", "organisation", "organization", "web", "version", "official", "f"}
    if all(normalize_lookup(word) in generic_tokens for word in words):
        return True
    return lowered in {"our corporate", "our strategy", "our corporate strategy", "web version corporate plan", "official f", "official"}


def _extract_org_name(lines: List[str], attachment_names: List[str]) -> str:
    joined = " ".join(lines[:30])
    patterns = (
        r"([A-Z][A-Za-z&'().,\- ]{3,80}?)\s+(?:20\d{2}\s*(?:[-–to]+\s*20\d{2})?\s+)?(?:strategic plan|strategy|annual report)\b",
        r"([A-Z][A-Za-z&'().,\- ]{3,80}?)\s+(?:strategic plan|strategy|annual report)\b",
        r"([A-Z][A-Za-z&'().,\- ]{3,80}?)\s+(?:strategic direction)\b",
        r"([A-Z][A-Za-z&'().,\- ]{3,80}?)\s+(?:statement of strategic priorities)\b",
    )
    matches: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, joined, flags=re.IGNORECASE):
            cleaned = _clean_org_candidate(match.group(1))
            if cleaned:
                matches.append(cleaned)
    if matches:
        matches = [item for item in matches if not _looks_like_generated_attachment_name(item)]
        if matches:
            return max(matches, key=_org_name_rank)
    for name in attachment_names:
        cleaned = Path(unquote(name)).stem.replace("_", " ").replace("-", " ")
        cleaned = re.sub(
            r"\b(annual report|strategic plan|strategic direction|strategy|report|compressed|draft|final|copy|\d{4})\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
        if _looks_like_generated_attachment_name(cleaned):
            continue
        if len(cleaned.split()) >= 2:
            return cleaned
    return ""


def _should_keep_extracted_stakeholder(item: Dict[str, Any], org_name: str) -> bool:
    name = re.sub(r"\s+", " ", str(item.get("name") or "").strip()).strip(" ,.;:-")
    role = normalize_lookup(str(item.get("current_role") or "").strip())
    lowered_name = normalize_lookup(name)
    org_key = normalize_lookup(org_name)
    if not name or not lowered_name:
        return False
    if org_key and lowered_name == org_key:
        return False
    if any(
        marker in lowered_name
        for marker in (
            "responsible ministers",
            "independent auditor",
            "organisational chart",
            "organizational chart",
            "evaluation panel",
            "and information",
        )
    ):
        return False
    if any(
        marker in role
        for marker in (
            "board committees",
            "independent auditor",
            "evaluation panel",
            "organisational chart",
            "organizational chart",
        )
    ):
        return False
    return True


def _extract_marked_items(lines: List[str], markers: tuple[str, ...], limit: int = 6) -> List[str]:
    items: List[str] = []
    for index, line in enumerate(lines[:250]):
        lowered = normalize_lookup(line)
        if not lowered:
            continue
        if any(marker in lowered for marker in markers):
            if 4 <= len(line) <= 140 and not _looks_like_noisy_theme_line(line):
                items.append(line)
            if index + 1 < len(lines):
                next_line = lines[index + 1]
                if (
                    4 <= len(next_line) <= 160
                    and normalize_lookup(next_line) != lowered
                    and not _looks_like_noisy_theme_line(next_line)
                ):
                    items.append(next_line)
        if len(items) >= limit * 2:
            break
    return _dedupe_keep_order(items)[:limit]


def _looks_like_strategy_heading(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip(" -"))
    normalized = normalize_lookup(text)
    if not text or normalized in _STRATEGY_HEADING_STOPWORDS:
        return False
    if len(text) < 8 or len(text) > 80:
        return False
    if any(char.isdigit() for char in text):
        return False
    if _looks_like_noisy_theme_line(text):
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text) if word]
    if len(words) < 2 or len(words) > 8:
        return False
    if normalize_lookup(words[0]) in {"our", "the", "this", "that", "to", "we"}:
        return False
    if normalize_lookup(words[-1]) in {"and", "the", "for", "of", "around", "to"}:
        return False
    if not any(word[0].isupper() for word in words if word):
        return False
    if not any(word[0].islower() for word in words[1:] if word):
        return False
    uppercase_positions = [index for index, word in enumerate(words) if word and word[0].isupper()]
    if len(uppercase_positions) > 2:
        return False
    if any(position > 0 and position != len(words) - 1 for position in uppercase_positions):
        return False
    if any(
        normalize_lookup(word) in {"strategy", "strategic", "plan", "report", "contents", "context", "purpose", "values"}
        for word in words
    ):
        return False
    return True


def _looks_like_section_boundary(value: str) -> bool:
    lowered = normalize_lookup(value)
    return lowered in {
        "strategic outcomes",
        "focus areas",
        "performance",
        "opportunities",
        "purpose",
        "vision",
        "strategic challenges",
        "contribution to queensland government objectives for the community",
    }


def _looks_like_objective_fragment(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip(" -"))
    lowered = normalize_lookup(text)
    if not text or len(text) > 45 or len(text) < 8:
        return False
    if text.endswith("."):
        return False
    if any(char.isdigit() for char in text):
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text) if word]
    if len(words) < 2 or len(words) > 6:
        return False
    if lowered in {"strategic objectives", "focus areas", "strategic outcomes"}:
        return False
    if any(normalize_lookup(word) in _INLINE_STRATEGY_SENTENCE_TOKENS for word in words):
        return False
    return True


def _looks_like_objective_continuation(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip(" -"))
    if not text or text.endswith("."):
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text) if word]
    if not words or len(words) > 4:
        return False
    return all(word[:1].islower() for word in words if word)


def _looks_like_focus_area_heading(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip(" -"))
    if not text or len(text) < 8 or len(text) > 80:
        return False
    if text.endswith("."):
        return False
    lowered = normalize_lookup(text)
    if lowered in {
        "our strategic focus areas",
        "our 5 strategic focus areas",
        "our focus areas",
        "example of a focus area",
        "intent statements and measures",
        "measured by",
    }:
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text) if word]
    if len(words) < 2 or len(words) > 7:
        return False
    uppercase_words = [word for word in words if word.isupper()]
    if len(uppercase_words) >= max(2, len(words) - 1):
        return True
    if len(text) > 42:
        return False
    if any(normalize_lookup(word) in _INLINE_STRATEGY_SENTENCE_TOKENS for word in words):
        return False
    if sum(1 for word in words if word[:1].isupper()) == 0:
        return False
    lowered_words = [normalize_lookup(word) for word in words]
    if lowered_words[0] in {"our", "what", "example", "measured", "intent"}:
        return False
    if lowered_words[0] in {"we", "were", "as", "through", "committed", "building", "develop"}:
        return False
    return True


def _looks_like_incomplete_focus_heading(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip(" -"))
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text) if word]
    if not words:
        return False
    return normalize_lookup(words[-1]) in {"and", "of", "for", "our", "at"}


def _looks_like_focus_area_cluster_heading(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip(" -"))
    if not text or len(text) > 36 or text.endswith("."):
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text) if word]
    if len(words) < 1 or len(words) > 4:
        return False
    lowered_words = [normalize_lookup(word) for word in words]
    if lowered_words[0] in {"our", "what", "example", "intent", "measured", "we", "were", "as", "through", "committed"}:
        return False
    if any(normalize_lookup(word) in _INLINE_STRATEGY_SENTENCE_TOKENS for word in words):
        return False
    return True


def _looks_like_focus_area_heading_continuation(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip(" -"))
    if not text or len(text) > 24 or text.endswith("."):
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text) if word]
    if len(words) < 1 or len(words) > 3:
        return False
    return all(not any(char.isdigit() for char in word) for word in words)


def _extract_structured_strategy_sections(lines: List[str], limit: int = 6) -> List[Dict[str, str]]:
    compact = [re.sub(r"\s+", " ", str(line or "").strip()) for line in lines[:500] if str(line or "").strip()]
    themes: List[Dict[str, str]] = []
    seen: set[str] = set()

    focus_area_markers = {"focus areas", "our strategic focus areas", "our 5 strategic focus areas", "our focus areas"}
    focus_indices = [index for index, line in enumerate(compact) if normalize_lookup(line) in focus_area_markers]
    start = focus_indices[-1] if focus_indices else -1
    if start >= 0:
        cluster_probe = start + 1
        cluster_headings: List[str] = []
        while cluster_probe < len(compact) and len(cluster_headings) < limit:
            current = compact[cluster_probe]
            if _looks_like_section_boundary(current):
                break
            if not _looks_like_focus_area_cluster_heading(current):
                if cluster_headings:
                    break
                cluster_probe += 1
                continue
            headline = current
            consumed = 1
            if cluster_probe + 1 < len(compact) and (
                _looks_like_incomplete_focus_heading(current)
                or len(re.findall(r"[A-Za-z][A-Za-z'’.\-]+", current)) <= 2
            ) and _looks_like_focus_area_heading_continuation(compact[cluster_probe + 1]):
                headline = f"{headline} {compact[cluster_probe + 1]}".strip()
                consumed = 2
            cluster_headings.append(headline.title())
            cluster_probe += consumed
        index = start + 1
        if len(cluster_headings) >= 4:
            description_blocks: List[str] = []
            paragraph_parts: List[str] = []
            while cluster_probe < len(compact):
                follow = compact[cluster_probe]
                if _looks_like_section_boundary(follow):
                    break
                if 20 <= len(follow) <= 260:
                    paragraph_parts.append(follow)
                    if follow.endswith("."):
                        description_blocks.append(" ".join(paragraph_parts).strip())
                        paragraph_parts = []
                cluster_probe += 1
                if len(description_blocks) >= len(cluster_headings):
                    break
            if paragraph_parts and len(description_blocks) < len(cluster_headings):
                description_blocks.append(" ".join(paragraph_parts).strip())
            for idx, headline in enumerate(cluster_headings):
                key = normalize_lookup(headline)
                if not key or key in seen:
                    continue
                seen.add(key)
                snippet = description_blocks[idx] if idx < len(description_blocks) else ""
                themes.append({"headline": headline, "snippet": snippet[:420].strip()})
            index = len(compact)
        while index < len(compact) and len(themes) < limit:
            line = compact[index]
            if _looks_like_section_boundary(line):
                break
            if not _looks_like_focus_area_heading(line):
                index += 1
                continue
            headings: List[str] = []
            while index < len(compact):
                current = compact[index]
                if _looks_like_section_boundary(current) or not _looks_like_focus_area_heading(current):
                    break
                headline = current
                consumed = 1
                if (
                    index + 1 < len(compact)
                    and _looks_like_incomplete_focus_heading(current)
                    and _looks_like_focus_area_heading(compact[index + 1])
                ):
                    headline = f"{headline} {compact[index + 1]}".strip()
                    consumed = 2
                headings.append(headline.title())
                index += consumed

            description_blocks: List[str] = []
            paragraph_parts: List[str] = []
            while index < len(compact):
                follow = compact[index]
                if _looks_like_section_boundary(follow) or _looks_like_focus_area_heading(follow):
                    break
                if 2 <= len(follow) <= 220:
                    paragraph_parts.append(follow)
                    if follow.endswith("."):
                        description_blocks.append(" ".join(paragraph_parts).strip())
                        paragraph_parts = []
                index += 1
            if paragraph_parts:
                description_blocks.append(" ".join(paragraph_parts).strip())

            for idx, headline in enumerate(headings):
                key = normalize_lookup(headline)
                if not key or key in seen:
                    continue
                seen.add(key)
                snippet = description_blocks[idx] if idx < len(description_blocks) else ""
                themes.append({"headline": headline, "snippet": snippet[:420].strip()})
                if len(themes) >= limit:
                    break

    try:
        start = next(index for index, line in enumerate(compact) if normalize_lookup(line) == "strategic objectives")
    except StopIteration:
        start = -1
    if start >= 0 and len(themes) < limit:
        index = start + 1
        headings: List[str] = []
        while index < len(compact) and len(headings) < limit:
            line = compact[index]
            if _looks_like_section_boundary(line):
                break
            if not _looks_like_objective_fragment(line):
                if headings and len(line) > 45:
                    break
                index += 1
                continue
            headline = line
            consumed = 1
            if index + 1 < len(compact) and (
                _looks_like_objective_fragment(compact[index + 1])
                or _looks_like_objective_continuation(compact[index + 1])
            ):
                headline = f"{headline} {compact[index + 1]}".strip()
                consumed = 2
            headings.append(headline)
            index += consumed
        for headline in headings:
            key = normalize_lookup(headline)
            if key and key not in seen:
                seen.add(key)
                themes.append({"headline": headline, "snippet": ""})
                if len(themes) >= limit:
                    break

    return themes[:limit]


def _extract_explicit_strategy_themes(lines: List[str], limit: int = 6) -> List[Dict[str, str]]:
    structured = _extract_structured_strategy_sections(lines, limit=limit)
    if len(structured) >= min(limit, 3):
        return structured
    themes: List[Dict[str, str]] = []
    seen: set[str] = set()
    compact = [re.sub(r"\s+", " ", str(line or "").strip()) for line in lines[:400] if str(line or "").strip()]

    def _looks_truncated(fragment: str) -> bool:
        words = [normalize_lookup(word) for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", fragment)]
        if not words:
            return False
        return words[-1] in {"and", "the", "for", "of", "around", "our"}

    index = 0
    while index < len(compact):
        line = compact[index]
        candidate = line
        consumed = 1
        if index + 1 < len(compact):
            merged_two = f"{line} {compact[index + 1]}".strip()
            if _looks_like_strategy_heading(merged_two):
                candidate = merged_two
                consumed = 2
            if index + 2 < len(compact) and _looks_truncated(line):
                merged_three = f"{line} {compact[index + 1]} {compact[index + 2]}".strip()
                if _looks_like_strategy_heading(merged_three):
                    candidate = merged_three
                    consumed = 3
        if not _looks_like_strategy_heading(candidate):
            index += 1
            continue
        key = normalize_lookup(candidate)
        if key in seen:
            index += consumed
            continue
        snippet_parts: List[str] = []
        for follow in compact[index + consumed : index + consumed + 3]:
            if _looks_like_strategy_heading(follow):
                break
            if 20 <= len(follow) <= 220 and not _looks_like_noisy_theme_line(follow):
                snippet_parts.append(re.sub(r"\s+", " ", follow).strip())
        seen.add(key)
        themes.append(
            {
                "headline": candidate,
                "snippet": " ".join(snippet_parts)[:420].strip(),
            }
        )
        index += consumed
        if len(themes) >= limit:
            break
    if len(themes) >= min(limit, 3):
        return themes

    blob = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if not blob:
        return themes
    org_hint = _extract_org_name(lines, [])

    def _looks_like_inline_heading(candidate: str) -> bool:
        if not _looks_like_strategy_heading(candidate):
            return False
        candidate_tokens = [normalize_lookup(token) for token in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", candidate)]
        if any(token in _INLINE_STRATEGY_SENTENCE_TOKENS for token in candidate_tokens):
            return False
        return True

    tokens = re.findall(r"[A-Za-z][A-Za-z'’.\-]+", blob)
    index = 0
    while index < len(tokens) and len(themes) < limit:
        token = tokens[index]
        if not token or not token[0].isupper():
            index += 1
            continue
        best_candidate = ""
        best_end = index + 1
        phrase_tokens: List[str] = []
        max_end = min(len(tokens), index + 8)
        for end in range(index, max_end):
            phrase_tokens.append(tokens[end])
            candidate = " ".join(phrase_tokens)
            if _looks_like_inline_heading(candidate):
                next_token = tokens[end + 1] if end + 1 < len(tokens) else ""
                if not next_token or next_token[0].isupper():
                    best_candidate = candidate
                    best_end = end + 1
        normalized_candidate = normalize_lookup(best_candidate)
        if (
            best_candidate
            and normalized_candidate not in seen
            and normalized_candidate != normalize_lookup(org_hint)
            and "strategy" not in normalized_candidate
            and "plan" not in normalized_candidate
            and "report" not in normalized_candidate
        ):
            seen.add(normalized_candidate)
            themes.append({"headline": best_candidate, "snippet": ""})
            index = best_end
            continue
        index += 1
    return themes


def _headline_case(value: str) -> str:
    parts = re.split(r"(\s+)", re.sub(r"\s+", " ", str(value or "").strip()))
    return "".join(part[:1].upper() + part[1:] if part and not part.isspace() else part for part in parts).strip()


def _looks_like_ambition_noise_line(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip(" -•*"))
    lowered = normalize_lookup(text)
    if not lowered:
        return True
    if lowered in {"who we are", "measured by"}:
        return True
    if "output measure" in lowered or "five year customer commitment" in lowered:
        return True
    if re.fullmatch(r"[*•-]+", text):
        return True
    if len(re.findall(r"[A-Za-z]", text)) < 2:
        return True
    return False


def _looks_like_ambition_subheading(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip(" -•*"))
    lowered = normalize_lookup(text)
    if not text or _looks_like_ambition_noise_line(text):
        return False
    if lowered.startswith(("our ambitions for", "intent statements and measures")):
        return False
    if any(char.isdigit() for char in text):
        return False
    if text.endswith(".") or len(text) > 42:
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’&/\-]*", text) if word]
    if len(words) < 1 or len(words) > 5:
        return False
    if any(normalize_lookup(word) in _INLINE_STRATEGY_SENTENCE_TOKENS for word in words):
        return False
    return True


def _looks_like_ambition_subheading_continuation(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip(" -•*"))
    if not text or _looks_like_ambition_noise_line(text):
        return False
    if any(char.isdigit() for char in text):
        return False
    if text.endswith(".") or len(text) > 28:
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’&/\-]*", text) if word]
    if len(words) < 1 or len(words) > 4:
        return False
    return not any(normalize_lookup(word) in _INLINE_STRATEGY_SENTENCE_TOKENS for word in words)


def _clean_ambition_narrative_line(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip(" -•*"))
    lowered = normalize_lookup(text)
    if not text or _looks_like_ambition_noise_line(text):
        return ""
    if lowered.startswith(("our ambitions for", "intent statements and measures")):
        return ""
    if text.startswith(("<", ">", "$")) and len(text) < 80:
        return ""
    if any(marker in lowered for marker in ("five year customer commitment", "output measure")):
        return ""
    return text


def _build_ambition_section_snippet(section_lines: List[str]) -> str:
    narrative_lines: List[str] = []
    index = 0
    while index < len(section_lines):
        line = section_lines[index]
        if _looks_like_ambition_subheading(line):
            if index + 1 < len(section_lines) and _looks_like_ambition_subheading_continuation(section_lines[index + 1]):
                index += 2
            else:
                index += 1
            continue
        cleaned = _clean_ambition_narrative_line(line)
        if cleaned:
            narrative_lines.append(cleaned)
        index += 1

    text = re.sub(r"\s+", " ", " ".join(narrative_lines)).strip(" ,.;:-")
    if not text:
        return ""

    sentences: List[str] = []
    for fragment in re.split(r"(?<=[.!?])\s+", text):
        cleaned = re.sub(r"\s+", " ", fragment).strip(" ,.;:-")
        if not cleaned or len(cleaned) < 40:
            continue
        if cleaned[:1] in "<>$" or re.match(r"^\d", cleaned):
            continue
        sentences.append(cleaned)
        if len(sentences) >= 2:
            break
    snippet = " ".join(sentences) if sentences else text
    if len(snippet) > 420:
        snippet = snippet[:417].rstrip(" ,.;:-") + "..."
    return snippet


def _build_ambition_detail_snippet(narrative_lines: List[str]) -> str:
    text = re.sub(r"\s+", " ", " ".join(str(item or "").strip() for item in narrative_lines if str(item or "").strip())).strip(" ,.;:-")
    if not text:
        return ""
    sentences: List[str] = []
    for fragment in re.split(r"(?<=[.!?])\s+", text):
        cleaned = re.sub(r"\s+", " ", fragment).strip(" ,.;:-")
        if not cleaned or len(cleaned) < 25:
            continue
        if cleaned[:1] in "<>$" or re.match(r"^\d", cleaned):
            continue
        sentences.append(cleaned)
        if len(sentences) >= 2:
            break
    snippet = " ".join(sentences) if sentences else text
    if len(snippet) > 320:
        snippet = snippet[:317].rstrip(" ,.;:-") + "..."
    return snippet


def _extract_ambition_detail_points(section_lines: List[str], limit: int = 4) -> List[Dict[str, str]]:
    compact = [re.sub(r"\s+", " ", str(line or "").strip()) for line in section_lines if str(line or "").strip()]
    points: List[Dict[str, str]] = []
    seen: set[str] = set()
    index = 0
    while index < len(compact) and len(points) < limit:
        current = compact[index]
        if not _looks_like_ambition_subheading(current):
            index += 1
            continue
        raw_heading = current
        heading = current
        if index + 1 < len(compact) and _looks_like_ambition_subheading_continuation(compact[index + 1]):
            raw_heading = f"{raw_heading} {compact[index + 1]}".strip()
            heading = f"{heading} {compact[index + 1]}".strip()
            index += 2
        else:
            index += 1
        narrative: List[str] = []
        while index < len(compact) and not _looks_like_ambition_subheading(compact[index]):
            cleaned = _clean_ambition_narrative_line(compact[index])
            if cleaned:
                narrative.append(cleaned)
            index += 1
        snippet = _build_ambition_detail_snippet(narrative)
        heading = _headline_case(heading)
        key = normalize_lookup(heading)
        if not heading or not snippet or key in seen:
            continue
        seen.add(key)
        points.append({"heading": heading, "raw_heading": raw_heading, "snippet": snippet})
    return points


def _extract_ambition_focus_area_signals(lines: List[str], limit: int = 8) -> List[Dict[str, str]]:
    compact = [re.sub(r"\s+", " ", str(line or "").strip()) for line in lines if str(line or "").strip()]
    signals: List[Dict[str, str]] = []
    seen: set[str] = set()
    index = 0
    while index < len(compact) and len(signals) < limit:
        if not normalize_lookup(compact[index]).startswith("our ambitions for"):
            index += 1
            continue
        index += 1
        title_lines: List[str] = []
        while index < len(compact):
            current = compact[index]
            lowered = normalize_lookup(current)
            if lowered == "intent statements and measures":
                index += 1
                break
            if _looks_like_ambition_noise_line(current):
                index += 1
                continue
            if len(title_lines) >= 4:
                break
            title_lines.append(current)
            index += 1
        headline = _headline_case(" ".join(title_lines))
        if not headline:
            continue
        section_lines: List[str] = []
        while index < len(compact) and not normalize_lookup(compact[index]).startswith("our ambitions for"):
            section_lines.append(compact[index])
            index += 1
        key = normalize_lookup(headline)
        if not key or key in seen:
            continue
        seen.add(key)
        snippet = _build_ambition_section_snippet(section_lines)
        detail_points = _extract_ambition_detail_points(section_lines)
        signals.append(
            {
                "headline": headline,
                "category": "strategic_theme",
                "snippet": snippet,
                "evidence": snippet or headline,
                "detail_points": detail_points,
            }
        )
    return signals


def _looks_like_priority_plan_heading(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip(" -•*"))
    lowered = normalize_lookup(text)
    if not text or len(text) < 6 or len(text) > 40:
        return False
    if text.endswith(".") or any(char.isdigit() for char in text):
        return False
    if lowered in {
        "official",
        "our strategic priorities",
        "our strategic priorities framework",
        "our 2050 vision",
        "our commercial businesses",
        "financial projections",
        "what we do",
        "strategic context",
        "our current challenges",
    }:
        return False
    if lowered.startswith(("our ", "this ", "we ", "the ")):
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’&/\-]*", text) if word]
    if len(words) < 1 or len(words) > 4:
        return False
    if any(normalize_lookup(word) in _INLINE_STRATEGY_SENTENCE_TOKENS for word in words):
        return False
    if normalize_lookup(words[0]) in {"delivery", "development", "establishment", "refreshing", "replacement", "embedding", "engagement", "undertaking"}:
        return False
    return True


def _clean_priority_plan_line(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^[•*\-]+\s*", "", text).strip()
    lowered = normalize_lookup(text)
    if not text:
        return ""
    if lowered.startswith(("our strategic priorities", "our 2024-29 strategic priorities plan outlines priorities including")):
        return ""
    if lowered in {"official", "5", "supervisory control and data acquisition"}:
        return ""
    if len(re.findall(r"[A-Za-z]", text)) < 3:
        return ""
    return text


def _build_priority_plan_snippet(section_lines: List[str]) -> str:
    sentences: List[str] = []
    for raw_line in section_lines:
        cleaned = _clean_priority_plan_line(raw_line)
        if not cleaned:
            continue
        if len(cleaned) < 24:
            continue
        if not cleaned.endswith((".", "!", "?")):
            cleaned = cleaned.rstrip(" ,;:") + "."
        sentences.append(cleaned)
        if len(sentences) >= 2:
            break
    snippet = " ".join(sentences).strip()
    if len(snippet) > 320:
        snippet = snippet[:317].rstrip(" ,.;:-") + "..."
    return snippet


def _extract_priority_plan_signals(lines: List[str], limit: int = 8) -> List[Dict[str, str]]:
    compact = [re.sub(r"\s+", " ", str(line or "").strip()) for line in lines if str(line or "").strip()]
    signals: List[Dict[str, str]] = []
    seen: set[str] = set()

    try:
        start = next(index for index, line in enumerate(compact) if normalize_lookup(line) == "our strategic priorities")
    except StopIteration:
        return signals

    index = start + 1
    while index < len(compact) and len(signals) < limit:
        current = compact[index]
        lowered = normalize_lookup(current)
        if lowered in {"our 2050 vision", "our commercial businesses", "financial projections"}:
            break
        if lowered.startswith("our 2024-") and "strategic priorities plan outlines priorities" in lowered:
            index += 1
            continue
        if not _looks_like_priority_plan_heading(current):
            index += 1
            continue

        headline = _headline_case(current)
        key = normalize_lookup(headline)
        if key in seen:
            index += 1
            continue
        seen.add(key)
        index += 1

        detail_lines: List[str] = []
        while index < len(compact):
            follow = compact[index]
            lowered_follow = normalize_lookup(follow)
            if lowered_follow in {"our 2050 vision", "our commercial businesses", "financial projections"}:
                break
            if _looks_like_priority_plan_heading(follow):
                break
            cleaned = _clean_priority_plan_line(follow)
            if cleaned:
                detail_lines.append(cleaned)
            index += 1

        snippet = _build_priority_plan_snippet(detail_lines)
        signals.append(
            {
                "headline": headline,
                "category": "strategic_theme",
                "snippet": snippet,
                "evidence": snippet or headline,
            }
        )
    return signals


def _collect_signal_snippets(lines: List[str], markers: tuple[str, ...], limit: int = 2) -> List[str]:
    candidates: List[tuple[int, str]] = []
    seen: set[str] = set()
    for index, line in enumerate(lines[:500]):
        lowered = normalize_lookup(line)
        if not lowered or not any(marker in lowered for marker in markers):
            continue
        parts = [line]
        if index + 1 < len(lines):
            next_line = lines[index + 1]
            next_normalized = normalize_lookup(next_line)
            if (
                25 <= len(next_line) <= 240
                and not _looks_like_noisy_theme_line(next_line)
                and next_normalized != lowered
            ):
                parts.append(next_line)
        snippet = re.sub(r"\s+", " ", " ".join(parts)).strip()
        snippet_key = normalize_lookup(snippet)
        if not snippet_key or snippet_key in seen:
            continue
        seen.add(snippet_key)
        score = sum(2 for marker in markers if marker in lowered) + min(len(snippet) // 90, 3)
        candidates.append((score, snippet))
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    return [snippet for _, snippet in candidates[:limit]]


def _sentence_window(blob: str, start_idx: int, end_idx: int, max_chars: int = 360, min_chars: int = 120) -> str:
    if not blob:
        return ""
    left_boundary = max(blob.rfind(marker, 0, start_idx) for marker in (". ", "! ", "? ", "\n"))
    snippet_start = left_boundary + 1 if left_boundary >= 0 else max(0, start_idx - 40)
    if 0 < snippet_start < len(blob) and blob[snippet_start - 1].isalpha() and blob[snippet_start].isalpha():
        prev_space = blob.rfind(" ", 0, snippet_start)
        if prev_space >= 0 and snippet_start - prev_space <= 20:
            snippet_start = prev_space + 1
        else:
            next_space = blob.find(" ", snippet_start)
            if next_space >= 0:
                snippet_start = next_space + 1

    right_candidates = [blob.find(marker, end_idx) for marker in (". ", "! ", "? ", "\n")]
    first_right = min(item for item in right_candidates if item >= 0) if any(item >= 0 for item in right_candidates) else -1
    snippet_end = first_right + 1 if first_right >= 0 else min(len(blob), end_idx + 220)

    if snippet_end - snippet_start > max_chars and start_idx - snippet_start > max_chars // 3:
        snippet_start = max(snippet_start, start_idx - 80)
        if 0 < snippet_start < len(blob) and blob[snippet_start - 1].isalpha() and blob[snippet_start].isalpha():
            prev_space = blob.rfind(" ", 0, snippet_start)
            if prev_space >= 0 and snippet_start - prev_space <= 20:
                snippet_start = prev_space + 1
            else:
                next_space = blob.find(" ", snippet_start)
                if next_space >= 0:
                    snippet_start = next_space + 1

    snippet = re.sub(r"\s+", " ", blob[snippet_start:snippet_end]).strip(" ,.;:-")
    if len(snippet) < min_chars and first_right >= 0:
        second_candidates = [blob.find(marker, first_right + 1) for marker in (". ", "! ", "? ", "\n")]
        second_right = min(item for item in second_candidates if item >= 0) if any(item >= 0 for item in second_candidates) else -1
        extended_end = second_right + 1 if second_right >= 0 else min(len(blob), snippet_end + 220)
        snippet = re.sub(r"\s+", " ", blob[snippet_start:extended_end]).strip(" ,.;:-")

    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip(" ,.;:-")
    return snippet


def _trim_signal_snippet_lead(snippet: str, marker: str, max_prefix_words: int = 10, prefix_chars: int = 60) -> str:
    text = re.sub(r"\s+", " ", str(snippet or "")).strip(" ,.;:-")
    if not text or not marker:
        return text
    match = re.search(re.escape(marker), text, flags=re.IGNORECASE)
    if not match:
        return text
    prefix = text[:match.start()].strip()
    prefix_words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", prefix) if word]
    if len(prefix_words) <= max_prefix_words:
        return text
    start = max(0, match.start() - prefix_chars)
    if 0 < start < len(text) and text[start - 1].isalpha() and text[start].isalpha():
        prev_space = text.rfind(" ", 0, start)
        if prev_space >= 0 and start - prev_space <= 20:
            start = prev_space + 1
        else:
            next_space = text.find(" ", start)
            if next_space >= 0:
                start = next_space + 1
    trimmed = text[start:]
    capitalized_restart = re.search(r"\b[A-Z][A-Za-z'’.\-]{3,}\b", trimmed[:30])
    if capitalized_restart and capitalized_restart.start() > 0:
        leading = trimmed[: capitalized_restart.start()].strip()
        if leading and leading[:1].islower() and len(leading.split()) <= 2:
            trimmed = trimmed[capitalized_restart.start() :]
    return trimmed.strip(" ,.;:-")


def _collect_blob_signal_snippets(blob: str, markers: tuple[str, ...], limit: int = 2) -> List[str]:
    candidates: List[tuple[int, str]] = []
    seen: set[str] = set()
    text = str(blob or "")
    if not text:
        return []
    for marker in markers:
        if not marker:
            continue
        for match in re.finditer(re.escape(marker), text, flags=re.IGNORECASE):
            snippet = _trim_signal_snippet_lead(_sentence_window(text, match.start(), match.end()), marker)
            snippet_key = normalize_lookup(snippet)
            if not snippet_key or snippet_key in seen:
                continue
            seen.add(snippet_key)
            score = len(marker) + min(len(snippet) // 80, 4)
            candidates.append((score, snippet))
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    return [snippet for _, snippet in candidates[:limit]]


def _looks_like_person_name(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text or len(text) > 80:
        return False
    lowered = normalize_lookup(text)
    if any(char.isdigit() for char in text) or "%" in text:
        return False
    if "(" in text or ")" in text:
        return False
    if lowered in {"key management personnel", "official", "official f"}:
        return False
    if any(token in lowered for token in ("foreword", "comment", "statement", "vision", "purpose", "values")):
        return False
    if any(token in lowered for token in _ROLE_KEYWORDS):
        return False
    if not re.search(r"[A-Z]", text):
        return False
    raw_words = re.findall(r"[A-Za-z][A-Za-z'’.\-]+", text)
    if len(raw_words) < 2 or len(raw_words) > 7:
        return False
    eval_words = [
        word.strip(".,")
        for word in raw_words
        if normalize_lookup(word.strip(".,"))
        not in _PERSON_CREDENTIAL_TOKENS
    ]
    if eval_words and normalize_lookup(" ".join(eval_words[:2])) in _PERSON_PREFIXES:
        eval_words = eval_words[2:]
    elif eval_words and normalize_lookup(eval_words[0]) in _PERSON_PREFIXES:
        eval_words = eval_words[1:]
    if len(eval_words) < 2 or len(eval_words) > 4:
        return False
    if len(eval_words) == 4 and all(re.match(r"^[A-Z][a-z]{2,}$", word) for word in eval_words):
        return False
    if sum(1 for word in eval_words if re.match(r"^[A-Z][a-z]{2,}$", word)) < 2:
        return False
    lowered_words = [normalize_lookup(word) for word in eval_words]
    if any(word in _NON_PERSON_NAME_TOKENS for word in lowered_words):
        return False
    title_case_words = sum(1 for word in eval_words if word and word[0].isupper())
    if title_case_words < len(eval_words):
        return False
    if normalize_lookup(raw_words[0]) in _PERSON_PREFIXES or normalize_lookup(" ".join(raw_words[:2])) in _PERSON_PREFIXES:
        return True
    return True


def _looks_like_role_line(value: str) -> bool:
    lowered = normalize_lookup(value)
    if not lowered:
        return False
    if len(str(value or "").split()) > 9:
        return False
    if any(marker in lowered for marker in (" has ", " is ", " responsible for ", "experience", "including", "see ")):
        return False
    if any(marker in lowered for marker in _ROLE_LINE_EXCLUSION_MARKERS):
        return False
    if any(char.isdigit() for char in str(value or "")) or "%" in str(value or ""):
        return False
    words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", str(value or "")) if word]
    if len(words) >= 2 and len({normalize_lookup(word) for word in words}) == 1:
        return False
    return any(keyword in lowered for keyword in _ROLE_KEYWORDS)


def _extract_leadership_people(lines: List[str], org_name: str) -> List[Dict[str, Any]]:
    people: List[Dict[str, Any]] = []
    seen: set[str] = set()
    org_key = normalize_lookup(org_name)
    for index, line in enumerate(lines):
        if not _looks_like_person_name(line):
            continue
        role = lines[index + 1] if index + 1 < len(lines) and _looks_like_role_line(lines[index + 1]) else ""
        clean_name = re.sub(r"\s+", " ", line).strip(" ,.;:")
        clean_role = clean_strategic_role_label(role, org_name)
        employer = ""
        if index + 2 < len(lines):
            candidate_org = lines[index + 2]
            if org_key and org_key in normalize_lookup(candidate_org):
                employer = org_name
        evidence_parts = [clean_name]
        if clean_role:
            evidence_parts.append(clean_role)
        if employer:
            evidence_parts.append(org_name)
        if not clean_role:
            continue
        key = _person_identity_key(clean_name)
        if not key or key in seen:
            continue
        seen.add(key)
        people.append(
            {
                "name": clean_name,
                "current_role": clean_role,
                "current_employer": employer or org_name,
                "confidence": 0.92,
                "evidence": "Strategic document signatory block: " + ", ".join(evidence_parts),
                "extraction_method": "strategic_doc_signatory",
            }
        )
    return people


def _person_identity_key(value: str) -> str:
    words = [
        normalize_lookup(word.strip(".,()"))
        for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", str(value or ""))
        if normalize_lookup(word.strip(".,()")) not in _PERSON_CREDENTIAL_TOKENS
        and normalize_lookup(word.strip(".,()")) not in _PERSON_PREFIXES
    ]
    return " ".join(words[:4]).strip()


def _extract_strategic_signals(lines: List[str], blob: str = "") -> List[Dict[str, str]]:
    signals: List[Dict[str, str]] = []
    for spec in _STRATEGIC_SIGNAL_SPECS:
        snippets = _collect_blob_signal_snippets(blob, spec["markers"])
        if not snippets:
            snippets = _collect_signal_snippets(lines, spec["markers"])
        if not snippets:
            continue
        signals.append(
            {
                "headline": spec["headline"],
                "category": spec["category"],
                "snippet": " ".join(snippets)[:420],
                "evidence": snippets[0],
            }
        )
    return signals


def _org_short_forms(org_name: str) -> List[str]:
    words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", str(org_name or ""))
        if normalize_lookup(word) not in _ORG_NAME_STOPWORDS
    ]
    forms: List[str] = []
    acronym = "".join(word[0].upper() for word in words if word and word[0].isalpha())
    if 2 <= len(acronym) <= 8:
        forms.append(acronym)
    return forms


def _annual_report_framework_keywords(heading: str) -> List[str]:
    lowered = normalize_lookup(heading)
    keywords = [token for token in re.findall(r"[a-z]+", lowered) if token]
    if "future" in lowered or "solution" in lowered:
        keywords.extend(["project", "projects", "capital", "infrastructure", "growth", "capacity"])
    if "healthy" in lowered or "country" in lowered:
        keywords.extend(["environment", "waste", "wastewater", "recycle", "renewable", "country", "traditional owners"])
    if "affordable" in lowered or "bill" in lowered:
        keywords.extend(["bill", "bills", "price", "prices", "cpi", "value", "customer", "cost"])
    if "climate" in lowered or "preparedness" in lowered:
        keywords.extend(["climate", "renewable", "energy", "emissions", "carbon", "resilience"])
    return _dedupe_keep_order([str(item).strip() for item in keywords if str(item).strip()])


def _build_annual_report_framework_snippet(blob: str, heading: str) -> str:
    text = re.sub(r"\s+", " ", str(blob or "")).strip()
    if not text:
        return ""
    keyword_tokens = {normalize_lookup(item) for item in _annual_report_framework_keywords(heading) if str(item).strip()}
    heading_key = normalize_lookup(heading)
    sentences = [re.sub(r"\s+", " ", part).strip(" ,.;:-") for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    scored: List[tuple[int, str]] = []
    for sentence in sentences:
        lowered = normalize_lookup(sentence)
        if not lowered or "four pillars of our strategic framework" in lowered:
            continue
        if len(sentence) < 40 or len(sentence) > 320:
            continue
        score = 0
        if heading_key and heading_key in lowered:
            score += 4
        for token in keyword_tokens:
            if token and token in lowered:
                score += 1
        if score <= 0:
            continue
        scored.append((score, sentence))
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    chosen: List[str] = []
    seen: set[str] = set()
    for _, sentence in scored:
        key = normalize_lookup(sentence)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(sentence.rstrip(".") + ".")
        if len(chosen) >= 2:
            break
    snippet = " ".join(chosen).strip()
    if len(snippet) > 320:
        snippet = snippet[:317].rstrip(" ,.;:-") + "..."
    return snippet


def _extract_annual_report_framework_signals(lines: List[str], blob: str, limit: int = 6) -> List[Dict[str, str]]:
    compact = [re.sub(r"\s+", " ", str(line or "").strip()) for line in lines if str(line or "").strip()]
    headings: List[str] = []
    for index, line in enumerate(compact[:220]):
        lowered = normalize_lookup(line)
        if "four pillars of our strategic framework" not in lowered:
            continue
        window = " ".join(compact[index : min(index + 3, len(compact))])
        match = re.search(r"four pillars of our strategic framework,?\s*(.+)", window, flags=re.IGNORECASE)
        if not match:
            continue
        remainder = match.group(1)
        remainder = re.split(
            r"\b(?:another notable achievement|we delivered|it is a pleasure to present|board chair|managing director)\b",
            remainder,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        parts: List[str] = []
        for chunk in re.split(r",", remainder):
            parts.extend(re.split(r"\band\b", chunk, flags=re.IGNORECASE))
        for part in parts:
            cleaned = re.sub(r"\s+", " ", str(part or "").strip(" ,.;:-"))
            lowered_clean = normalize_lookup(cleaned)
            lowered_words = set(re.findall(r"[a-z]+", lowered_clean))
            if not cleaned or len(cleaned.split()) > 4 or len(cleaned.split()) < 2:
                continue
            if lowered_clean.endswith(" and"):
                continue
            if lowered_words & _INLINE_STRATEGY_SENTENCE_TOKENS:
                continue
            if lowered_clean in {"our strategic framework", "delivering on our expectations"}:
                continue
            headings.append(_headline_case(cleaned))
        for candidate in re.findall(r"[A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+){1,2}", remainder):
            cleaned = re.sub(r"\s+", " ", candidate).strip(" ,.;:-")
            lowered_clean = normalize_lookup(cleaned)
            lowered_words = set(re.findall(r"[a-z]+", lowered_clean))
            if not cleaned or len(cleaned.split()) > 3 or len(cleaned.split()) < 2:
                continue
            if lowered_clean in {normalize_lookup(item) for item in headings}:
                continue
            if lowered_clean.startswith(("another notable", "it is", "we delivered", "board chair", "managing director")):
                continue
            if lowered_clean.endswith(" and"):
                continue
            if lowered_words & _INLINE_STRATEGY_SENTENCE_TOKENS:
                continue
            headings.append(_headline_case(cleaned))
        for candidate in re.findall(r"\band\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+){1,2})", remainder, flags=re.IGNORECASE):
            cleaned = re.sub(r"\s+", " ", candidate).strip(" ,.;:-")
            lowered_clean = normalize_lookup(cleaned)
            lowered_words = set(re.findall(r"[a-z]+", lowered_clean))
            if not cleaned or len(cleaned.split()) > 3 or len(cleaned.split()) < 2:
                continue
            if lowered_clean in {normalize_lookup(item) for item in headings}:
                continue
            if lowered_words & _INLINE_STRATEGY_SENTENCE_TOKENS:
                continue
            headings.append(_headline_case(cleaned))
        if headings:
            break

    signals: List[Dict[str, str]] = []
    seen: set[str] = set()
    for heading in headings:
        key = normalize_lookup(heading)
        if not key or key in seen:
            continue
        seen.add(key)
        snippet = _build_annual_report_framework_snippet(blob, heading)
        signals.append(
            {
                "headline": heading,
                "category": "strategic_theme",
                "snippet": snippet,
                "evidence": snippet or heading,
            }
        )
        if len(signals) >= limit:
            break
    return signals


def clean_strategic_role_label(role: str, org_name: str = "") -> str:
    text = re.sub(r"\s+", " ", str(role or "").strip()).strip(" ,.;:-")
    if not text:
        return ""
    candidate = text
    if org_name:
        org_pattern = re.escape(str(org_name or "").strip())
        if org_pattern:
            candidate = re.sub(rf"^\b{org_pattern}\b[\s,:|-]+", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(rf"[\s,:|-]+\b{org_pattern}\b$", "", candidate, flags=re.IGNORECASE)
        for short_form in _org_short_forms(org_name):
            short_pattern = re.escape(short_form)
            candidate = re.sub(rf"^\b{short_pattern}\b[\s,:|-]+", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(rf"[\s,:|-]+\b{short_pattern}\b$", "", candidate, flags=re.IGNORECASE)
    acronym_match = re.search(r"\(([^)]+)\)\s*$", candidate)
    if acronym_match:
        acronym = normalize_lookup(acronym_match.group(1))
        expanded = normalize_lookup(re.sub(r"\([^)]+\)\s*$", "", candidate))
        acronym_map = {
            "ceo": "chief executive officer",
            "cfo": "chief financial officer",
            "coo": "chief operating officer",
            "cio": "chief information officer",
            "cto": "chief technology officer",
        }
        if acronym and acronym_map.get(acronym) and acronym_map[acronym] in expanded:
            candidate = re.sub(r"\s*\([^)]+\)\s*$", "", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" ,.;:-")
    if candidate and any(keyword in normalize_lookup(candidate) for keyword in _ROLE_KEYWORDS):
        return candidate
    return text


def clean_indicator_evidence_text(text: str, max_words: int = 28) -> str:
    snippet = re.sub(r"\s+", " ", str(text or "")).strip(" ,.;:-")
    if not snippet:
        return ""
    words = snippet.split()
    kept: List[str] = []
    numeric_heavy_run = 0
    for index, word in enumerate(words):
        normalized = re.sub(r"[^A-Za-z0-9%$]", "", word)
        alpha_chars = sum(char.isalpha() for char in normalized)
        digit_chars = sum(char.isdigit() for char in normalized)
        is_upper_artifact = alpha_chars >= 3 and normalized.isupper()
        is_symbol_artifact = any(char in word for char in {"#", "Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ"})
        is_numeric_heavy = digit_chars >= max(2, alpha_chars)
        if index >= 8 and (is_upper_artifact or is_symbol_artifact):
            break
        if is_numeric_heavy:
            numeric_heavy_run += 1
        else:
            numeric_heavy_run = 0
        if index >= 10 and numeric_heavy_run >= 3:
            break
        kept.append(word)
        if len(kept) >= max_words:
            break
    cleaned = " ".join(kept).strip(" ,.;:-")
    return cleaned or snippet


def _sentence_snippet(blob: str, match: re.Match[str]) -> str:
    left_scan = max(0, match.start() - 180)
    left_boundary = max(blob.rfind(marker, left_scan, match.start()) for marker in (". ", "! ", "? ", "; ", "\n"))
    start = left_boundary + 1 if left_boundary >= 0 else max(0, match.start() - 40)

    matched_text = re.sub(r"\s+", " ", match.group(0)).strip()
    if re.match(r"^[\d$]", matched_text):
        start = max(start, match.start())

    raw_prefix = re.sub(r"\s+", " ", blob[start:match.start()]).strip()
    prefix_words = [word for word in re.findall(r"[A-Za-z][A-Za-z'’.\-]+", raw_prefix) if word]
    alpha_count = sum(char.isalpha() for char in raw_prefix)
    digit_count = sum(char.isdigit() for char in raw_prefix)
    if raw_prefix and (
        (left_boundary < 0 and len(prefix_words) <= 4)
        or raw_prefix[:1].islower()
        or digit_count > max(6, alpha_count // 2)
        or alpha_count < 10
    ):
        start = max(start, match.start() - 12)

    right_scan = min(len(blob), match.end() + 220)
    right_candidates = [blob.find(marker, match.end(), right_scan) for marker in (". ", "! ", "? ", "; ", "\n")]
    right_boundary = min(item for item in right_candidates if item >= 0) if any(item >= 0 for item in right_candidates) else -1
    end = right_boundary + 1 if right_boundary >= 0 else min(len(blob), match.end() + 180)

    snippet = re.sub(r"\s+", " ", blob[start:end]).strip(" ,.;:-")
    if snippet and snippet[0].islower():
        snippet = re.sub(r"\s+", " ", blob[max(0, match.start() - 24):end]).strip(" ,.;:-")
    return clean_indicator_evidence_text(snippet[:240])


def _format_indicator_value(template: str, groups: tuple[str, ...]) -> str:
    value = template
    for index, group in enumerate(groups):
        value = value.replace("${" + str(index) + "}", "$" + str(group or "").strip())
        value = value.replace("{" + str(index) + "}", str(group or "").strip())
    return re.sub(r"\s+", " ", value).strip(" -")


def _extract_performance_indicators(blob: str, doc_type: str) -> List[Dict[str, str]]:
    if doc_type != "annual_report":
        return []

    indicators: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    normalized_blob = re.sub(r"\s+", " ", str(blob or "")).strip()
    for spec in _PERFORMANCE_INDICATOR_SPECS:
        for pattern in spec["patterns"]:
            match = re.search(pattern, normalized_blob, flags=re.IGNORECASE)
            if not match:
                continue
            groups = tuple(part.strip() for part in match.groups() if str(part or "").strip())
            value = _format_indicator_value(spec["value_template"], groups) if groups else spec["label"]
            evidence = _sentence_snippet(normalized_blob, match)
            key = (spec["label"], normalize_lookup(value))
            if key in seen:
                break
            seen.add(key)
            indicators.append(
                {
                    "label": spec["label"],
                    "category": spec["category"],
                    "value": value,
                    "evidence": evidence,
                }
            )
            break
    return indicators[:12]


def _clean_project_name(name: str) -> str:
    text = re.sub(r"\s+", " ", str(name or "").strip(" ,.;:-"))
    text = re.sub(r"^(?:and\s+the|and|the)\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" ,.;:-")


def _normalise_project_value(amount: str, unit: str) -> str:
    numeric = str(amount or "").strip()
    unit_text = str(unit or "").strip().lower()
    if not numeric or not unit_text:
        return ""
    if unit_text == "m":
        unit_text = "million"
    elif unit_text == "bn":
        unit_text = "billion"
    return f"${numeric} {unit_text}"


def _extract_major_projects(blob: str, doc_type: str) -> List[Dict[str, str]]:
    if doc_type != "annual_report":
        return []

    text = re.sub(r"\s+", " ", str(blob or "")).strip()
    if not text:
        return []

    project_windows = re.finditer(
        r"(?:largest [^.]{0,120}? investments? (?:included|include)|largest investments? include)\s+(.{0,700}?)(?:\.(?=\s+[A-Z]|\s*$)|Table \d+:)",
        text,
        flags=re.IGNORECASE,
    )
    projects: List[Dict[str, str]] = []
    seen: set[str] = set()
    for match in project_windows:
        window_text = match.group(0).strip()
        segment = match.group(1).strip()
        for project_match in _PROJECT_AMOUNT_RE.finditer(segment):
            name = _clean_project_name(project_match.group(1))
            amount = _normalise_project_value(project_match.group(2), project_match.group(3))
            key = normalize_lookup(name)
            if not name or not amount or key in seen:
                continue
            seen.add(key)
            projects.append(
                {
                    "name": name,
                    "value": amount,
                    "evidence": clean_indicator_evidence_text(window_text, max_words=40),
                }
            )
    return projects[:8]


def _extract_kpi_focuses(lines: List[str], doc_type: str) -> List[str]:
    if doc_type != "annual_report":
        return []
    focuses: List[str] = []
    capture = False
    for line in lines:
        lowered = normalize_lookup(line)
        if "measure of success" in lowered or "key performance indicators" in lowered:
            capture = True
            continue
        if capture and len(focuses) >= 5:
            break
        if not capture:
            continue
        if len(line) > 70 or not re.search(r"[A-Za-z]", line):
            continue
        if any(token in lowered for token in ("operation plan", "financials", "annual report")):
            continue
        focuses.append(line)
    return _dedupe_keep_order(focuses)[:5]


def _doc_type(subject: str, filenames: List[str], text: str) -> str:
    haystack = normalize_lookup(" ".join([subject, *filenames, text[:2000]]))
    if "annual report" in haystack:
        return "annual_report"
    if any(token in haystack for token in ("strategic plan", "strategic direction", "strategy", "statement of strategic priorities", "roadmap")):
        return "strategic_plan"
    if any(token in haystack for token in ("industry report", "sector report")):
        return "industry_report"
    return "general"


def analyse_strategic_documents(
    attachments: List[Dict[str, Any]],
    extracted_summary: str,
    subject: str = "",
    raw_text: str = "",
    extraction_depth: str = "default",
) -> Dict[str, Any]:
    depth = str(extraction_depth or "default").strip().lower()
    if depth not in {"brief", "default", "detailed"}:
        depth = "default"
    processed = [item for item in attachments or [] if str(item.get("status") or "").strip().lower() == "processed"]
    names = [str(item.get("filename") or "").strip() for item in processed]
    text_blocks = [_read_document_text(item) for item in processed]
    combined_text = "\n\n".join(block for block in text_blocks if str(block or "").strip())
    document_lines = _compact_lines("\n".join([combined_text, extracted_summary]))
    lines = _compact_lines("\n".join([subject, raw_text, combined_text, extracted_summary]))
    doc_type = _doc_type(subject, names, combined_text or extracted_summary)
    vision = _extract_label_value(lines, "Vision")
    mission = _extract_label_value(lines, "Mission")
    values = _extract_label_value(lines, "Values")
    themes = _extract_marked_items(lines, _THEME_MARKERS, limit=6)
    initiatives = _extract_marked_items(lines, _INITIATIVE_MARKERS, limit=6)
    strategic_signals = _extract_strategic_signals(lines, "\n".join(lines))
    priorities = [str(item.get("headline") or "").strip() for item in strategic_signals if str(item.get("headline") or "").strip()]
    org_name = _extract_org_name(document_lines or lines, names)
    subject_org_name = _extract_org_name(_compact_lines(subject), names)
    if subject_org_name and (
        _looks_like_generated_attachment_name(org_name)
        or _looks_like_generic_org_placeholder(org_name)
        or _looks_like_noisy_org_candidate(org_name)
        or len(org_name.split()) < 2
        or normalize_lookup(org_name) in {"managing", "director"}
    ):
        org_name = subject_org_name
    priority_plan_signals = _extract_priority_plan_signals(document_lines or lines, limit=8) if doc_type == "strategic_plan" else []
    explicit_themes = _extract_explicit_strategy_themes(document_lines or lines, limit=6) if doc_type == "strategic_plan" else []
    ambition_signals = _extract_ambition_focus_area_signals(document_lines or lines, limit=8) if doc_type == "strategic_plan" else []
    annual_report_framework_signals = _extract_annual_report_framework_signals(document_lines or lines, combined_text or extracted_summary, limit=6) if doc_type == "annual_report" else []
    if priority_plan_signals and len(priority_plan_signals) >= 3:
        themes = [str(item.get("headline") or "").strip() for item in priority_plan_signals if str(item.get("headline") or "").strip()]
        strategic_signals = priority_plan_signals
        priorities = themes[:]
    elif annual_report_framework_signals and len(annual_report_framework_signals) >= 3:
        themes = [str(item.get("headline") or "").strip() for item in annual_report_framework_signals if str(item.get("headline") or "").strip()]
        strategic_signals = annual_report_framework_signals
        priorities = themes[:]
    elif ambition_signals and len(ambition_signals) >= 3:
        themes = [str(item.get("headline") or "").strip() for item in ambition_signals if str(item.get("headline") or "").strip()]
        strategic_signals = ambition_signals
        priorities = themes[:]
    elif explicit_themes and len(explicit_themes) >= 3:
        themes = [str(item.get("headline") or "").strip() for item in explicit_themes if str(item.get("headline") or "").strip()]
        strategic_signals = [
            {
                "headline": str(item.get("headline") or "").strip(),
                "category": "strategic_theme",
                "snippet": str(item.get("snippet") or "").strip(),
                "evidence": str(item.get("snippet") or item.get("headline") or "").strip(),
            }
            for item in explicit_themes
            if str(item.get("headline") or "").strip()
        ]
        priorities = [str(item.get("headline") or "").strip() for item in strategic_signals if str(item.get("headline") or "").strip()]
    leadership_people = _extract_leadership_people(document_lines or lines, org_name)
    performance_indicators = _extract_performance_indicators("\n".join(lines), doc_type)
    major_projects = _extract_major_projects("\n".join(lines), doc_type)
    kpi_focuses = _extract_kpi_focuses(lines, doc_type)
    key_stakeholders = [
        {
            "name": str(item.get("name") or "").strip(),
            "current_role": str(item.get("current_role") or "").strip(),
            "current_employer": str(item.get("current_employer") or "").strip(),
            "evidence": str(item.get("evidence") or "").strip(),
        }
        for item in leadership_people
        if str(item.get("name") or "").strip() and _should_keep_extracted_stakeholder(item, org_name)
    ]

    if depth == "brief":
        themes = themes[:3]
        initiatives = initiatives[:3]
        strategic_signals = strategic_signals[:2]
        priorities = priorities[:2]
        leadership_people = leadership_people[:4]
        performance_indicators = performance_indicators[:3]
        major_projects = major_projects[:3]
        kpi_focuses = kpi_focuses[:2]
        key_stakeholders = key_stakeholders[:4]
    elif depth == "default":
        strategic_signals = strategic_signals[:7]
        priorities = priorities[:7]
        leadership_people = leadership_people[:8]
        performance_indicators = performance_indicators[:8]
        major_projects = major_projects[:8]
        kpi_focuses = kpi_focuses[:4]
        key_stakeholders = key_stakeholders[:6]
    else:
        strategic_signals = strategic_signals[:8]
        priorities = priorities[:8]
        leadership_people = leadership_people[:12]
        performance_indicators = performance_indicators[:10]
        major_projects = major_projects[:10]
        kpi_focuses = kpi_focuses[:6]
        key_stakeholders = key_stakeholders[:10]

    summary_parts: List[str] = []
    if org_name:
        summary_parts.append(f"Document appears to relate to {org_name}.")
    if strategic_signals:
        summary_parts.append(
            f"Key strategic signals: {', '.join(item['headline'] for item in strategic_signals[:(2 if depth == 'brief' else 4 if depth == 'default' else 6)])}."
        )
    elif themes:
        summary_parts.append(f"Detected strategy themes: {', '.join(themes[:(2 if depth == 'brief' else 3 if depth == 'default' else 5)])}.")
    if initiatives:
        summary_parts.append(f"Possible key initiatives: {', '.join(initiatives[:(2 if depth == 'brief' else 3 if depth == 'default' else 5)])}.")
    if performance_indicators:
        summary_parts.append(
            "Performance snapshot: "
            + ", ".join(
                f"{item['label']} ({item['value']})"
                for item in performance_indicators[:(2 if depth == 'brief' else 4 if depth == 'default' else 6)]
                if str(item.get("value") or "").strip()
            )
            + "."
        )
    if major_projects:
        summary_parts.append(
            "Key projects: "
            + ", ".join(
                f"{item['name']} ({item['value']})"
                for item in major_projects[:(2 if depth == 'brief' else 4 if depth == 'default' else 6)]
                if str(item.get("name") or "").strip() and str(item.get("value") or "").strip()
            )
            + "."
        )
    if key_stakeholders:
        summary_parts.append(
            "Key stakeholders: "
            + ", ".join(
                f"{item['name']} ({item['current_role']})"
                for item in key_stakeholders[:(2 if depth == 'brief' else 3 if depth == 'default' else 5)]
                if str(item.get("current_role") or "").strip()
            )
            + "."
        )

    return {
        "attachment_count": len(processed),
        "attachment_names": names,
        "doc_type": doc_type,
        "org_name": org_name,
        "vision": vision,
        "mission": mission,
        "values": values,
        "themes": themes,
        "initiatives": initiatives,
        "priorities": priorities,
        "leadership_people": leadership_people,
        "key_stakeholders": key_stakeholders,
        "performance_indicators": performance_indicators,
        "major_projects": major_projects,
        "kpi_focuses": kpi_focuses,
        "strategic_signals": strategic_signals,
        "extraction_depth": depth,
        "has_strategy_markers": bool(strategic_signals or themes or initiatives or vision or mission or values or doc_type in {"strategic_plan", "annual_report"}),
        "strategic_summary": " ".join(summary_parts).strip(),
    }
