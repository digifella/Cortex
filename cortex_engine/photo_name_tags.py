"""Substitute known people's names into generated photo descriptions.

A vision model describing a family photo writes "A man smiles at the camera".
When the photo already carries a person keyword (e.g. ``Paul_C``) the subject is
known, and the description should say "Paul smiles at the camera".

This runs *after* the model, as a text transform. That is deliberate: asking a
small local model to use a supplied name reliably is exactly the kind of
instruction-following these models fail at, whereas a post-hoc rewrite is
deterministic and testable.

The transform is conservative by design — when it cannot map names to subjects
unambiguously it leaves the description untouched rather than guessing. A wrong
name in a photo archive is worse than a generic one.
"""
import re
from typing import Dict, Iterable, List

# Default keyword -> display name mapping. Keys are compared case-insensitively.
DEFAULT_NAME_TAGS: Dict[str, str] = {
    "paul_c": "Paul",
    "jacqui_c": "Jacqui",
}

# Up to three descriptive words may sit between the article and the noun,
# e.g. "a solitary figure", "an elderly man", "two smiling adults".
_ADJ = r"(?:[a-z]+(?:ly)?\s+){0,3}"

# Nouns that stand in for an unnamed person.
_PERSON = r"(?:man|woman|person|figure|individual|adult|hiker|walker|surfer|guy|lady)"

# Two-person phrases, tried first when two names are known.
_PAIR_PATTERNS = (
    rf"\b(?:a|an|one)\s+{_ADJ}{_PERSON}\s+and\s+(?:a|an|one)\s+{_ADJ}{_PERSON}\b",
    rf"\b(?:two|both)\s+{_ADJ}(?:{_PERSON}s|people|adults|figures|individuals)\b",
    rf"\b(?:a|an)\s+{_ADJ}couple\b",
)

# Single-person phrases.
_SINGLE_PATTERNS = (
    rf"\b(?:a|an|one|the)\s+{_ADJ}{_PERSON}\b",
)


def parse_name_tags(raw: str) -> Dict[str, str]:
    """Parse a ``Tag=Name, Tag=Name`` string into a mapping.

    Invalid entries are skipped. Returns the defaults when *raw* is empty.
    """
    text = (raw or "").strip()
    if not text:
        return dict(DEFAULT_NAME_TAGS)
    mapping: Dict[str, str] = {}
    for chunk in text.split(","):
        if "=" not in chunk:
            continue
        tag, name = chunk.split("=", 1)
        tag, name = tag.strip().lower(), name.strip()
        if tag and name:
            mapping[tag] = name
    return mapping or dict(DEFAULT_NAME_TAGS)


def names_from_keywords(
    keywords: Iterable[str],
    name_tags: Dict[str, str] = None,
) -> List[str]:
    """Return display names for any person-tags present, in mapping order."""
    tags = name_tags if name_tags is not None else DEFAULT_NAME_TAGS
    lowered = {str(k).strip().lower() for k in (keywords or [])}
    return [name for tag, name in tags.items() if tag in lowered]


def _sub_first(patterns: Iterable[str], text: str, replacement: str):
    """Replace the first match of the first matching pattern."""
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return text[: match.start()] + replacement + text[match.end():], True
    return text, False


def _tidy(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text[:1].upper() + text[1:] if text else text


def apply_names(
    description: str,
    keywords: Iterable[str],
    name_tags: Dict[str, str] = None,
) -> str:
    """Rewrite *description* to name people identified by *keywords*.

    Returns the description unchanged when no person-tag is present, when the
    description is a placeholder, or when no generic person-reference is found.
    """
    text = (description or "").strip()
    if not text or text.startswith("[Image:"):
        return description

    names = names_from_keywords(keywords, name_tags)
    if not names:
        return description

    if len(names) >= 2:
        joined = f"{names[0]} and {names[1]}"
        text, matched = _sub_first(_PAIR_PATTERNS, text, joined)
        if matched:
            return _tidy(text)
        # No pair phrase — name the first person mentioned instead.
        text, matched = _sub_first(_SINGLE_PATTERNS, text, names[0])
        return _tidy(text) if matched else description

    text, matched = _sub_first(_SINGLE_PATTERNS, text, names[0])
    return _tidy(text) if matched else description
