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
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List

# Default keyword -> display name mapping. Keys are compared case-insensitively.
DEFAULT_NAME_TAGS: Dict[str, str] = {
    "paul_c": "Paul",
    "jacqui_c": "Jacqui",
    "greg_r": "Greg",
    "steve_c": "Steve",
    "jenny_s": "Jenny",
    "kay_w": "Kay",
    "mike_s": "Mike",
}

# Up to three descriptive words may sit between the article and the noun,
# e.g. "a solitary figure", "an elderly man", "two smiling adults".
#
# Articles and "and" are excluded so the run cannot bridge two person-phrases:
# without that, "A man and a woman" matched as article "A" + adjectives
# "man and a " + noun "woman", so a single known name replaced the whole phrase
# and the second person disappeared ("Paul hold drinks and pose together").
_ADJ = r"(?:(?!(?:and|a|an|the)\s)[a-z]+(?:ly)?\s+){0,3}"

# Nouns that stand in for an unnamed person.
_PERSON = r"(?:man|woman|person|figure|individual|adult|hiker|walker|surfer|guy|lady)"

# Grammatical gender of those nouns, so a single known name is never written
# onto the wrong person. A photo tagged only ``Paul_C`` whose caption reads
# "a woman in a red coat" is describing somebody else; renaming her "Paul" is
# worse than leaving the caption generic.
_FEMALE = re.compile(r"\b(?:woman|lady|girl)\b", re.IGNORECASE)
_MALE = re.compile(r"\b(?:man|guy|gentleman|boy)\b", re.IGNORECASE)

# Gender by display name. Names absent here impose no constraint, so every name
# added to DEFAULT_NAME_TAGS belongs here too — otherwise the guard above
# silently stops applying to that person.
DEFAULT_NAME_GENDER: Dict[str, str] = {
    "Paul": "M",
    "Jacqui": "F",
    "Greg": "M",
    "Steve": "M",
    "Jenny": "F",
    "Kay": "F",
    "Mike": "M",
}


def _phrase_gender(phrase: str) -> str:
    """Return 'M', 'F' or '' for a matched person-phrase.

    ``_FEMALE`` is tested first because "woman" contains "man".
    """
    if _FEMALE.search(phrase):
        return "F"
    if _MALE.search(phrase):
        return "M"
    return ""

# Plural forms standing in for two unnamed people. ``men`` and ``women`` must be
# listed explicitly: the singular nouns are pluralised as ``{_PERSON}s``, which
# yields "mans"/"womans" and so matched neither. That gap left captions like
# "Two men smile warmly at a community center" generic even when both people
# were tagged and known.
_PEOPLE = (rf"(?:{_PERSON}s|women|men|gentlemen|ladies|people|adults|figures"
           r"|individuals|friends)")

# Two-person phrases, tried first when two names are known.
_PAIR_PATTERNS = (
    rf"\b(?:a|an|one)\s+{_ADJ}{_PERSON}\s+and\s+(?:a|an|one)\s+{_ADJ}{_PERSON}\b",
    rf"\b(?:two|both)\s+{_ADJ}{_PEOPLE}\b",
    rf"\b(?:a|an)\s+{_ADJ}couple\b",
)

# Single-person phrases.
_SINGLE_PATTERNS = (
    rf"\b(?:a|an|one|the)\s+{_ADJ}{_PERSON}\b",
)


# The roster of real people lives OUTSIDE this repo, in the people registry:
# it is personal data, and this package has a public git remote. The names kept
# above are a minimal fallback so the module works with no registry present.
REGISTRY_PATH = Path(
    os.environ.get("CORTEX_PEOPLE_REGISTRY",
                   Path.home() / "vault-rag-db" / "people-registry.json"))


def load_registry(path: Path = None) -> int:
    """Merge the people registry into the module's name and gender maps.

    Callers opt in explicitly rather than this loading at import time, so the
    pure functions stay testable without a registry on disk. Returns the number
    of tags merged; 0 when the file is absent, which is a normal no-op.

    Each tag carries its own display name: a person tagged under both a maiden
    and a married name is one identity in the registry, but the caption should
    use whichever name that tag was given.
    """
    path = Path(path) if path else REGISTRY_PATH
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    merged = 0
    for person in data.get("people", []):
        gender = (person.get("gender") or "").strip().upper()[:1]
        for tag, name in (person.get("tag_names") or {}).items():
            if not tag or not name:
                continue
            DEFAULT_NAME_TAGS[_normalise_tag(tag)] = name
            if gender:
                DEFAULT_NAME_GENDER[name] = gender
            merged += 1
    return merged


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


def _normalise_tag(tag: str) -> str:
    """Fold a keyword to a comparable form.

    Real libraries carry the same person as ``Paul_C``, ``paul c`` and
    ``paul-c`` — Lightroom keyword entry is inconsistent about separators, and a
    tag that only differs by a space would otherwise silently never match.
    """
    return re.sub(r"[\s\-_]+", "_", str(tag or "").strip().lower())


def names_from_keywords(
    keywords: Iterable[str],
    name_tags: Dict[str, str] = None,
) -> List[str]:
    """Return display names for any person-tags present, in mapping order."""
    tags = name_tags if name_tags is not None else DEFAULT_NAME_TAGS
    present = {_normalise_tag(k) for k in (keywords or [])}
    return [name for tag, name in tags.items() if _normalise_tag(tag) in present]


def _sub_first(patterns: Iterable[str], text: str, replacement: str):
    """Replace the first match of the first matching pattern."""
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return text[: match.start()] + replacement + text[match.end():], True
    return text, False


def _sub_person(text: str, name: str):
    """Substitute *name* onto the first person-phrase it can plausibly be.

    The leading phrase is the subject, so a gender-neutral noun there ("a
    person") is accepted. Once that phrase is rejected on gender, only an
    explicitly matching noun later in the sentence is used: a neutral noun
    further along is usually not a person at all — "An elderly man using a
    walker" would otherwise become "An elderly man using Jacqui".
    """
    want = DEFAULT_NAME_GENDER.get(name, "")
    first = True
    for pattern in _SINGLE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            gender = _phrase_gender(match.group(0))
            if not want:
                ok = True
            elif first:
                ok = gender in ("", want)
            else:
                ok = gender == want
            first = False
            if ok:
                return text[: match.start()] + name + text[match.end():], True
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

    # A name already in the caption must not be substituted again. The vision
    # model often names the subject and then refers to a *different* person
    # generically ("Jacqui and two children ... A blurred figure appears in the
    # foreground"), and filling her name into that phrase asserts something
    # false. Filtering is per-name, not per-caption, so "Paul and a woman"
    # tagged Paul_C + jenny_s still resolves to "Paul and Jenny".
    names = [n for n in names
             if not re.search(rf"\b{re.escape(n)}\b", text, flags=re.IGNORECASE)]
    if not names:
        return description

    if len(names) >= 2:
        joined = f"{names[0]} and {names[1]}"
        text, matched = _sub_first(_PAIR_PATTERNS, text, joined)
        if matched:
            return _tidy(text)
        # No pair phrase — name whichever person is actually described.
        for name in names:
            text, matched = _sub_person(text, name)
            if matched:
                return _tidy(text)
        return description

    text, matched = _sub_person(text, names[0])
    return _tidy(text) if matched else description
