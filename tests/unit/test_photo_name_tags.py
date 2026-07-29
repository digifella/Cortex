"""Tests for post-hoc person-name substitution in photo descriptions."""
import pytest

from cortex_engine.photo_name_tags import (
    DEFAULT_NAME_TAGS,
    apply_names,
    names_from_keywords,
    parse_name_tags,
)


class TestParseNameTags:
    def test_empty_returns_defaults(self):
        assert parse_name_tags("") == DEFAULT_NAME_TAGS

    def test_parses_pairs_and_lowercases_tags(self):
        assert parse_name_tags("Paul_C=Paul, Jacqui_C=Jacqui") == {
            "paul_c": "Paul",
            "jacqui_c": "Jacqui",
        }

    def test_skips_malformed_entries(self):
        assert parse_name_tags("Paul_C=Paul, garbage, =NoTag, Tag=") == {"paul_c": "Paul"}

    def test_all_malformed_falls_back_to_defaults(self):
        assert parse_name_tags("garbage, more garbage") == DEFAULT_NAME_TAGS


class TestNamesFromKeywords:
    def test_matches_case_insensitively(self):
        # Real libraries tag "Paul_C"; the mapping keys are lowercased.
        assert names_from_keywords(["Burleigh_Heads", "Paul_C"]) == ["Paul"]

    def test_returns_both_in_mapping_order(self):
        assert names_from_keywords(["jacqui_c", "paul_c"]) == ["Paul", "Jacqui"]

    def test_no_person_tags(self):
        assert names_from_keywords(["beach", "sunset"]) == []


class TestApplyNames:
    def test_names_single_subject(self):
        assert apply_names("A man smiles at the camera.", ["Paul_C"]) == \
            "Paul smiles at the camera."

    def test_names_subject_mid_sentence(self):
        out = apply_names(
            "A sandy path leads to where a solitary figure walks toward the ocean.",
            ["Paul_C"],
        )
        assert "Paul walks toward the ocean" in out

    def test_names_pair_from_two_tags(self):
        assert apply_names(
            "Two smiling adults sit at a table.", ["Paul_C", "Jacqui_C"]
        ) == "Paul and Jacqui sit at a table."

    def test_names_explicit_man_and_woman_pair(self):
        assert apply_names(
            "A man and a woman stand together.", ["paul_c", "jacqui_c"]
        ) == "Paul and Jacqui stand together."

    def test_leaves_plural_subject_alone_when_only_one_name(self):
        # Two women but only Jacqui tagged — naming one would be a guess.
        text = "Two women with gray hair stand smiling."
        assert apply_names(text, ["Jacqui_C"]) == text

    def test_unchanged_without_person_phrase(self):
        text = "A dark cocktail sits garnished with cucumber."
        assert apply_names(text, ["Paul_C"]) == text

    def test_unchanged_without_name_tags(self):
        text = "Surfers ride waves at dawn."
        assert apply_names(text, ["beach", "dawn"]) == text

    def test_placeholder_untouched(self):
        assert apply_names("[Image: logo/icon omitted]", ["Paul_C"]) == \
            "[Image: logo/icon omitted]"

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_description(self, value):
        assert apply_names(value, ["Paul_C"]) == value

    def test_custom_mapping(self):
        out = apply_names("A person waves.", ["dog_c"], {"dog_c": "Rex"})
        assert out == "Rex waves."


class TestSeparatorTolerance:
    """Lightroom keyword entry is inconsistent about separators.

    A real library carried Paul_C, paul c and jacqui_c simultaneously; the space
    variant silently never matched, so those photos were never named.
    """

    @pytest.mark.parametrize("tag", ["Paul_C", "paul_c", "paul c", "paul-c",
                                     "PAUL C", " Paul_C "])
    def test_variants_all_resolve(self, tag):
        assert names_from_keywords([tag]) == ["Paul"]

    @pytest.mark.parametrize("tag", ["paulc", "paul", "paula_c", "paul_d"])
    def test_near_misses_do_not_match(self, tag):
        # Guarding against false positives on unrelated tags.
        assert names_from_keywords([tag]) == []

    def test_mixed_separators_across_two_people(self):
        assert apply_names("A man and a woman walk the beach.",
                           ["paul c", "Jacqui-C"]) == "Paul and Jacqui walk the beach."

    def test_custom_mapping_keys_are_also_normalised(self):
        assert apply_names("A person waves.", ["dog c"], {"dog_c": "Rex"}) == "Rex waves."


class TestRetroApplication:
    """Photos tagged after captioning keep generic phrasing until re-applied."""

    def test_generic_caption_becomes_named(self):
        caption = ("The photograph captures a lively outdoor market scene in Burleigh "
                   "Waters, Queensland, Australia. A woman is walking past several stalls.")
        out = apply_names(caption, ["jacqui_c"])
        assert "Jacqui is walking past several stalls" in out
        assert "A woman is walking" not in out

    def test_already_named_caption_is_untouched(self):
        caption = "There's Paul walking, trees on the sides, ocean in the background."
        assert apply_names(caption, ["Paul_C"]) == caption

    def test_tagged_photo_without_a_person_phrase_is_untouched(self):
        caption = "A table draped with a white tablecloth serves as the centrepiece."
        assert apply_names(caption, ["jacqui_c"]) == caption
