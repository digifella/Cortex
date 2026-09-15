"""Tests for post-hoc person-name substitution in photo descriptions."""
import pytest

from cortex_engine.photo_name_tags import (
    DEFAULT_NAME_GENDER,
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

    def test_one_name_on_a_pair_phrase_keeps_the_other_person(self):
        # Only Paul is tagged but the photo shows two people. The adjective run
        # must not swallow "man and a " and bind to "woman", which dropped the
        # second person entirely and left "Paul hold drinks and pose together".
        assert apply_names(
            "A man and a woman hold drinks and pose together.", ["paul_c"]
        ) == "Paul and a woman hold drinks and pose together."

    def test_one_name_does_not_consume_trailing_person(self):
        assert apply_names(
            "A woman and a man walk along the beach.", ["jacqui_c"]
        ) == "Jacqui and a man walk along the beach."

    def test_does_not_name_a_woman_paul(self):
        # Photo tagged Paul_C only, but the caption describes a woman — the
        # subject is somebody else and must stay generic.
        text = "A woman in a red coat waits by the door."
        assert apply_names(text, ["paul_c"]) == text

    def test_does_not_name_a_man_jacqui(self):
        text = "An elderly man using a walker moves along a brick pathway."
        assert apply_names(text, ["jacqui_c"]) == text

    def test_still_names_matching_gender(self):
        assert apply_names("A woman in a red coat waits.", ["jacqui_c"]) == \
            "Jacqui in a red coat waits."
        assert apply_names("An elderly man reads.", ["paul_c"]) == "Paul reads."

    def test_gender_neutral_phrase_still_named(self):
        assert apply_names("A person stands by the lake.", ["paul_c"]) == \
            "Paul stands by the lake."

    def test_skips_mismatched_noun_and_uses_a_later_valid_one(self):
        # "a woman" must be passed over for Paul, but "a man" later still names him.
        assert apply_names("A woman waves as a man boards the train.", ["paul_c"]) == \
            "A woman waves as Paul boards the train."

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


class TestExtendedNameRoster:
    """People beyond Paul/Jacqui, added 2026-08-18.

    A photo tagged ``Paul_C`` + ``greg_r`` kept the caption "Two men smile
    warmly at a community center" because only one of the two was a known name:
    the pair patterns are tried only when two names resolve, and a lone name
    falls through to the single-person patterns, which cannot match "Two men".
    """

    def test_pair_phrase_names_both_people(self):
        caption = ("Two men smile warmly at a community center in Melbourne, "
                   "with one wearing a Delphi cap.")
        out = apply_names(caption, ["Paul_C", "greg_r"])
        assert out.startswith("Paul and Greg smile warmly")
        assert "Two men" not in out

    @pytest.mark.parametrize("tag,expected", [
        ("greg_r", "Greg"), ("steve_c", "Steve"), ("jenny_s", "Jenny"),
        ("kay_w", "Kay"), ("mike_s", "Mike"),
    ])
    def test_each_new_tag_resolves(self, tag, expected):
        assert names_from_keywords([tag]) == [expected]

    @pytest.mark.parametrize("tag", ["Mike_s", "MIKE_S", "mike s", "Mike-S", " mike_s "])
    def test_new_tags_are_case_and_separator_insensitive(self, tag):
        assert names_from_keywords([tag]) == ["Mike"]

    def test_every_default_name_has_a_gender(self):
        # A name missing from DEFAULT_NAME_GENDER imposes no constraint, which
        # silently disables the wrong-gender guard for that person.
        missing = [n for n in DEFAULT_NAME_TAGS.values()
                   if n not in DEFAULT_NAME_GENDER]
        assert missing == [], f"names without a gender: {missing}"

    def test_new_male_name_is_not_written_onto_a_woman(self):
        caption = "A woman in a red coat waits by the door."
        assert apply_names(caption, ["greg_r"]) == caption

    def test_new_female_name_is_not_written_onto_a_man(self):
        caption = "A man in a red coat waits by the door."
        assert apply_names(caption, ["kay_w"]) == caption

    @pytest.mark.parametrize("plural", [
        "Two men", "Two women", "Both men", "Two smiling men",
        "Two elderly women", "Two people", "Two adults", "Two friends",
    ])
    def test_irregular_and_regular_plurals_all_match(self, plural):
        # "men"/"women" are irregular: the singular nouns pluralise to
        # "mans"/"womans", so they matched nothing before 2026-08-18.
        out = apply_names(f"{plural} stand by the gate.", ["Paul_C", "jacqui_c"])
        assert out.startswith("Paul and Jacqui stand by the gate")

    def test_plural_without_two_known_names_is_untouched(self):
        # Only one name resolves, so naming a pair would be a guess.
        caption = "Two men stand by the gate."
        assert apply_names(caption, ["Paul_C"]) == caption


class TestNameAlreadyPresent:
    """A name already in the caption must not be substituted a second time.

    The 2026-08-18 dry run proposed 94 such rewrites out of 367, and some
    asserted something false: "Jacqui and two children stand outdoors. A
    blurred figure appears in the foreground." became "... Jacqui appears in
    the foreground", naming a background figure who is demonstrably someone
    else.
    """

    def test_does_not_name_a_second_person_with_the_same_name(self):
        caption = ("Jacqui and two children stand outdoors in a garden setting. "
                   "A blurred figure appears in the foreground.")
        assert apply_names(caption, ["jacqui_c"]) == caption

    def test_does_not_rename_a_later_generic_reference_to_the_subject(self):
        caption = "Jacqui and a young child stand on steps. The woman wears a light coat."
        assert apply_names(caption, ["jacqui_c"]) == caption

    def test_other_tagged_person_is_still_named(self):
        # Paul is already named; Jenny is not, so the generic phrase is hers.
        caption = "Paul and a woman pose together indoors."
        out = apply_names(caption, ["paul_c", "jenny_s"])
        assert out == "Paul and Jenny pose together indoors."

    def test_pair_substitution_still_works_when_neither_is_named(self):
        assert apply_names("Two men stand by the gate.", ["paul_c", "greg_r"]) == \
            "Paul and Greg stand by the gate."

    def test_matching_is_case_insensitive(self):
        caption = "PAUL waves from the shore while a man rows past."
        assert apply_names(caption, ["paul_c"]) == caption


class TestRegistryLoading:
    """The roster of real people lives outside the repo (personal data)."""

    @pytest.fixture(autouse=True)
    def _restore_maps(self):
        # load_registry mutates the module-level maps, which would otherwise
        # leak into whatever test runs next.
        tags, gender = dict(DEFAULT_NAME_TAGS), dict(DEFAULT_NAME_GENDER)
        yield
        DEFAULT_NAME_TAGS.clear(); DEFAULT_NAME_TAGS.update(tags)
        DEFAULT_NAME_GENDER.clear(); DEFAULT_NAME_GENDER.update(gender)

    def _registry(self, tmp_path):
        import json
        p = tmp_path / "people-registry.json"
        p.write_text(json.dumps({"people": [
            {"id": "gemma_c", "display_name": "Gemma Crow", "gender": "F",
             "tag_names": {"gemma_c": "Gemma Crow", "gemma_d": "Gemma Dunstall"}},
            {"id": "donald_c", "display_name": "Donald Cooper", "gender": "M",
             "tag_names": {"donald_c": "Donald Cooper"}},
        ]}))
        return p

    def test_missing_registry_is_a_no_op(self, tmp_path):
        from cortex_engine.photo_name_tags import load_registry
        assert load_registry(tmp_path / "nope.json") == 0

    def test_tags_and_genders_are_merged(self, tmp_path):
        from cortex_engine.photo_name_tags import load_registry
        assert load_registry(self._registry(tmp_path)) == 3
        assert names_from_keywords(["donald_c"]) == ["Donald Cooper"]
        assert DEFAULT_NAME_GENDER["Donald Cooper"] == "M"

    def test_each_tag_keeps_its_own_display_name(self, tmp_path):
        # One identity, two tags: the caption uses the name that tag was given.
        from cortex_engine.photo_name_tags import load_registry
        load_registry(self._registry(tmp_path))
        assert names_from_keywords(["gemma_c"]) == ["Gemma Crow"]
        assert names_from_keywords(["gemma_d"]) == ["Gemma Dunstall"]

    def test_registry_gender_guard_applies(self, tmp_path):
        from cortex_engine.photo_name_tags import load_registry
        load_registry(self._registry(tmp_path))
        text = "A man in a red coat waits by the door."
        assert apply_names(text, ["gemma_c"]) == text          # female name, male noun
        assert apply_names(text, ["donald_c"]) == "Donald Cooper in a red coat waits by the door."
