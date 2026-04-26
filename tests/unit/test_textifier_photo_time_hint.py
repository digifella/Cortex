import datetime

import pytest

from cortex_engine.textifier import DocumentTextifier


# ------------------------------------------------------------------
# Filename datetime extraction
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("2026-04-05T16-27_study.jpg", datetime.datetime(2026, 4, 5, 16, 27, 0)),
        ("2026-04-05T16-27-03_handoff.json", datetime.datetime(2026, 4, 5, 16, 27, 3)),
        ("IMG_20260405_061423.jpg", datetime.datetime(2026, 4, 5, 6, 14, 23)),
        ("PXL_20260405_061423123.jpg", datetime.datetime(2026, 4, 5, 6, 14, 23)),
        ("DSC_20260405_061423.jpg", datetime.datetime(2026, 4, 5, 6, 14, 23)),
        ("20260405_061423.jpg", datetime.datetime(2026, 4, 5, 6, 14, 23)),
        ("Screenshot_20260405-061423.png", datetime.datetime(2026, 4, 5, 6, 14, 23)),
    ],
)
def test_parse_filename_capture_datetime_recognises_common_patterns(name, expected):
    result = DocumentTextifier._parse_filename_capture_datetime(name)
    assert result is not None
    assert result["datetime_naive"] == expected
    assert result["offset_minutes"] is None
    assert result["source"].startswith("filename:")


@pytest.mark.parametrize(
    "name",
    [
        "",
        "random_file.jpg",
        "Annual_Report_2024.pdf",
        "report_2024Q1.pdf",  # year only, should not match
        "2026-13-45T99-99.jpg",  # invalid date components
    ],
)
def test_parse_filename_capture_datetime_rejects_non_datetime_names(name):
    assert DocumentTextifier._parse_filename_capture_datetime(name) is None


# ------------------------------------------------------------------
# EXIF timezone offset parser
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset, expected",
    [
        ("+10:00", 600),
        ("-05:30", -330),
        ("+0000", 0),
        ("+1000", 600),
        ("-0800", -480),
    ],
)
def test_parse_exif_tz_offset_minutes_parses_common_formats(offset, expected):
    assert DocumentTextifier._parse_exif_tz_offset_minutes(offset) == expected


@pytest.mark.parametrize("offset", ["", "bad", "+25:00", "1000", None])
def test_parse_exif_tz_offset_minutes_returns_none_for_invalid(offset):
    assert DocumentTextifier._parse_exif_tz_offset_minutes(offset or "") is None


# ------------------------------------------------------------------
# Sun-phase classification with astral
# ------------------------------------------------------------------


_PT_LONSDALE_GPS = (-38.28, 144.62)
_AEST_OFFSET_MINUTES = 10 * 60


def _phase_at(hour, minute, *, gps=_PT_LONSDALE_GPS, offset_minutes=_AEST_OFFSET_MINUTES):
    ctx = DocumentTextifier._compute_photo_time_context(
        datetime.datetime(2026, 4, 22, hour, minute, 0),
        offset_minutes=offset_minutes,
        gps=gps,
    )
    return ctx["phase"], ctx["source"]


def test_compute_time_context_labels_morning_twilight_and_sunrise_correctly():
    """At Pt Lonsdale in April, sunrise is ~06:54 AEST.

    Photos 40 min before sunrise must NOT be labelled evening — they need
    a morning-phase label so the VLM doesn't confuse dawn with dusk.
    """
    for hh, mm, expected_group in [
        (4, 30, {"night"}),
        (6, 0, {"pre-dawn"}),
        (6, 14, {"pre-dawn"}),
        (6, 30, {"dawn"}),
        (7, 5, {"sunrise"}),
        (10, 0, {"morning"}),
        (12, 30, {"midday"}),
        (15, 0, {"afternoon"}),
        (17, 40, {"sunset"}),
        (18, 0, {"dusk"}),
        (18, 30, {"post-dusk"}),
        (22, 0, {"night"}),
    ]:
        phase, source = _phase_at(hh, mm)
        assert source == "astral", (hh, mm, source)
        assert phase in expected_group, (hh, mm, phase, expected_group)


def test_compute_time_context_falls_back_to_hour_bucket_without_gps():
    ctx = DocumentTextifier._compute_photo_time_context(
        datetime.datetime(2026, 4, 22, 6, 14, 0),
        offset_minutes=None,
        gps=None,
    )
    assert ctx["source"] == "hour-bucket"
    assert ctx["phase"] == "dawn"
    assert ctx["sunrise_str"] is None
    assert ctx["sunset_str"] is None


# ------------------------------------------------------------------
# Hint formatting — verifies the prompt guidance resolves dawn/dusk ambiguity
# ------------------------------------------------------------------


def test_format_photo_time_hint_morning_phases_steer_away_from_dusk():
    for phase in ("pre-dawn", "dawn", "sunrise"):
        hint = DocumentTextifier._format_photo_time_hint(
            {
                "phase": phase,
                "local_time_str": "06:30 on 2026-04-22",
                "sunrise_str": "06:54",
                "sunset_str": "17:44",
                "source": "astral",
            }
        )
        assert "morning" in hint.lower()
        assert "dawn/sunrise" in hint
        assert "never as dusk/sunset" in hint


def test_format_photo_time_hint_evening_phases_steer_away_from_dawn():
    for phase in ("sunset", "dusk", "post-dusk"):
        hint = DocumentTextifier._format_photo_time_hint(
            {
                "phase": phase,
                "local_time_str": "18:00 on 2026-04-22",
                "sunrise_str": "06:54",
                "sunset_str": "17:44",
                "source": "astral",
            }
        )
        assert "evening" in hint.lower()
        assert "sunset/dusk" in hint
        assert "never as sunrise/dawn" in hint


def test_format_photo_time_hint_empty_context_returns_empty_string():
    assert DocumentTextifier._format_photo_time_hint({}) == ""
    assert DocumentTextifier._format_photo_time_hint(
        {"phase": "", "local_time_str": "12:00 on 2026-04-22"}
    ) == ""


# ------------------------------------------------------------------
# Public build_photo_time_hint: EXIF missing -> filename fallback
# ------------------------------------------------------------------


def test_build_photo_time_hint_falls_back_to_filename_without_exif(tmp_path, monkeypatch):
    """When EXIF read yields no datetime, the filename pattern must drive the hint."""
    monkeypatch.setattr(
        DocumentTextifier,
        "_read_exif_capture_datetime",
        staticmethod(lambda _path: None),
    )
    # Create an empty file whose name encodes an evening capture time
    photo = tmp_path / "2026-04-22T18-00_beach.jpg"
    photo.write_bytes(b"")
    hint = DocumentTextifier.build_photo_time_hint(str(photo), gps=_PT_LONSDALE_GPS)
    # 18:00 at Pt Lonsdale in April is after sunset (17:44) -> "dusk"
    assert "evening" in hint.lower()
    assert "sunset/dusk" in hint


def test_build_photo_time_hint_returns_empty_when_no_exif_and_no_pattern(tmp_path, monkeypatch):
    monkeypatch.setattr(
        DocumentTextifier,
        "_read_exif_capture_datetime",
        staticmethod(lambda _path: None),
    )
    photo = tmp_path / "random_photo.jpg"
    photo.write_bytes(b"")
    assert DocumentTextifier.build_photo_time_hint(str(photo), gps=_PT_LONSDALE_GPS) == ""


# ------------------------------------------------------------------
# _describe_with_model prompt integration
# ------------------------------------------------------------------


def test_describe_with_model_appends_context_hint_to_prompt(monkeypatch):
    textifier = DocumentTextifier()

    captured_prompts = []

    def fake_call(self, model, prompt, encoded):  # noqa: ARG001
        captured_prompts.append(prompt)
        return {"message": {"content": "A coastline at sunrise with soft warm light."}}

    monkeypatch.setattr(DocumentTextifier, "_call_ollama_chat_http", fake_call)

    textifier._describe_with_model(
        "qwen3-vl:8b",
        "fake-base64",
        simple_prompt=True,
        context_hint="Context: local capture time is 06:14 on 2026-04-22 (pre-dawn). Lighting is a morning scene.",
    )

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "06:14 on 2026-04-22" in prompt
    assert "morning scene" in prompt
    # Base prompt is still present
    assert "Describe this photograph" in prompt


def test_describe_with_model_without_hint_uses_base_prompt_only(monkeypatch):
    textifier = DocumentTextifier()
    captured_prompts = []

    def fake_call(self, model, prompt, encoded):  # noqa: ARG001
        captured_prompts.append(prompt)
        return {"message": {"content": "A building."}}

    monkeypatch.setattr(DocumentTextifier, "_call_ollama_chat_http", fake_call)

    textifier._describe_with_model("qwen3-vl:8b", "fake-base64", simple_prompt=True)
    assert len(captured_prompts) == 1
    assert "Context:" not in captured_prompts[0]
