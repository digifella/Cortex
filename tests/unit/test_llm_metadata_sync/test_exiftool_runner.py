from pathlib import Path

from cortex_engine.llm_metadata_sync import exiftool_runner
from cortex_engine.llm_metadata_sync.models import TargetType


def test_clear_keyword_lists_png_uses_xmp_only(monkeypatch, tmp_path):
    target = tmp_path / "shot.png"
    target.touch()
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(exiftool_runner, "exiftool_path", lambda: "/usr/bin/exiftool")

    def fake_run(args: list[str]):
        captured["args"] = args
        return exiftool_runner.RunResult(0, "", "", args)

    monkeypatch.setattr(exiftool_runner, "_run", fake_run)

    exiftool_runner.clear_keyword_lists(target, TargetType.EMBEDDED, keep_backups=True)

    assert captured["args"] == [
        "/usr/bin/exiftool",
        "-xmp-dc:subject=",
        str(target),
    ]


def test_write_metadata_png_uses_xmp_only(monkeypatch, tmp_path):
    jpg = tmp_path / "source.jpg"
    target = tmp_path / "shot.png"
    jpg.touch()
    target.touch()
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(exiftool_runner, "exiftool_path", lambda: "/usr/bin/exiftool")

    def fake_run(args: list[str]):
        captured["args"] = args
        return exiftool_runner.RunResult(0, "", "", args)

    monkeypatch.setattr(exiftool_runner, "_run", fake_run)

    exiftool_runner.write_metadata(
        jpg=jpg,
        target=target,
        target_type=TargetType.EMBEDDED,
        keywords=["bird", "wetland"],
        description="Two birds.",
        keep_backups=True,
        location_fields={"city", "country"},
        rating=5,
    )

    args = captured["args"]
    assert args[0] == "/usr/bin/exiftool"
    assert "-overwrite_original_in_place" not in args
    assert "-xmp-dc:subject+=bird" in args
    assert "-xmp-dc:subject+=wetland" in args
    assert "-iptc:Keywords+=bird" not in args
    assert "-iptc:Keywords+=wetland" not in args
    assert "-xmp-dc:description<iptc:Caption-Abstract" in args
    assert "-XMP-xmp:Rating<XMP-xmp:Rating" in args
    assert "-iptc:Caption-Abstract<iptc:Caption-Abstract" not in args
    assert "-XMP-photoshop:City<XMP-photoshop:City" in args
    assert "-IPTC:City<XMP-photoshop:City" not in args
    assert "-XMP-photoshop:Country<XMP-photoshop:Country" in args
    assert "-IPTC:Country-PrimaryLocationName<XMP-photoshop:Country" not in args
    assert args[-1] == str(target)


def test_clear_keyword_lists_without_backups_uses_in_place_overwrite(monkeypatch, tmp_path):
    target = tmp_path / "shot.png"
    target.touch()
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(exiftool_runner, "exiftool_path", lambda: "/usr/bin/exiftool")

    def fake_run(args: list[str]):
        captured["args"] = args
        return exiftool_runner.RunResult(0, "", "", args)

    monkeypatch.setattr(exiftool_runner, "_run", fake_run)

    exiftool_runner.clear_keyword_lists(target, TargetType.EMBEDDED, keep_backups=False)

    assert captured["args"] == [
        "/usr/bin/exiftool",
        "-overwrite_original_in_place",
        "-xmp-dc:subject=",
        str(target),
    ]
