from pathlib import Path
from unittest.mock import MagicMock, patch

from cortex_engine.photo_metadata_copy import (
    MetadataTargetType,
    PhotoMetadataCopyAction,
    PhotoMetadataCopyConfig,
    PhotoMetadataPayload,
    _write_args_for_target,
    run_photo_metadata_copy,
    scan_photo_metadata_copy,
)


def _cfg(tmp_path: Path, **kwargs) -> PhotoMetadataCopyConfig:
    defaults = {"folder": tmp_path}
    defaults.update(kwargs)
    return PhotoMetadataCopyConfig(**defaults)


def test_scan_matches_jpg_to_tif_dng_and_raw_sidecar(tmp_path):
    (tmp_path / "shot.jpg").touch()
    (tmp_path / "shot.tif").touch()
    (tmp_path / "shot.dng").touch()
    (tmp_path / "shot.RAF").touch()
    (tmp_path / "shot.xmp").touch()

    report = scan_photo_metadata_copy(_cfg(tmp_path))

    targets = {action.target_path.name for action in report.actions}
    assert targets == {"shot.tif", "shot.dng", "shot.xmp"}
    assert report.orphaned_jpgs == []


def test_scan_preserves_existing_xmp_as_target_for_raw(tmp_path):
    (tmp_path / "rawpair.jpg").touch()
    raw_path = tmp_path / "rawpair.RAF"
    raw_path.touch()
    xmp_path = tmp_path / "rawpair.xmp"
    xmp_path.touch()

    report = scan_photo_metadata_copy(_cfg(tmp_path))

    assert len(report.actions) == 1
    action = report.actions[0]
    assert action.target_type == MetadataTargetType.SIDECAR
    assert action.raw_path == raw_path
    assert action.target_path == xmp_path
    assert action.sidecar_exists is True


def test_scan_can_strip_rating_suffix_from_jpg_name(tmp_path):
    (tmp_path / "shot-5.jpg").touch()
    (tmp_path / "shot.RAF").touch()

    report = scan_photo_metadata_copy(_cfg(tmp_path, strip_rating_suffix=True))

    assert len(report.actions) == 1
    assert report.actions[0].target_path == tmp_path / "shot.xmp"


def test_existing_sidecar_write_args_are_surgical():
    action = PhotoMetadataCopyAction(
        jpg_path=Path("shot.jpg"),
        target_path=Path("shot.xmp"),
        target_type=MetadataTargetType.SIDECAR,
        raw_path=Path("shot.RAF"),
        sidecar_exists=True,
    )
    metadata = PhotoMetadataPayload(
        keywords=["alpha", "beta"],
        description="New description",
        caption="New caption",
        image_description="",
    )

    args = _write_args_for_target(action, metadata)

    assert args == [
        "-XMP-dc:Subject=",
        "-XMP-dc:Subject+=alpha",
        "-XMP-dc:Subject+=beta",
        "-XMP-dc:Description=New description",
    ]
    assert not any(arg in {"-all=", "-XMP:all="} for arg in args)
    assert not any(arg.startswith("-crs:") or arg.startswith("-XMP-crs:") for arg in args)


def test_embedded_write_args_copy_description_caption_and_keywords():
    action = PhotoMetadataCopyAction(
        jpg_path=Path("shot.jpg"),
        target_path=Path("shot.tif"),
        target_type=MetadataTargetType.EMBEDDED,
    )
    metadata = PhotoMetadataPayload(
        keywords=["alpha"],
        description="XMP description",
        caption="IPTC caption",
        image_description="",
    )

    args = _write_args_for_target(action, metadata)

    assert "-XMP-dc:Subject=" in args
    assert "-IPTC:Keywords=" in args
    assert "-XMP-dc:Subject+=alpha" in args
    assert "-IPTC:Keywords+=alpha" in args
    assert "-XMP-dc:Description=XMP description" in args
    assert "-EXIF:ImageDescription=XMP description" in args
    assert "-IPTC:Caption-Abstract=IPTC caption" in args


def test_run_sets_jpg_rating_only_once_for_multiple_targets(tmp_path):
    jpg = tmp_path / "shot.jpg"
    jpg.touch()
    (tmp_path / "shot.tif").touch()
    (tmp_path / "shot.dng").touch()
    cfg = _cfg(tmp_path, dry_run=False, set_jpg_rating_to_one=True)

    completed = MagicMock()
    completed.returncode = 0
    completed.stderr = ""

    with patch(
        "cortex_engine.photo_metadata_copy.read_jpg_metadata",
        return_value=PhotoMetadataPayload(["alpha"], "desc", "caption", ""),
    ), patch("cortex_engine.photo_metadata_copy.exiftool_path", return_value="exiftool"), patch(
        "cortex_engine.photo_metadata_copy.subprocess.run", return_value=completed
    ) as run_mock:
        results = list(run_photo_metadata_copy(cfg))

    assert len(results) == 2
    rating_calls = [
        call for call in run_mock.call_args_list
        if "-XMP-xmp:Rating=1" in call.args[0]
    ]
    assert len(rating_calls) == 1
