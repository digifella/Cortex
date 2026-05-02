from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generator, Iterable


class MetadataTargetType(Enum):
    SIDECAR = "xmp_sidecar"
    EMBEDDED = "embedded"


@dataclass(frozen=True)
class PhotoMetadataPayload:
    keywords: list[str]
    description: str
    caption: str
    image_description: str

    @property
    def has_metadata(self) -> bool:
        return bool(
            self.keywords
            or self.description.strip()
            or self.caption.strip()
            or self.image_description.strip()
        )

    @property
    def xmp_description(self) -> str:
        return (self.description or self.caption or self.image_description or "").strip()

    @property
    def iptc_caption(self) -> str:
        return (self.caption or self.description or self.image_description or "").strip()


@dataclass(frozen=True)
class PhotoMetadataCopyAction:
    jpg_path: Path
    target_path: Path
    target_type: MetadataTargetType
    raw_path: Path | None = None
    sidecar_exists: bool = False


@dataclass(frozen=True)
class PhotoMetadataCopyResult:
    action: PhotoMetadataCopyAction
    success: bool
    keywords_written: int
    description_written: bool
    caption_written: bool
    rating_written: bool
    error: str | None = None


@dataclass(frozen=True)
class PhotoMetadataCopyReport:
    actions: list[PhotoMetadataCopyAction]
    orphaned_jpgs: list[Path]


@dataclass(frozen=True)
class PhotoMetadataCopyConfig:
    folder: Path
    recursive: bool = False
    dry_run: bool = True
    keep_backups: bool = True
    set_jpg_rating_to_one: bool = False
    strip_rating_suffix: bool = False
    rating_suffix_range: tuple[int, int] = (1, 5)
    jpg_extensions: tuple[str, ...] = ("jpg", "jpeg")
    embedded_extensions: tuple[str, ...] = ("tif", "tiff", "dng")
    raw_sidecar_extensions: tuple[str, ...] = (
        "raf", "raw", "nef", "cr2", "cr3", "arw", "rw2", "orf", "pef", "srw",
    )


def exiftool_path() -> str:
    path = shutil.which("exiftool")
    if not path:
        raise RuntimeError(
            "exiftool not found on PATH. Install ExifTool before running metadata copy."
        )
    return path


def is_exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


def _iter_files(folder: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    return (p for p in folder.glob(pattern) if p.is_file())


def _normal_exts(exts: Iterable[str]) -> set[str]:
    return {e.lower().lstrip(".") for e in exts}


def _strip_rating_suffix(stem: str, enabled: bool, suffix_range: tuple[int, int]) -> str:
    if not enabled:
        return stem
    lo, hi = suffix_range
    for rating in range(lo, hi + 1):
        suffix = f"-{rating}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _key_for_source(path: Path, config: PhotoMetadataCopyConfig) -> tuple[Path, str]:
    stem = _strip_rating_suffix(path.stem, config.strip_rating_suffix, config.rating_suffix_range)
    return path.parent, stem.casefold()


def scan_photo_metadata_copy(config: PhotoMetadataCopyConfig) -> PhotoMetadataCopyReport:
    jpg_exts = _normal_exts(config.jpg_extensions)
    embedded_exts = _normal_exts(config.embedded_extensions)
    raw_exts = _normal_exts(config.raw_sidecar_extensions)

    jpgs: list[Path] = []
    targets_by_key: dict[tuple[Path, str], list[PhotoMetadataCopyAction]] = {}

    for path in sorted(_iter_files(config.folder, config.recursive)):
        ext = path.suffix.lower().lstrip(".")
        key = (path.parent, path.stem.casefold())

        if ext in jpg_exts:
            jpgs.append(path)
            continue

        if ext in embedded_exts:
            targets_by_key.setdefault(key, []).append(
                PhotoMetadataCopyAction(
                    jpg_path=Path(),
                    target_path=path,
                    target_type=MetadataTargetType.EMBEDDED,
                    raw_path=None,
                    sidecar_exists=True,
                )
            )
            continue

        if ext in raw_exts:
            sidecar = path.with_suffix(".xmp")
            targets_by_key.setdefault(key, []).append(
                PhotoMetadataCopyAction(
                    jpg_path=Path(),
                    target_path=sidecar,
                    target_type=MetadataTargetType.SIDECAR,
                    raw_path=path,
                    sidecar_exists=sidecar.exists(),
                )
            )

    actions: list[PhotoMetadataCopyAction] = []
    orphaned: list[Path] = []
    for jpg in sorted(jpgs):
        matches = targets_by_key.get(_key_for_source(jpg, config), [])
        if not matches:
            orphaned.append(jpg)
            continue
        for match in matches:
            actions.append(
                PhotoMetadataCopyAction(
                    jpg_path=jpg,
                    target_path=match.target_path,
                    target_type=match.target_type,
                    raw_path=match.raw_path,
                    sidecar_exists=match.sidecar_exists,
                )
            )

    return PhotoMetadataCopyReport(actions=actions, orphaned_jpgs=orphaned)


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def read_jpg_metadata(jpg_path: Path) -> PhotoMetadataPayload:
    et = exiftool_path()
    result = subprocess.run(
        [
            et,
            "-json",
            "-s",
            "-XMP-dc:Subject",
            "-IPTC:Keywords",
            "-XMP-dc:Description",
            "-IPTC:Caption-Abstract",
            "-EXIF:ImageDescription",
            str(jpg_path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return PhotoMetadataPayload([], "", "", "")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return PhotoMetadataPayload([], "", "", "")

    if not payload:
        return PhotoMetadataPayload([], "", "", "")

    row = payload[0]
    keywords = _dedupe(_as_list(row.get("Subject")) + _as_list(row.get("Keywords")))
    description = str(row.get("Description") or "").strip()
    caption = str(row.get("Caption-Abstract") or "").strip()
    image_description = str(row.get("ImageDescription") or "").strip()
    return PhotoMetadataPayload(keywords, description, caption, image_description)


def _backup_args(keep_backups: bool) -> list[str]:
    return [] if keep_backups else ["-overwrite_original"]


def _write_args_for_target(
    action: PhotoMetadataCopyAction,
    metadata: PhotoMetadataPayload,
) -> list[str]:
    args: list[str] = []

    if action.target_type == MetadataTargetType.SIDECAR:
        if metadata.keywords:
            args.append("-XMP-dc:Subject=")
            args.extend(f"-XMP-dc:Subject+={kw}" for kw in metadata.keywords)
        if metadata.xmp_description:
            args.append(f"-XMP-dc:Description={metadata.xmp_description}")
        return args

    if metadata.keywords:
        args.extend(["-XMP-dc:Subject=", "-IPTC:Keywords="])
        for kw in metadata.keywords:
            args.append(f"-XMP-dc:Subject+={kw}")
            args.append(f"-IPTC:Keywords+={kw}")

    if metadata.xmp_description:
        args.append(f"-XMP-dc:Description={metadata.xmp_description}")
        args.append(f"-EXIF:ImageDescription={metadata.xmp_description}")
    if metadata.iptc_caption:
        args.append(f"-IPTC:Caption-Abstract={metadata.iptc_caption}")

    return args


def _create_sidecar(action: PhotoMetadataCopyAction) -> subprocess.CompletedProcess[str]:
    """Create a minimal XMP sidecar from the JPG metadata when one is absent."""
    et = exiftool_path()
    return subprocess.run(
        [
            et,
            "-tagsfromfile",
            str(action.jpg_path),
            "-XMP-dc:Description<XMP-dc:Description",
            "-XMP-dc:Description<IPTC:Caption-Abstract",
            "-XMP-dc:Subject<XMP-dc:Subject",
            "-XMP-dc:Subject<IPTC:Keywords",
            "-o",
            str(action.target_path),
            str(action.jpg_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _set_jpg_rating_to_one(jpg_path: Path, keep_backups: bool) -> subprocess.CompletedProcess[str]:
    et = exiftool_path()
    return subprocess.run(
        [et, *_backup_args(keep_backups), "-XMP-xmp:Rating=1", str(jpg_path)],
        capture_output=True,
        text=True,
        timeout=20,
    )


def apply_photo_metadata_copy_action(
    action: PhotoMetadataCopyAction,
    config: PhotoMetadataCopyConfig,
    set_rating_override: bool | None = None,
) -> PhotoMetadataCopyResult:
    metadata = read_jpg_metadata(action.jpg_path)
    description_written = bool(metadata.xmp_description)
    caption_written = bool(metadata.iptc_caption)
    should_set_rating = (
        config.set_jpg_rating_to_one if set_rating_override is None else set_rating_override
    )

    if config.dry_run:
        return PhotoMetadataCopyResult(
            action=action,
            success=True,
            keywords_written=len(metadata.keywords),
            description_written=description_written,
            caption_written=caption_written,
            rating_written=should_set_rating,
        )

    if not metadata.has_metadata:
        return PhotoMetadataCopyResult(
            action=action,
            success=True,
            keywords_written=0,
            description_written=False,
            caption_written=False,
            rating_written=False,
        )

    if action.target_type == MetadataTargetType.SIDECAR and not action.target_path.exists():
        create_result = _create_sidecar(action)
        if create_result.returncode != 0:
            return PhotoMetadataCopyResult(
                action=action,
                success=False,
                keywords_written=0,
                description_written=False,
                caption_written=False,
                rating_written=False,
                error=create_result.stderr.strip() or "Could not create XMP sidecar.",
            )

    write_args = _write_args_for_target(action, metadata)
    if write_args:
        et = exiftool_path()
        result = subprocess.run(
            [et, *_backup_args(config.keep_backups), *write_args, str(action.target_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return PhotoMetadataCopyResult(
                action=action,
                success=False,
                keywords_written=0,
                description_written=False,
                caption_written=False,
                rating_written=False,
                error=result.stderr.strip() or "ExifTool metadata write failed.",
            )

    rating_written = False
    if should_set_rating:
        rating_result = _set_jpg_rating_to_one(action.jpg_path, config.keep_backups)
        if rating_result.returncode != 0:
            return PhotoMetadataCopyResult(
                action=action,
                success=False,
                keywords_written=len(metadata.keywords),
                description_written=description_written,
                caption_written=caption_written,
                rating_written=False,
                error=rating_result.stderr.strip() or "ExifTool rating write failed.",
            )
        rating_written = True

    return PhotoMetadataCopyResult(
        action=action,
        success=True,
        keywords_written=len(metadata.keywords),
        description_written=description_written,
        caption_written=caption_written,
        rating_written=rating_written,
    )


def run_photo_metadata_copy(
    config: PhotoMetadataCopyConfig,
) -> Generator[PhotoMetadataCopyResult, None, None]:
    report = scan_photo_metadata_copy(config)
    rated_jpgs: set[Path] = set()
    for action in report.actions:
        should_set_rating = config.set_jpg_rating_to_one and action.jpg_path not in rated_jpgs
        result = apply_photo_metadata_copy_action(action, config, should_set_rating)
        if result.rating_written:
            rated_jpgs.add(action.jpg_path)
        yield result
