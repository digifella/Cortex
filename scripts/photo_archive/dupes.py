"""Stage 4 - duplicate grouping and keeper selection."""
import csv
import os
import re

DEMOTE_MARKERS = ("dupes", "backup", "_dupes")

# Google Drive names a re-upload "IMG_0176 (1).JPG"; Windows uses "x - Copy".
# macOS uses "_N4A5939 2". All anchored to the END of the stem, and the bare
# space-number form is capped at TWO digits so real catalog names like
# "Family 158.jpg" and "Crowfam 2011.jpg" are not eaten. This only ever
# discriminates within a byte-identical group, so a false positive costs
# nothing but a filename preference.
_COPY_RE = re.compile(
    r"(?:\s*-\s*Copy)?\s*\(\d+\)$"   # Google Drive / Windows "(1)"
    r"|\s*-\s*Copy$"                    # Windows "- Copy"
    r"|\s\d{1,2}$",                     # macOS "_N4A5939 2"
    re.I)


def _is_copy(filename: str) -> bool:
    """True if the name carries a duplicate-upload suffix."""
    return bool(_COPY_RE.search(os.path.splitext(filename)[0]))


def _demoted(row) -> int:
    haystack = f"{row['top_folder']}/{row['rel_dir']}".lower()
    return 1 if any(m in haystack for m in DEMOTE_MARKERS) else 0


def _sort_key(row):
    """Lower sorts better. Mirrors the spec's keeper rule, in order.

    The copy-suffix test sits second because mtime cannot decide it: a bulk
    Google Drive download stamps every copy within minutes in arbitrary order,
    so "oldest wins" picked "BH_2010_278 (3).jpg" over the original.
    """
    return (
        _demoted(row),
        1 if _is_copy(row["filename"]) else 0,
        0 if row["exif_dt"] else 1,
        0 if row["camera_model"] else 1,
        len([p for p in str(row["rel_dir"]).split("\\") if p]),
        row["mtime"],
        row["path"],
    )


def choose_keeper(rows):
    return min(rows, key=_sort_key)


def tier1_groups(conn):
    """Byte-identical groups. RAW never groups with non-RAW; sidecars never group."""
    shas = [r[0] for r in conn.execute(
        "SELECT sha256 FROM files WHERE sha256 IS NOT NULL AND kind != 'sidecar' "
        "GROUP BY sha256 HAVING COUNT(*) > 1")]
    for sha in shas:
        rows = conn.execute(
            "SELECT * FROM files WHERE sha256 = ? AND kind != 'sidecar'",
            (sha,)).fetchall()
        for kind_bucket in ("raw", "image", "video", "other"):
            members = [r for r in rows if r["kind"] == kind_bucket]
            if len(members) > 1:
                yield members


FIELDS = ["group_id", "role", "path", "top_folder", "rel_dir", "size",
          "sha256", "exif_dt", "camera_model", "mtime"]


def plan_tier1(conn, out_csv: str) -> dict[str, int]:
    stats = {"groups": 0, "to_quarantine": 0, "bytes_reclaimed": 0}
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for gid, members in enumerate(tier1_groups(conn), start=1):
            keeper = choose_keeper(members)
            stats["groups"] += 1
            for row in sorted(members, key=_sort_key):
                role = "keep" if row["id"] == keeper["id"] else "quarantine"
                if role == "quarantine":
                    stats["to_quarantine"] += 1
                    stats["bytes_reclaimed"] += row["size"]
                writer.writerow({k: row[k] for k in FIELDS if k in row.keys()}
                                | {"group_id": gid, "role": role})
    return stats
