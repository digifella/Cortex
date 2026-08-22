# scripts/photo_archive/perceptual.py
"""Tier 3 - perceptual duplicate detection.

Byte hashing (tier 1) cannot see that a 5640x3760 scan and its 1280x720
rendition are the same photograph. This decodes the image and hashes its
structure instead, so a resize or a re-encode still matches.

DECODABLE STILLS ONLY. RAW is excluded deliberately: decoding 70,671 RAF files
would cost more than the rest of the pipeline combined, and two RAWs that
differ byte-wise are in practice different exposures. Video is excluded too.

Candidates are restricted to files that SHARE AN EXIF CAPTURE INSTANT with at
least one other file. Two photographs taken at the same second by the same
camera are a plausible rendition pair; two taken years apart are not, however
similar they look. That restriction is what keeps this to a few hundred
thousand decodes rather than half a million, and it also stops the hash
collapsing genuinely different frames.

REPORT ONLY. This never moves a file and never changes a row's state - the
spec makes tiers 2 and 3 reviewable, because a perceptual match is a judgement
and a byte match is a fact.
"""
import csv

from .paths import win_long

STILL_EXT = ("'.jpg'", "'.jpeg'", "'.png'", "'.tif'", "'.tiff'",
             "'.heic'", "'.bmp'", "'.gif'")
_EXT_SQL = "(" + ", ".join(STILL_EXT) + ")"

HASH_SIZE = 8


def perceptual_hash(path: str, hash_size: int = HASH_SIZE) -> str:
    """Structure hash of the decoded image, robust to resize and re-encode."""
    from PIL import Image
    import imagehash

    with Image.open(win_long(path)) as im:
        # draft() lets libjpeg decode straight to a reduced size - several
        # times faster, and irrelevant at an 8x8 hash.
        try:
            im.draft("L", (hash_size * 8, hash_size * 8))
        except (AttributeError, ValueError):
            pass
        return str(imagehash.dhash(im.convert("L"), hash_size=hash_size))


def _candidate_sql(extra: str = "") -> str:
    return f"""
        SELECT id, path FROM files
        WHERE ext IN {_EXT_SQL}
          AND state != 'quarantined'
          AND exif_dt IS NOT NULL
          AND exif_dt IN (SELECT exif_dt FROM files
                          WHERE ext IN {_EXT_SQL} AND state != 'quarantined'
                            AND exif_dt IS NOT NULL
                          GROUP BY exif_dt HAVING COUNT(*) > 1)
          {extra}
        ORDER BY id
    """


def hash_candidates(conn, limit: int | None = None,
                    progress_every: int = 0) -> dict:
    """Decode and hash every candidate still. Resumable: skips rows already done."""
    stats = {"hashed": 0, "errors": 0, "candidates": 0}
    rows = conn.execute(_candidate_sql("AND percept_hash IS NULL")).fetchall()
    stats["candidates"] = len(rows)
    if limit:
        rows = rows[:limit]

    for n, row in enumerate(rows, 1):
        try:
            h = perceptual_hash(row["path"])
        except Exception:
            # A corrupt or unreadable image must not abort the stage; it is one
            # file, not one folder. It simply never joins a group.
            stats["errors"] += 1
            continue
        conn.execute("UPDATE files SET percept_hash = ? WHERE id = ?",
                     (h, row["id"]))
        stats["hashed"] += 1
        if progress_every and n % progress_every == 0:
            conn.commit()
            print(f"   ...{n:,}/{len(rows):,} hashed", flush=True)
    conn.commit()
    return stats


FIELDS = ["group_id", "role", "path", "top_folder", "rel_dir", "size",
          "ext", "exif_dt", "camera_model", "percept_hash"]


def plan_tier3(conn, out_csv: str) -> dict:
    """Report groups of same-instant files whose decoded structure matches.

    Keeper is the largest file - for renditions of one photograph the biggest
    is the least degraded. That differs from tier 1, where every member is
    byte-identical and size cannot discriminate.
    """
    stats = {"groups": 0, "candidates_for_review": 0, "bytes_if_applied": 0}
    groups = conn.execute(f"""
        SELECT exif_dt, percept_hash FROM files
        WHERE percept_hash IS NOT NULL AND state != 'quarantined'
          AND ext IN {_EXT_SQL}
        GROUP BY exif_dt, percept_hash HAVING COUNT(*) > 1
        ORDER BY exif_dt
    """).fetchall()

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for gid, g in enumerate(groups, 1):
            members = conn.execute(f"""
                SELECT * FROM files
                WHERE exif_dt = ? AND percept_hash = ? AND state != 'quarantined'
                  AND ext IN {_EXT_SQL}
                ORDER BY size DESC, path
            """, (g["exif_dt"], g["percept_hash"])).fetchall()
            if len(members) < 2:
                continue
            stats["groups"] += 1
            keeper = members[0]
            for row in members:
                role = "keep" if row["id"] == keeper["id"] else "review"
                if role == "review":
                    stats["candidates_for_review"] += 1
                    stats["bytes_if_applied"] += row["size"]
                writer.writerow({k: row[k] for k in FIELDS if k in row.keys()}
                                | {"group_id": gid, "role": role})
    return stats
