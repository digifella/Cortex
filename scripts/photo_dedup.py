#!/usr/bin/env python3
"""Same-scene dedup + enrichment for poorly-scanned photo folders.

Built for folders whose filenames cannot be joined to the Lightroom catalog by
stem (1996 scans: 198 exports over 35 distinct timestamps), so identity is
established from pixels rather than names.

Design: docs/superpowers/specs/2026-08-13-vietnam-scan-dedup-design.md

Stages (one subcommand each, all resumable):
    describe  hash every export + one Haiku vision call per image
    geo       reverse-geocode each export's GPS to city/state/country
    link      match exports to catalog originals by perceptual hash
    group     cluster same-scene shots and pick a keeper
    sheet     HTML contact sheet for review
    apply     write keywords/caption/location (and `Duplicate`) back
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from photo_batch import load_dotenv_keys  # noqa: E402  (shared .env handling)
from cortex_engine.photo_name_tags import apply_names  # noqa: E402

CHECKPOINT = ".photo_dedup.json"
GEOCACHE = ".photo_dedup_geo.json"
VISION_MODEL = "claude-haiku-4-5-20251001"  # matches textifier.CLAUDE_VISION_MODEL
MAX_EDGE = 1024

PROMPT = """This is a photograph from {year}, typically a scanned film photo of
variable quality.

Return ONLY a JSON object, no prose and no code fences, with exactly these keys:

"caption": 1-2 plain declarative sentences, max 35 words, describing what is shown. No self-directives, no analysis. When one or two people are the subject, describe them with direct phrases ("a man", "a woman", "a man and a woman", "a couple") rather than collective wording like "two adults" or "a group of people" — but only where that is accurate; genuine group shots should still read as groups. Never invent or guess a person's name.
"keywords": 5-12 lowercase keyword strings.
"subject": a 1-3 word canonical subject category, e.g. "cham tower", "pagoda", "market stall", "rice paddy", "river boat", "group portrait", "street scene", "beach".
"scene_signature": one short phrase identifying THIS SPECIFIC scene, including distinguishing detail (structures present, composition, who or what is in frame). Two photos of the same subject taken moments apart from the same viewpoint must produce the SAME signature. Different sites, or clearly different compositions of the same site, must produce DIFFERENT signatures.
"landmark": the name of the specific site ONLY if you genuinely recognise it from a distinctive structure, skyline or feature (e.g. "Eiffel Tower", "Uluru", "Po Nagar", "Sydney Opera House"). If you are not certain, use null. Do not guess.
"city": the city or town, ONLY if "landmark" is non-null. Otherwise null.
"country": the country, ONLY if you can tell with real confidence from an identified landmark or from unmistakable visual evidence (signage in a specific language, distinctive architecture, recognisable landscape). Otherwise null. A generic beach, garden, room interior or portrait must return null — do not infer a country from people's appearance.
"""


# Boilerplate that scanners and upscalers leave in the description field. It is
# long enough to pass the "already captioned" gate, so without this it would be
# preserved as a caption and then written onto the catalog master.
JUNK_DESC = re.compile(r"^(file written by|upscaled with|created with|"
                       r"scanned (with|by)|adobe photoshop)", re.I)


def dhash(path: Path, size: int = 8) -> str | None:
    """64-bit difference hash. Uses JPEG draft mode so large scans decode fast."""
    try:
        with Image.open(path) as img:
            img.draft("L", (size * 8, size * 8))  # 1/8-scale decode where possible
            img = img.convert("L").resize((size + 1, size), Image.LANCZOS)
            px = list(img.getdata())
    except Exception:
        return None
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | int(px[base + col] < px[base + col + 1])
    return f"{bits:016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def read_exif(directory: Path) -> dict:
    """One batch exiftool call for the whole folder."""
    out = subprocess.run(
        ["exiftool", "-j", "-FileName", "-Rating", "-DateTimeOriginal", "-ImageSize",
         "-Description", "-Subject", str(directory)],
        capture_output=True, text=True,
    )
    rows = json.loads(out.stdout) if out.stdout.strip() else []
    return {r["FileName"]: r for r in rows if "FileName" in r}


def encode_jpeg(path: Path) -> str | None:
    try:
        with Image.open(path) as img:
            img.draft("RGB", (MAX_EDGE * 2, MAX_EDGE * 2))
            img = img.convert("RGB")
            img.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
    except Exception:
        return None
    return base64.standard_b64encode(buf.getvalue()).decode()


def parse_json_reply(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.strip("`")
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def describe_one(client, path: Path, year: str = "the 1990s", retries: int = 3) -> dict | None:
    encoded = encode_jpeg(path)
    if not encoded:
        return None
    prompt = PROMPT.format(year=year)
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=VISION_MODEL,
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64",
                                                     "media_type": "image/jpeg",
                                                     "data": encoded}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            raw = ""
            for block in resp.content or []:
                if hasattr(block, "text"):
                    raw = block.text or ""
                    break
            parsed = parse_json_reply(raw)
            if parsed:
                return parsed
        except Exception as exc:
            if attempt == retries - 1:
                print(f"    API error: {exc}", flush=True)
            time.sleep(2 * (attempt + 1))
    return None


def save_checkpoint(state: dict, path: Path) -> None:
    """Direct write, not os.replace: OneDrive drvfs intermittently EPERMs on
    replace-over-existing (see photo_batch save_checkpoint)."""
    path.write_text(json.dumps(state, indent=1))


def cmd_describe(args) -> None:
    directory = args.export_dir
    ckpt_path = directory / CHECKPOINT
    state = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {}

    jpgs = sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})
    exif = read_exif(directory)
    print(f"{len(jpgs)} images; {len(state)} already described", flush=True)

    load_dotenv_keys()
    import anthropic
    client = anthropic.Anthropic()

    year = re.match(r"(\d{4})", directory.name)
    year = year.group(1) if year else "the 1990s"

    done = skipped = 0
    for i, jpg in enumerate(jpgs, start=1):
        rec = state.get(jpg.name, {})
        meta = exif.get(jpg.name, {})
        if not rec.get("phash"):
            rec["phash"] = dhash(jpg)
            rec["rating"] = meta.get("Rating")
            rec["timestamp"] = meta.get("DateTimeOriginal")
            rec["size"] = meta.get("ImageSize")
        if rec.get("caption"):
            state[jpg.name] = rec
            continue
        # Already carries a real caption on disk (e.g. the enriched Vietnam
        # survivors) — leave it alone rather than paying to redescribe it.
        existing = str(meta.get("Description") or "").strip()
        if JUNK_DESC.match(existing):
            existing = ""
        if len(existing) >= args.min_desc_len and not args.redescribe_all:
            rec["preexisting"] = True
            state[jpg.name] = rec
            skipped += 1
            continue
        if args.limit and done >= args.limit:
            break

        parsed = describe_one(client, jpg, year=year)
        if parsed:
            # Lightroom person-tags turn "a man and a woman" into "Paul and
            # Jacqui". Deterministic post-transform, not a model instruction —
            # and it must run now, because it reads the keywords present when
            # the caption is written.
            existing_kw = meta.get("Subject") or []
            if isinstance(existing_kw, str):
                existing_kw = [existing_kw]
            caption = apply_names(str(parsed.get("caption") or "").strip(),
                                  [str(k) for k in existing_kw])
            rec.update({
                "caption": caption,
                # Keep the pre-substitution text so name handling can be redone
                # later without paying for vision again.
                "caption_raw": str(parsed.get("caption") or "").strip(),
                "named": caption != str(parsed.get("caption") or "").strip(),
                "keywords": [str(k).lower().strip() for k in (parsed.get("keywords") or [])],
                "subject": str(parsed.get("subject") or "").lower().strip(),
                "scene_signature": str(parsed.get("scene_signature") or "").lower().strip(),
                "landmark": parsed.get("landmark") or None,
                "city": parsed.get("city") or None,
                "country": parsed.get("country") or None,
            })
            print(f"[{i}/{len(jpgs)}] {jpg.name[:40]:42} {rec['subject'][:18]:20} "
                  f"{(rec['landmark'] or '-')[:16]:18} {(rec['country'] or '-')[:14]}",
                  flush=True)
        else:
            rec["error"] = "vision failed"
            print(f"[{i}/{len(jpgs)}] {jpg.name[:44]:46} FAILED", flush=True)

        state[jpg.name] = rec
        save_checkpoint(state, ckpt_path)
        done += 1

    save_checkpoint(state, ckpt_path)
    ok = sum(1 for r in state.values() if r.get("caption"))
    named = sum(1 for r in state.values() if r.get("landmark"))
    ctry = sum(1 for r in state.values() if r.get("country"))
    print(f"\ndescribe complete: {ok}/{len(jpgs)} described ({skipped} already had captions), "
          f"{named} landmarks, {ctry} countries")
    print(f"checkpoint: {ckpt_path}")


# --------------------------------------------------------------------------
# Stage 2b — reverse-geocode GPS to a real place
# --------------------------------------------------------------------------

# describe() only fills city/country when Haiku recognises a landmark, which on
# ordinary family scans is almost never (3 of 89 on the 1996 set). Where the
# photo carries GPS — Lightroom map placement on these Pre-Dig years — that is
# both available and far more trustworthy, so it wins outright.

GEO_PRECISION = 4       # ~11 m; collapses the 400-photo set to ~72 lookups
NOMINATIM_DELAY = 1.1   # Nominatim asks for <= 1 req/sec


# textifier.reverse_geocode stops at city/town/village/suburb, which leaves rural
# coordinates with no place at all — 103 photos here, including a Wilsons Promontory
# set that Nominatim returns as `city_district`. Widened rather than changing the
# shared helper, whose chain the rest of the photo pipeline is calibrated against.
CITY_FIELDS = ("city", "town", "municipality", "city_district", "suburb",
               "village", "county", "state_district")


def geocode(lat: float, lon: float) -> dict:
    try:
        from geopy.geocoders import Nominatim
        loc = Nominatim(user_agent="cortex_suite", timeout=10).reverse(
            (lat, lon), exactly_one=True, language="en")
        if loc is None:
            return {"country": "", "state": "", "city": ""}
        addr = loc.raw.get("address", {})
        return {
            "country": addr.get("country", ""),
            "state": addr.get("state", "") or addr.get("region", ""),
            "city": next((addr[f] for f in CITY_FIELDS if addr.get(f)), ""),
        }
    except Exception as exc:
        print(f"    geocode failed for ({lat}, {lon}): {exc}", flush=True)
        return {"country": "", "state": "", "city": ""}


def read_gps(directory: Path) -> dict:
    out = subprocess.run(
        ["exiftool", "-j", "-n", "-FileName", "-GPSLatitude", "-GPSLongitude", str(directory)],
        capture_output=True, text=True,
    )
    rows = json.loads(out.stdout) if out.stdout.strip() else []
    return {r["FileName"]: r for r in rows if "FileName" in r}


def cmd_geo(args) -> None:
    directory = args.export_dir
    ckpt_path = directory / CHECKPOINT
    state = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {}

    gps = read_gps(directory)
    cache_path = directory / GEOCACHE
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    jpgs = sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})
    resolved = nogps = lookups = 0
    for i, jpg in enumerate(jpgs, start=1):
        rec = state.setdefault(jpg.name, {})
        meta = gps.get(jpg.name, {})
        lat, lon = meta.get("GPSLatitude"), meta.get("GPSLongitude")
        if lat is None or lon is None:
            rec["geo"] = None
            nogps += 1
            continue

        key = f"{round(float(lat), GEO_PRECISION)},{round(float(lon), GEO_PRECISION)}"
        # A cached miss is worth retrying: the city chain was widened after the
        # first pass, and a coordinate with no city is exactly what it fixes.
        if key not in cache or not cache[key].get("city"):
            cache[key] = geocode(float(lat), float(lon))
            cache_path.write_text(json.dumps(cache, indent=1))
            lookups += 1
            time.sleep(NOMINATIM_DELAY)
        place = cache[key]
        rec["geo"] = place
        if any(place.values()):
            resolved += 1
        print(f"[{i}/{len(jpgs)}] {jpg.name[:44]:46} "
              f"{(place.get('city') or '-')[:18]:20} {(place.get('state') or '-')[:16]:18} "
              f"{(place.get('country') or '-')[:14]}", flush=True)

    save_checkpoint(state, ckpt_path)
    print(f"\ngeo complete: {resolved}/{len(jpgs)} placed, {nogps} without GPS "
          f"({lookups} new Nominatim lookups, {len(cache)} cached coordinates)")


# --------------------------------------------------------------------------
# Stage 0 — link exports to catalog originals by image content
# --------------------------------------------------------------------------

CATALOG_CACHE = ".photo_dedup_catalog.json"
LINKS = ".photo_dedup_links.json"
CATALOG_EXT = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".dng"}

LINK_MAX_DIST = 10      # accept a link at or below this Hamming distance
VARIANT_DIST = 8        # sibling catalog files (jpg + -Edit.tif of one photo)
RATIO_TOL = 0.02        # aspect ratio agreement


def catalog_token(name: str) -> str | None:
    """The human subject token embedded in Pre-Dig filenames.

    Handles both conventions seen in the catalog:
        1996-04-23 06-12-36_Market_1562 x 1044_CanoScan...jpg
        2000-01-01 00-00-23-Crowfam-5184 x 3456-Canon EOS 550D.JPG
    """
    m = re.match(r"^[\d-]+ [\d-]+[_-]([A-Za-z][A-Za-z ]*?)[_-]", name)
    return m.group(1).strip() if m else None


def catalog_timestamp(name: str) -> str | None:
    m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})", name)
    return m.group(1) if m else None


def hash_catalog(catalog_dir: Path, cache_path: Path) -> dict:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    files = sorted(p for p in catalog_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in CATALOG_EXT)
    todo = [p for p in files if p.name not in cache]
    print(f"catalog: {len(files)} files, {len(todo)} to hash", flush=True)
    for i, path in enumerate(todo, start=1):
        entry = {"phash": dhash(path), "token": catalog_token(path.name),
                 "timestamp": catalog_timestamp(path.name), "path": str(path)}
        try:
            with Image.open(path) as img:
                entry["w"], entry["h"] = img.size
        except Exception:
            entry["w"] = entry["h"] = None
        cache[path.name] = entry
        if i % 25 == 0:
            print(f"  hashed {i}/{len(todo)}", flush=True)
            cache_path.write_text(json.dumps(cache, indent=1))
    cache_path.write_text(json.dumps(cache, indent=1))
    return {k: v for k, v in cache.items() if v.get("phash")}


def ratio_of(w, h) -> float | None:
    if not w or not h:
        return None
    return max(w, h) / min(w, h)


def cmd_link(args) -> None:
    directory = args.export_dir
    state = json.loads((directory / CHECKPOINT).read_text())
    catalog = hash_catalog(args.catalog_dir, directory / CATALOG_CACHE)

    cat_items = [(n, e) for n, e in catalog.items() if e.get("phash")]
    links, unlinked = {}, []
    stale = 0
    for name, rec in state.items():
        eh = rec.get("phash")
        if not eh:
            continue
        # A checkpoint outlives folder reorganisations: the 1993 export carried 18
        # records for files that had since been renamed away. Linking those would
        # let a caption with no surviving source photo claim a catalog master.
        if not (directory / name).exists():
            stale += 1
            continue
        size = (rec.get("size") or "x").split("x")
        try:
            er = ratio_of(int(size[0]), int(size[1]))
        except (ValueError, IndexError):
            er = None

        best, best_d = None, 99
        for cname, centry in cat_items:
            cr = ratio_of(centry.get("w"), centry.get("h"))
            if er and cr and abs(er - cr) / cr > RATIO_TOL:
                continue
            d = hamming(eh, centry["phash"])
            if d < best_d:
                best, best_d = cname, d

        if best is None or best_d > LINK_MAX_DIST:
            unlinked.append(name)
            continue
        # Siblings are derivatives of the SAME catalog photo (a .jpg and its
        # -Edit.tif), which share the filename timestamp. Without that guard a
        # keeper would also claim independent near-identical scans, and those
        # are exactly the catalog duplicates we need to flag rather than enrich.
        siblings = [c for c, e in cat_items
                    if c != best
                    and e.get("timestamp") == catalog[best].get("timestamp")
                    and hamming(e["phash"], catalog[best]["phash"]) <= VARIANT_DIST]
        links[name] = {"catalog": best, "distance": best_d,
                       "token": catalog[best].get("token"),
                       "variants": siblings,
                       "paths": [catalog[best]["path"]] + [catalog[c]["path"] for c in siblings]}

    (directory / LINKS).write_text(json.dumps(links, indent=1))

    # Validation: Haiku never saw the filename, so token/subject agreement is an
    # independent estimate of linkage accuracy (see design doc).
    agree = checked = 0
    for name, link in links.items():
        token = (link.get("token") or "").lower()
        if not token or token in {"vietnam", "holiday"}:  # too generic to test
            continue
        blob = f"{state[name].get('subject','')} {state[name].get('scene_signature','')} " \
               f"{state[name].get('landmark') or ''} {' '.join(state[name].get('keywords',[]))}".lower()
        checked += 1
        agree += int(token.split()[0] in blob)

    print(f"\nlinked {len(links)}/{len(state) - stale}; {len(unlinked)} unlinked"
          f"{f'; {stale} stale checkpoint entries skipped' if stale else ''}")
    dists = sorted(l["distance"] for l in links.values())
    if dists:
        print(f"hamming distance: min {dists[0]}, median {dists[len(dists)//2]}, max {dists[-1]}")
    if checked:
        print(f"site-token agreement: {agree}/{checked} ({100*agree/checked:.0f}%) "
              f"on specific tokens (Market/Mekong/Cham/Cuchi/Cao Dai/...)")
    print(f"manifest: {directory / LINKS}")


# --------------------------------------------------------------------------
# Stage 3 — same-scene grouping
# --------------------------------------------------------------------------

GROUPS = ".photo_dedup_groups.json"
SAME_FRAME_DIST = 6     # certainly the same frame, no semantics needed
NEAR_DIST = 22          # visually close enough to be the same moment
SCENE_SIM = 0.5         # scene-signature word overlap required to merge
STOPWORDS = {"a", "an", "the", "with", "and", "of", "in", "on", "at", "from",
             "to", "by", "over", "under", "near", "for", "is", "are"}


def words(text: str) -> set:
    return {w.strip(",.;:()") for w in (text or "").lower().split()} - STOPWORDS


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cmd_group(args) -> None:
    directory = args.export_dir
    state = json.loads((directory / CHECKPOINT).read_text())
    links_path = directory / LINKS
    links = json.loads(links_path.read_text()) if links_path.exists() else {}

    names = [n for n, r in state.items() if r.get("phash") and r.get("caption")]
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    scenes = {n: words(state[n].get("scene_signature")) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = hamming(state[a]["phash"], state[b]["phash"])
            if d <= SAME_FRAME_DIST:
                union(a, b)
                continue
            # Site tokens from the catalog veto a merge outright.
            ta = (links.get(a, {}).get("token") or "").lower()
            tb = (links.get(b, {}).get("token") or "").lower()
            if ta and tb and ta != tb and {ta, tb} - {"vietnam", "holiday"}:
                continue
            same_subject = state[a].get("subject") == state[b].get("subject")
            close = d <= NEAR_DIST or state[a].get("timestamp") == state[b].get("timestamp")
            if same_subject and close and jaccard(scenes[a], scenes[b]) >= SCENE_SIM:
                union(a, b)

    clusters = {}
    for n in names:
        clusters.setdefault(find(n), []).append(n)

    def sort_key(n):
        rec = state[n]
        rating = int(rec.get("rating") or 0)
        try:
            w, h = (rec.get("size") or "0x0").split("x")
            area = int(w) * int(h)
        except ValueError:
            area = 0
        return (-rating, -area, n)

    groups = []
    for members in clusters.values():
        members.sort(key=sort_key)
        groups.append({"keeper": members[0], "duplicates": members[1:]})
    groups.sort(key=lambda g: -len(g["duplicates"]))

    (directory / GROUPS).write_text(json.dumps(groups, indent=1))
    dupes = sum(len(g["duplicates"]) for g in groups)
    multi = [g for g in groups if g["duplicates"]]
    print(f"{len(names)} described -> {len(groups)} groups; "
          f"{len(groups)} keepers, {dupes} marked Duplicate")
    print(f"{len(multi)} groups have >1 member; largest {len(multi[0]['duplicates'])+1 if multi else 0}")
    print("\nlargest groups:")
    for g in multi[:8]:
        rec = state[g["keeper"]]
        print(f"  {len(g['duplicates'])+1:2} x {rec.get('subject','')[:22]:24} "
              f"{(rec.get('landmark') or '-')[:18]:20} keeper={g['keeper'][:40]}")
    print(f"\ngroups: {directory / GROUPS}")


# --------------------------------------------------------------------------
# Stage 4 — contact sheet for review
# --------------------------------------------------------------------------

THUMB_EDGE = 230


def thumb_data_uri(path: Path) -> str:
    try:
        with Image.open(path) as img:
            img.draft("RGB", (THUMB_EDGE * 3, THUMB_EDGE * 3))
            img = img.convert("RGB")
            img.thumbnail((THUMB_EDGE, THUMB_EDGE), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=72)
    except Exception:
        return ""
    return "data:image/jpeg;base64," + base64.standard_b64encode(buf.getvalue()).decode()


def cmd_sheet(args) -> None:
    directory = args.export_dir
    state = json.loads((directory / CHECKPOINT).read_text())
    groups_path = directory / GROUPS
    groups = json.loads(groups_path.read_text()) if groups_path.exists() else []
    auto_dupes = {n for g in groups for n in g["duplicates"]}

    by_subject = {}
    for name, rec in state.items():
        if rec.get("caption"):
            by_subject.setdefault(rec.get("subject") or "(none)", []).append(name)
    ordered = sorted(by_subject.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    print(f"rendering {sum(len(v) for v in by_subject.values())} thumbnails...", flush=True)
    parts = []
    for subject, members in ordered:
        members.sort()
        cards = []
        for n in members:
            rec = state[n]
            uri = thumb_data_uri(directory / n)
            flag = ' <span class="dup">auto-duplicate</span>' if n in auto_dupes else ""
            land = f' · {rec["landmark"]}' if rec.get("landmark") else ""
            cards.append(
                f'<figure><img loading="lazy" src="{uri}" alt="">'
                f'<figcaption><b>{n}</b>{flag}<br>'
                f'<span class="meta">{rec.get("rating") or "?"}★{land}</span><br>'
                f'{rec.get("scene_signature","")}</figcaption></figure>'
            )
        parts.append(f'<section><h2>{subject} <span class="n">{len(members)}</span></h2>'
                     f'<div class="grid">{"".join(cards)}</div></section>')

    html = f"""<title>1996 Vietnam Review</title>
<style>
:root {{ --bg:#fbfaf8; --fg:#1c1b19; --mut:#6c6862; --line:#e2ded7; --card:#fff; --dup:#b4531f; }}
@media (prefers-color-scheme:dark) {{ :root:not([data-theme=light]) {{
  --bg:#16151a; --fg:#eceaf0; --mut:#9d98a6; --line:#2e2c35; --card:#1e1d23; --dup:#e0894f; }} }}
:root[data-theme=dark] {{ --bg:#16151a; --fg:#eceaf0; --mut:#9d98a6; --line:#2e2c35; --card:#1e1d23; --dup:#e0894f; }}
body {{ background:var(--bg); color:var(--fg); margin:0; padding:2rem 1.5rem 4rem;
  font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; }}
h1 {{ font-size:1.6rem; margin:0 0 .3rem; }}
.lede {{ color:var(--mut); margin:0 0 2rem; max-width:60ch; }}
h2 {{ font-size:1.05rem; margin:2.4rem 0 .8rem; padding-bottom:.4rem;
  border-bottom:1px solid var(--line); text-transform:capitalize; }}
.n {{ color:var(--mut); font-weight:400; }}
.grid {{ display:grid; gap:1rem; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); }}
figure {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:8px;
  overflow:hidden; display:flex; flex-direction:column; }}
img {{ width:100%; height:170px; object-fit:cover; display:block; background:var(--line); }}
figcaption {{ padding:.55rem .6rem; font-size:11.5px; line-height:1.4; color:var(--mut); }}
figcaption b {{ color:var(--fg); font-weight:600; font-size:11px; word-break:break-all; }}
.meta {{ color:var(--mut); }}
.dup {{ color:var(--dup); font-weight:600; }}
</style>
<h1>1996 Vietnam — review sheet</h1>
<p class="lede">{len(state)} scans grouped by subject, largest groups first.
{len(auto_dupes)} are flagged as automatic same-scene duplicates. Nothing has been
written yet — this is for deciding which subjects to collapse further.</p>
{''.join(parts)}
"""
    out = args.out or (Path.home() / "vietnam-review-sheet.html")
    out.write_text(html)
    print(f"contact sheet: {out}  ({out.stat().st_size/1e6:.1f} MB)")


# --------------------------------------------------------------------------
# Stage 5 — write metadata back
# --------------------------------------------------------------------------

DUPLICATE_KEYWORD = "Duplicate"


def read_keywords_bulk(paths: list[Path]) -> dict:
    """One exiftool call for every target; avoids a read round-trip per file on L:."""
    if not paths:
        return {}
    args = ["exiftool", "-j", "-SourceFile", "-XMP-dc:Subject"]
    args += [str(p) for p in paths]
    out = subprocess.run(args, capture_output=True, text=True)
    result = {}
    if out.stdout.strip():
        for row in json.loads(out.stdout):
            subj = row.get("Subject") or []
            if isinstance(subj, str):
                subj = [subj]
            result[row.get("SourceFile", "")] = [str(s) for s in subj]
    return result


def read_descriptions_bulk(paths: list) -> dict:
    """On-disk captions, keyed by path. A `preexisting` record holds no caption in
    the checkpoint (describe declined to pay for one), so without this the catalog
    master never receives the caption its export already carries — 5 masters on the
    2001 run had no description at all."""
    if not paths:
        return {}
    out = subprocess.run(["exiftool", "-j", "-SourceFile", "-XMP-dc:Description"]
                         + [str(p) for p in paths], capture_output=True, text=True)
    if not out.stdout.strip():
        return {}
    return {r.get("SourceFile", ""): str(r.get("Description") or "").strip()
            for r in json.loads(out.stdout)}


def place_of(rec: dict) -> dict:
    """City/state/country for a photo. GPS wins over Haiku's landmark guess."""
    geo = rec.get("geo") or {}
    if any(geo.values()):
        return {"city": geo.get("city") or "", "state": geo.get("state") or "",
                "country": geo.get("country") or ""}
    return {"city": rec.get("city") or "", "state": "",
            "country": rec.get("country") or ""}


def place_keywords(rec: dict) -> list:
    """Location as tags, matching textifier.keyword_image: city/state/country
    lowercased, or `nogps` where the photo carries no coordinates."""
    if "geo" not in rec:                      # geo stage never run for this photo
        return []
    if rec.get("geo") is None:
        return ["nogps"]
    return [v.lower() for v in place_of(rec).values() if v]


def write_tags(path: Path, keywords: list, description: str, place: dict,
               keep_backup: bool) -> tuple[bool, str]:
    city, state, country = place.get("city"), place.get("state"), place.get("country")
    args = ["exiftool", "-m"]           # -m: ignore minor warnings, else writes fail
    if not keep_backup:
        args.append("-overwrite_original")
    args += ["-XMP-dc:Subject=", "-IPTC:Keywords="]      # clear, then re-add merged set
    for kw in keywords:
        args += [f"-XMP-dc:Subject={kw}", f"-IPTC:Keywords={kw}"]
    if description:
        args += [f"-XMP-dc:Description={description}",
                 f"-IPTC:Caption-Abstract={description}",
                 f"-EXIF:ImageDescription={description}"]
    if country:
        args += [f"-XMP-photoshop:Country={country}",
                 f"-IPTC:Country-PrimaryLocationName={country}"]
    if state:
        args += [f"-XMP-photoshop:State={state}",
                 f"-IPTC:Province-State={state}"]
    if city:
        args += [f"-XMP-photoshop:City={city}", f"-IPTC:City={city}"]
    # Lightroom decides "changed externally" from MetadataDate, not file mtime.
    args.append("-XMP-xmp:MetadataDate=now")
    args.append(str(path))
    out = subprocess.run(args, capture_output=True, text=True)
    return out.returncode == 0, (out.stderr or "").strip()


def cmd_apply(args) -> None:
    directory = args.export_dir
    state = json.loads((directory / CHECKPOINT).read_text())
    # Both are optional: enrichment-only runs have no catalog linkage and no
    # dedup pass, in which case every photo is a keeper and nothing is marked.
    links_file, groups_file = directory / LINKS, directory / GROUPS
    links = json.loads(links_file.read_text()) if links_file.exists() else {}
    groups = json.loads(groups_file.read_text()) if groups_file.exists() else []

    keepers = {g["keeper"] for g in groups}
    dup_exports = {n for g in groups for n in g["duplicates"]}

    # A catalog file claimed by ANY keeper must never be marked Duplicate: the
    # export folder holds the same photo twice, but the catalog holds one copy.
    keeper_targets = set()
    for n in keepers:
        keeper_targets.update(links.get(n, {}).get("paths", []))

    # Every export claiming each catalog file, so a keeper's claim can outrank
    # a duplicate's on the same file.
    claims: dict[str, list] = {}
    for name, link in links.items():
        for target in link["paths"]:
            claims.setdefault(target, []).append(name)

    catalog_plan, protected = [], 0
    for target, sources in claims.items():
        keeper_sources = [s for s in sources if s in keepers]
        if keeper_sources:
            # A keeper wants this file, so it is a photo being kept: enrich, never mark.
            if any(s in dup_exports for s in sources):
                protected += 1
            catalog_plan.append((Path(target), keeper_sources[0], False))
        else:
            # Mark only where a claiming export is a KNOWN duplicate. Deducing it
            # from "no keeper claims this" silently flags every master on an
            # enrichment-only run, where `groups` is empty so nothing is a keeper
            # — 343 catalog files on the 1993-1998 pass, which is the set the user
            # then filters on `Duplicate` and moves out of the catalog.
            catalog_plan.append(
                (Path(target), sources[0], any(s in dup_exports for s in sources)))

    # A photo whose caption already existed on disk carries no "caption" in the
    # state, so write_tags leaves its description untouched — but it still needs
    # its location and keywords written, so it belongs in the plan.
    export_plan = [(directory / n, n, n in dup_exports) for n in state
                   if (directory / n).exists()
                   and (state[n].get("caption") or place_keywords(state[n]))]

    marked = sum(1 for _, _, m in catalog_plan if m)
    print(f"catalog targets : {len(catalog_plan)}  ({marked} to mark {DUPLICATE_KEYWORD})")
    print(f"export files    : {len(export_plan)}  ({len(dup_exports)} flagged in the export folder)")
    print(f"protected       : {protected} catalog file(s) shared with a keeper — NOT marked")
    print(f"unlinked        : {len([n for n in state if n not in links])} export(s) with no catalog match")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        for path, source, mark in catalog_plan[:10]:
            print(f"  {'DUP ' if mark else '    '}{path.name[:62]:64} <- {source[:40]}")
        return

    existing = read_keywords_bulk([p for p, _, _ in catalog_plan])
    # Captions that were already on the export before this run live only on disk.
    on_disk = read_descriptions_bulk(
        [directory / s for _, s, _ in catalog_plan if state[s].get("preexisting")])
    ok = fail = 0
    for i, (path, source, mark) in enumerate(catalog_plan, start=1):
        rec = state[source]
        place = place_of(rec)
        kws = list(dict.fromkeys(
            existing.get(str(path), []) + rec.get("keywords", []) +
            place_keywords(rec) + ([DUPLICATE_KEYWORD] if mark else [])))
        caption = rec.get("caption") or on_disk.get(str(directory / source), "")
        good, err = write_tags(path, kws, caption, place, keep_backup=True)
        ok, fail = ok + good, fail + (not good)
        flag = "DUP" if mark else "   "
        print(f"[{i}/{len(catalog_plan)}] {flag} {'OK ' if good else 'FAIL'} "
              f"{path.name[:58]}{'' if good else ' :: ' + err[:60]}", flush=True)

    # The exports carry Paul's own Lightroom keywords (person tags, place tags)
    # and are the set indexed by kb-photos, so they must be merged in — write_tags
    # clears the keyword fields before re-adding, so anything omitted here is lost.
    export_existing = read_keywords_bulk([p for p, _, _ in export_plan])
    for path, name, is_dup in export_plan:
        rec = state[name]
        place = place_of(rec)
        kws = list(dict.fromkeys(
            export_existing.get(str(path), []) + rec.get("keywords", []) +
            place_keywords(rec) + ([DUPLICATE_KEYWORD] if is_dup else [])))
        good, _ = write_tags(path, kws, rec.get("caption", ""), place,
                             keep_backup=False)
        ok, fail = ok + good, fail + (not good)

    print(f"\napply complete: {ok} written, {fail} failed")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="photo_dedup")
    sub = parser.add_subparsers(dest="command", required=True)

    pd = sub.add_parser("describe", help="Hash + Haiku-describe every image in a folder.")
    pd.add_argument("export_dir", type=Path)
    pd.add_argument("--limit", type=int, default=0,
                    help="Stop after N new descriptions (0 = no limit).")
    pd.add_argument("--min-desc-len", type=int, default=40,
                    help="Photos already carrying a description this long are left alone.")
    pd.add_argument("--redescribe-all", action="store_true",
                    help="Redescribe even photos that already have a caption on disk.")

    pgeo = sub.add_parser("geo", help="Reverse-geocode each export's GPS to city/state/country.")
    pgeo.add_argument("export_dir", type=Path)

    pl = sub.add_parser("link", help="Match exports to catalog originals by image content.")
    pl.add_argument("export_dir", type=Path)
    pl.add_argument("catalog_dir", type=Path)

    pg = sub.add_parser("group", help="Cluster same-scene shots and pick keepers.")
    pg.add_argument("export_dir", type=Path)

    ps = sub.add_parser("sheet", help="Render an HTML contact sheet for review.")
    ps.add_argument("export_dir", type=Path)
    ps.add_argument("--out", type=Path, default=None)

    pa = sub.add_parser("apply", help="Write keywords/caption/location back (dry-run by default).")
    pa.add_argument("export_dir", type=Path)
    pa.add_argument("--apply", action="store_true", help="Perform the write.")

    args = parser.parse_args(argv)
    if not args.export_dir.is_dir():
        parser.error(f"Not a directory: {args.export_dir}")
    if args.command == "describe":
        cmd_describe(args)
    elif args.command == "geo":
        cmd_geo(args)
    elif args.command == "link":
        if not args.catalog_dir.is_dir():
            parser.error(f"Not a directory: {args.catalog_dir}")
        cmd_link(args)
    elif args.command == "group":
        cmd_group(args)
    elif args.command == "sheet":
        cmd_sheet(args)
    elif args.command == "apply":
        cmd_apply(args)


if __name__ == "__main__":
    main()
