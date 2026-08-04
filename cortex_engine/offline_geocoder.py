"""Offline reverse geocoding for photo location enrichment.

Nominatim requires network access and asks for ~1 request/second, which makes
it unusable when travelling without a connection. This module resolves
coordinates from a local GeoNames dataset instead — no network, no rate limit.

Trade-off: the offline dataset returns the nearest *populated place*, which is
usually a suburb rather than the metro name. Coordinates in Toowong resolve to
"Toowong" offline and "Brisbane" via Nominatim. State and country match either
way, so keyword search by state/country is unaffected.

The `reverse_geocoder` dependency is optional. When it is absent this module
reports that clearly rather than silently returning empty location fields —
silent degradation is the failure mode this pipeline has been bitten by before.
"""
from typing import Dict, Optional, Tuple

from .utils import get_logger

logger = get_logger(__name__)

_EMPTY: Dict[str, str] = {"country": "", "state": "", "city": ""}

_rg = None
_pycountry = None
_import_failed = False


def _load() -> bool:
    """Import the offline geocoding stack once. Returns True when available."""
    global _rg, _pycountry, _import_failed
    if _rg is not None:
        return True
    if _import_failed:
        return False
    try:
        import reverse_geocoder  # noqa: WPS433 — optional dependency
        import pycountry  # noqa: WPS433
    except ImportError as exc:
        _import_failed = True
        logger.debug(
            "Offline geocoding unavailable (%s). Install with: "
            "pip install reverse_geocoder pycountry",
            exc,
        )
        return False
    _rg = reverse_geocoder
    _pycountry = pycountry
    return True


def is_available() -> bool:
    """True when offline geocoding can be used."""
    return _load()


def reverse_geocode_offline(lat: float, lon: float) -> Dict[str, str]:
    """Resolve coordinates to city/state/country without network access.

    Returns empty fields when the dataset is unavailable or the lookup fails,
    matching the shape of DocumentTextifier.reverse_geocode().
    """
    if not _load():
        return dict(_EMPTY)
    try:
        result = _rg.search([(float(lat), float(lon))], mode=1)[0]
    except Exception as exc:  # dataset load or lookup failure
        logger.warning("Offline reverse geocode failed for (%s, %s): %s", lat, lon, exc)
        return dict(_EMPTY)

    code = (result.get("cc") or "").strip()
    country = ""
    if code:
        match = _pycountry.countries.get(alpha_2=code)
        country = match.name if match else code

    return {
        "city": (result.get("name") or "").strip(),
        "state": (result.get("admin1") or "").strip(),
        "country": country,
    }


def resolve(
    lat: float,
    lon: float,
    mode: str = "auto",
    online_fn=None,
) -> Tuple[Dict[str, str], str]:
    """Reverse geocode using the requested strategy.

    mode:
        "online"  — use online_fn only
        "offline" — use the local dataset only
        "auto"    — try online_fn first, fall back offline when it yields nothing

    Returns (location_dict, source) where source is "online", "offline" or "none".
    """
    mode = (mode or "auto").strip().lower()

    if mode == "offline":
        return reverse_geocode_offline(lat, lon), "offline"

    if mode == "online":
        location = online_fn(lat, lon) if online_fn else dict(_EMPTY)
        return (location or dict(_EMPTY)), "online"

    # auto — prefer the richer online result, fall back when offline/unreachable
    if online_fn:
        try:
            location = online_fn(lat, lon) or {}
        except Exception as exc:
            logger.info("Online geocode failed (%s); falling back to offline", exc)
            location = {}
        if any((location or {}).values()):
            return location, "online"

    location = reverse_geocode_offline(lat, lon)
    return location, ("offline" if any(location.values()) else "none")


def describe_mode(mode: str) -> Optional[str]:
    """Return a warning string when the requested mode cannot be satisfied."""
    mode = (mode or "auto").strip().lower()
    if mode in {"offline", "auto"} and not is_available():
        return (
            "Offline geocoding is not installed — location lookup will need a "
            "network connection. Install with: pip install reverse_geocoder pycountry"
        )
    return None
