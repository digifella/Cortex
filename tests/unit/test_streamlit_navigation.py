from pathlib import Path

from cortex_ui.navigation import HIDDEN_LEGACY_PAGES, NAVIGATION_SECTIONS


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_navigation_routes_are_unique_and_exist():
    routes = [
        path
        for _section, definitions in NAVIGATION_SECTIONS
        for path, _title, _icon in definitions
    ]

    assert len(routes) == len(set(routes))
    assert all((PROJECT_ROOT / route).is_file() for route in routes)


def test_navigation_labels_are_unique():
    labels = [
        title
        for _section, definitions in NAVIGATION_SECTIONS
        for _path, title, _icon in definitions
    ]

    assert len(labels) == len(set(labels))


def test_legacy_pages_are_retained_but_not_registered():
    registered_routes = {
        path
        for _section, definitions in NAVIGATION_SECTIONS
        for path, _title, _icon in definitions
    }

    assert registered_routes.isdisjoint(HIDDEN_LEGACY_PAGES)
    assert all((PROJECT_ROOT / route).is_file() for route in HIDDEN_LEGACY_PAGES)
