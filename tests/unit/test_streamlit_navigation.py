import ast
from pathlib import Path
import tomllib

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


def test_explicit_navigation_does_not_use_streamlits_reserved_pages_folder():
    assert not (PROJECT_ROOT / "pages").exists()
    assert all(
        not path.startswith("pages/")
        for _section, definitions in NAVIGATION_SECTIONS
        for path, _title, _icon in definitions
    )


def test_file_watcher_is_disabled_for_torch_compatibility():
    config = tomllib.loads(
        (PROJECT_ROOT / ".streamlit/config.toml").read_text(encoding="utf-8")
    )
    assert config["server"]["fileWatcherType"] == "none"


def _calls_main(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and (
        isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "main"
    )


def test_registered_pages_execute_main_under_streamlit_navigation():
    """Explicit Streamlit navigation executes page scripts as ``__page__``."""
    routes = [
        path
        for _section, definitions in NAVIGATION_SECTIONS
        for path, _title, _icon in definitions
    ]

    for route in routes:
        tree = ast.parse((PROJECT_ROOT / route).read_text(encoding="utf-8"))
        defines_main = any(
            isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == "main"
            for statement in tree.body
        )
        if not defines_main:
            continue

        calls_main_directly = any(_calls_main(statement) for statement in tree.body)
        calls_main_as_page = any(
            isinstance(statement, ast.If)
            and any(_calls_main(child) for child in statement.body)
            and any(
                isinstance(node, ast.Constant) and node.value == "__page__"
                for node in ast.walk(statement.test)
            )
            for statement in tree.body
        )
        assert calls_main_directly or calls_main_as_page, (
            f"{route} defines main() but does not invoke it when Streamlit runs "
            "the file as __page__"
        )
