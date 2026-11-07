"""Import rules for the component layout (docs/PRD_RESTRUCTURE.md, section 6).

The five component folders never import each other; only the pipeline (and
the transport layer above it) compose them. Observability and storage are
leaves. Enforced here so a convenience import cannot quietly re-create the
hub-and-spoke coupling the restructure removed.
"""

import ast
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"

COMPONENTS = {"context", "guardrails", "gateway", "actions"}
LEAVES = {"observability", "storage"}
COMPOSERS = {"pipeline", "api"}
ROOT_MODULES = {"config", "main"}

# folder -> the folders it may import from (its own folder is always allowed).
ALLOWED = {
    **{name: {"config", *LEAVES} for name in COMPONENTS},
    "observability": {"config", "storage"},
    "storage": {"config"},
    "pipeline": {"config", *COMPONENTS, *LEAVES},
    "api": {"config", *COMPONENTS, *LEAVES, "pipeline"},
    "config": set(),
    "main": {"config", *COMPONENTS, *LEAVES, *COMPOSERS},
}


def _modules() -> dict[str, Path]:
    return {
        ".".join(["app", *path.relative_to(APP_DIR).with_suffix("").parts]): path
        for path in APP_DIR.rglob("*.py")
        if "__pycache__" not in path.parts
    }


def _folder(module: str) -> str:
    """'app.pipeline.nodes.generate' -> 'pipeline'; 'app.config' -> 'config'."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def _imports(module: str, path: Path) -> set[str]:
    """Folders this module imports from within the app package."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module.split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                target = ".".join([*base, *(node.module.split(".") if node.module else [])])
            else:
                target = node.module or ""
            if target.startswith("app."):
                targets.add(_folder(target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("app."):
                    targets.add(_folder(alias.name))
    return targets


@pytest.mark.xfail(strict=True, reason="until the component layout lands (PRD phase 8)")
def test_app_has_only_component_folders_at_top_level():
    top_level = {path.stem for path in APP_DIR.glob("*.py")} - {"__init__"}
    assert top_level == ROOT_MODULES
    folders = {
        path.name for path in APP_DIR.iterdir() if path.is_dir() and path.name != "__pycache__"
    }
    assert folders == COMPONENTS | LEAVES | COMPOSERS


@pytest.mark.xfail(strict=True, reason="until the component layout lands (PRD phase 8)")
def test_components_only_import_what_the_rules_allow():
    violations = []
    for module, path in sorted(_modules().items()):
        folder = _folder(module)
        if folder not in ALLOWED:
            violations.append(f"{module}: unknown folder {folder!r}")
            continue
        for target in sorted(_imports(module, path) - {folder}):
            if target not in ALLOWED[folder]:
                violations.append(f"{module} imports app.{target}")
    assert violations == []
