"""S3.2 -- no tracked Python module may be dead.

Deleting a module leaves the tree *working*: nothing imports it any more, so
nothing breaks, and the dead file just sits there growing stale until someone
notices it. This scan is the brake. It walks the import graph from the
repository's entry points and fails on any Python file nothing reaches.

Scope and known gaps, stated rather than hidden:

* ``src/chatterbox/**`` is **excluded**. It is the vendored upstream package:
  imported package-relatively (``..src.chatterbox.tts``), mocked wholesale
  during collection, and deliberately free of ComfyUI-API awareness.
  Its integrity is asserted elsewhere, by content.
* Package ``__init__.py`` files are treated as roots. Python's import
  machinery loads them implicitly as the parent of any submodule, so they are
  reachable by construction and scanning them for inedges is meaningless.
* ``modules/**`` is reachable again. S3.1/S3.2 deleted the only importers (the
  V1 ``nodes.py``) and left the directory as a temporary root; S6 landed the
  replacement node layer, which imports the modules it needs, and
  ``chatterbox_nodes.py`` imports the node package from ``__init__.py``.
"""

from __future__ import annotations

import ast
import os

import pytest

from conftest import ROOT_DIR, PKG_ALIAS

# Paths relative to the repository root that are never scanned.
EXCLUDED_PREFIXES: tuple[str, ...] = (
    "src/chatterbox/",  # vendored upstream; imported as `..src.chatterbox.*`
    "src/chatterbox_flash/",  # vendored Flash sampler; imported as `..src.chatterbox_flash.*`
    "tests/",  # the test suite itself, and its helpers
)

# Entry points: something outside the repository must import them, so they have
# no inedge by design.
ALWAYS_ROOTS: tuple[str, ...] = (
    "__init__.py",  # the ComfyUI custom-node entry point
    "conftest.py",  # the pytest harness
)

# Temporary roots, each with the step that makes it unnecessary. See S3.2.
PHASE_ROOTS: dict[str, str] = {
    "nodes/": "S6.1/S6.2 node implementations; imported by chatterbox_nodes.py.",
}

# Files the scan would call dead, each with the step that deletes it.
#
# Empty, and deliberately kept as an empty map rather than deleted outright.
# Its one entry -- ``modules/utils.py``, a ``set_seed`` re-export shim orphaned
# when S6 replaced the V1 ``nodes.py`` -- was retired in B11 by deleting the
# file. The map survives so the escape hatch has an owner: an entry whose file
# is already gone is stale bookkeeping (see
# ``test_the_pending_deletion_list_is_not_a_graveyard``), which is how a
# retirement map turns into a place where dead code stops being reported.
PENDING_DELETION: dict[str, str] = {}


def _iter_python_files() -> list[str]:
    """Repo-relative POSIX paths of every scannable ``.py`` file.

    The filesystem is walked rather than ``git ls-files`` because the modules
    under construction are not committed yet, and a scan that only sees
    committed history cannot see the thing it exists to catch.

    Dot-directories (``.git``, ``.dev``, ``.codegraph``, ``.zcodeignore`` and any
    future tooling directory) are skipped wholesale: none of them ship.
    """
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        dirnames[:] = sorted(
            name for name in dirnames if not name.startswith(".") and name != "__pycache__"
        )
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            absolute = os.path.join(dirpath, filename)
            relative = os.path.relpath(absolute, ROOT_DIR).replace(os.sep, "/")
            if relative.startswith(EXCLUDED_PREFIXES):
                continue
            found.append(relative)
    return found


def _module_name(relative: str) -> str:
    """Dotted name of ``relative``, e.g. ``modules/loader.py`` ->
    ``modules.loader``; ``src/hf_progress/__init__.py`` ->
    ``src.hf_progress``."""
    parts = relative[:-3].split("/")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_of(relative: str) -> str:
    """The *enclosing* package of a module, i.e. its directory as a dotted name.

    This is what :func:`_resolve_relative` needs, and it differs from
    :func:`_module_name` for one case that matters: a module inside a package
    (``src/hf_progress/tracker.py``) belongs to ``src.hf_progress``, even though
    its own dotted name is ``src.hf_progress.tracker``. Using the module name
    here would resolve ``from .types import X`` inside that file to
    ``src.hf_progress.tracker.types``, which exists nowhere, and the walk would
    then report a perfectly well-imported module as dead.

    For a package's own ``__init__.py`` both readings agree.
    """
    parts = relative.split("/")[:-1]
    return ".".join(parts)


def _parse(relative: str) -> ast.Module:
    with open(os.path.join(ROOT_DIR, relative), "r", encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=relative)


def _resolve_relative(package: str, node: ast.ImportFrom) -> str:
    """Absolute dotted name a ``from .x import y`` statement refers to.

    ``package`` is the enclosing package, so ``level == 1`` keeps it whole and
    each extra dot steps one level out.
    """
    parts = package.split(".") if package else []
    if node.level > 1:
        parts = parts[: len(parts) - (node.level - 1)]
    if node.module:
        parts = parts + node.module.split(".")
    return ".".join(parts)


def _imported_names(module: ast.Module) -> set[str]:
    """Every dotted name the module imports, relative ones already resolved."""
    names: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                continue  # resolved by the caller, which knows the package
            if node.module:
                names.add(node.module)
                # `from a.b import c` also references `a.b.c`.
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
    return names


def _build_graph() -> tuple[dict[str, str], dict[str, set[str]], list[str]]:
    """Return (name -> relative path, relative path -> imports, all files)."""
    files = _iter_python_files()
    name_to_path: dict[str, str] = {}
    for relative in files:
        name = _module_name(relative)
        # The repository root package is importable under the harness alias
        # only; register both so either spelling resolves.
        name_to_path.setdefault(name, relative)
        if name not in (PKG_ALIAS, "src", "conftest"):
            name_to_path.setdefault(f"{PKG_ALIAS}.{name}", relative)

    imports: dict[str, set[str]] = {}
    for relative in files:
        module = _parse(relative)
        package = _package_of(relative)
        targets = _imported_names(module)
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom) and node.level:
                targets.add(_resolve_relative(package, node))
        imports[relative] = targets
    return name_to_path, imports, files


def _is_root(relative: str) -> bool:
    if relative in ALWAYS_ROOTS or relative.endswith("/__init__.py"):
        return True
    return any(relative.startswith(prefix) for prefix in PHASE_ROOTS)


@pytest.fixture(scope="module")
def graph() -> dict:
    name_to_path, imports, files = _build_graph()
    reached: set[str] = set()
    frontier = [relative for relative in files if _is_root(relative)]
    while frontier:
        current = frontier.pop()
        if current in reached:
            continue
        reached.add(current)
        for target in imports.get(current, ()):
            resolved = name_to_path.get(target)
            if resolved and resolved not in reached:
                frontier.append(resolved)
    return {
        "files": files,
        "reached": reached,
        "roots": sorted(relative for relative in files if _is_root(relative)),
        "name_to_path": name_to_path,
        "imports": imports,
    }


class TestDeletedSurfaceIsGone:
    """The files the restructure deleted must not have come back."""

    @pytest.mark.parametrize(
        "relative",
        ["nodes.py", "modules/chatterbox_handler.py", "modules/utils.py"],
    )
    def test_file_is_absent(self, relative: str) -> None:
        assert not os.path.exists(os.path.join(ROOT_DIR, relative)), (
            f"{relative} was deleted; a resurrected copy is either a failed "
            f"deletion or an unrecorded reintroduction of the V1 surface"
        )

    def test_nothing_names_the_deleted_handler(self, graph: dict) -> None:
        """S3 left the entry point importing a file that no longer existed.

        It was allowed to, and only to, until S6.5 rewrote it. That rewrite
        landed with the V3 node layer, so the transitional allowance is over:
        nothing may name ``chatterbox_handler`` any more, and its constants
        live in ``modules/constants.py``.
        """
        offenders = sorted(
            relative
            for relative, targets in graph["imports"].items()
            if any("chatterbox_handler" in name for name in targets)
        )
        assert offenders == [], (
            f"{offenders} import the deleted handler; its constants now live in "
            f"modules/constants.py (S4/S5)"
        )


class TestNoDeadModules:
    def test_the_scan_actually_looks_at_something(self, graph: dict) -> None:
        """A vacuous pass is worse than no test: the scan must find modules."""
        assert len(graph["files"]) >= 2
        assert "__init__.py" in graph["files"]
        assert graph["roots"], "no entry points found -- the scan cannot be trusted"

    def test_every_module_is_reachable(self, graph: dict) -> None:
        dead = sorted(set(graph["files"]) - set(graph["reached"]) - set(PENDING_DELETION))
        assert not dead, (
            "nothing imports these modules:\n  "
            + "\n  ".join(dead)
            + "\nEither they are dead code (delete them) or their importer is "
            "missing. See the module docstring for the temporary root set."
        )

    def test_the_pending_deletion_list_is_not_a_graveyard(self) -> None:
        """The retirement map must be empty, and must stay empty.

        An entry whose file is already gone is stale bookkeeping, and a map that
        accumulates such entries is how dead code stops being reported. B11
        retired the map's only entry (``modules/utils.py``) by deleting the
        file, so ``PENDING_DELETION`` is now ``{}``. This asserts that
        directly: an empty map is the only state in which the exemption cannot
        hide anything, and adding an entry is a deliberate act that has to come
        with the step that empties it again.
        """
        assert PENDING_DELETION == {}, (
            "PENDING_DELETION is a retirement map, not a graveyard. Every entry "
            "must name a file that still exists AND delete it in the same batch; "
            f"the live entries are {sorted(PENDING_DELETION)}"
        )

    def test_the_vendored_tree_is_excluded_not_forgotten(self, graph: dict) -> None:
        """The exclusion is deliberate; assert it is still scoped tightly."""
        assert not any(
            path.startswith(("src/chatterbox/", "src/chatterbox_flash/"))
            for path in graph["files"]
        )
        assert "src/hf_progress/__init__.py" not in EXCLUDED_PREFIXES

    def test_entry_point_reaches_a_real_module(self, graph: dict) -> None:
        """Sanity: a real import edge exists, so the walk is not vacuous."""
        assert graph["reached"] >= set(graph["roots"])
        imports = graph["imports"]["__init__.py"]
        assert any("chatterbox_nodes" in name for name in imports), (
            "the entry point must import chatterbox_nodes.py; "
            f"it imports {sorted(imports)}"
        )

    def test_conftest_imports_are_resolvable(self, graph: dict) -> None:
        """`from conftest import ROOT_DIR` resolves to the repo-root conftest."""
        assert graph["name_to_path"]["conftest"] == "conftest.py"