"""Pytest configuration for ComfyUI-Chatterbox.

Responsibilities, in the order they run:

1. Locate a real ComfyUI checkout (:func:`_resolve_comfyui_root`) and put it on
   ``sys.path``. ``comfy`` and ``folder_paths`` are imported **for real** -- the
   nodes under test are ComfyUI nodes, so a fake ComfyUI would test nothing.
2. Mock the optional web layer (``server`` / ``aiohttp``) *only if* it cannot
   be imported in the target environment.
3. Mock the whole vendored ``chatterbox`` tree so collection never drags in
   ``perth``, ``diffusers``, ``s3tokenizer``, ``conformer`` or the model stack.
4. Register the custom-node package under the importable alias
   ``ComfyUI_Chatterbox`` -- the checkout directory contains a hyphen, so
   ``import ComfyUI-Chatterbox`` is syntactically impossible.

Public names other batches rely on are listed in ``__all__`` at the bottom.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import traceback
from types import ModuleType
from typing import Iterable, Sequence
from unittest.mock import MagicMock

import pytest

__all__ = [
    "ROOT_DIR",
    "COMFYUI_ROOT",
    "PKG_ALIAS",
    "ALIAS_SUBMODULES",
    "VENDORED_MODULES",
    "MOCKED_MODULES",
    "REAL_MODULES",
    "load_package_alias",
    "register_package_alias",
    "ensure_alias_submodules",
    "reset_comfyui_root_cache",
    "mock_prompt_server",
    "repo_root",
]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ====================================================================
# 1. COMFYUI ROOT RESOLUTION
# ====================================================================

# Filled by the first successful resolution; two successive calls must agree
# even if the environment changes in between (plan S0.3).
_COMFYUI_ROOT_CACHE: str | None = None


def reset_comfyui_root_cache() -> None:
    """Drop the memoised ComfyUI root (test-only hook for S0.3)."""
    global _COMFYUI_ROOT_CACHE
    _COMFYUI_ROOT_CACHE = None


def _is_comfyui_root(path: str) -> bool:
    """True when ``path`` is a ComfyUI root carrying its top-level modules."""
    return all(
        os.path.isfile(os.path.join(path, name))
        for name in ("nodes.py", "execution.py")
    )


def _comfyui_search_dirs() -> list[str]:
    """Directories to scan for a ComfyUI checkout: parent first, then siblings.

    Derived entirely from :data:`ROOT_DIR` so the result moves with the
    checkout instead of being baked in.
    """
    search_dirs: list[str] = []
    comfy_nodes_parent = os.path.dirname(os.path.dirname(ROOT_DIR))
    if os.path.isdir(comfy_nodes_parent):
        search_dirs.append(comfy_nodes_parent)
    repo_parent = os.path.dirname(ROOT_DIR)
    if os.path.isdir(repo_parent):
        search_dirs.append(repo_parent)
    return search_dirs


def _resolve_comfyui_root() -> str:
    """Locate a ComfyUI checkout for the suite to import.

    An explicit ``COMFYUI_ROOT`` always wins and is returned verbatim -- even
    when the path does not exist. A bad override must surface as a loud import
    error, not be silently "helpfully" replaced by a different checkout.

    Otherwise the parent of this repo and its siblings are scanned for a
    directory carrying ComfyUI's top-level ``nodes.py`` and ``execution.py``.

    There is deliberately NO hardcoded fallback. A stale absolute path on
    ``sys.path`` does not fail here; it fails later as an opaque
    "attempted relative import beyond top-level package" (this repo's own
    ``nodes`` shadowing ComfyUI's) inside an unrelated test file. Failing by
    name is the only actionable outcome.

    The result is memoised: repeated calls return the identical string, so a
    test that mutates the environment must call :func:`reset_comfyui_root_cache`.
    """
    global _COMFYUI_ROOT_CACHE
    if _COMFYUI_ROOT_CACHE is not None:
        return _COMFYUI_ROOT_CACHE

    explicit = os.environ.get("COMFYUI_ROOT")
    if explicit:
        _COMFYUI_ROOT_CACHE = explicit
        return _COMFYUI_ROOT_CACHE

    search_dirs = _comfyui_search_dirs()
    for base in search_dirs:
        try:
            entries = sorted(os.listdir(base))
        except OSError:
            continue
        for entry in entries:
            candidate = os.path.join(base, entry)
            if _is_comfyui_root(candidate):
                _COMFYUI_ROOT_CACHE = candidate
                return _COMFYUI_ROOT_CACHE
            nested = os.path.join(candidate, "ComfyUI")
            if _is_comfyui_root(nested):
                _COMFYUI_ROOT_CACHE = nested
                return _COMFYUI_ROOT_CACHE

    raise RuntimeError(
        "Could not locate a ComfyUI checkout.\n"
        "These tests import ComfyUI itself (`comfy`, `folder_paths`), so a "
        "real installation is required.\n"
        "Point COMFYUI_ROOT at it and re-run, e.g.:\n"
        "    COMFYUI_ROOT=/path/to/ComfyUI python -m pytest tests/\n"
        "A directory qualifies when it contains ComfyUI's top-level nodes.py "
        "and execution.py.\n"
        f"Searched: {search_dirs}"
    )


COMFYUI_ROOT = _resolve_comfyui_root()

# ComfyUI first so `import nodes` resolves to ComfyUI's, never this repo's.
for _path in (ROOT_DIR, COMFYUI_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)


# ====================================================================
# 2. MOCK THE OPTIONAL WEB LAYER (ONLY WHAT IS NOT IMPORTABLE)
# ====================================================================
# ``server`` and ``aiohttp`` are ComfyUI's optional web dependencies. They pull
# in a socket server and a live event loop, which has no place in collection.
# They are mocked *conditionally*: when the environment provides them they are
# imported for real, because mocking something that works only hides breakage.

MOCKED_MODULES: dict[str, ModuleType] = {}
REAL_MODULES: list[str] = ["comfy", "folder_paths"]


def _ensure_optional_web_module(name: str) -> None:
    """Import ``name`` for real; fall back to a ``MagicMock`` on ImportError."""
    try:
        importlib.import_module(name)
    except Exception:  # pragma: no cover - depends on the target environment
        mock = MagicMock()
        MOCKED_MODULES[name] = mock
        sys.modules[name] = mock
    else:
        REAL_MODULES.append(name)


for _web_mod in ("server", "aiohttp", "aiohttp.web"):
    _ensure_optional_web_module(_web_mod)


# ====================================================================
# 3. MOCK THE VENDORED chatterbox TREE
# ====================================================================
# Collection must never load the model stack. ``perth`` is not installed,
# ``diffusers`` is installed but broken (raises on import), and importing the
# real vendor package drags in librosa/transformers/torch. Every entry is
# mocked, so the day a vendored file gains a new heavy import, collection
# still works and the heavy dependency is still absent from sys.modules.
#
# The mocks are registered under the node package's alias, not under the
# top-level ``chatterbox`` name: see :func:`_install_vendor_mocks`.

PKG_ALIAS = "ComfyUI_Chatterbox"
# The alias finder materialises `modules/` and `src/` as namespace-style
# stand-ins. On disk the two are not alike: `src/` has no `__init__.py` at all,
# while `modules/` has an empty one and is a real package at runtime (see
# `tests/test_vendor_loader.py`). `nodes/` became a real package in S6.1: it is
# resolved by the ordinary path finder through the alias package's `__path__`,
# so listing it here would only shadow that with a namespace-style stand-in.
ALIAS_SUBMODULES: tuple[str, ...] = ("modules", "src")

# Entry points upstream v0.1.7 ships. Registered unconditionally, so the mock
# set is correct even if one of them is missing from the tree -- a re-sync that
# drops a module then shows up as a failing name in the file set, not as an
# unmocked import reaching for ``perth`` during collection.
_VENDOR_ENTRY_POINTS: tuple[str, ...] = (
    "chatterbox",
    "chatterbox.tts",
    "chatterbox.vc",
    "chatterbox.mtl_tts",
    "chatterbox.tts_turbo",
    "chatterbox.models",
)


def vendored_module_names() -> list[str]:
    """Every importable dotted name under ``src/chatterbox/``, sorted.

    Derived from the tree on disk rather than hardcoded, so the v0.1.7 re-sync
    (which added ``mtl_tts``, ``tts_turbo`` and
    ``models/s3gen/utils/intmeanflow`` and dropped
    ``models/t3/inference/alignment_stream_analyzer``) is picked up here with no
    edit to this file. ``tests/test_vendor_signatures.py`` pins the result so the
    walk cannot silently stop covering a module.
    """
    src_dir = os.path.join(ROOT_DIR, "src")
    names: set[str] = set(_VENDOR_ENTRY_POINTS)
    vend_root = os.path.join(src_dir, "chatterbox")
    for dirpath, dirnames, filenames in os.walk(vend_root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        rel_dir = os.path.relpath(dirpath, src_dir).replace(os.sep, ".")
        if rel_dir == ".":
            continue
        names.add(rel_dir)
        for filename in sorted(filenames):
            if filename.endswith(".py") and filename != "__init__.py":
                names.add(f"{rel_dir}.{filename[:-3]}")
    return sorted(names)


VENDORED_MODULES: tuple[str, ...] = tuple(vendored_module_names())


def _install_vendor_mocks(names: Iterable[str]) -> None:
    """Register a fresh ``MagicMock`` for every name under the aliased prefix.

    Only the aliased form is registered. The top-level ``chatterbox`` name is
    the one the shipped code must not use, so a harness that fabricates it
    cannot see a regression back to ``import chatterbox`` -- it would keep
    passing against a module name the runtime is forbidden to occupy.
    """
    for name in names:
        sys.modules[f"{PKG_ALIAS}.src.{name}"] = MagicMock()


_install_vendor_mocks(VENDORED_MODULES)


# ====================================================================
# 4. ALIAS FOR THE HYPHENATED CUSTOM-NODE PACKAGE
# ====================================================================
# ComfyUI loads custom nodes by directory name, so the real module is called
# "ComfyUI-Chatterbox" -- not a valid identifier. Everything below registers it
# under a stable alias instead.


def load_package_alias(package_name: str, init_path: str) -> ModuleType:
    """Load ``init_path`` as the package ``package_name`` and register it.

    Strict: any failure inside the module is re-raised as a ``RuntimeError``
    whose message names the offending file, because "collection broke" without
    a path is not actionable.
    """
    spec = importlib.util.spec_from_file_location(
        package_name, init_path, submodule_search_locations=[os.path.dirname(init_path)]
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Cannot build an import spec for {package_name} at {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(package_name, None)
        raise RuntimeError(
            f"Failed to import package alias {package_name!r} from {init_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return module


def register_package_alias(package_name: str, init_path: str) -> ModuleType:
    """Load ``init_path`` as ``package_name``, degrading instead of raising.

    On failure the alias is still registered -- as a namespace package with an
    empty ``__path__``-style ``__spec__`` -- carrying ``_alias_import_error``
    with the full traceback. Losing the submodules would be worse than losing
    the root module: everything downstream can still be imported and the
    reason is inspectable.
    """
    try:
        return load_package_alias(package_name, init_path)
    except RuntimeError as exc:
        module = ModuleType(package_name)
        module.__path__ = [os.path.dirname(init_path)]  # type: ignore[attr-defined]
        module.__spec__ = importlib.machinery.ModuleSpec(package_name, None, is_package=True)
        module.__spec__.submodule_search_locations = [os.path.dirname(init_path)]
        module._alias_import_error = str(exc)
        sys.modules[package_name] = module
        return module


class _NamespaceAliasLoader(importlib.abc.Loader):
    """Exec-less loader that yields a namespace-style package.

    Sets ``__path__`` to the real directory and deliberately leaves
    ``__file__`` unset, exactly like :mod:`types`-style namespace packages.
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        module.__path__ = [self._path]  # type: ignore[attr-defined]


class _AliasSubmoduleLoader(importlib.abc.Loader):
    """Loads a real ``<name>.py`` and degrades to a bare module on failure.

    The source is executed in place rather than through a nested
    ``spec.loader.exec_module`` so the module keeps the outer spec -- with a
    correct ``origin`` -- and therefore keeps working relative imports via
    ``__package__``. A broken module must not stop collection, but silently
    losing the symbol is worse than failing: the traceback is kept on the
    module as ``_alias_import_error`` so a test can report it.
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as handle:
                source = handle.read()
            code = compile(source, self._path, "exec")
            exec(code, module.__dict__)  # noqa: S102 -- importing repo source
        except Exception:
            module._alias_import_error = (
                f"Failed to import {module.__name__!r} from {self._path}:\n"
                f"{traceback.format_exc()}"
            )


class _AliasSubmoduleFinder(importlib.abc.MetaPathFinder):
    """Materialises ``ComfyUI_Chatterbox.{modules,nodes,src}`` on first use."""

    def __init__(self, root_dir: str, package_name: str, submodules: Sequence[str]) -> None:
        self._root_dir = root_dir
        self._prefix = package_name + "."
        self._submodules = frozenset(submodules)

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(self._prefix):
            return None
        name = fullname[len(self._prefix) :]
        if name not in self._submodules:
            return None

        module_file = os.path.join(self._root_dir, f"{name}.py")
        package_dir = os.path.join(self._root_dir, name)
        if os.path.isfile(module_file):
            spec = importlib.util.spec_from_file_location(fullname, module_file)
            if spec is not None:
                spec.loader = _AliasSubmoduleLoader(module_file)
                return spec
        if os.path.isdir(package_dir):
            spec = importlib.machinery.ModuleSpec(
                fullname, _NamespaceAliasLoader(package_dir), is_package=True
            )
            spec.submodule_search_locations = [package_dir]
            return spec
        return None  # pragma: no cover - defensive


_ALIAS_FINDER = _AliasSubmoduleFinder(ROOT_DIR, PKG_ALIAS, ALIAS_SUBMODULES)
sys.meta_path.insert(0, _ALIAS_FINDER)


def ensure_alias_submodules() -> dict[str, ModuleType]:
    """Import and return every aliased submodule, keyed by dotted name.

    Idempotent: the finder is a no-op once ``sys.modules`` holds the entry.
    """
    for name in ALIAS_SUBMODULES:
        importlib.import_module(f"{PKG_ALIAS}.{name}")
    return {name: sys.modules[f"{PKG_ALIAS}.{name}"] for name in ALIAS_SUBMODULES}


# The repo-root alias is loaded eagerly so a broken ``__init__.py`` surfaces at
# conftest import, but a failure degrades to a bare namespace package instead of
# killing collection -- later batches still need the submodules.
#
# ``sys.path`` is snapshotted and restored around the exec: the entry point
# does not write to ``sys.path`` any more (the vendored tree is reached
# package-relatively), but the harness must not inherit a path entry from any
# import -- a test asserting the mutation is gone has to start from a clean
# slate (S2.4). Entries added later by pytest are unaffected: only the delta is
# removed.
_repo_init = os.path.join(ROOT_DIR, "__init__.py")
if PKG_ALIAS not in sys.modules:
    _sys_path_before = list(sys.path)
    try:
        register_package_alias(PKG_ALIAS, _repo_init)
    finally:
        for _entry in list(sys.path):
            if _entry not in _sys_path_before:
                sys.path.remove(_entry)


def _make_dunder_init_alias() -> ModuleType:
    """Build the stand-in pytest expects under the bare name ``__init__``.

    pytest's ``Package`` collector imports *every* ``__init__.py`` it meets --
    including this repository's, which is a ComfyUI custom-node entry point
    built from relative imports and ``folder_paths``. Imported parentless (as
    pytest does: ``import_module("__init__")``) it dies with "attempted
    relative import with no known parent package" before a single test runs.

    Pre-registering the name makes that collection step a no-op. The module is
    given ``__package__`` and ``__path__`` so anything that *does* import it
    resolves relative imports through the alias registered above, and the real
    entry point stays reachable as ``ComfyUI_Chatterbox``.
    """
    module = ModuleType("__init__")
    module.__file__ = _repo_init
    module.__path__ = [ROOT_DIR]  # type: ignore[attr-defined]
    module.__package__ = PKG_ALIAS
    module.__spec__ = importlib.machinery.ModuleSpec("__init__", None, is_package=True)
    module.__spec__.submodule_search_locations = [ROOT_DIR]
    return module


sys.modules.setdefault("__init__", _make_dunder_init_alias())


# ``comfy`` and ``folder_paths`` are imported for real, after the vendored tree
# is mocked, so a broken vendor import can never break ComfyUI itself.
import comfy  # noqa: E402
import folder_paths  # noqa: E402


# ====================================================================
# 5. FIXTURES
# ====================================================================


@pytest.fixture
def repo_root() -> str:
    """Absolute path of the custom-node checkout under test."""
    return ROOT_DIR


@pytest.fixture
def mock_prompt_server():
    """A resettable ``PromptServer.instance`` stand-in, or None when server is real."""
    server = sys.modules.get("server")
    if server is None or isinstance(server, MagicMock):  # pragma: no branch
        instance = MagicMock()
        if server is not None:
            server.PromptServer.instance = instance
        instance.send_sync.reset_mock()
        yield instance
    else:  # pragma: no cover - only when a real server module is present
        yield None