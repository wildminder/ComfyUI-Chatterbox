"""The vendored packages are private to this node.

The vendored ``chatterbox`` tree used to be published as a top-level import
root, so it competed for a name another ComfyUI plugin may also ship. The
node now reaches its own tree package-relatively, and four properties keep
that true:

1. a foreign top-level ``chatterbox`` on ``sys.path`` does not answer for
   ours -- the loader binds our classes;
2. no shipped file imports a top-level ``chatterbox`` / ``chatterbox_flash``;
3. the flash package's own imports resolve to our tree;
4. the harness does not fabricate the top-level name itself.

The suite stays light: nothing here loads the model stack at collection
time. Two of the checks need the *real* vendored modules -- a mock has no
honest ``__module__`` -- and lift the harness mocks for their duration only,
through :func:`conftest_helpers.real_vendor_modules`, which puts them back.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import os
import sys

import pytest

from conftest import PKG_ALIAS, ROOT_DIR

#: The vendored packages must never occupy a top-level name: another ComfyUI
#: plugin may ship its own ``chatterbox``, and the two must not collide.
TOP_LEVEL_VENDOR_NAMES = ("chatterbox", "chatterbox_flash")

LOADER_PATH = os.path.join(ROOT_DIR, "modules", "loader.py")

#: Third-party modules the vendored tree legitimately needs. One of them being
#: absent means the check cannot run here -- it never means the check passed.
KNOWN_EXTERNAL_REQUIREMENTS = (
    "torch",
    "torchaudio",
    "numpy",
    "huggingface_hub",
    "safetensors",
    "conformer",
    "s3tokenizer",
    "tokenizers",
    "librosa",
    "einops",
    "transformers",
    "tqdm",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_top_level_vendor(name: str) -> bool:
    """True for a top-level vendor name and everything below it."""
    return any(
        name == top or name.startswith(top + ".")
        for top in TOP_LEVEL_VENDOR_NAMES
    )


def _top_level_vendor_modules() -> list[str]:
    """Every top-level vendor name currently in ``sys.modules``."""
    return sorted(name for name in sys.modules if _is_top_level_vendor(name))


@contextlib.contextmanager
def _no_top_level_vendor_modules():
    """No top-level vendor name is registered while the block runs.

    Whatever was registered before is put back afterwards, so the block cannot
    leak a name into the rest of the suite.
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if _is_top_level_vendor(name)
    }
    for name in saved:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in [n for n in sys.modules if _is_top_level_vendor(n)]:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def _loader_import_statement() -> str:
    """The loader's own import of ``ChatterboxTTS``, verbatim.

    Read from the source rather than hardcoded here: a regression back to a
    top-level ``from chatterbox.tts import ...`` has to be what the test
    executes, not a string this file happens to carry.
    """
    with open(LOADER_PATH, encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source, filename=LOADER_PATH)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not any(alias.name == "ChatterboxTTS" for alias in node.names):
            continue
        parts = (node.module or "").split(".")
        if any(part in TOP_LEVEL_VENDOR_NAMES for part in parts):
            return ast.unparse(node)
    raise AssertionError(f"{LOADER_PATH} no longer imports ChatterboxTTS")


def _loader_bound_class():
    """The class the loader binds, by running the loader's own import."""
    loader = importlib.import_module(f"{PKG_ALIAS}.modules.loader")
    namespace = {"__name__": loader.__name__, "__package__": loader.__package__}
    exec(  # noqa: S102 -- the loader's own import statement, verbatim
        compile(_loader_import_statement(), "<loader-probe>", "exec"),
        namespace,
    )
    return namespace["ChatterboxTTS"]


def _shipped_python_files() -> list[str]:
    """Every shipped ``.py`` file: ``modules/``, ``nodes/``, ``src/``, the root."""
    files: list[str] = []
    for directory in ("modules", "nodes", "src"):
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT_DIR, directory)):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for filename in sorted(filenames):
                if filename.endswith(".py"):
                    files.append(os.path.join(dirpath, filename))
    for filename in sorted(os.listdir(ROOT_DIR)):
        if filename.endswith(".py"):
            files.append(os.path.join(ROOT_DIR, filename))
    return files


def _relative(path: str) -> str:
    """``path`` relative to the repository root, with forward slashes."""
    return os.path.relpath(path, ROOT_DIR).replace(os.sep, "/")


def _top_level_vendor_imports(path: str) -> list[str]:
    """Every import in ``path`` that reaches a top-level vendor name.

    The scan is AST-based on purpose: a docstring that shows an import is not
    an import -- ``src/chatterbox_flash/attention_flashinfer.py`` and
    ``src/hf_progress/integrations/sse.py`` both carry such examples.
    """
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders.extend(
                f"import {alias.name} (line {node.lineno})"
                for alias in node.names
                if alias.name.split(".")[0] in TOP_LEVEL_VENDOR_NAMES
            )
        elif isinstance(node, ast.ImportFrom):
            if (
                node.level == 0
                and node.module
                and node.module.split(".")[0] in TOP_LEVEL_VENDOR_NAMES
            ):
                offenders.append(f"from {node.module} import ... (line {node.lineno})")
        elif isinstance(node, ast.Call):
            # ``importlib.import_module("chatterbox.tts")`` is an import too.
            function = node.func
            name = getattr(function, "attr", None) or getattr(function, "id", None)
            if name not in ("import_module", "__import__") or not node.args:
                continue
            first = node.args[0]
            if (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value.split(".")[0] in TOP_LEVEL_VENDOR_NAMES
            ):
                offenders.append(f"import_module({first.value!r}) (line {node.lineno})")
    return offenders


# ---------------------------------------------------------------------------
# 1. A foreign top-level package is present
# ---------------------------------------------------------------------------


class TestForeignPackagePresent:
    """Another plugin's ``chatterbox`` must not answer for ours."""

    def test_the_loader_binds_our_class(self, tmp_path, monkeypatch) -> None:
        foreign_root = tmp_path / "foreign_plugin"
        package = foreign_root / "chatterbox"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            "WHO = 'foreign'\n\n\nclass ChatterboxTTS:\n    WHO = 'foreign'\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(foreign_root))

        with _no_top_level_vendor_modules():
            foreign = importlib.import_module("chatterbox")
            assert foreign.__file__.startswith(str(foreign_root)), (
                "the foreign package did not take the top-level name; the "
                "collision this test is about never happened"
            )

            from .conftest_helpers import real_vendor_modules

            try:
                with real_vendor_modules(["tts"]):
                    bound = _loader_bound_class()
                    assert bound.__module__.startswith(f"{PKG_ALIAS}.src.chatterbox"), (
                        f"the loader bound {bound.__module__!r}; it must belong "
                        "to this node's package"
                    )
                    assert bound.__module__ != foreign.ChatterboxTTS.__module__, (
                        "the loader bound the foreign class"
                    )
                    assert bound is not foreign.ChatterboxTTS
            except ModuleNotFoundError as exc:
                # The vendored tree needs the heavy extras. Without them the
                # check cannot run here -- it never means the check passed.
                missing = (exc.name or "").split(".")[0]
                if missing in KNOWN_EXTERNAL_REQUIREMENTS:
                    pytest.skip(
                        f"the vendored tree needs {exc.name!r}, which is not "
                        "installed here"
                    )
                raise


# ---------------------------------------------------------------------------
# 2. No top-level names are left in the shipped source
# ---------------------------------------------------------------------------


class TestNoTopLevelImports:
    """No shipped file reaches a top-level vendor name."""

    def test_no_shipped_file_imports_a_top_level_vendor_name(self) -> None:
        offenders: dict[str, list[str]] = {}
        for path in _shipped_python_files():
            hits = _top_level_vendor_imports(path)
            if hits:
                offenders[_relative(path)] = hits
        assert offenders == {}, (
            f"these files import a top-level vendor name: {offenders}"
        )

    def test_the_scan_covers_the_shipped_tree(self) -> None:
        """A scan that looked at nothing would pass by default."""
        scanned = {_relative(path) for path in _shipped_python_files()}
        for expected in (
            "modules/loader.py",
            "nodes/tts_node.py",
            "src/chatterbox/tts.py",
            "src/chatterbox_flash/tts.py",
            "src/hf_progress/integrations/sse.py",
            "__init__.py",
        ):
            assert expected in scanned, f"the scan did not reach {expected}"


# ---------------------------------------------------------------------------
# 3. The flash package resolves to our tree
# ---------------------------------------------------------------------------


class TestFlashImportsResolve:
    """The flash package's own imports land in our package."""

    def test_the_flash_package_binds_our_s3gen(self) -> None:
        from .conftest_helpers import real_vendor_modules

        prefix = f"{PKG_ALIAS}.src.chatterbox_flash"
        saved = {
            name: module
            for name, module in sys.modules.items()
            if name == prefix or name.startswith(prefix + ".")
        }
        for name in saved:
            sys.modules.pop(name, None)
        try:
            # The mocks have to go: with them in place ``S3Gen`` is a mock and
            # its ``__module__`` says nothing about where the class came from.
            with real_vendor_modules(["models.s3gen"]):
                try:
                    flash = importlib.import_module(f"{prefix}.tts")
                except ModuleNotFoundError as exc:
                    missing = (exc.name or "").split(".")[0]
                    if missing in KNOWN_EXTERNAL_REQUIREMENTS:
                        pytest.skip(
                            f"the vendored tree needs {exc.name!r}, which is not "
                            "installed here"
                        )
                    raise
                s3gen = flash.S3Gen
                assert isinstance(s3gen.__module__, str), (
                    "S3Gen is still a mock: the harness mocks must be lifted for "
                    "this check"
                )
                assert s3gen.__module__.startswith(f"{PKG_ALIAS}.src.chatterbox"), (
                    f"S3Gen is bound to {s3gen.__module__!r}; it must belong to "
                    "this node's package"
                )
        finally:
            for name in [n for n in sys.modules if n == prefix or n.startswith(prefix + ".")]:
                sys.modules.pop(name, None)
            sys.modules.update(saved)


# ---------------------------------------------------------------------------
# 4. The harness does not fabricate the top-level name
# ---------------------------------------------------------------------------


class TestHarnessDoesNotFabricateTheTopLevelName:
    """The harness must not occupy the name the shipped code may not use."""

    def test_no_top_level_vendor_name_is_registered(self) -> None:
        for name in TOP_LEVEL_VENDOR_NAMES:
            assert name not in sys.modules, (
                f"the harness registered a top-level {name!r}; the shipped code "
                "must not occupy that name, and the harness must not fabricate "
                "it either"
            )
        assert _top_level_vendor_modules() == [], (
            "a top-level vendor name is registered: "
            f"{_top_level_vendor_modules()!r}"
        )
