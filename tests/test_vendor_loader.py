"""The vendored tree is reached through the node package, never at top level.

The vendored ``chatterbox`` tree used to be published as a top-level import
root: the node appended this repository's ``src/`` to ``sys.path`` and every
shipped import read ``from chatterbox.tts import ...``. That made this node
compete for a name another ComfyUI plugin may also ship -- and it made the test
harness fabricate the same name, so the suite passed against a module name the
fixed runtime must not use.

Three properties hold now, and each is cheap to check:

* ``src/`` is **not** an import root: the node reaches the vendored tree
  package-relatively (``from ..src.chatterbox.tts import ...``);
* no top-level ``chatterbox`` / ``chatterbox_flash`` is left in
  ``sys.modules``;
* the loader module, imported through the node package, resolves **our**
  classes.

The in-process tests run against the harness's own interpreter. The end-to-end
check needs a clean one (the harness mocks the whole vendor tree), so it runs
in a subprocess and *skips with a named reason* when a dependency is genuinely
missing -- never silently, and never by swallowing the error.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

from conftest import PKG_ALIAS, ROOT_DIR, VENDORED_MODULES

SRC_DIR = os.path.join(ROOT_DIR, "src")

MODULES_DIR = os.path.join(ROOT_DIR, "modules")

#: The vendored packages must never occupy a top-level name: another ComfyUI
#: plugin may ship its own ``chatterbox``, and the two must not collide.
TOP_LEVEL_VENDOR_NAMES = ("chatterbox", "chatterbox_flash")


def _bare_vendor_modules() -> list[str]:
    """Every top-level vendor name currently in ``sys.modules``."""
    return sorted(
        name
        for name in sys.modules
        if any(
            name == top or name.startswith(top + ".")
            for top in TOP_LEVEL_VENDOR_NAMES
        )
    )


def _loader_module():
    """The loader module, imported through the node package."""
    return importlib.import_module(f"{PKG_ALIAS}.modules.loader")


def _parse(path: str) -> ast.Module:
    with open(path, encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=path)


def _top_level_vendor_imports(tree: ast.AST) -> list[str]:
    """Every import statement in ``tree`` that reaches a top-level vendor name.

    Docstrings are not import statements, so a prose mention cannot trip this.
    """
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders.extend(
                alias.name
                for alias in node.names
                if alias.name.split(".")[0] in TOP_LEVEL_VENDOR_NAMES
            )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and node.module.split(".")[0] in TOP_LEVEL_VENDOR_NAMES
        ):
            offenders.append(node.module)
    return offenders


def _sys_path_writes(tree: ast.AST) -> list[str]:
    """Every place in ``tree`` that writes to ``sys.path``."""
    offenders: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"append", "extend", "insert"}
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "path"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "sys"
        ):
            offenders.append(ast.unparse(node))
    return offenders


class TestNoVendorPathOnSysPath:
    """``src/`` is not an import root any more."""

    def test_src_is_not_on_sys_path(self) -> None:
        assert SRC_DIR not in sys.path, (
            "the vendored tree is reached through the node package; its "
            "directory must not be an import root"
        )

    def test_importing_the_node_package_leaves_sys_path_alone(self) -> None:
        before = list(sys.path)
        _loader_module()
        assert sys.path == before, (
            "importing the node package changed sys.path"
        )

    def test_the_entry_point_never_writes_to_sys_path(self) -> None:
        tree = _parse(os.path.join(ROOT_DIR, "__init__.py"))
        assert _sys_path_writes(tree) == [], (
            "the entry point still writes to sys.path; the vendored tree is "
            "reached package-relatively"
        )

    def test_no_shipped_module_writes_to_sys_path(self) -> None:
        offenders: dict[str, list[str]] = {}
        for name in sorted(os.listdir(MODULES_DIR)):
            if not name.endswith(".py"):
                continue
            writes = _sys_path_writes(_parse(os.path.join(MODULES_DIR, name)))
            if writes:
                offenders[name] = writes
        assert offenders == {}, (
            f"these shipped modules write to sys.path: {offenders}"
        )


class TestNoTopLevelVendorNames:
    """The top-level names belong to whoever else ships them."""

    def test_no_top_level_vendor_module_is_registered(self) -> None:
        assert _bare_vendor_modules() == [], (
            "a top-level vendor name is registered: the shipped code must not "
            "occupy it, and the harness must not fabricate it either"
        )

    def test_the_harness_registers_the_tree_under_the_alias(self) -> None:
        for name in VENDORED_MODULES:
            assert f"{PKG_ALIAS}.src.{name}" in sys.modules, name


class TestLoaderPackage:
    def test_modules_is_a_real_package(self) -> None:
        init = os.path.join(MODULES_DIR, "__init__.py")
        assert os.path.isfile(init), (
            "modules/ has no __init__.py; `from .modules.loader import ...` "
            "cannot resolve without it"
        )

    def test_modules_init_is_empty(self) -> None:
        """An empty ``__init__.py``: re-exporting would drag the handler in."""
        init = os.path.join(MODULES_DIR, "__init__.py")
        with open(init, "r", encoding="utf-8") as handle:
            assert handle.read().strip() == "", "modules/__init__.py must stay empty"


class TestLoaderResolvesOurClasses:
    """The loader, imported through the node package, binds our classes."""

    def test_the_loader_is_imported_through_the_node_package(self) -> None:
        loader = _loader_module()
        assert loader.__name__ == f"{PKG_ALIAS}.modules.loader"
        assert loader.__package__ == f"{PKG_ALIAS}.modules"

    def test_the_loader_never_imports_a_top_level_vendor_name(self) -> None:
        tree = _parse(os.path.join(MODULES_DIR, "loader.py"))
        assert _top_level_vendor_imports(tree) == [], (
            "the loader reaches the vendored tree by a top-level name; it must "
            "go through the node package"
        )

    def test_the_loader_resolves_our_vendored_modules(self) -> None:
        loader = _loader_module()
        resolved = importlib.util.resolve_name(
            "..src.chatterbox.tts", loader.__package__
        )
        assert resolved == f"{PKG_ALIAS}.src.chatterbox.tts"
        assert importlib.import_module(resolved) is sys.modules[resolved]

    def test_the_class_the_loader_binds_is_ours(self) -> None:
        """The loader's own import statement, executed in its own package."""
        loader = _loader_module()
        namespace = {"__name__": loader.__name__, "__package__": loader.__package__}
        exec(  # noqa: S102 -- the loader's own import statement, verbatim
            compile(
                "from ..src.chatterbox.tts import ChatterboxTTS",
                "<loader-probe>",
                "exec",
            ),
            namespace,
        )
        module = sys.modules[f"{PKG_ALIAS}.src.chatterbox.tts"]
        assert namespace["ChatterboxTTS"] is module.ChatterboxTTS


# ---------------------------------------------------------------------------
# End to end, in a clean interpreter
# ---------------------------------------------------------------------------

PROBE = textwrap.dedent(
    """
    import importlib.machinery, json, sys, types

    # The checkout directory carries a hyphen, so the child registers the same
    # alias the harness uses and reaches the vendored tree through it.
    alias = {alias!r}
    package = types.ModuleType(alias)
    package.__path__ = [{root!r}]
    package.__package__ = alias
    package.__spec__ = importlib.machinery.ModuleSpec(alias, None, is_package=True)
    package.__spec__.submodule_search_locations = [{root!r}]
    sys.modules[alias] = package

    before = list(sys.path)
    result = {{}}

    try:
        from {alias}.modules.loader import ChatterboxLoader
    except BaseException as exc:
        result['loader_error'] = '{{}}: {{}}'.format(type(exc).__name__, exc)
    else:
        result['loader_error'] = None
        result['loader_module'] = ChatterboxLoader.__module__

    try:
        from {alias}.src.chatterbox.tts import ChatterboxTTS
    except BaseException as exc:
        result['import_kind'] = type(exc).__name__
        result['import_detail'] = str(exc)[:300]
        result['missing'] = getattr(exc, 'name', None)
    else:
        result['import_kind'] = 'ok'
        result['class_module'] = ChatterboxTTS.__module__

    result['added'] = [entry for entry in sys.path if entry not in before]
    result['src_on_path'] = {src_dir!r} in sys.path
    result['bare_modules'] = sorted(
        name for name in sys.modules
        if name.split('.')[0] in ('chatterbox', 'chatterbox_flash')
    )

    print('__PROBE__' + json.dumps(result))
    """
)


def run_probe() -> dict:
    script = PROBE.format(alias=PKG_ALIAS, root=ROOT_DIR, src_dir=SRC_DIR)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=900,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("__PROBE__"):
            return json.loads(line[len("__PROBE__"):])
    pytest.fail(
        "the vendor-namespace probe produced no result\n"
        f"--- stdout ---\n{completed.stdout[-2000:]}\n"
        f"--- stderr ---\n{completed.stderr[-2000:]}"
    )


@pytest.fixture(scope="module")
def probe() -> dict:
    return run_probe()


#: Third-party modules the vendored tree legitimately needs. If one of these is
#: absent the probe skips -- by name, so "why did this not run?" is answerable.
KNOWN_EXTERNAL_REQUIREMENTS = (
    "librosa",
    "torchaudio",
    "transformers",
    "s3tokenizer",
    "conformer",
    "huggingface_hub",
    "safetensors",
    "einops",
    "pyloudnorm",
)


class TestCleanInterpreter:
    def test_the_loader_imports_through_the_node_package(self, probe: dict) -> None:
        assert probe["loader_error"] is None, (
            f"importing the loader through the node package failed: "
            f"{probe['loader_error']}"
        )
        assert probe["loader_module"] == f"{PKG_ALIAS}.modules.loader"

    def test_src_is_not_added_to_sys_path(self, probe: dict) -> None:
        assert probe["added"] == [], (
            f"importing the node package added {probe['added']!r} to sys.path"
        )
        assert probe["src_on_path"] is False

    def test_no_top_level_vendor_name_is_registered(self, probe: dict) -> None:
        assert probe["bare_modules"] == [], (
            f"a top-level vendor name was registered: {probe['bare_modules']!r}"
        )

    def test_the_vendored_class_belongs_to_our_package(self, probe: dict) -> None:
        if probe["import_kind"] != "ok":
            pytest.skip(
                "the vendored chatterbox does not import in this environment: "
                f"{probe['import_kind']} on {probe.get('missing')!r} "
                f"({probe.get('import_detail')}); install "
                f"{', '.join(KNOWN_EXTERNAL_REQUIREMENTS)} to run this check"
            )
        assert probe["class_module"].startswith(f"{PKG_ALIAS}.src.chatterbox"), (
            f"the vendored class is bound to {probe['class_module']!r}; it must "
            "belong to this node's package"
        )

    def test_a_missing_dependency_is_never_reported_as_success(self, probe: dict) -> None:
        """A vacuous pass is the failure mode this guards against."""
        if probe["import_kind"] != "ok":
            assert probe["import_kind"] in {"ModuleNotFoundError", "ImportError"}, (
                f"importing the vendored tree failed on {probe['import_kind']}: "
                f"{probe.get('import_detail')}"
            )
            assert probe.get("missing"), "the failure names no missing module"
