"""Tests for the conftest harness itself (plan S0.3 / S0.4 / S0.5).

These are the tests every other batch leans on: if the aliasing or the vendor
mocking silently stops working, the failures show up here first, as a clear
assertion, rather than inside an unrelated behavioural test.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# --------------------------------------------------------------------------
# Locate the loaded conftest module without assuming how pytest named it.
# --------------------------------------------------------------------------


def _conftest_module() -> ModuleType:
    """Return the imported root ``conftest`` module."""
    module = sys.modules.get("conftest")
    if module is None:  # pragma: no cover - pytest always imports it under this name
        raise RuntimeError("root conftest.py is not in sys.modules")
    return module


cf = _conftest_module()


@pytest.fixture
def clean_root_resolution(monkeypatch):
    """Unmemoise the ComfyUI root so each test controls its own environment."""
    cf.reset_comfyui_root_cache()
    monkeypatch.delenv("COMFYUI_ROOT", raising=False)
    yield
    cf.reset_comfyui_root_cache()


def _make_comfyui_stub(base, name="FakeComfyUI"):
    """Create a directory that satisfies the ComfyUI root marker test."""
    root = base / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "nodes.py").write_text("# comfy marker\n", encoding="utf-8")
    (root / "execution.py").write_text("# comfy marker\n", encoding="utf-8")
    return root


# ==========================================================================
# S0.3 -- ComfyUI root resolution
# ==========================================================================


class TestComfyRootResolution:
    """``_resolve_comfyui_root`` -- env wins, then scan, then fail by name."""

    def test_env_var_wins_verbatim(self, clean_root_resolution, monkeypatch, tmp_path):
        """An explicit COMFYUI_ROOT is returned unchanged, even if bogus."""
        bogus = str(tmp_path / "does-not-exist")
        monkeypatch.setenv("COMFYUI_ROOT", bogus)
        assert cf._resolve_comfyui_root() == bogus

    def test_two_calls_agree(self, clean_root_resolution, tmp_path):
        """The result is memoised: repeated calls return the same object."""
        _make_comfyui_stub(tmp_path)
        monkey_repo = tmp_path / "ComfyUI-Chatterbox"
        monkey_repo.mkdir()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cf, "ROOT_DIR", str(monkey_repo))
            first = cf._resolve_comfyui_root()
            second = cf._resolve_comfyui_root()
        assert first == second
        assert first is second  # cached, not merely equal

    def test_sibling_scan_finds_comfyui(self, clean_root_resolution, tmp_path):
        """With no env var, a sibling checkout carrying both markers is found."""
        expected = _make_comfyui_stub(tmp_path)
        repo = tmp_path / "ComfyUI-Chatterbox"
        repo.mkdir()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cf, "ROOT_DIR", str(repo))
            found = cf._resolve_comfyui_root()
        assert found == str(expected)

    def test_nested_comfyui_directory_found(self, clean_root_resolution, tmp_path):
        """``<sibling>/ComfyUI/nodes.py`` is also accepted (the common layout)."""
        holder = tmp_path / "ComfyUI_dev"
        holder.mkdir()
        nested = _make_comfyui_stub(holder, "ComfyUI")
        repo = tmp_path / "ComfyUI-Chatterbox"
        repo.mkdir()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cf, "ROOT_DIR", str(repo))
            found = cf._resolve_comfyui_root()
        assert found == str(nested)

    def test_missing_root_raises_naming_the_env_var(self, clean_root_resolution, tmp_path):
        """No checkout anywhere -> RuntimeError that names COMFYUI_ROOT."""
        repo = tmp_path / "lonely" / "ComfyUI-Chatterbox"
        repo.mkdir(parents=True)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cf, "ROOT_DIR", str(repo))
            with pytest.raises(RuntimeError) as excinfo:
                cf._resolve_comfyui_root()
        message = str(excinfo.value)
        assert "COMFYUI_ROOT" in message
        assert "nodes.py" in message and "execution.py" in message

    def test_directory_without_both_markers_is_rejected(self, clean_root_resolution, tmp_path):
        """A half-checkout (only nodes.py) must not be accepted."""
        decoy = tmp_path / "Decoy"
        decoy.mkdir()
        (decoy / "nodes.py").write_text("# not comfy\n", encoding="utf-8")
        repo = tmp_path / "ComfyUI-Chatterbox"
        repo.mkdir()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cf, "ROOT_DIR", str(repo))
            with pytest.raises(RuntimeError):
                cf._resolve_comfyui_root()

    def test_no_hardcoded_absolute_path_in_conftest(self):
        """No machine-specific literal may live in conftest.py (plan S0.3).

        A stale default silently puts a wrong tree on sys.path on any other
        machine; the repository must be portable.
        """
        source = open(cf.__file__, "r", encoding="utf-8").read()
        # Strip comments so prose about the rule does not trip the rule.
        code = "\n".join(
            line.split("#", 1)[0] for line in source.splitlines()
        )
        assert not re.search(r"[Cc]:[\\/]", code), (
            "conftest.py contains an absolute drive-letter literal"
        )

    def test_comfy_and_folder_paths_are_real(self):
        """Only the web layer is mocked; ComfyUI itself loads for real."""
        assert isinstance(sys.modules["comfy"], ModuleType)
        assert not isinstance(sys.modules["comfy"], MagicMock)
        assert not isinstance(sys.modules["folder_paths"], MagicMock)
        assert "comfy" in cf.REAL_MODULES
        assert "folder_paths" in cf.REAL_MODULES

    def test_web_layer_is_either_real_or_mocked(self):
        """``server``/``aiohttp`` must resolve to something usable either way."""
        for name in ("server", "aiohttp"):
            module = sys.modules.get(name)
            assert module is not None or name in cf.MOCKED_MODULES
            if name in cf.MOCKED_MODULES:
                assert isinstance(module, MagicMock)


# ==========================================================================
# S0.4 -- hyphenated package alias
# ==========================================================================


class TestPackageAlias:
    """``ComfyUI_Chatterbox`` and its submodules must be importable."""

    def test_alias_is_registered(self):
        module = sys.modules[cf.PKG_ALIAS]
        assert isinstance(module, ModuleType)
        assert module.__spec__ is not None
        assert module.__spec__.name == cf.PKG_ALIAS

    def test_alias_maps_to_repo_root_init(self):
        """The alias is built from this checkout's ``__init__.py``.

        A degraded alias (S3.1 deleted the module it imports) has no origin --
        it is a namespace stand-in -- so fall back to the recorded reason plus
        the package path it was built for.
        """
        module = sys.modules[cf.PKG_ALIAS]
        origin = module.__spec__.origin
        if origin is None:
            assert list(module.__path__) == [cf.ROOT_DIR]
            assert getattr(module, "_alias_import_error", None)
            return
        assert os.path.realpath(origin) == os.path.realpath(
            os.path.join(cf.ROOT_DIR, "__init__.py")
        )

    def test_root_alias_state_is_explainable(self):
        """Either it loaded, or it degraded with the reason recorded.

        S3.1 deleted ``nodes.py``, which this repository's ``__init__.py``
        still imports; until S6.5 rewrites the entry point the degradation is
        expected, and it must name the missing module rather than fail
        silently.
        """
        module = sys.modules[cf.PKG_ALIAS]
        error = getattr(module, "_alias_import_error", None)
        if error is not None:
            assert "nodes" in error, f"repo-root alias degraded for another reason: {error}"

    def test_existing_submodules_materialise_with_specs(self):
        """``nodes`` disappeared with the V1 teardown (S3.1) and returns as a
        package in S6.1; until then only the submodules that exist on disk are
        asserted. ``conftest.ALIAS_SUBMODULES`` still lists all three, so
        :func:`ensure_alias_submodules` cannot be used while ``nodes`` is gone.
        """
        present = [
            name
            for name in cf.ALIAS_SUBMODULES
            if os.path.isfile(os.path.join(cf.ROOT_DIR, f"{name}.py"))
            or os.path.isdir(os.path.join(cf.ROOT_DIR, name))
        ]
        assert "modules" in present and "src" in present
        for name in present:
            module = importlib.import_module(f"{cf.PKG_ALIAS}.{name}")
            assert module.__spec__ is not None, name
            assert module.__spec__.name == f"{cf.PKG_ALIAS}.{name}"

    def test_directory_submodules_are_namespace_packages(self):
        """``modules``/``src`` are directories: namespace-style, ``__path__`` set."""
        for name in ("modules", "src"):
            module = importlib.import_module(f"{cf.PKG_ALIAS}.{name}")
            expected = os.path.join(cf.ROOT_DIR, name)
            assert list(module.__path__) == [expected]
            assert getattr(module, "__file__", None) is None, (
                f"{name} must be a namespace package"
            )

    def test_root_alias_exposes_no_v1_node_mappings(self):
        """S3.1 deleted the V1 nodes, so there is no V1 surface left to expose.

        S6.5 replaces this with the V3 ``comfy_entrypoint``; until then the
        alias either degrades (no mappings at all) or imports without them.
        """
        root = sys.modules[cf.PKG_ALIAS]
        assert not hasattr(root, "NODE_CLASS_MAPPINGS"), (
            "the V1 node mappings survived the teardown of nodes.py"
        )

    def test_repo_init_does_not_leak_sys_path_entries(self):
        """``__init__.py`` no longer touches sys.path; conftest keeps it clean.

        The vendored tree is reached through the node package, so the entry
        point has no reason to write to ``sys.path``. The harness still
        snapshots and restores it around the alias registration, so a path
        entry added by any import cannot leak into the suite.

        Phase 2 (S2.4) asserts the mutation is gone from the source. That
        assertion is only meaningful if the harness starts from a clean slate.
        """
        src_dir = os.path.join(cf.ROOT_DIR, "src")
        assert src_dir not in sys.path

    def test_pytest_dunder_init_alias_is_package_aware(self):
        """pytest imports the root ``__init__.py`` as the bare name ``__init__``.

        The pre-registered stand-in must be package-aware, otherwise pytest's
        package collector blows up before any test runs.
        """
        module = sys.modules["__init__"]
        assert module.__package__ == cf.PKG_ALIAS
        assert list(module.__path__) == [cf.ROOT_DIR]
        assert os.path.realpath(module.__file__) == os.path.realpath(
            os.path.join(cf.ROOT_DIR, "__init__.py")
        )

    def test_submodule_import_succeeds(self):
        """A submodule import must not raise (a broken one degrades, not fail)."""
        module = importlib.import_module(f"{cf.PKG_ALIAS}.modules")
        assert module is sys.modules[f"{cf.PKG_ALIAS}.modules"]

    def test_broken_module_raises_naming_the_file(self, tmp_path):
        """Strict loading of a deliberately broken package names its file."""
        broken_dir = tmp_path / "broken_pkg"
        broken_dir.mkdir()
        broken_init = broken_dir / "__init__.py"
        broken_init.write_text("raise ValueError('boom')\n", encoding="utf-8")

        alias = "ConftestBrokenAlias"
        sys.modules.pop(alias, None)
        try:
            with pytest.raises(RuntimeError) as excinfo:
                cf.load_package_alias(alias, str(broken_init))
            message = str(excinfo.value)
            assert str(broken_init) in message
            assert "ValueError" in message and "boom" in message
            # A failed strict load must not leave a half-built module behind.
            assert alias not in sys.modules
        finally:
            sys.modules.pop(alias, None)

    def test_broken_module_degrades_with_a_diagnosable_error(self, tmp_path):
        """The repo-root alias must never take collection down with it."""
        broken_dir = tmp_path / "degraded_pkg"
        broken_dir.mkdir()
        broken_init = broken_dir / "__init__.py"
        broken_init.write_text("import a_module_that_does_not_exist\n", encoding="utf-8")

        alias = "ConftestDegradedAlias"
        sys.modules.pop(alias, None)
        try:
            module = cf.register_package_alias(alias, str(broken_init))
            assert sys.modules[alias] is module
            assert module.__spec__ is not None
            assert list(module.__path__) == [str(broken_dir)]
            error = getattr(module, "_alias_import_error", None)
            assert error is not None, "a degraded alias must say why"
            assert str(broken_init) in error
            assert "ModuleNotFoundError" in error
        finally:
            sys.modules.pop(alias, None)


# ==========================================================================
# S0.5 -- vendored chatterbox tree is mocked
# ==========================================================================


class TestVendorMocking:
    """Collection must never load perth/diffusers/s3tokenizer/conformer."""

    def test_core_vendor_modules_are_mocks(self):
        for name in (
            "chatterbox",
            "chatterbox.tts",
            "chatterbox.vc",
            "chatterbox.mtl_tts",
            "chatterbox.tts_turbo",
            "chatterbox.models",
        ):
            assert isinstance(sys.modules[f"{cf.PKG_ALIAS}.src.{name}"], MagicMock), name

    def test_every_vendored_leaf_is_mocked(self):
        """One dotted name per real file under src/chatterbox/."""
        for name in cf.VENDORED_MODULES:
            assert isinstance(sys.modules[f"{cf.PKG_ALIAS}.src.{name}"], MagicMock), name
            assert name not in sys.modules, (
                f"{name} is registered under its bare top-level name; the "
                "harness must not fabricate the name the shipped code may not use"
            )

    def test_vendored_module_list_matches_the_tree(self):
        """The mocked list is derived from the tree, not hand-maintained."""
        vend_root = os.path.join(cf.ROOT_DIR, "src", "chatterbox")
        on_disk = {
            "chatterbox.tts",
            "chatterbox.vc",
            "chatterbox.models.s3gen.flow",
            "chatterbox.models.t3.t3",
            "chatterbox.models.voice_encoder.voice_encoder",
        }
        assert on_disk <= set(cf.VENDORED_MODULES)
        assert all(name.startswith("chatterbox") for name in cf.VENDORED_MODULES)
        assert os.path.isdir(vend_root)

    def test_import_helper_using_vendor_tree_succeeds(self):
        """The shape the plan cares about: importing the entry point is cheap."""

        class _Helper:
            from ComfyUI_Chatterbox.src.chatterbox.tts import ChatterboxTTS
            from ComfyUI_Chatterbox.src.chatterbox.vc import ChatterboxVC

        assert isinstance(
            sys.modules[f"{cf.PKG_ALIAS}.src.chatterbox.tts"].ChatterboxTTS, MagicMock
        )
        assert isinstance(
            sys.modules[f"{cf.PKG_ALIAS}.src.chatterbox.vc"].ChatterboxVC, MagicMock
        )

    def test_diffusers_never_loaded(self):
        """diffusers is installed but broken in this environment."""
        assert "diffusers" not in sys.modules

    def test_perth_is_never_the_real_library(self):
        """S2.4 removed the stub; nothing may put a ``perth`` in ``sys.modules``.

        This used to allow a flagged dummy that the repo entry point injected.
        That injection is gone -- the vendored entry points import ``perth``
        themselves, guarded -- so the unconditional form applies: anything here
        is a regression, not a pass.
        """
        assert "perth" not in sys.modules, (
            "a 'perth' module is registered; the suite must not depend on it and "
            "the node package must never fabricate one"
        )

    def test_heaviest_optional_deps_not_loaded_by_vendor_aliases(self):
        """s3tokenizer/conformer are mocked away under the vendored tree."""
        name = "chatterbox.models.s3tokenizer.s3tokenizer"
        assert name in cf.VENDORED_MODULES
        assert isinstance(sys.modules[f"{cf.PKG_ALIAS}.src.{name}"], MagicMock)