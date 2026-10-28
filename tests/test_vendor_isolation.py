"""The vendored ``chatterbox`` package must coexist with other nodes' ``chatterbox``.

Python allows one top-level package per name, so a second ComfyUI node shipping
its own ``chatterbox`` (``ComfyUI_ChatterBox_Voice`` does) and importing it at
start-up makes ``from chatterbox.tts import ...`` in this node resolve to *that*
node's older classes. ``modules/vendor_isolation.py`` swaps this node's modules
in for the duration of a call and puts the other node's back afterwards.

These tests load the module straight from its file and build two throw-away
``chatterbox`` packages in ``tmp_path``, so they need neither ComfyUI nor torch.
The last group checks, from source, that the entry points are actually wrapped.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULES = REPO / "modules"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Two ``chatterbox`` packages, a fresh isolation module, and clean globals."""
    foreign_root = tmp_path / "other_node"
    ours_src = tmp_path / "this_node" / "src"
    _write(foreign_root / "chatterbox" / "__init__.py", "WHO = 'foreign'\n")
    _write(foreign_root / "chatterbox" / "tts.py", "WHO = 'foreign'\n")
    _write(ours_src / "chatterbox" / "__init__.py", "WHO = 'ours'\n")
    _write(ours_src / "chatterbox" / "sub.py", "WHO = 'ours-sub'\n")
    _write(
        ours_src / "chatterbox" / "tts.py",
        """
        WHO = 'ours'

        def generate():
            from .sub import WHO as sub  # lazy import at call time
            return ("ours", sub)
        """,
    )

    saved_path = list(sys.path)
    saved_modules = {
        n: m for n, m in sys.modules.items() if n == "chatterbox" or n.startswith("chatterbox.")
    }
    for name in saved_modules:
        monkeypatch.delitem(sys.modules, name)

    spec = importlib.util.spec_from_file_location(
        "vendor_isolation_under_test", MODULES / "vendor_isolation.py"
    )
    isolation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolation)
    monkeypatch.setattr(isolation, "_SRC_DIR", str(ours_src))

    # Start-up as ComfyUI leaves it: the other node's package is imported and its
    # directory is on the path; this node's src/ is appended at the tail.
    sys.path.insert(0, str(foreign_root))
    sys.path.append(str(ours_src))
    import chatterbox.tts  # noqa: F401  (registers the foreign package)

    class World:
        mod = isolation
        src = str(ours_src)
        foreign_pkg = sys.modules["chatterbox"]
        foreign_tts = sys.modules["chatterbox.tts"]
        path_before = list(sys.path)

    try:
        yield World
    finally:
        for name in [n for n in sys.modules if n == "chatterbox" or n.startswith("chatterbox.")]:
            del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


class TestSwap:
    def test_unwrapped_import_gets_the_other_nodes_package(self, world):
        """The bug itself: without the context manager, the wrong package wins."""
        import chatterbox.tts as tts

        assert tts.WHO == "foreign"

    def test_context_serves_this_nodes_package(self, world):
        with world.mod.vendored_chatterbox:
            from chatterbox.tts import generate

            assert sys.modules["chatterbox"].WHO == "ours"
            assert generate() == ("ours", "ours-sub")

    def test_other_nodes_modules_restored_exactly(self, world):
        with world.mod.vendored_chatterbox:
            import chatterbox.tts  # noqa: F401
        assert sys.modules["chatterbox"] is world.foreign_pkg
        assert sys.modules["chatterbox.tts"] is world.foreign_tts
        assert "chatterbox.sub" not in sys.modules
        assert sys.path == world.path_before

    def test_exception_restores_state(self, world):
        with pytest.raises(RuntimeError):
            with world.mod.vendored_chatterbox:
                import chatterbox.tts  # noqa: F401
                raise RuntimeError("boom")
        assert sys.modules["chatterbox"] is world.foreign_pkg
        assert sys.path == world.path_before
        assert world.mod._depth == 0

    def test_lazy_import_needs_the_wrapper(self, world):
        """Why every call into vendored code is wrapped, not just the import."""
        with world.mod.vendored_chatterbox:
            from chatterbox.tts import generate
        with pytest.raises(ModuleNotFoundError):
            generate()  # lazy ``from .sub import`` now runs against the other node

    def test_decorator_form_and_reentrancy(self, world):
        @world.mod.vendored_chatterbox
        def outer():
            import chatterbox

            with world.mod.vendored_chatterbox:  # nested: must not swap twice
                assert sys.modules["chatterbox"] is chatterbox
            assert sys.modules["chatterbox"] is chatterbox
            return chatterbox.WHO

        assert outer() == "ours"
        assert sys.modules["chatterbox"] is world.foreign_pkg

    def test_our_modules_are_reused_not_reexecuted(self, world):
        with world.mod.vendored_chatterbox:
            import chatterbox.tts as first
        with world.mod.vendored_chatterbox:
            import chatterbox.tts as second
        assert first is second

    def test_src_already_first_on_path_is_left_alone(self, world):
        sys.path.insert(0, world.src)
        with world.mod.vendored_chatterbox:
            pass
        assert sys.path[0] == world.src


class TestNoOp:
    def test_nothing_registered(self, world):
        for name in [n for n in sys.modules if n == "chatterbox" or n.startswith("chatterbox.")]:
            del sys.modules[name]
        before = list(sys.path)
        with world.mod.vendored_chatterbox:
            assert "chatterbox" not in sys.modules
        assert sys.path == before

    def test_stub_without_file_is_not_swapped(self, world):
        """Test suites register stubs for ``chatterbox``; leave those alone."""
        stub = MagicMock()
        sys.modules["chatterbox"] = stub
        before = list(sys.path)
        with world.mod.vendored_chatterbox:
            assert sys.modules["chatterbox"] is stub
        assert sys.modules["chatterbox"] is stub
        assert sys.path == before

    def test_own_package_already_registered(self, world):
        for name in [n for n in sys.modules if n == "chatterbox" or n.startswith("chatterbox.")]:
            del sys.modules[name]
        # Other ``chatterbox`` trees may sit earlier on the path (the repo's own
        # conftest adds one); put this test's tree first so the import is ours.
        sys.path.insert(0, world.src)
        importlib.invalidate_caches()
        import chatterbox

        assert chatterbox.WHO == "ours"
        with world.mod.vendored_chatterbox:
            assert sys.modules["chatterbox"] is chatterbox
        assert sys.modules["chatterbox"] is chatterbox


def _decorators(source_path: Path) -> dict[str, list[str]]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    return {
        node.name: [ast.unparse(d) for d in node.decorator_list]
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


class TestEntryPointsAreWrapped:
    """Source-level: a regression here silently brings the name clash back."""

    LOADER = [
        "load_model",  # appears on the handler and on ChatterboxLoader
        "load_multilingual_model",
        "load_turbo_model",
        "load_flash_model",
    ]
    GENERATION = [
        "generate_audio",
        "convert_voice_audio",
        "generate_mtl_audio",
        "generate_turbo_audio",
        "generate_flash_audio",
    ]

    def test_loader_entry_points(self):
        source = MODULES / "loader.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        wrapped = {
            n.name
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and "vendored_chatterbox" in [ast.unparse(d) for d in n.decorator_list]
        }
        assert set(self.LOADER) <= wrapped

    def test_generation_entry_points(self):
        decorators = _decorators(MODULES / "generation.py")
        for name in self.GENERATION:
            assert "vendored_chatterbox" in decorators[name], name
