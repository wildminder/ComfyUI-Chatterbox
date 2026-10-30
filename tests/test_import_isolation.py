"""S2.4 -- importing the node package must not mutate global import state.

Before this change, importing the ComfyUI node package:

* prepended its own ``src/`` to ``sys.path``, shadowing anything the user had
  installed; and
* built a fake ``perth`` module and injected it into ``sys.modules``, so a real
  installation of the watermarking library was invisible to the rest of the
  process.

Both are gone. ``sys.path`` is left exactly as it was found: the vendored tree
is reached through the node's own package, so nothing is added to it.
``perth`` is never fabricated; ``diffusers`` -- installed but unimportable in
this environment -- is never pulled in.

The in-process assertions run against this suite's own interpreter. The
end-to-end check needs a clean one (the harness mocks the vendor tree), so it
runs in a subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from conftest import ROOT_DIR

SRC_DIR = os.path.join(ROOT_DIR, "src")


def read_repo_init() -> str:
    with open(os.path.join(ROOT_DIR, "__init__.py"), "r", encoding="utf-8") as handle:
        return handle.read()


class TestThisProcessWasNotPolluted:
    def test_perth_is_not_in_sys_modules(self) -> None:
        """Not even a stub: patching the source made one unnecessary."""
        assert "perth" not in sys.modules, (
            "a 'perth' module was injected into sys.modules; the entry point must "
            "never fabricate an optional dependency"
        )

    def test_diffusers_is_not_in_sys_modules(self) -> None:
        assert "diffusers" not in sys.modules, (
            "diffusers was imported; it is no longer a dependency of anything "
            "this repository ships"
        )

    def test_no_marked_perth_stub_exists(self) -> None:
        module = sys.modules.get("perth")
        assert module is None or not getattr(module, "_is_mock", False), (
            "the old stub module is still being installed somewhere"
        )


class TestSourceNoLongerMutatesGlobals:
    def test_entry_point_does_not_touch_sys_modules(self) -> None:
        source = read_repo_init()
        assert "sys.modules[" not in source, (
            "the entry point still assigns into sys.modules; nothing at import "
            "time may fake an optional dependency"
        )

    def test_entry_point_does_not_insert_on_sys_path(self) -> None:
        assert "sys.path.insert" not in read_repo_init()

    def test_entry_point_has_no_types_module_stub(self) -> None:
        source = read_repo_init()
        assert "types.ModuleType" not in source
        assert "import types" not in source, (
            "types was only needed to build the perth stub"
        )


# ---------------------------------------------------------------------------
# End to end, in a clean interpreter
# ---------------------------------------------------------------------------

PROBE = textwrap.dedent(
    """
    import importlib.util, json, sys

    spec = importlib.util.spec_from_file_location(
        'comfy_chatterbox_probe',
        {init_path!r},
        submodule_search_locations=[{root!r}],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['comfy_chatterbox_probe'] = module

    before = list(sys.path)
    try:
        spec.loader.exec_module(module)
        error = None
    except BaseException as exc:
        error = '{{}}: {{}}'.format(type(exc).__name__, exc)

    after = list(sys.path)
    result = {{
        'error': error,
        'added': [entry for entry in after if entry not in before],
        'prefix_changed': after[: len(before)] != before,
        'src_count': after.count({src_dir!r}),
        'src_index': after.index({src_dir!r}) if {src_dir!r} in after else None,
        'sys_path_len': len(after),
        'path_len_before': len(before),
        'perth_in_modules': 'perth' in sys.modules,
        'diffusers_in_modules': 'diffusers' in sys.modules,
        'registered': module.__name__ in sys.modules,
    }}
    result['has_mappings'] = hasattr(module, 'NODE_CLASS_MAPPINGS')
    print('__PROBE__' + json.dumps(result))
    """
)


def run_probe() -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            PROBE.format(
                init_path=os.path.join(ROOT_DIR, "__init__.py"),
                root=ROOT_DIR,
                src_dir=SRC_DIR,
            ),
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "COMFY_PROBE": "1"},
    )
    for line in completed.stdout.splitlines():
        if line.startswith("__PROBE__"):
            return json.loads(line[len("__PROBE__"):])
    pytest.fail(
        "the import-isolation probe produced no result\n"
        f"--- stdout ---\n{completed.stdout[-2000:]}\n"
        f"--- stderr ---\n{completed.stderr[-2000:]}"
    )


@pytest.fixture(scope="module")
def probe() -> dict:
    return run_probe()


class TestCleanInterpreterImport:
    def test_sys_path_is_untouched(self, probe: dict) -> None:
        """Importing the node package adds nothing to ``sys.path``."""
        assert probe["added"] == [], (
            f"importing the node package added {probe['added']!r} to sys.path"
        )
        assert probe["prefix_changed"] is False, (
            "an existing sys.path entry was reordered by the node package"
        )

    def test_src_is_not_an_import_root(self, probe: dict) -> None:
        """``src/`` is reached through the node package, not through sys.path."""
        assert probe["src_count"] == 0, (
            f"importing the node package put src/ on sys.path "
            f"{probe['src_count']} time(s)"
        )
        assert probe["src_index"] is None, (
            f"src/ sits at index {probe['src_index']} of sys.path; the vendored "
            "tree must not be an import root"
        )

    def test_perth_is_not_injected(self, probe: dict) -> None:
        assert probe["perth_in_modules"] is False, (
            "importing the node package fabricated a 'perth' module"
        )

    def test_diffusers_is_never_imported(self, probe: dict) -> None:
        assert probe["diffusers_in_modules"] is False

    def test_no_v1_node_surface_survives_the_teardown(self, probe: dict) -> None:
        """S3.1 deleted ``nodes.py``; the entry point that imported it cannot
        import until S6.5 rewrites it.

        Both states are legitimate -- "still importing the removed V1 module" and
        "importing cleanly with no V1 mappings" -- because the entry point is
        mid-rewrite. What is never acceptable is a working ``NODE_CLASS_MAPPINGS``
        pointing at the deleted V1 classes.
        """
        if probe["error"] is not None:
            assert "nodes" in probe["error"], (
                f"importing the node package failed for an unrelated reason: "
                f"{probe['error']}"
            )
        assert probe["has_mappings"] is False