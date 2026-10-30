"""S6.4 -- the extension object ComfyUI actually calls.

A V3 custom node is discovered through exactly two calls: ComfyUI awaits
``comfy_entrypoint()``, then calls ``get_node_list()`` on what it gets back.
Both are asserted here against the real ``comfy_api``, because a node package
that registers cleanly in isolation and fails in the loader is worthless.

``get_node_list`` order is the Add-Node-menu order, so it is treated as part of
the surface rather than as an implementation detail.
"""

from __future__ import annotations

import asyncio
import inspect
import sys

import pytest
from comfy_api.latest import ComfyExtension, io

from ComfyUI_Chatterbox.chatterbox_nodes import (
    ChatterboxDialogNode,
    ChatterboxExtension,
    ChatterboxMTLTTSNode,
    ChatterboxSRTNode,
    ChatterboxTTSNode,
    ChatterboxVCNode,
    comfy_entrypoint,
)

from .conftest_helpers import load_frozen_snapshot

# Menu order, and the node ids it must resolve to. Both frozen: the order is
# what a user sees in the search box, the ids are what every saved graph names.
# The two original ids keep their positions at the head and the tail -- the four
# Phase 8 nodes are additive and were inserted between them, never in front of
# ``ChatterboxTTS`` or after ``ChatterboxVC``.
EXPECTED_NODES = (
    ChatterboxTTSNode,
    ChatterboxMTLTTSNode,
    ChatterboxSRTNode,
    ChatterboxDialogNode,
    ChatterboxVCNode,
)
EXPECTED_NODE_IDS = (
    "ChatterboxTTS",
    "ChatterboxMTLTTS",
    "ChatterboxSRT",
    "ChatterboxDialog",
    "ChatterboxVC",
)

#: The two ids every saved graph in the wild names, and the positions they must
#: keep in the menu order above.
FROZEN_NODE_IDS = ("ChatterboxTTS", "ChatterboxVC")


@pytest.fixture(scope="module")
def extension() -> ChatterboxExtension:
    return asyncio.run(comfy_entrypoint())


def run(coro):
    """Drive one coroutine from a synchronous test."""
    return asyncio.run(coro)


class TestEntrypoint:
    def test_comfy_entrypoint_returns_the_extension(self, extension) -> None:
        assert isinstance(extension, ChatterboxExtension)

    def test_the_extension_is_a_real_ComfyExtension(self, extension) -> None:
        """Not a duck type: ComfyUI dispatches on the base class."""
        assert isinstance(extension, ComfyExtension)
        # The abstract method is genuinely implemented, not merely inherited.
        assert type(extension).get_node_list is not ComfyExtension.get_node_list

    def test_comfy_entrypoint_is_coroutine(self) -> None:
        assert inspect.iscoroutinefunction(comfy_entrypoint)


class TestNodeList:
    def test_node_order_is_stable(self, extension) -> None:
        assert tuple(run(extension.get_node_list())) == EXPECTED_NODES

    def test_node_ids_match_the_frozen_snapshot(self, extension) -> None:
        """The menu order resolves to the pinned ids, and the frozen two are among them.

        The snapshot names exactly the two original nodes, so the comparison is
        a **subset**, not an equality: Phase 8 added four ids, and a check
        demanding an exhaustive match would make every additive release a
        documentation change. What must never happen is the reverse -- one of
        the frozen ids going missing, or moving position in the menu.
        """
        node_ids = tuple(
            node_cls.GET_SCHEMA().node_id for node_cls in run(extension.get_node_list())
        )
        assert node_ids == EXPECTED_NODE_IDS
        assert set(FROZEN_NODE_IDS) <= set(node_ids)
        assert set(FROZEN_NODE_IDS) == set(load_frozen_snapshot()["nodes"])

    def test_the_frozen_ids_keep_their_positions(self, extension) -> None:
        node_ids = tuple(
            node_cls.GET_SCHEMA().node_id for node_cls in run(extension.get_node_list())
        )
        assert node_ids[0] == "ChatterboxTTS"
        assert node_ids[-1] == "ChatterboxVC"

    def test_the_new_ids_are_additive(self, extension) -> None:
        """None of the four Phase 8 nodes reuses a frozen id."""
        node_ids = {
            node_cls.GET_SCHEMA().node_id for node_cls in run(extension.get_node_list())
        }
        assert {
            "ChatterboxMTLTTS",
            "ChatterboxSRT",
            "ChatterboxDialog",
        } <= node_ids
        assert not node_ids & {"ChatterboxSRT2", "ChatterboxDialogV2"}

    def test_every_entry_is_a_ComfyNode_subclass(self, extension) -> None:
        for node_cls in run(extension.get_node_list()):
            assert issubclass(node_cls, io.ComfyNode)
            # Genuinely overridden, not the abstract base raising NotImplemented.
            assert node_cls.define_schema.__func__ is not io.ComfyNode.define_schema.__func__
            assert node_cls.execute.__func__ is not io.ComfyNode.execute.__func__

    def test_the_node_ids_are_unique(self, extension) -> None:
        node_ids = [
            node_cls.GET_SCHEMA().node_id for node_cls in run(extension.get_node_list())
        ]
        assert len(node_ids) == len(set(node_ids))


class TestOnLoad:
    def test_on_load_is_idempotent(self, extension) -> None:
        """ComfyUI may call it more than once; a later call must be a no-op.

        ``on_load`` has no one-time work left to do: the vendored tree is
        imported package-relatively, so nothing is put on ``sys.path``. What
        must not happen is a call growing the global import state, so the
        snapshot is taken *after* the first.
        """
        run(extension.on_load())
        before = list(sys.path)
        run(extension.on_load())
        run(extension.on_load())
        assert sys.path == before, (
            "on_load grew sys.path on a repeat call; the extension is "
            "supposed to be idempotent"
        )

    def test_on_load_returns_none(self, extension) -> None:
        assert run(extension.on_load()) is None


class TestNoV1Surface:
    """The V1 registration pair must not reappear on the extension."""

    def test_the_module_exports_no_v1_mappings(self) -> None:
        import ComfyUI_Chatterbox.chatterbox_nodes as module

        assert not hasattr(module, "NODE_CLASS_MAPPINGS")
        assert not hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS")

    def test_the_extension_declares_no_web_directory(self, extension) -> None:
        assert not hasattr(extension, "WEB_DIRECTORY")
        assert not hasattr(extension, "web_directory")