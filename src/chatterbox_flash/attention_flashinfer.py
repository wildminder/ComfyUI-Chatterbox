"""Backwards-compatibility shim — :class:`FlashInferEngine` now lives in
:mod:`.engines.flashinfer`.

The vendored tree is private to this ComfyUI node, so ``chatterbox_flash``
is no longer a top-level package: this module is reachable only through the
node's own package. How many leading dots that takes depends on where the
importer sits -- one from the package root (``chatterbox_nodes.py``), two from a
module inside the package (``modules/loader.py``):

    from .src.chatterbox_flash.attention_flashinfer import FlashInferEngine
    from ..src.chatterbox_flash.attention_flashinfer import FlashInferEngine

New code should import from the ``engines`` subpackage (or call
:func:`.engines.build_engine`) instead.
"""

from __future__ import annotations

from .engines.flashinfer import (  # noqa: F401
    FlashInferEngine,
    flashinfer_available,
)

__all__ = ["FlashInferEngine", "flashinfer_available"]
