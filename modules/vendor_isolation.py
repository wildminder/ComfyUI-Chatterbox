"""Keep this node's vendored ``chatterbox`` package apart from any other node's.

Why this exists
---------------
Python allows exactly one top-level package called ``chatterbox`` per process.
Other ComfyUI nodes (``ComfyUI_ChatterBox_Voice``, for one) ship their own
package under the same name and import it when ComfyUI starts, so
``sys.modules["chatterbox"]`` already points at *their* code by the time this
node runs ``from chatterbox.tts import ChatterboxTTS``. Python then hands back
the other node's (older) classes, and loading or generation fails with errors
such as ``unexpected keyword argument 'repetition_penalty'`` or a strict
``load_state_dict`` complaining about ``tokenizer._mel_filters``.

What it does
------------
:data:`vendored_chatterbox` is a re-entrant context manager and decorator.
While it is active, ``sys.modules`` holds *this* node's ``chatterbox`` modules
and ``src/`` is first on ``sys.path``; on exit the other node's modules are put
back exactly as they were and this node's are stashed for the next entry.

Wrap every code path that imports from, or runs code inside, the vendored
package -- loading a pack and calling ``generate`` / ``convert`` -- because the
vendored code can import its own submodules lazily at call time.

Only a *real* foreign package triggers the swap: a ``chatterbox`` that is
file-backed and lives outside this node's ``src/``. If nothing is registered,
if this node's own copy is already registered, or if the registered module is a
stub without a ``__file__`` (as test suites install), the context manager does
nothing at all.

Limits
------
* Not safe against another *thread* importing ``chatterbox`` while a wrapped
  call is running. ComfyUI runs one prompt at a time, so this does not arise in
  normal use; wrapped calls are serialised against each other by a lock.
* Only top-level names ``chatterbox`` and ``chatterbox.*`` are swapped.
  ``chatterbox_flash`` and ``hf_progress`` are unique to this node.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from types import ModuleType

_SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
_PACKAGE = "chatterbox"

_lock = threading.RLock()
_depth = 0
_foreign: dict[str, ModuleType] = {}  # other nodes' modules, held while we are active
_ours: dict[str, ModuleType] = {}  # this node's modules, held while we are inactive
_inserted_path = False
_active = False  # True only for an outermost entry that actually swapped modules


def _foreign_package_registered() -> bool:
    """Is a real, file-backed ``chatterbox`` from outside our ``src/`` loaded?"""
    module = sys.modules.get(_PACKAGE)
    file = getattr(module, "__file__", None)
    if not isinstance(file, str):
        return False
    root = os.path.dirname(os.path.dirname(os.path.abspath(file)))
    return not _same_dir(root, _SRC_DIR)


def _is_chatterbox(name: str) -> bool:
    return name == _PACKAGE or name.startswith(_PACKAGE + ".")


def _same_dir(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


class _VendoredChatterbox(contextlib.ContextDecorator):
    """Context manager *and* decorator; see the module docstring."""

    def __enter__(self) -> "_VendoredChatterbox":
        global _depth, _inserted_path, _active
        _lock.acquire()
        try:
            if _depth == 0:
                _active = _foreign_package_registered()
            if _depth == 0 and _active:
                # Park whatever ``chatterbox*`` is currently registered (normally
                # another node's package) and bring back our own, if we have one.
                _foreign.clear()
                for name in [n for n in sys.modules if _is_chatterbox(n)]:
                    _foreign[name] = sys.modules.pop(name)
                sys.modules.update(_ours)
                _ours.clear()
                # Needed for the first import; harmless afterwards.
                if not (sys.path and _same_dir(sys.path[0], _SRC_DIR)):
                    sys.path.insert(0, _SRC_DIR)
                    _inserted_path = True
                else:
                    _inserted_path = False
            _depth += 1
        except BaseException:
            _lock.release()
            raise
        return self

    def __exit__(self, *exc_info: object) -> None:
        global _depth, _inserted_path, _active
        try:
            _depth -= 1
            if _depth == 0 and _active:
                # Stash everything ``chatterbox*`` that is registered now (ours,
                # including submodules imported lazily during the call), then
                # give the other node its modules back untouched.
                _ours.clear()
                for name in [n for n in sys.modules if _is_chatterbox(n)]:
                    _ours[name] = sys.modules.pop(name)
                sys.modules.update(_foreign)
                _foreign.clear()
                if _inserted_path:
                    for i, entry in enumerate(sys.path):
                        if _same_dir(entry, _SRC_DIR):
                            del sys.path[i]
                            break
                    _inserted_path = False
                _active = False
        finally:
            _lock.release()


#: Use as ``with vendored_chatterbox:`` or as ``@vendored_chatterbox``.
vendored_chatterbox = _VendoredChatterbox()
