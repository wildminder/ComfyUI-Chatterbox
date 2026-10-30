"""ComfyUI-Chatterbox entry point.

ComfyUI imports this module for one reason: to find ``comfy_entrypoint``, the
async factory that hands back the :class:`~comfy_api.latest.ComfyExtension`
holding this package's nodes. That is all a V3 custom node exposes -- the V1
``NODE_CLASS_MAPPINGS`` / ``NODE_DISPLAY_NAME_MAPPINGS`` pair is gone. The one
other name it does expose is ``WEB_DIRECTORY``: ``js/`` ships ``extension.js``,
a plain-ESM extension that turns a failed Chatterbox node execution into a
web-UI toast, and ComfyUI serves that directory at
``/extensions/ComfyUI-Chatterbox/`` for its loader to import at boot.

What happens here, in order, each step degrading rather than raising:

1. **Pytest guard.** pytest imports every ``__init__.py`` it meets during
   collection -- including this one -- but imports it parentless, so the
   relative imports below would die with "attempted relative import with no
   known parent package" before a single test runs. Under pytest the module
   therefore exports nothing and returns.
2. **Folder registration.** ``tts`` and ``chatterbox`` are registered with
   ComfyUI's ``folder_paths`` so voice packs are discoverable.
3. **Model discovery.** The pack directory is scanned once so the dropdowns are
   populated on the first request instead of the first click.
4. **Routes.** The ``/chatterbox/*`` API is registered when a ``PromptServer``
   is available. A headless or test process has none, and that is not an error.
"""

import sys

if "pytest" in sys.modules:
    # Imported parentless by pytest's collection pass; see the module docstring.
    __all__: list[str] = []

else:
    import logging
    import os

    import folder_paths

    from .modules.api_routes import register_routes
    from .modules.folder_registration import register_chatterbox_folders
    from .modules.model_discovery import get_chatterbox_model_pack_names

    # Declared with the bootstrap, never above the pytest guard: the guard is
    # what makes this module importable during collection, and a web directory
    # bound there would be a claim the test process cannot back up. ComfyUI
    # resolves it against this module's own directory and serves whatever it
    # finds at /extensions/ComfyUI-Chatterbox/.
    WEB_DIRECTORY = "./js"

    # 1. Where voice packs are looked for: `models/tts/chatterbox`.
    register_chatterbox_folders(folder_paths)

    # 2. One scan now, so the pack dropdown is populated on the frontend's first
    #    `/object_info` request instead of its first click.
    logging.debug(
        "[Chatterbox TTS] Found %d Chatterbox voice pack(s).",
        len(get_chatterbox_model_pack_names()),
    )

    # 3. The `/chatterbox/*` API. Only a running server has a `PromptServer`;
    #    a headless or partially-booted process simply gets no routes.
    try:
        from server import PromptServer

        register_routes(
            PromptServer.instance.routes, folder_paths, os.path.dirname(__file__)
        )
    except Exception as exc:
        logging.debug(
            "[Chatterbox TTS] The /chatterbox API routes were not registered "
            "on this process: %s",
            exc,
        )

    from .chatterbox_nodes import comfy_entrypoint  # noqa: E402 (after bootstrap)

    __all__ = ["comfy_entrypoint"]