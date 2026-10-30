"""Chatterbox HTTP API routes for the ComfyUI server.

Ported from ``ComfyUI-VoxCPM/modules/api_routes.py`` (plan S4.4): the same
:func:`register_routes` shape, the same ``folder_paths_module`` injection, and
the same directory-scanning flow -- with the ``voxcpm`` prefix renamed to
``chatterbox`` and the endpoint set reduced to the seven the Chatterbox
frontend actually calls:

======================================  ======  ==============================
Route                                   Method  Handler
======================================  ======  ==============================
``/chatterbox/models``                  GET     ``chatterbox_models_handler``
``/chatterbox/cancel_download``         POST    ``chatterbox_cancel_download_handler``
``/chatterbox/download_status``         GET     ``chatterbox_download_status_handler``
``/chatterbox/tts_search_paths``        GET     ``chatterbox_tts_search_paths_handler``
``/chatterbox/validate_directory``      POST    ``chatterbox_validate_directory_handler``
``/chatterbox/register_model_path``     POST    ``chatterbox_register_model_path_handler``
``/chatterbox/settings``                DELETE  ``chatterbox_clear_settings_handler``
======================================  ======  ==============================

Deliberate differences from the borrowed file:

1. **No lazy-JS endpoint.** The borrowed ``GET /voxcpm/heavy_extension/{filename}``
   served ``js_lazy/voxcpmHeavy.js``; Chatterbox has no lazy bundle, so the
   route is gone. ``current_dir`` stays in the signature -- batch B10 calls
   positionally -- and is reported back as ``extension_dir`` in the models
   payload so a future asset route has the value at hand.
2. **The scan endpoints are POSTs with a JSON body.** The borrowed
   ``validate_directory`` was a GET reading ``?path=``, which cannot carry an
   absolute Windows path comfortably through a query string. Both it and
   ``register_model_path`` now read ``{"path": ...}``.
3. **``/chatterbox/models`` merges the two borrowed model endpoints.**
   VoxCPM had ``POST /voxcpm/models`` (scan a directory) *and*
   ``GET /voxcpm/model_info`` (the registry). Chatterbox has one GET that
   answers both: the registry by default, plus a ``scan`` block when a ``path``
   query parameter is supplied.
4. **Settings are embedded in the models response.** This is the compensation
   the plan requires in place of the ``register_config_on_connect`` no-op
   (§3.1.2): a client that connects after the scheduled config broadcast still
   receives ``settings`` and the feature flags over plain HTTP.
5. **A directory missing weights is rejected with an actionable message.**
   The borrowed ``validate_directory`` accepted any registered directory,
   including an empty one, and said nothing about what was missing.

Imports
-------
``aiohttp`` is imported **lazily** through :func:`_resolve_web`, and
``server``/``PromptServer`` is never imported here at all: the download
progress events are pushed by :mod:`modules.downloader`. Nothing in this
module imports the ComfyUI API surface, ``comfy.*`` or ``folder_paths`` at
module scope, so it imports and unit-tests without a running ComfyUI.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from . import model_discovery, model_info
from .downloader import PROGRESS_CHANNEL as DOWNLOAD_EVENT_CHANNEL
from .folder_registration import CHATTERBOX_SUBDIR, TTS_FOLDER_KEY
from .model_info import (
    CHATTERBOX_TOKENIZER_FILE,
    CHATTERBOX_WEIGHT_FILES,
    MODEL_CONFIGS,
    canonical_pack_name,
    describe_model_dir,
    is_chatterbox_model_dir,
    pack_spellings,
    scan_chatterbox_models,
)


__all__ = [
    "API_PREFIX",
    "DOWNLOAD_EVENT_CHANNEL",
    "DOWNLOAD_EVENT_TYPES",
    "prime_registry",
    "register_routes",
]

# Every route hangs off this prefix. Kept in one constant so the frontend's
# base URL and the route table cannot drift apart.
API_PREFIX = "/chatterbox"

# The sub-directory a Chatterbox pack lives in, under every registered TTS
# root. Owned by :mod:`modules.folder_registration`, which registers it with
# ComfyUI -- imported rather than re-declared so the two agree.
CHATTERBOX_SUBDIR_NAME = CHATTERBOX_SUBDIR

# Wire vocabulary of the download progress channel. ``cancelled`` is mandatory
# and distinct from ``error`` (plan §3.1.4): the frontend must be able to tell
# "you stopped this" from "try again", and ``modules.downloader`` emits these
# four values on :data:`DOWNLOAD_EVENT_CHANNEL`.
DOWNLOAD_EVENT_TYPES: Tuple[str, ...] = ("progress", "complete", "error", "cancelled")

# How many weight files to name in an actionable "you are missing files"
# message. Three is enough to show the shape of a pack without a wall of text.
_WEIGHT_HINT_COUNT = 3


# ── Lazily resolved web layer ────────────────────────────────────────

def _resolve_web() -> Optional[Any]:
    """Return ``aiohttp.web``, or ``None`` when aiohttp is unavailable.

    Lazy on purpose: the module must import in a bare interpreter (and under
    the test harness, which may have replaced ``aiohttp`` with a mock), and
    resolving it per response keeps the injection point obvious.
    """
    try:
        from aiohttp import web
    except ImportError as exc:  # pragma: no cover - aiohttp ships with ComfyUI
        logging.debug(
            "[Chatterbox TTS] aiohttp is not importable; cannot build an "
            "HTTP response: %s",
            exc,
        )
        return None
    return web


def _json_response(payload: Dict[str, Any], status: int = 200) -> Any:
    """Build a JSON HTTP response through the lazily resolved ``web`` module."""
    web = _resolve_web()
    if web is None:  # pragma: no cover - aiohttp ships with ComfyUI
        raise RuntimeError("aiohttp is not available; cannot build an HTTP response")
    return web.json_response(payload, status=status)


async def _json_body(request: Any) -> Dict[str, Any]:
    """Parse a JSON object body.

    Raises ``ValueError`` for a malformed body *and* for a well-formed one that
    is not an object (``json.JSONDecodeError`` subclasses ``ValueError``), so
    the caller needs a single ``except ValueError`` for both 400s.
    """
    data = await request.json()
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


# ── folder_paths access (always through the injected module) ─────────

def _tts_roots(folder_paths_module: Any) -> List[str]:
    """Every registered TTS root, or ``[]`` when the key is unregistered.

    Asking ComfyUI for a folder key that nobody registered raises; that is a
    normal state on a fresh install and must not turn a frontend poll into a
    500.
    """
    getter = getattr(folder_paths_module, "get_folder_paths", None)
    if not callable(getter):
        logging.debug(
            "[Chatterbox TTS] The injected folder_paths object has no "
            "get_folder_paths()"
        )
        return []
    try:
        roots = getter(TTS_FOLDER_KEY)
    except Exception as exc:  # noqa: BLE001 -- an unregistered folder is normal
        logging.debug(
            "[Chatterbox TTS] No '%s' folder registered with ComfyUI: %s",
            TTS_FOLDER_KEY,
            exc,
        )
        return []
    return [os.fspath(root) for root in (roots or [])]


def _default_tts_root(folder_paths_module: Any) -> str:
    """``<models>/tts`` -- the root a stock ComfyUI install uses."""
    models_dir = getattr(folder_paths_module, "models_dir", "") or ""
    if not models_dir:
        return ""
    return os.path.join(os.fspath(models_dir), TTS_FOLDER_KEY)


def _registered_parent(path: str, roots: List[str]) -> Tuple[bool, Optional[str]]:
    """Whether ``path`` is inside one of ``roots``; the matching root if so."""
    normalized = os.path.normpath(path)
    for root in roots:
        norm_root = os.path.normpath(root)
        if normalized == norm_root or normalized.startswith(norm_root + os.sep):
            return True, root
    return False, None


def _pack_subdir(path: str) -> Optional[str]:
    """``<path>/chatterbox`` when it exists, else ``None``."""
    candidate = os.path.join(path, CHATTERBOX_SUBDIR_NAME)
    return candidate if os.path.isdir(candidate) else None


def _discover_packs(path: str) -> List[dict]:
    """Every Chatterbox pack reachable from ``path``.

    Delegates to :func:`modules.model_info.scan_custom_model_path`, so a path
    that *is* a pack comes back as a one-element list, and a path that holds
    packs directly is scanned. When that finds nothing and a ``chatterbox/``
    sub-directory does exist, the sub-directory is scanned instead -- that is
    the layout a registered TTS root uses, and the two-deep nesting the direct
    scan does not see.
    """
    models = model_info.scan_custom_model_path(path)
    if models:
        return list(models)
    subdir = _pack_subdir(path)
    if subdir is not None:
        return scan_chatterbox_models(subdir)
    return []


def _pack_files_present(model_dir: str, required_files: Any) -> bool:
    """True when ``model_dir`` carries every file the registry declares.

    Falls back to :func:`is_chatterbox_model_dir` for an entry that declares no
    file list, so a community pack is still recognised.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return False
    files = list(required_files or ())
    if files:
        return all(os.path.isfile(os.path.join(model_dir, name)) for name in files)
    return is_chatterbox_model_dir(model_dir)


def _is_official_pack_present(
    model_name: str, config: Dict[str, Any], folder_paths_module: Any
) -> bool:
    """Whether the official pack ``model_name`` is already on disk.

    The directory an entry downloads into is its ``folder``, which for the
    Turbo-family Nano member is the shared ``chatterbox-turbo`` folder rather
    than the entry's own name.

    An alias is resolved to its canonical name first, so the old spelling of a
    renamed pack asks about the very directory the current name asks about --
    and a config that names no folder of its own inherits the canonical
    entry's.

    Every spelling of that folder is probed, not just the current one: an
    install that predates a rename still has its weights in a folder named
    after the old id, and reporting that pack as "not downloaded" would offer
    the user a download they have already paid for. The canonical spelling is
    tried first, so a machine holding both folders reports the current name's.
    """
    canonical = canonical_pack_name(model_name)
    folder = str(config.get("folder") or "")
    if not folder:
        official = MODEL_CONFIGS.get(canonical, {})
        folder = str(official.get("folder") or canonical)
    folders = sorted(pack_spellings(folder), key=lambda name: name != folder)
    for root in _tts_roots(folder_paths_module):
        for candidate in folders:
            model_dir = os.path.join(root, CHATTERBOX_SUBDIR_NAME, candidate)
            if _pack_files_present(model_dir, config.get("files")):
                return True
    return False


def _describe_missing_pack_files(path: str) -> str:
    """An actionable message naming what ``path`` lacks to be a pack.

    The point is that the user can act on it: naming the tokenizer and the
    weights is the difference between "invalid directory" and "this folder is
    not a Chatterbox model -- it has no ``tokenizer.json`` and no
    ``.safetensors``".
    """
    try:
        entries = set(os.listdir(path))
    except OSError as exc:
        return f"Could not read the directory: {exc}"

    missing_tokenizer = CHATTERBOX_TOKENIZER_FILE not in entries
    missing_weights = not any(entry.endswith(".safetensors") for entry in entries)
    if not (missing_tokenizer or missing_weights):
        return ""

    weight_hint = ", ".join(CHATTERBOX_WEIGHT_FILES[:_WEIGHT_HINT_COUNT])
    prefix = f"Not a Chatterbox model directory: missing"
    if missing_tokenizer and missing_weights:
        return f"{prefix} {CHATTERBOX_TOKENIZER_FILE} and safetensors weights (e.g. {weight_hint})"
    if missing_tokenizer:
        return f"{prefix} {CHATTERBOX_TOKENIZER_FILE}"
    return f"{prefix} safetensors weights (e.g. {weight_hint})"


# ── Registry priming ─────────────────────────────────────────────────

def prime_registry() -> None:
    """Refresh the feature flags and mirror the on-disk packs into the registry.

    Called once from :func:`register_routes` so the very first
    ``GET /chatterbox/models`` already reports the packs the user has on disk,
    rather than a registry that only fills in on the second poll. Neither call
    may take the node down: a failure here is logged and the route still
    serves the static entries.
    """
    model_info.refresh_feature_flags()
    try:
        model_info.refresh_local_models()
    except Exception as exc:  # noqa: BLE001 -- a bad models root is not fatal
        logging.warning(
            "[Chatterbox TTS] Could not scan the local Chatterbox model packs: %s",
            exc,
        )


# ── Route registration ───────────────────────────────────────────────

def register_routes(routes: Any, folder_paths_module: Any, current_dir: str) -> None:
    """Register the seven Chatterbox HTTP routes on the ComfyUI server.

    Args:
        routes: ``PromptServer.instance.routes`` -- the aiohttp route
            decorator target.
        folder_paths_module: the ``folder_paths`` module (or a stand-in
            exposing ``get_folder_paths``, ``add_model_folder_path``,
            ``models_dir`` and ``get_user_directory``). Injected so the
            handlers are unit-testable without booting ComfyUI.
        current_dir: the package root, reported back as ``extension_dir``.
    """
    prime_registry()

    # ── GET /chatterbox/models ─────────────────────────────────────

    @routes.get(f"{API_PREFIX}/models")
    async def chatterbox_models_handler(request: Any) -> Any:
        """The model registry, the feature flags and the user settings.

        Optionally scans a directory as well: pass ``?path=<absolute path>`` to
        get a ``scan`` block alongside the registry, which is what the
        "browse for a model" dialog needs.

        Returns::

            {
              "models": [{"name", "type", "architecture", "sample_rate",
                          "size_gb", "is_downloaded", "repo_id"?, "path"?}],
              "model_count": int,
              "normalization_available": bool,
              "turbo_available": bool,
              "model_packs": [str],
              "models_root": str,
              "extension_dir": str,
              "settings": {...},
              "scan": {...}   # only when ?path= was given
            }
        """
        try:
            entries = []
            for model_name, model_data in model_info.AVAILABLE_CHATTERBOX_MODELS.items():
                model_type = model_data.get("type", "local")
                info: Dict[str, Any] = {
                    "name": model_name,
                    "type": model_type,
                    "architecture": model_data.get("architecture", "unknown"),
                    "sample_rate": model_data.get("sample_rate", 0),
                    "size_gb": model_data.get("size_gb", 0),
                }
                if model_type == "official":
                    config = MODEL_CONFIGS.get(model_name, {})
                    info["is_downloaded"] = _is_official_pack_present(
                        model_name, config, folder_paths_module
                    )
                else:
                    info["is_downloaded"] = _pack_files_present(
                        model_data.get("path", ""), model_data.get("files")
                    )
                    if model_data.get("path"):
                        info["path"] = model_data["path"]
                if model_data.get("repo_id"):
                    info["repo_id"] = model_data["repo_id"]
                entries.append(info)

            payload: Dict[str, Any] = {
                "models": entries,
                "model_count": len(entries),
                # Read through the module, not a from-import: priming refreshed
                # these globals in place.
                "normalization_available": model_info.TEXT_NORMALIZATION_AVAILABLE,
                "turbo_available": model_info.TURBO_AVAILABLE,
                "model_packs": model_discovery.get_chatterbox_model_pack_names(),
                "models_root": model_discovery.get_models_root(),
                "extension_dir": current_dir,
            }

            scan_path = ""
            query = getattr(request, "query", None)
            if query is not None:
                scan_path = query.get("path", "") or ""
            if scan_path:
                payload["scan"] = _scan_payload(scan_path)

            # The config broadcast is best-effort and can be missed entirely
            # (a client connecting after it, plan §3.1.2), so the same payload
            # rides along in the response body.
            payload["settings"] = _settings_payload()

            return _json_response(payload)
        except Exception as exc:  # noqa: BLE001 -- a poll must not 500 the UI
            logging.error("[Chatterbox TTS] Error handling the model list request: %s", exc)
            return _json_response({"error": str(exc)}, status=500)

    # ── POST /chatterbox/cancel_download ───────────────────────────

    @routes.post(f"{API_PREFIX}/cancel_download")
    async def chatterbox_cancel_download_handler(request: Any) -> Any:
        """Cancel an in-flight model download.

        Body: ``{"model_name": "<pack>"}``.

        Always answers ``200`` with a ``success`` flag -- asking to cancel a
        download that already finished (or was never started) is a no-op the
        frontend resolves silently, not an error::

            {"success": true,  "message": "Download cancelled for '<name>'"}
            {"success": false, "message": "No active download for '<name>'"}
        """
        try:
            from .downloader import get_download_manager

            data = await _json_body(request)
            model_name = data.get("model_name", "")
            if not model_name:
                return _json_response({"error": "model_name is required"}, status=400)

            cancelled = get_download_manager().cancel_download(model_name)
            if cancelled:
                return _json_response(
                    {"success": True, "message": f"Download cancelled for '{model_name}'"}
                )
            return _json_response(
                {"success": False, "message": f"No active download for '{model_name}'"}
            )
        except ValueError as exc:
            return _json_response({"error": f"Invalid JSON body: {exc}"}, status=400)
        except Exception as exc:  # noqa: BLE001 -- report, do not crash the server
            logging.error("[Chatterbox TTS] Error handling the cancel-download request: %s", exc)
            return _json_response({"error": str(exc)}, status=500)

    # ── GET /chatterbox/download_status ────────────────────────────

    @routes.get(f"{API_PREFIX}/download_status")
    async def chatterbox_download_status_handler(request: Any) -> Any:
        """Snapshot of every tracked download.

        Returns ``{"downloads": {"<model_name>": {...DownloadState.to_dict...}}}``.
        Empty while nothing is downloading.
        """
        try:
            from .downloader import get_download_manager

            return _json_response({"downloads": get_download_manager().get_all_statuses()})
        except Exception as exc:  # noqa: BLE001 -- a poll must not 500 the UI
            logging.error("[Chatterbox TTS] Error handling the download-status request: %s", exc)
            return _json_response({"error": str(exc)}, status=500)

    # ── GET /chatterbox/tts_search_paths ──────────────────────────

    @routes.get(f"{API_PREFIX}/tts_search_paths")
    async def chatterbox_tts_search_paths_handler(request: Any) -> Any:
        """Every registered TTS root, with the Chatterbox packs inside it.

        Returns ``{"paths": [{"path", "is_default", "chatterbox_subdir",
        "is_direct_model", "direct_model_info", "models"}]}``.
        """
        try:
            roots = _tts_roots(folder_paths_module)
            default_tts = _default_tts_root(folder_paths_module)
            paths_info = []

            for root in roots:
                is_direct_model = is_chatterbox_model_dir(root)
                models: List[dict] = []
                direct_model_info = None
                subdir = _pack_subdir(root)

                if subdir is not None:
                    # The convention: <tts root>/chatterbox/<pack>/
                    models = scan_chatterbox_models(subdir)
                elif is_direct_model:
                    # The registered root *is* a pack.
                    direct_model_info = describe_model_dir(root)
                    models = [direct_model_info]
                else:
                    models = scan_chatterbox_models(root)

                paths_info.append(
                    {
                        "path": root,
                        "is_default": root == default_tts,
                        "chatterbox_subdir": subdir,
                        "is_direct_model": is_direct_model,
                        "direct_model_info": direct_model_info,
                        "models": models,
                    }
                )

            return _json_response({"paths": paths_info})
        except Exception as exc:  # noqa: BLE001 -- a poll must not 500 the UI
            logging.error("[Chatterbox TTS] Error handling the tts-search-paths request: %s", exc)
            return _json_response({"error": str(exc)}, status=500)

    # ── POST /chatterbox/validate_directory ────────────────────────

    @routes.post(f"{API_PREFIX}/validate_directory")
    async def chatterbox_validate_directory_handler(request: Any) -> Any:
        """Is ``path`` usable as a Chatterbox model directory?

        Body: ``{"path": "<absolute path>"}``.

        Returns::

            {
              "valid": bool,
              "is_registered": bool,
              "registered_parent": str | None,
              "exists": bool,
              "is_direct_model": bool,
              "direct_model_info": {...} | None,
              "models": [...],
              "error": str | None
            }

        ``valid`` is stricter than the borrowed version: a registered but empty
        directory is **not** valid, and ``error`` then names what is missing
        (the tokenizer, the weights, or both).
        """
        try:
            data = await _json_body(request)
            path = data.get("path", "")
            if not path:
                return _json_response({"error": "path is required"}, status=400)
            if not os.path.isabs(path):
                return _json_response({"error": "Path must be absolute"}, status=400)

            normalized = os.path.normpath(path)
            is_registered, registered_parent = _registered_parent(
                normalized, _tts_roots(folder_paths_module)
            )
            exists = os.path.isdir(normalized)
            is_direct_model = exists and is_chatterbox_model_dir(normalized)
            direct_model_info = describe_model_dir(normalized) if is_direct_model else None
            models = [] if is_direct_model else (_discover_packs(normalized) if exists else [])

            # A path that *is* a pack is valid wherever it lives -- it is the
            # model, not a parent directory that happens to contain one.
            is_valid = bool(is_direct_model or (is_registered and exists and models))
            error: Optional[str] = None
            if not is_valid:
                if not exists:
                    error = f"Directory does not exist: {path}"
                elif not is_registered:
                    error = (
                        f"Not within a registered '{TTS_FOLDER_KEY}' model directory: "
                        f"{path}"
                    )
                else:
                    error = (
                        _describe_missing_pack_files(normalized)
                        or f"No Chatterbox model found in {path}"
                    )

            return _json_response(
                {
                    "valid": is_valid,
                    "is_registered": is_registered,
                    "registered_parent": registered_parent,
                    "exists": exists,
                    "is_direct_model": is_direct_model,
                    "direct_model_info": direct_model_info,
                    "models": models,
                    "error": error,
                }
            )
        except ValueError as exc:
            return _json_response({"error": f"Invalid JSON body: {exc}"}, status=400)
        except Exception as exc:  # noqa: BLE001 -- report, do not crash the server
            logging.error(
                "[Chatterbox TTS] Error handling the validate-directory request: %s",
                exc,
            )
            return _json_response({"error": str(exc)}, status=500)

    # ── POST /chatterbox/register_model_path ───────────────────────

    @routes.post(f"{API_PREFIX}/register_model_path")
    async def chatterbox_register_model_path_handler(request: Any) -> Any:
        """Add a directory as a TTS model search path and register its packs.

        Body: ``{"path": "<absolute path>"}``. Calls
        ``folder_paths.add_model_folder_path("tts", path)`` so ComfyUI's own
        discovery sees the new root, then mirrors the packs found into
        :data:`modules.model_info.AVAILABLE_CHATTERBOX_MODELS` so they pass
        prompt validation straight away.

        Returns::

            {
              "success": bool,
              "path": str,
              "error": str | None,
              "already_registered": bool,
              "models": [...],
              "model_count": int,
              "warning": str | None
            }

        An empty directory is still registered -- the user may be laying out
        where to download to -- but the response says so in ``warning``.
        """
        try:
            data = await _json_body(request)
            path = data.get("path", "")
            if not path:
                return _json_response({"error": "path is required"}, status=400)
            if not os.path.isabs(path):
                return _json_response({"error": "Path must be absolute"}, status=400)
            if not os.path.isdir(path):
                return _json_response(
                    {"error": f"Directory does not exist: {path}"}, status=400
                )

            already_registered, _parent = _registered_parent(
                os.path.normpath(path), _tts_roots(folder_paths_module)
            )

            if not already_registered:
                adder = getattr(folder_paths_module, "add_model_folder_path", None)
                if not callable(adder):
                    return _json_response(
                        {
                            "error": (
                                "folder_paths does not expose add_model_folder_path(); "
                                "the directory cannot be registered"
                            )
                        },
                        status=500,
                    )
                adder(TTS_FOLDER_KEY, path)
                logging.info(
                    "[Chatterbox TTS] Registered a new '%s' model path: %s",
                    TTS_FOLDER_KEY,
                    path,
                )

            models = _discover_packs(path)
            for model in models:
                name = model.get("name", "")
                if not name:
                    continue
                existing = model_info.AVAILABLE_CHATTERBOX_MODELS.get(name)
                if existing is not None and existing.get("type") == "official":
                    # Never demote an official entry to a local one: its repo_id
                    # and download button live in that metadata.
                    continue
                model_info.register_local_model(name, model["path"])

            # The models root moved, so the cached pack listing is stale.
            model_discovery.invalidate_cache()
            _broadcast_config()

            return _json_response(
                {
                    "success": True,
                    "path": path,
                    "error": None,
                    "already_registered": already_registered,
                    "models": models,
                    "model_count": len(models),
                    "warning": (
                        f"No Chatterbox models found in '{path}'"
                        if not models
                        else None
                    ),
                }
            )
        except ValueError as exc:
            return _json_response({"error": f"Invalid JSON body: {exc}"}, status=400)
        except Exception as exc:  # noqa: BLE001 -- report, do not crash the server
            logging.error(
                "[Chatterbox TTS] Error handling the register-model-path request: %s",
                exc,
            )
            return _json_response({"error": str(exc)}, status=500)

    # ── DELETE /chatterbox/settings ────────────────────────────────

    @routes.delete(f"{API_PREFIX}/settings")
    async def chatterbox_clear_settings_handler(request: Any) -> Any:
        """Remove every ``chatterbox.*`` key from ``comfy.settings.json``.

        Used to reset to defaults and on uninstall. Keys belonging to other
        nodes are preserved.

        Returns ``{"success": bool, "removed": [str], "settings": {...}}``.
        """
        from .config_broadcast import send_config_to_client
        from .settings import SETTINGS_PREFIX, get_settings

        settings_path = _comfy_settings_path(folder_paths_module)
        if settings_path is None:
            return _json_response({"error": "User directory not found"}, status=500)

        removed: List[str] = []
        try:
            if os.path.exists(settings_path):
                with open(settings_path, "r", encoding="utf-8") as handle:
                    all_settings = json.load(handle)
                removed = [
                    key for key in list(all_settings) if key.startswith(SETTINGS_PREFIX)
                ]
                for key in removed:
                    del all_settings[key]
                with open(settings_path, "w", encoding="utf-8") as handle:
                    json.dump(all_settings, handle, indent=2)

            # Reload the singleton so the next broadcast carries the cleared
            # values, then push them: the frontend's own widgets are driven
            # from the broadcast, not from this response.
            chatterbox_settings = get_settings()
            chatterbox_settings.reload()
            send_config_to_client()

            logging.info("[Chatterbox TTS] Cleared the Chatterbox settings: %s", removed)
            return _json_response(
                {"success": True, "removed": removed, "settings": chatterbox_settings.to_dict()}
            )
        except Exception as exc:  # noqa: BLE001 -- report, do not crash the server
            logging.error("[Chatterbox TTS] Error clearing the Chatterbox settings: %s", exc)
            return _json_response({"error": str(exc)}, status=500)

    logging.debug("[Chatterbox TTS] Chatterbox API routes registered under %s", API_PREFIX)


# ── Shared response fragments ────────────────────────────────────────

def _settings_payload() -> Dict[str, Any]:
    """``ChatterboxSettings.to_dict()``, or ``{}`` if it cannot be read."""
    try:
        from .settings import get_settings

        chatterbox_settings = get_settings()
        # Re-read the file: the frontend may have written through
        # ``storeSetting`` since the backend last looked.
        chatterbox_settings.reload()
        return chatterbox_settings.to_dict()
    except Exception as exc:  # noqa: BLE001 -- settings are a convenience here
        logging.debug("[Chatterbox TTS] Could not read the Chatterbox settings: %s", exc)
        return {}


def _scan_payload(path: str) -> Dict[str, Any]:
    """The borrowed directory-scan block, used by ``GET /models?path=``."""
    if not os.path.isdir(path):
        return {
            "path": path,
            "exists": False,
            "models": [],
            "count": 0,
            "is_direct_model": False,
            "direct_model_info": None,
            "error": f"Directory does not exist: {path}",
        }
    is_direct_model = is_chatterbox_model_dir(path)
    direct_model_info = describe_model_dir(path) if is_direct_model else None
    models = _discover_packs(path)
    return {
        "path": path,
        "exists": True,
        "models": models,
        "count": len(models),
        "is_direct_model": is_direct_model,
        "direct_model_info": direct_model_info,
        "error": None if models else _describe_missing_pack_files(path) or None,
    }


def _broadcast_config() -> None:
    """Push the configuration to the frontend, tolerating a missing server."""
    try:
        from .config_broadcast import send_config_to_client

        send_config_to_client()
    except Exception as exc:  # noqa: BLE001 -- never let a toast break a request
        logging.debug("[Chatterbox TTS] Could not broadcast the Chatterbox config: %s", exc)


def _comfy_settings_path(folder_paths_module: Any) -> Optional[str]:
    """Path of ``comfy.settings.json``, via the injected ``folder_paths``.

    Falls back to the resolution :class:`modules.settings.ChatterboxSettings`
    uses itself, so the handler still works when the injected object does not
    expose ``get_user_directory``.
    """
    getter = getattr(folder_paths_module, "get_user_directory", None)
    if callable(getter):
        try:
            user_dir = getter()
        except Exception as exc:  # noqa: BLE001 -- a broken ComfyUI is not fatal
            logging.debug("[Chatterbox TTS] Could not resolve the ComfyUI user directory: %s", exc)
            user_dir = None
        if user_dir:
            return os.path.join(os.fspath(user_dir), "default", "comfy.settings.json")

    try:
        from .settings import ChatterboxSettings

        return ChatterboxSettings._get_comfy_settings_path()
    except Exception as exc:  # noqa: BLE001 -- degrade to "unknown"
        logging.debug("[Chatterbox TTS] Could not resolve the settings file path: %s", exc)
        return None