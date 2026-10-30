"""Cached discovery of the Chatterbox model packs on disk.

The pack combo on the TTS and VC nodes is a dynamic list: ComfyUI rebuilds
every node's schema whenever the frontend refreshes, and each rebuild used to
``listdir`` the models root again. This module exists so that walk happens
once per change and not once per widget render.

The cache key is the models root's ``st_mtime_ns`` (plus its inode, so a
directory replaced by a fresh one at the same path is not mistaken for the
old one). Any change to the root's entries updates that timestamp, so the
check is a single ``stat`` -- far cheaper than the ``scandir`` it guards.

Ported from the pack scanning in ``ComfyUI-VoxCPM/modules/api_routes.py`` and
the deleted V1 ``chatterbox_handler.py`` (plan S4.7), with the cache added.

This module never imports the ComfyUI node-API package, ``comfy.*`` or ``server``.
``folder_paths`` is imported lazily, inside :func:`get_models_root`, so the
module works without ComfyUI running.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from .constants import CHATTERBOX_MODEL_SUBDIR, DEFAULT_MODEL_PACK_NAME


__all__ = [
    "get_models_root",
    "get_chatterbox_model_pack_names",
    "get_model_pack_path",
    "get_model_pack_entries",
    "invalidate_cache",
]

# Cache state. ``_packs`` is None before the first scan. The key is
# (st_mtime_ns, st_ino) of the models root; a missing root keys on None, which
# is stable, so a missing directory is also cached rather than re-probed.
_cache: Dict[str, object] = {"key": "unset", "packs": None}


def get_models_root() -> str:
    """``<models>/tts/chatterbox``, resolved lazily.

    Falls back to a relative ``models/tts/chatterbox`` when ``folder_paths`` is
    unavailable, so the module imports and runs outside ComfyUI.
    """
    try:
        import folder_paths
    except ImportError:
        logging.debug(
            "[Chatterbox TTS] folder_paths is not importable; resolving the "
            "Chatterbox models root relative to the working directory"
        )
        return os.path.join("models", CHATTERBOX_MODEL_SUBDIR)

    try:
        tts_paths = folder_paths.get_folder_paths("tts")
    except Exception as exc:  # noqa: BLE001 -- an unregistered folder is normal
        logging.debug("[Chatterbox TTS] No 'tts' folder registered with ComfyUI: %s", exc)
        tts_paths = None
    if tts_paths:
        return os.path.join(tts_paths[0], os.path.basename(CHATTERBOX_MODEL_SUBDIR))

    try:
        return os.path.join(folder_paths.models_dir, CHATTERBOX_MODEL_SUBDIR)
    except Exception as exc:  # noqa: BLE001 -- a broken ComfyUI must not break the schema
        logging.debug("[Chatterbox TTS] Could not resolve models_dir: %s", exc)

    return os.path.join("models", CHATTERBOX_MODEL_SUBDIR)


def _root_key(root: str) -> Optional[Tuple[int, int]]:
    """``(st_mtime_ns, st_ino)`` of ``root``, or ``None`` when it is absent."""
    try:
        stat_result = os.stat(root)
    except OSError:
        return None
    return (stat_result.st_mtime_ns, stat_result.st_ino)


def _scan(root: str) -> List[Tuple[str, str]]:
    """Every subdirectory of ``root``, as sorted ``(name, path)`` pairs.

    Uses ``os.scandir`` rather than ``os.listdir`` so the directory is read
    once and each entry's type comes from the readdir data already in hand.
    Every subdirectory counts as a pack: a partially downloaded one must stay
    selectable so the download can finish it. The reserved shared-weight
    store (``SHARED_ASSET_DIR_NAME``) is not a pack and never appears.

    Each directory is offered under its **canonical** name, because the combo
    may only show names that resolve: a folder still carrying a pack's old id
    is the pack it now names, and showing two spellings of one pack would
    offer a value nothing else in the repository accepts. When a canonical
    directory and a legacy one both exist the canonical directory wins, so
    the pick always reads the same as the folder it loaded -- the legacy
    folder stays loadable, but as the fallback :func:`get_model_pack_path`
    resolves rather than as a rival entry.
    """
    from .model_info import SHARED_ASSET_DIR_NAME, canonical_pack_name

    packs: Dict[str, Tuple[str, str]] = {}
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                try:
                    if entry.name == SHARED_ASSET_DIR_NAME:
                        continue
                    if entry.is_dir():
                        name = canonical_pack_name(entry.name)
                        if name in packs and entry.name != name:
                            continue
                        packs[name] = (name, entry.path)
                except OSError as exc:
                    # A pack deleted mid-scan is not a reason to lose the rest.
                    logging.debug(
                        "[Chatterbox TTS] Skipping '%s' during the pack scan: %s",
                        entry.name,
                        exc,
                    )
    except OSError as exc:
        logging.debug("[Chatterbox TTS] Could not scan '%s': %s", root, exc)
        return []

    # Sort by name, then hoist the default pack so a freshly saved workflow
    # that carries no widget value lands on the official voice rather than
    # whatever sorts first alphabetically.
    found = sorted(packs.values(), key=lambda pair: pair[0])
    for index, (name, _path) in enumerate(found):
        if name == DEFAULT_MODEL_PACK_NAME:
            default_pack = found.pop(index)
            found.insert(0, default_pack)
            break
    return found


def get_model_pack_entries() -> List[Tuple[str, str]]:
    """``(name, path)`` for every pack, served from cache when unchanged.

    A Turbo-family folder carrying both t3 checkpoints is listed twice -- once
    under its own name, once under the canonical upstream name of the other
    member -- because the dropdown entry, not the folder, picks the weights.
    The rule (and the name) live in
    :func:`modules.model_info.pack_dir_entries`; this pass only borrows the
    names, and skips a synthesized name that a real directory already owns.

    Every name here is canonical: a directory carrying a renamed pack's old
    spelling is listed under the name that pack now has, against its own
    path. Nothing above this pass needs to know the folder was renamed.
    """
    from .model_info import pack_dir_entries

    root = get_models_root()
    key = _root_key(root)

    if _cache["packs"] is None or _cache["key"] != key:
        _cache["packs"] = _scan(root)
        _cache["key"] = key
        logging.debug(
            "[Chatterbox TTS] Scanned %d Chatterbox pack(s) under '%s'",
            len(_cache["packs"]), root,
        )

    packs = list(_cache["packs"])  # type: ignore[arg-type]
    taken = {name for name, _path in packs}
    offered: List[Tuple[str, str]] = []
    for name, path in packs:
        offered.append((name, path))
        for entry_name, _architecture in pack_dir_entries(path):
            if entry_name != name and entry_name not in taken:
                offered.append((entry_name, path))
    return offered


def get_chatterbox_model_pack_names() -> List[str]:
    """Every selectable pack name, default pack first.

    The list holds the packs that are **on disk** *and* the official packs
    that **can be downloaded**. A fresh install has no models folder at all,
    and a dropdown built from the filesystem alone then offers nothing to
    pick -- and a pack that cannot be picked cannot be downloaded. The
    registry knows every official pack, so it supplies the names the disk
    does not have yet.

    Local names come first, in the order the scan produced them (the
    synthesized Turbo-family entry next to its folder), then the official
    names that are not on disk yet. The default pack is hoisted to index 0
    either way, so a saved workflow that carries no widget value lands on the
    official voice rather than on whatever sorts first.

    Never returns an empty list: the default pack is always offered, because
    selecting it is what triggers the download.
    """
    from .model_info import official_pack_names

    names: List[str] = []
    seen: set = set()
    for name, _path in get_model_pack_entries():
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    for name in official_pack_names():
        if name in seen:
            continue
        seen.add(name)
        names.append(name)

    if DEFAULT_MODEL_PACK_NAME in names:
        names.remove(DEFAULT_MODEL_PACK_NAME)
        names.insert(0, DEFAULT_MODEL_PACK_NAME)
    if not names:
        return [DEFAULT_MODEL_PACK_NAME]
    return names


def get_model_pack_path(model_pack_name: Optional[str]) -> Optional[str]:
    """The directory ``model_pack_name`` lives in -- or should be downloaded into.

    ``None`` for empty input is deliberate: a saved workflow can carry an
    empty combo value, and the loader turns ``None`` into a clear message
    instead of resolving to the models root and loading the wrong thing.

    The name is canonicalised first, so an old spelling asks about the very
    directory the current name asks about. The answer comes from the cached
    scan, which is what makes an **existing install** keep working rather
    than re-download 2.7 GB: a folder still carrying the old spelling is
    listed under the canonical name against its own path, and that path is
    what this hands back. A pack with no directory yet -- a fresh install, or
    a pack the user has not chosen before -- resolves to the canonical path,
    which is where a download puts it.
    """
    if not model_pack_name:
        return None
    from .model_info import canonical_pack_name

    name = canonical_pack_name(model_pack_name)
    for entry_name, path in get_model_pack_entries():
        if entry_name == name:
            return path
    return os.path.join(get_models_root(), name)


def invalidate_cache() -> None:
    """Force the next scan to hit the filesystem."""
    _cache["key"] = "unset"
    _cache["packs"] = None