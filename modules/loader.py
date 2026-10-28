"""Loading Chatterbox voice packs off disk, and the handler ComfyUI manages.

Two objects live here:

* :class:`ChatterboxModelHandler` -- an ``nn.Module`` wrapper around a voice
  pack. It exists because ``ChatterboxTTS`` / ``ChatterboxVC`` are plain
  classes, not modules: ComfyUI's memory manager can only move, measure and
  patch ``nn.Module``s. The handler loads the pack lazily and re-exposes the
  weight submodules (``t3``, ``s3gen``, ``ve``) as genuine children, which is
  what makes ``ModelPatcher._load_list()`` see anything at all.
* :class:`ChatterboxLoader` -- the static side: where a pack lives, whether it
  needs downloading, and the ``from_local`` calls that build it.

**Size accounting (D4).** The handler's ``size`` is the sum of the *real
on-disk* ``*.safetensors`` bytes of the pack directory, not a constant. V1
hardcoded ``int(1.5 * 1024**3)`` at two places (``nodes.py:148`` and
``:272``); a pack is 2.7 GB on disk, so ComfyUI was told the model was half
the size it is, agreed to hold it, and the overflow surfaced later as a CUDA
OOM somewhere inside generation. After a load, :meth:`_refine_size` replaces
the estimate with the measured parameter bytes, which is what the weights
actually cost once cast to the requested dtype.

``_ensure_downloaded`` is folded in here rather than kept in a separate
``base_loader``: Chatterbox needs exactly one download helper, and the
subclassing that justified the split does not exist in this project.

Nothing at module scope imports ``folder_paths``, ``comfy.model_management`` or
the vendored ``chatterbox`` package.
"""

from __future__ import annotations

import inspect
import logging
import os
import shutil
from typing import Any, Optional

import torch

from .constants import (
    CHATTERBOX_FILES_TO_DOWNLOAD,
    DEFAULT_MODEL_PACK_NAME,
    MTL_T3_MODELS,
)
from .model_discovery import get_model_pack_path
from .vendor_isolation import vendored_chatterbox
from .model_info import (
    AVAILABLE_CHATTERBOX_MODELS,
    CHATTERBOX_WEIGHT_FILES,
    MODEL_CONFIGS,
    MTL_TOKENIZER_FILE,
    SHARED_ASSET_DIR_NAME,
    SHARED_ASSET_FILES,
    canonical_pack_name,
    is_chatterbox_model_dir,
)

__all__ = [
    "WEIGHT_SUFFIXES",
    "MTL_CHECKPOINT_FILES",
    "TURBO_CHECKPOINT_FILES",
    "ChatterboxModelHandler",
    "ChatterboxLoader",
    "pack_size_on_disk",
]

# Suffixes that carry real weights. ``.safetensors`` is what every Chatterbox
# pack ships; ``.bin`` is here only so a community pack converted to the torch
# archive format is measured rather than reported as zero bytes. ``conds.pt``
# is deliberately **excluded**: it holds the pack's built-in voice conditioning
# (a few MB), not model weights, and counting it would report a 50-byte file as
# half a gigabyte of headroom on a pack whose safetensors are missing.
WEIGHT_SUFFIXES = (".safetensors", ".bin")

# Weight-bearing attributes on ``ChatterboxTTS`` / ``ChatterboxVC``. They are
# ``nn.Module`` instances even though their owning class is not.
_WEIGHT_ATTRIBUTES = ("t3", "s3gen", "ve")

# The multilingual and Turbo entry points read *different* file layouts from the
# classic one -- ``.pt`` weights and a grapheme tokenizer for multilingual,
# safetensors plus a GPT-2 tokenizer for Turbo -- so completeness is checked
# against these lists rather than against
# :data:`modules.model_info.CHATTERBOX_WEIGHT_FILES`, which would re-download a
# perfectly good checkpoint on every run. The version- or variant-selected T3
# checkpoint is appended by the caller, because its name is not fixed -- and
# so is the s3gen decoder's: the Single Language Packs ship ``s3gen_v3.pt``
# (P-036), the shared repo ships ``s3gen.pt``.
#: Files every multilingual pack holds *besides* the resolved T3, voice
#: encoder and s3gen checkpoints (P-036 resolves those per pack): just the
#: grapheme tokenizer.
MTL_CHECKPOINT_FILES: tuple[str, ...] = (MTL_TOKENIZER_FILE,)

#: The s3gen filenames a multilingual pack may carry, in resolution order:
#: the Single Language Packs' v3 retrain (safetensors first), then the shared
#: repo's canonical decoder.
MTL_S3GEN_CHECKPOINTS: tuple[str, ...] = (
    "s3gen_v3.safetensors",
    "s3gen_v3.pt",
    "s3gen.safetensors",
    "s3gen.pt",
)

#: Pack folders that historically hold the shared repo's weights. Read-only
#: search candidates for shared assets; downloads never write into them.
LEGACY_SHARED_DIRS: tuple[str, ...] = ("chatterbox", DEFAULT_MODEL_PACK_NAME)

#: Turbo and Nano share one layout, and so share one completeness list. The
#: tokenizer is checked separately: both families publish the GPT2-style
#: HuggingFace layout (``vocab.json`` + ``merges.txt``), and
#: ``s3gen_meanflow.safetensors`` is here because the vendored Turbo loader
#: builds ``S3Gen(meanflow=True)`` from it unconditionally.
TURBO_CHECKPOINT_FILES: tuple[str, ...] = (
    "ve.safetensors",
    "s3gen.safetensors",
    "s3gen_meanflow.safetensors",
)

# T3 checkpoint names that identify a pack's variant. A pack carries exactly
# one T3 checkpoint, and its file name is the only variant marker Chatterbox
# packs ship -- there is no config.json.
_TURBO_T3_FILES = ("t3_turbo_v1.safetensors", "t3_nano_v1.safetensors")
_FLASH_T3_FILE = "t3_flash.safetensors"


def _refuse_wrong_variant_pack(pack_dir: str, model_name: str, entry_point: str) -> None:
    """Raise when the pack's T3 checkpoint names a variant this loader can't run.

    The pack dropdown is a directory scan, so every loader is offered every
    pack. The error the wrong pairing produced before was a missing-file list
    that read like a broken download, sending the user hunting for files their
    pack was never supposed to hold. Flash is refused by every entry point:
    it is a block-diffusion model only the Chatterbox TTS node (and its
    vendored Flash sampler) can run.
    """
    try:
        entries = set(os.listdir(pack_dir))
    except OSError:
        return

    if _FLASH_T3_FILE in entries and entry_point != "flash":
        raise FileNotFoundError(
            f"The pack '{model_name}' at '{pack_dir}' is Chatterbox Flash, a "
            "block-diffusion model. Only the Chatterbox TTS node runs it."
        )

    turbo_shaped = any(name in entries for name in _TURBO_T3_FILES)
    classic_shaped = "t3_cfg.safetensors" in entries
    if entry_point == "classic" and turbo_shaped:
        raise FileNotFoundError(
            f"The pack '{model_name}' at '{pack_dir}' is a Chatterbox "
            "Turbo/Nano pack (GPT2-style T3 weights) reaching the classic "
            "loader directly. The Chatterbox TTS node routes Turbo-family "
            "packs automatically -- load them through that node, or this "
            "loader's Turbo entry point."
        )
    if entry_point == "turbo" and classic_shaped:
        raise FileNotFoundError(
            f"The pack '{model_name}' at '{pack_dir}' is a classic Chatterbox "
            "pack (llama-style T3 weights). The Turbo entry point cannot run "
            "it -- the Chatterbox TTS node loads classic packs on its own "
            "path."
        )


# ---------------------------------------------------------------------------
# On-disk measurement
# ---------------------------------------------------------------------------


def pack_size_on_disk(pack_dir: Optional[str]) -> int:
    """Total bytes of the weight files in ``pack_dir``.

    A plain ``os.scandir`` sum of the files' sizes. Deliberately not a safetensors
    header parse: the point is what the loader will have to *read*, and reading
    the file is what costs the bytes. A header parse would answer a different
    question (tensor storage only) and would disagree with the download size the
    user was shown.

    Args:
        pack_dir: The pack directory, or ``None``.

    Returns:
        The summed size in bytes, or ``0`` for a missing/unreadable directory.
    """
    if not pack_dir or not os.path.isdir(pack_dir):
        return 0

    total = 0
    try:
        with os.scandir(pack_dir) as entries:
            for entry in entries:
                if not entry.name.endswith(WEIGHT_SUFFIXES):
                    continue
                try:
                    if entry.is_file():
                        total += entry.stat().st_size
                except OSError as exc:
                    # A file disappearing mid-scan is not a reason to report a
                    # partial pack as complete.
                    logging.debug(
                        "[Chatterbox TTS] Could not stat '%s' while sizing pack "
                        "'%s': %s",
                        entry.path,
                        pack_dir,
                        exc,
                    )
    except OSError as exc:
        logging.debug("[Chatterbox TTS] Could not scan pack '%s': %s", pack_dir, exc)
        return 0
    return total


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class ChatterboxModelHandler(torch.nn.Module):
    """A ComfyUI-manageable wrapper around one Chatterbox voice pack.

    ``ComfyUI`` sees an ordinary module: ``named_parameters()`` walks through to
    the real weights, ``.to(device)`` moves them, and ``ModelPatcher`` can patch
    and free them. The Chatterbox-specific state -- the pack name, the requested
    attention mode, the lazily-loaded ``tts_model`` / ``vc_model`` objects --
    hangs off it as ordinary attributes.
    """

    def __init__(
        self,
        model_pack_name: str,
        *,
        attention_mode: str = "sdpa",
        dtype_str: str = "auto",
        model_path: Optional[str] = None,
        load_vc: bool = True,
        entry_point: str = "classic",
        variant: str = "",
    ) -> None:
        super().__init__()
        self.model_pack_name = model_pack_name
        self.attention_mode = attention_mode
        self.dtype_str = dtype_str
        self.model_path = model_path
        self.load_vc = load_vc
        self.entry_point = entry_point
        self.cache_key = f"chatterbox_{model_pack_name}_attn_{attention_mode}"
        # ``entry_point`` has one meaningful value for two of the three: the
        # multilingual node's resolved T3 checkpoint filename, and ``"nano"`` /
        # ``"turbo"``. Empty for the classic node, which is what upstream loads.
        self.variant: str = variant

        # The loaded pack. Plain attributes, NOT submodules: Chatterbox's own
        # classes are not nn.Modules, and registering a non-module here would
        # raise. Their weight submodules are registered separately.
        self.tts_model: Any = None
        self.vc_model: Any = None
        self.device: Optional[torch.device] = None

        self.size = pack_size_on_disk(model_path)
        if self.size <= 0:
            self.size = _registry_size_estimate(model_pack_name)

    # -- weight plumbing ----------------------------------------------

    def attach_weights(self) -> None:
        """Register the pack's weight modules as submodules of this handler.

        Called after a load. ``ChatterboxTTS.t3`` / ``.s3gen`` / ``.ve`` and
        ``ChatterboxVC.s3gen`` are ``nn.Module`` instances; registering them is
        what gives ``ModelPatcher._load_list()`` something to enumerate and
        ``.to(device)`` something to move.

        A component already registered is skipped rather than registered twice
        under a second name: the VC model loads its own copy of ``s3gen`` from
        the same checkpoint, so counting both would double the model's reported
        size for no gain.
        """
        for owner in (self.tts_model, self.vc_model):
            if owner is None:
                continue
            for attr in _WEIGHT_ATTRIBUTES:
                if attr in self._modules:
                    continue
                module = getattr(owner, attr, None)
                if isinstance(module, torch.nn.Module):
                    self.add_module(attr, module)

    def detach_weights(self) -> None:
        """Drop every registered submodule so the weights really are freed.

        Nulling ``tts_model`` / ``vc_model`` on the wrapper is not enough: the
        ``add_module`` registrations from :meth:`attach_weights` hold the very
        same objects, and a module registry entry is as strong a reference as
        the attribute it came from.
        """
        for name in list(self._modules):
            del self._modules[name]

    def to_device(self, device: torch.device) -> "ChatterboxModelHandler":
        """Move the handler *and* the non-module state the modules do not own.

        ``ChatterboxTTS.conds`` and ``ChatterboxVC.ref_dict`` are plain dicts
        of tensors sitting outside any submodule. Leaving them behind is how a
        "warm offload" ends up still holding GPU memory after it claimed not to.

        Returns:
            ``self``, so the call chains.
        """
        self.to(device)
        self.device = device
        for owner in (self.tts_model, self.vc_model):
            if owner is None:
                continue
            for attr in ("conds", "ref_dict"):
                holder = getattr(owner, attr, None)
                if isinstance(holder, dict):
                    for key, value in holder.items():
                        if torch.is_tensor(value):
                            holder[key] = value.to(device)
        return self

    # -- loading -------------------------------------------------------

    @vendored_chatterbox
    def load_model(
        self,
        device: torch.device,
        attention_mode: str = "sdpa",
        client_id: Optional[str] = None,
    ) -> None:
        """Load the pack onto ``device`` and measure what it actually cost.

        Args:
            device: Where the weights should live.
            attention_mode: The resolved attention mode, recorded on the pack.
            client_id: Websocket id for first-run download progress, forwarded
                from :class:`modules.patcher.ChatterboxPatcher`. ``None`` means
                "broadcast", which is what an API call (no browser tab) wants.
        """
        self.attention_mode = attention_mode
        if self.entry_point == "multilingual":
            # The multilingual entry point is one model, not a TTS/VC pair, and
            # it takes the resolved T3 checkpoint filename in its cache stem.
            tts_model, vc_model = (
                ChatterboxLoader.load_multilingual_model(
                    self.model_pack_name,
                    self.variant,
                    device,
                    client_id=client_id,
                    model_path=self.model_path,
                ),
                None,
            )
        elif self.entry_point == "turbo":
            tts_model, vc_model = (
                ChatterboxLoader.load_turbo_model(
                    self.model_pack_name,
                    device,
                    nano=self.variant == "nano",
                    client_id=client_id,
                    model_path=self.model_path,
                ),
                None,
            )
        elif self.entry_point == "flash":
            tts_model, vc_model = (
                ChatterboxLoader.load_flash_model(
                    self.model_pack_name,
                    device,
                    client_id=client_id,
                    model_path=self.model_path,
                ),
                None,
            )
        else:
            tts_model, vc_model = ChatterboxLoader.load_model(
                self.model_pack_name,
                device,
                attention_mode=attention_mode,
                dtype_str=self.dtype_str,
                model_path=self.model_path,
                load_vc=self.load_vc,
                client_id=client_id,
            )
        self.tts_model = tts_model
        self.vc_model = vc_model
        self.device = device
        self.attach_weights()
        self._refine_size()

    @staticmethod
    def _estimate_size(model: Any) -> int:
        """Measured byte size of ``model``'s tensors, or ``0`` when unmeasurable.

        Two routes, because the two things worth measuring have different
        shapes:

        * an ``nn.Module`` (this handler, or anything exposing ``parameters()``)
          -- summed directly, with ``named_parameters``' own de-duplication
          keeping a component registered twice from counting twice;
        * a Chatterbox pack object, which is a *plain* class holding ``nn.Module``
          attributes -- walked attribute by attribute.

        Unlike the V1 constant this never guesses: a model it cannot measure
        reports zero, and the caller decides what to do with that.
        """
        total = _tensor_bytes(model)
        if total > 0:
            return total

        total = 0
        seen: set[int] = set()
        for owner in (model, getattr(model, "tts_model", None), getattr(model, "vc_model", None)):
            if owner is None:
                continue
            for attr in _WEIGHT_ATTRIBUTES:
                component = getattr(owner, attr, None)
                if not isinstance(component, torch.nn.Module):
                    continue
                if id(component) in seen:
                    continue
                seen.add(id(component))
                total += _tensor_bytes(component)
        return total

    def _refine_size(self) -> None:
        """Replace the on-disk estimate with the measured parameter bytes.

        The on-disk sum is the right answer for ComfyUI's *planning* (it is what
        the read costs). The measured sum is the right answer for occupancy (it
        is what the tensors cost once cast to the requested dtype, and it
        covers anything the checkpoint did not need to store). After a load the
        measurement is strictly better, so it wins.

        Measured on the handler rather than on the pack objects so the
        components registered once by :meth:`attach_weights` are counted once.

        Note: the built-in voice conditioning in ``conds.pt`` is a few megabytes
        of features, not parameters, and is not in this total. It is small
        enough that over- or under-reporting it by a few MB does not change any
        load decision.
        """
        total = self._estimate_size(self)
        if total > 0:
            self.size = total
            logging.debug(
                "[Chatterbox TTS] Refined '%s' size to %d bytes from the loaded "
                "parameters (on-disk estimate was %d).",
                self.model_pack_name,
                total,
                pack_size_on_disk(self.model_path),
            )


def _tensor_bytes(module: Any) -> int:
    """Summed bytes of ``module``'s parameters and buffers; ``0`` if it has none."""
    total = 0
    try:
        for tensor in module.parameters():
            total += tensor.numel() * tensor.element_size()
        for tensor in module.buffers():
            total += tensor.numel() * tensor.element_size()
    except Exception:
        return 0
    return total


def _registry_size_estimate(model_pack_name: str) -> int:
    """Registry ``size_gb`` as a last-resort estimate, or ``0``.

    Only reached for a pack with no weight files on disk yet -- mid-download, or
    pointed at by a stale combo value. Returning 0 would leave ComfyUI to
    measure the (still empty) module and conclude the model is free, which is
    worse than a rough guess.

    An alias is resolved to its canonical name first, so the old spelling of a
    renamed pack is measured against the pack it now names.
    """
    name = canonical_pack_name(model_pack_name)
    config = MODEL_CONFIGS.get(name) or AVAILABLE_CHATTERBOX_MODELS.get(name)
    if not config:
        return 0
    size_gb = config.get("size_gb")
    try:
        size_gb = float(size_gb)
    except (TypeError, ValueError):
        return 0
    if size_gb <= 0:
        return 0
    logging.debug(
        "[Chatterbox TTS] Pack '%s' has no weight files on disk yet; falling back "
        "to the registry estimate of %.2f GB.",
        name,
        size_gb,
    )
    return int(size_gb * (1024**3))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class ChatterboxLoader:
    """Filesystem and Hugging Face side of loading a Chatterbox pack."""

    # -- paths ---------------------------------------------------------

    @staticmethod
    def resolve_pack_dir(model_name: str) -> str:
        """Directory the named pack lives in, or should be downloaded into.

        An alias resolves to the pack it now names: a workflow saved before a
        pack was renamed still finds (and downloads into) the same directory,
        because the old spelling is answered from the canonical entry.

        Never raises for a missing directory: "not downloaded yet" is a normal
        state that the download step resolves, and a select-then-download flow
        asks this question before the files exist.

        Raises:
            ValueError: ``model_name`` is blank.
        """
        if not model_name:
            raise ValueError(
                "No Chatterbox model was selected. Pick a voice pack from the "
                "model dropdown."
            )

        name = canonical_pack_name(model_name)
        # The official table is the fallback: a local scan can demote an entry
        # to a local one, and an official entry is the only thing that knows
        # the download target -- the Turbo-family Nano member downloads into
        # the shared ``chatterbox-turbo`` folder, not into a folder of its own.
        entry = AVAILABLE_CHATTERBOX_MODELS.get(name) or MODEL_CONFIGS.get(name)
        if entry and entry.get("type") == "local" and entry.get("path"):
            return str(entry["path"])

        # An official entry may download into a folder other than its own
        # name: the Turbo-family members share one directory, so the Nano
        # entry's ``folder`` is ``chatterbox-turbo``.
        folder = str(entry.get("folder")) if entry and entry.get("folder") else name
        known = get_model_pack_path(folder)
        if known:
            return known

        if name == DEFAULT_MODEL_PACK_NAME:
            # The default pack has no directory until something creates one;
            # hand back the path it *would* live at.
            from .model_discovery import get_models_root

            return os.path.join(get_models_root(), DEFAULT_MODEL_PACK_NAME)

        return str(get_model_pack_path(name) or name)

    # -- completeness --------------------------------------------------

    @staticmethod
    def is_pack_complete(pack_dir: Optional[str]) -> bool:
        """Whether ``pack_dir`` holds a loadable pack."""
        return is_chatterbox_model_dir(pack_dir) if pack_dir else False

    # -- download ------------------------------------------------------

    @staticmethod
    def _ensure_downloaded(
        repo_id: str,
        local_dir: str,
        model_name: str = "",
        allow_patterns: Optional[list[str]] = None,
    ) -> None:
        """Fetch an official pack from ``repo_id`` when it is not on disk.

        The completeness check is :meth:`is_pack_complete` rather than a
        ``config.json`` probe: Chatterbox packs have no ``config.json``, and the
        check that actually matters is "can the loader build a model from this
        directory".

        The ``local_dir_use_symlinks`` kwarg was removed in
        ``huggingface_hub`` 0.23 and is absent in the 1.x installed here;
        passing it raises ``TypeError`` on the *first* download of a model. It
        is only passed when the installed ``snapshot_download`` still declares
        it, which is what the signature probe below buys.

        Args:
            repo_id: Hugging Face repository id. Empty means "nothing to fetch".
            local_dir: Destination directory.
            model_name: Friendly name for the log line.
        """
        if not repo_id:
            return
        if ChatterboxLoader.is_pack_complete(local_dir):
            return

        logging.info(
            "[Chatterbox TTS] Downloading voice pack '%s' from %s...",
            model_name or local_dir,
            repo_id,
        )

        from huggingface_hub import snapshot_download

        kwargs: dict[str, Any] = {"repo_id": repo_id, "local_dir": local_dir}
        if allow_patterns:
            kwargs["allow_patterns"] = list(allow_patterns)
        try:
            accepts_symlinks = (
                "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters
            )
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            accepts_symlinks = False
        if accepts_symlinks:
            kwargs["local_dir_use_symlinks"] = False

        snapshot_download(**kwargs)

    @staticmethod
    def download_pack(
        model_name: str,
        repo_id: str,
        local_dir: str,
        client_id: Optional[str] = None,
        allow_patterns: Optional[list[str]] = None,
    ) -> None:
        """Ensure the pack exists, preferring the progress-tracking manager.

        The download manager (S4.2) reports per-file progress and honours
        cancellation; a bare ``snapshot_download`` in a node's ``execute()``
        blocks the prompt with no way out. The manager is therefore the path
        taken, and :meth:`_ensure_downloaded` is the fallback for when it cannot
        be imported (or has been torn down).

        ``allow_patterns`` narrows the fetch to the named files -- the shared
        weights are deduplicated away (see ``SHARED_ASSET_FILES``), so a pack
        download must not drag every byte of the repo along. The manager's API
        has no patterns parameter, so a filtered fetch goes straight to
        ``snapshot_download`` (which resumes and skips complete files); the
        manager keeps serving the unfiltered legacy path.
        """
        if ChatterboxLoader.is_pack_complete(local_dir):
            return
        if not repo_id:
            return

        if allow_patterns:
            ChatterboxLoader._ensure_downloaded(
                repo_id, local_dir, model_name, allow_patterns
            )
            return

        manager = None
        try:
            from .downloader import get_download_manager

            manager = get_download_manager()
        except Exception as exc:
            logging.debug(
                "[Chatterbox TTS] Download manager unavailable (%s); falling back "
                "to a plain snapshot_download.",
                exc,
            )

        if manager is None:
            ChatterboxLoader._ensure_downloaded(repo_id, local_dir, model_name)
            return

        try:
            manager.download_model(model_name, repo_id, local_dir, client_id)
        except Exception as exc:
            logging.warning(
                "[Chatterbox TTS] The download manager failed for '%s' (%s); "
                "retrying with a plain snapshot_download.",
                model_name,
                exc,
            )
            ChatterboxLoader._ensure_downloaded(repo_id, local_dir, model_name)

    # -- loading -------------------------------------------------------

    @staticmethod
    @vendored_chatterbox
    def load_model(
        model_name: str,
        device: torch.device,
        attention_mode: str = "sdpa",
        dtype_str: str = "auto",
        model_path: Optional[str] = None,
        load_vc: bool = True,
        client_id: Optional[str] = None,
    ) -> tuple[Any, Optional[Any]]:
        """Build the pack's TTS (and VC) models onto ``device``.

        Args:
            model_name: Pack name from the model dropdown.
            device: Where the weights should live.
            attention_mode: Resolved attention mode. Recorded on the handler;
                the vendored models build their own attention internally, so
                this is carried for the patcher rather than passed down.
            dtype_str: The dtype widget value. Recorded; the cast itself happens
                in :meth:`modules.patcher.ChatterboxPatcher.patch_model`, after
                the load, so the weights reach the target device in the dtype
                they were stored in.
            model_path: Pack directory. Resolved from ``model_name`` when omitted.
            load_vc: Whether to build the voice-conversion model as well. Both
                come out of the same checkpoint, so skipping the VC half saves
                a second S3Gen load for a TTS-only workflow.
            client_id: Websocket id, forwarded to the download manager so a
                first-run download reports progress to the right browser tab.

        Returns:
            ``(tts_model, vc_model)``. ``vc_model`` is ``None`` when
            ``load_vc`` is false.

        Raises:
            FileNotFoundError: The pack is still not on disk after the download
                step.
        """
        pack_dir = model_path or ChatterboxLoader.resolve_pack_dir(model_name)
        _refuse_wrong_variant_pack(pack_dir, model_name, entry_point="classic")

        if not ChatterboxLoader.is_pack_complete(pack_dir):
            entry = AVAILABLE_CHATTERBOX_MODELS.get(model_name) or {}
            repo_id = str(entry.get("repo_id") or "")
            logging.info(
                "[Chatterbox TTS] Voice pack '%s' is not complete at '%s'; "
                "ensuring the download first.",
                model_name,
                pack_dir,
            )
            files = list(entry.get("files") or CHATTERBOX_FILES_TO_DOWNLOAD)
            unique = set(entry.get("unique_files") or ())
            ChatterboxLoader.download_pack(
                model_name,
                repo_id,
                pack_dir,
                client_id,
                allow_patterns=[
                    name
                    for name in files
                    if name not in SHARED_ASSET_FILES or name in unique
                ],
            )
            ChatterboxLoader._ensure_shared_assets(
                pack_dir, [name for name in files if name in SHARED_ASSET_FILES]
            )
            try:
                os.makedirs(pack_dir, exist_ok=True)
            except OSError as exc:
                logging.warning(
                    "[Chatterbox TTS] Could not create '%s': %s", pack_dir, exc
                )

        if not ChatterboxLoader.is_pack_complete(pack_dir):
            raise FileNotFoundError(
                f"The Chatterbox voice pack '{model_name}' is not available at "
                f"'{pack_dir}'. It needs a tokenizer and at least one "
                "'.safetensors' file -- select another pack, or download it from "
                "the model manager."
            )

        # Imported here, never at module scope: the vendored package drags in
        # librosa, transformers and the whole model stack, and collection must
        # never pay for that.
        from chatterbox.tts import ChatterboxTTS
        from chatterbox.vc import ChatterboxVC

        device_arg = str(device)
        logging.info(
            "[Chatterbox TTS] Building '%s' from '%s' on %s...",
            model_name,
            pack_dir,
            device_arg,
        )
        tts_model = ChatterboxTTS.from_local(pack_dir, device_arg)

        vc_model = None
        if load_vc:
            vc_model = ChatterboxVC.from_local(pack_dir, device_arg)

        logging.debug(
            "[Chatterbox TTS] Built '%s' (attention_mode=%s, dtype=%s, vc=%s)",
            model_name,
            attention_mode,
            dtype_str,
            "yes" if vc_model is not None else "no",
        )
        return tts_model, vc_model

    # -- other entry points (S8.1 / S8.2) --------------------------------

    @staticmethod
    def _missing_checkpoint_files(
        pack_dir: str,
        required: tuple[str, ...],
    ) -> list[str]:
        """Which of ``required`` are absent from ``pack_dir``.

        Named in the refusal message so a user staring at a half-downloaded pack
        knows exactly which file is missing, instead of getting "no such file or
        directory" from deep inside a ``torch.load``.
        """
        return [name for name in required if not os.path.isfile(os.path.join(pack_dir, name))]

    @staticmethod
    def _shared_store_dir() -> str:
        """The reserved ``_shared`` directory under the Chatterbox models root."""
        from .model_discovery import get_models_root

        return os.path.join(get_models_root(), SHARED_ASSET_DIR_NAME)

    @staticmethod
    def _ensure_shared_assets(pack_dir: str, names) -> None:
        """Materialize shared weights into ``pack_dir`` under canonical names.

        The shared files (``SHARED_ASSET_FILES``) are byte-identical across
        every repo that ships them, so each is fetched at most once. Search
        order: the pack itself (nothing to do), the reserved ``_shared``
        store, then the legacy full-repo folders a user may already have;
        only when the file exists nowhere does one ``hf_hub_download`` from
        the shared Chatterbox repo populate the store. The store copy is then
        hardlinked into the pack -- both live under the Chatterbox models
        root, so they share a volume and the link costs no extra disk -- with
        a plain copy as the fallback for exotic filesystems.
        """
        from huggingface_hub import hf_hub_download

        from .constants import CHATTERBOX_REPO_ID

        store = ChatterboxLoader._shared_store_dir()
        root = os.path.dirname(store)
        for name in names:
            target = os.path.join(pack_dir, name)
            if os.path.isfile(target):
                continue
            source = None
            candidates = [store]
            candidates.extend(os.path.join(root, d) for d in LEGACY_SHARED_DIRS)
            for candidate in candidates:
                probe = os.path.join(candidate, name)
                if os.path.isfile(probe):
                    source = probe
                    break
            if source is None:
                os.makedirs(store, exist_ok=True)
                source = hf_hub_download(
                    repo_id=CHATTERBOX_REPO_ID, filename=name, local_dir=store
                )
            try:
                os.link(source, target)
            except OSError:
                try:
                    os.makedirs(pack_dir, exist_ok=True)
                except OSError:
                    pass
                shutil.copyfile(source, target)

    @staticmethod
    def _ensure_entry_point_dir(
        pack_dir: str,
        repo_id: str,
        required: tuple[str, ...],
        model_name: str,
        client_id: Optional[str] = None,
    ) -> None:
        """Fetch ``repo_id`` into ``pack_dir`` when a required file is absent.

        The shared prologue for the multilingual, Turbo and Flash entry
        points. All are ordinary Hugging Face snapshots with their own file
        layouts, so the completeness test is ``required`` -- not
        :meth:`is_pack_complete`, which is written for the *classic* layout and
        would therefore re-download a perfectly good multilingual pack on every
        run.

        Community finetunes (Viterbox, Egyptian Arabic, anything else a user
        drops under ``models/tts/chatterbox``) satisfy the same test and take the
        same path: there is no special case for them anywhere in this class.

        Downloads are narrowed by the registry entry's ``files`` list minus
        the shared assets (``SHARED_ASSET_FILES``) -- those are byte-identical
        across the family and are materialized afterwards by
        :meth:`_ensure_shared_assets`, hardlinked out of the ``_shared``
        store. An entry-less (community) pack falls back to a whole-repo
        fetch: its layout is unknown.
        """
        if os.path.isdir(pack_dir) and not ChatterboxLoader._missing_checkpoint_files(
            pack_dir, required
        ):
            return
        if not repo_id:
            return
        entry = ChatterboxLoader._registry_entry(model_name) or {}
        unique = set(entry.get("unique_files") or ())
        shared_needed = [
            name for name in required if name in SHARED_ASSET_FILES and name not in unique
        ]
        patterns = None
        if entry.get("files"):
            # The registry's file list is the pack's usual content, but the
            # checkpoint the caller resolved is the one file the pack cannot be
            # assembled without -- and it is not always on that list (the
            # classic entry names the monolingual set, while the multilingual
            # loader asks for a T3 variant out of the same repository). So
            # whatever the caller requires that the list does not already name
            # is fetched too: a pack that is not on disk yet has to arrive
            # complete, not arrive and still be missing a checkpoint.
            candidates = list(entry["files"])
            for name in required:
                if name not in candidates:
                    candidates.append(name)
            patterns = [
                name
                for name in candidates
                if name not in SHARED_ASSET_FILES or name in unique
            ]
        logging.info(
            "[Chatterbox TTS] Fetching '%s' from %s into '%s'...",
            model_name,
            repo_id,
            pack_dir,
        )
        ChatterboxLoader.download_pack(
            model_name, repo_id, pack_dir, client_id, allow_patterns=patterns
        )
        ChatterboxLoader._ensure_shared_assets(pack_dir, shared_needed)

    @staticmethod
    def _registry_entry(model_name: str) -> Optional[dict]:
        """The registry config for ``model_name``, official or local, or ``None``.

        An alias is resolved to its canonical name first, so the old spelling
        of a renamed pack answers with the entry that describes it.
        """
        name = canonical_pack_name(model_name)
        entry = MODEL_CONFIGS.get(name)
        if entry is not None:
            return entry
        return AVAILABLE_CHATTERBOX_MODELS.get(name)

    @staticmethod
    def _resolve_mtl_t3_file(pack_dir: str, model_name: str, t3_model: str) -> str:
        """The T3 checkpoint a multilingual pack actually holds.

        Precedence: the registry's ``t3_checkpoint`` (a Single Language Pack
        entry names its own per-language blob), then the version combo's
        filename when the pack carries it, then -- for a community SLP
        directory no registry entry covers -- the single ``t3_*.safetensors``
        that is not one of the known per-entry-point checkpoints. Falls back
        to the combo filename so the missing-file message names what the user
        selected.
        """
        entry = ChatterboxLoader._registry_entry(model_name)
        registered = entry.get("t3_checkpoint") if entry else None
        if registered:
            return str(registered)
        if os.path.isdir(pack_dir):
            entries = set(os.listdir(pack_dir))
            if t3_model in entries:
                return t3_model
            known = set(MTL_T3_MODELS.values()) | set(CHATTERBOX_WEIGHT_FILES)
            language_t3 = sorted(
                name
                for name in entries
                if name.startswith("t3_")
                and name.endswith(".safetensors")
                and name not in known
            )
            if len(language_t3) == 1:
                return language_t3[0]
        return t3_model

    @staticmethod
    def _resolve_mtl_s3gen_file(pack_dir: str, model_name: str) -> str:
        """The s3gen decoder filename a multilingual pack actually holds.

        The directory wins (a local community pack is on disk when this is
        called with a ``model_path``), safetensors before any legacy ``.pt``
        twin; a registry entry whose ``files`` list names a decoder follows;
        the Single Language Packs' ``s3gen_v3.safetensors`` is the default,
        so the missing-file message keeps naming it.
        """
        if os.path.isdir(pack_dir):
            for name in MTL_S3GEN_CHECKPOINTS:
                if os.path.isfile(os.path.join(pack_dir, name)):
                    return name
        entry = ChatterboxLoader._registry_entry(model_name)
        files = entry.get("files", ()) if entry else ()
        for name in MTL_S3GEN_CHECKPOINTS:
            if name in files:
                return name
        return MTL_S3GEN_CHECKPOINTS[0]

    @vendored_chatterbox
    def load_multilingual_model(
        model_name: str,
        t3_model: str,
        device: torch.device,
        client_id: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> Any:
        """Build ``ChatterboxMultilingualTTS`` from ``model_name``.

        Args:
            model_name: Pack name or directory basename, from the dropdown.
            t3_model: The **resolved T3 checkpoint filename** from the
                version combo. A Single Language Pack overrides it with its
                own per-language checkpoint (see
                :meth:`_resolve_mtl_t3_file`).
            device: Where the weights should live.
            client_id: Websocket id for a first-run download.
            model_path: Pack directory; resolved from ``model_name`` when omitted.

        Returns:
            A loaded ``ChatterboxMultilingualTTS``.

        Raises:
            FileNotFoundError: The checkpoint is missing files, with the missing
                names in the message.
        """
        pack_dir = model_path or ChatterboxLoader.resolve_pack_dir(model_name)
        _refuse_wrong_variant_pack(pack_dir, model_name, entry_point="multilingual")
        t3_model = ChatterboxLoader._resolve_mtl_t3_file(pack_dir, model_name, t3_model)
        s3gen_file = ChatterboxLoader._resolve_mtl_s3gen_file(pack_dir, model_name)
        required = (*MTL_CHECKPOINT_FILES, "ve.safetensors", s3gen_file, t3_model)

        from .constants import CHATTERBOX_REPO_ID

        entry = ChatterboxLoader._registry_entry(model_name) or {}
        ChatterboxLoader._ensure_entry_point_dir(
            pack_dir,
            str(entry.get("repo_id") or CHATTERBOX_REPO_ID),
            required,
            model_name,
            client_id,
        )

        missing = ChatterboxLoader._missing_checkpoint_files(pack_dir, required)
        if missing:
            raise FileNotFoundError(
                f"The multilingual Chatterbox checkpoint at '{pack_dir}' is "
                f"incomplete. Missing: {', '.join(missing)}. A complete pack "
                f"holds {', '.join(required)}. Drop the files in, or pick a "
                f"pack that is fully downloaded."
            )

        # Imported here, never at module scope -- see the note in load_model.
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        logging.info(
            "[Chatterbox TTS] Building multilingual '%s' (%s) from '%s' on %s...",
            model_name,
            t3_model,
            pack_dir,
            device,
        )
        return ChatterboxMultilingualTTS.from_local(
            pack_dir, str(device), t3_model=t3_model
        )

    @staticmethod
    @vendored_chatterbox
    def load_turbo_model(
        model_name: str,
        device: torch.device,
        nano: bool = False,
        client_id: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> Any:
        """Build ``ChatterboxTurboTTS`` -- for Turbo **or** Nano.

        There is no ``chatterbox.tts_nano``: Nano ships as a second set of
        weights inside the same module and is reached through ``nano=True``.
        A separate node for it would be a separate file to keep in step with a
        model that is one flag away.

        Args:
            model_name: Pack name or directory basename.
            device: Where the weights should live.
            nano: Load the 110M Nano weights instead of the 350M Turbo ones.
            client_id: Websocket id for a first-run download.
            model_path: Pack directory; resolved from ``model_name`` when omitted.

        Returns:
            A loaded ``ChatterboxTurboTTS``.

        Raises:
            FileNotFoundError: The checkpoint is missing files, with the missing
                names in the message.
        """
        pack_dir = model_path or ChatterboxLoader.resolve_pack_dir(model_name)
        _refuse_wrong_variant_pack(pack_dir, model_name, entry_point="turbo")
        variant = "t3_nano_v1.safetensors" if nano else "t3_turbo_v1.safetensors"
        repo_id = "ResembleAI/chatterbox-nano" if nano else "ResembleAI/chatterbox-turbo"

        # Both families publish the GPT2-style HuggingFace layout (vocab.json +
        # merges.txt) rather than a tokenizer.json, but AutoTokenizer reads
        # either, so the completeness check accepts whichever layout the pack
        # carries -- and when neither is there yet, vocab.json is the name the
        # refusal lists.
        tokenizer_file = "tokenizer.json"
        if not os.path.isfile(os.path.join(pack_dir, "tokenizer.json")):
            tokenizer_file = "vocab.json"

        ChatterboxLoader._ensure_entry_point_dir(
            pack_dir,
            repo_id,
            (*TURBO_CHECKPOINT_FILES, tokenizer_file, variant),
            model_name,
            client_id,
        )

        required = (*TURBO_CHECKPOINT_FILES, tokenizer_file, variant)
        missing = ChatterboxLoader._missing_checkpoint_files(pack_dir, required)
        if missing:
            raise FileNotFoundError(
                f"The {'Nano' if nano else 'Turbo'} Chatterbox checkpoint at "
                f"'{pack_dir}' is incomplete. Missing: {', '.join(missing)}. A "
                f"complete pack holds {', '.join(required)}. Drop the files in, "
                f"or pick a pack that is fully downloaded."
            )

        # Imported here, never at module scope -- see the note in load_model.
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        logging.info(
            "[Chatterbox TTS] Building %s '%s' from '%s' on %s...",
            "Nano" if nano else "Turbo",
            model_name,
            pack_dir,
            device,
        )
        return ChatterboxTurboTTS.from_local(pack_dir, str(device), nano=nano)

    @staticmethod
    @vendored_chatterbox
    def load_flash_model(
        model_name: str,
        device: torch.device,
        client_id: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> Any:
        """Build ``ChatterboxFlashTTS`` from the vendored sampler package.

        Flash is block-diffusion T3 (a Fast-dLLM-style masked decoder) shipped
        in the ``chatterbox-flash`` package -- the vendored Chatterbox 0.1.7
        has no such decoder. That package is vendored by this node at
        ``src/chatterbox_flash/`` (never pip-installed): its dependency block
        would pull a pip copy of ``chatterbox-tts`` that shadows the vendored
        (patched) tree, while the code itself imports ``S3Gen`` /
        ``VoiceEncoder`` / ``T3Cond`` from ``chatterbox`` -- i.e. from this
        node's vendored sources, bf16 patches included.

        Flash ships no ``conds.pt``, so a reference clip is mandatory; the
        node-level dispatch refuses generation without one.

        Args:
            model_name: Pack name or directory basename.
            device: Where the weights should live.
            client_id: Websocket id for a first-run download.
            model_path: Pack directory; resolved from ``model_name`` when omitted.

        Returns:
            A loaded ``ChatterboxFlashTTS``.

        Raises:
            FileNotFoundError: The vendored package failed to import (with a
                never-pip-install warning), or the checkpoint is missing files.
        """
        pack_dir = model_path or ChatterboxLoader.resolve_pack_dir(model_name)
        required = (
            "ve.safetensors",
            "s3gen.safetensors",
            "t3_flash.safetensors",
            "tokenizer.json",
        )
        ChatterboxLoader._ensure_entry_point_dir(
            pack_dir,
            "ResembleAI/chatterbox-flash",
            required,
            model_name,
            client_id,
        )

        missing = ChatterboxLoader._missing_checkpoint_files(pack_dir, required)
        if missing:
            raise FileNotFoundError(
                f"The Chatterbox Flash checkpoint at '{pack_dir}' is "
                f"incomplete. Missing: {', '.join(missing)}. A complete pack "
                f"holds {', '.join(required)}."
            )

        try:
            from chatterbox_flash import ChatterboxFlashTTS
        except ImportError as exc:
            raise FileNotFoundError(
                f"The Flash pack '{model_name}' needs this node's vendored "
                "'chatterbox-flash' package (src/chatterbox_flash), which "
                "failed to import. The node checkout may be incomplete or "
                "damaged -- re-install or update the node. Do NOT pip-install "
                "'chatterbox-flash': its dependency block would pull a pip "
                "Chatterbox that shadows this node's vendored sources."
            ) from exc

        logging.info(
            "[Chatterbox TTS] Building Flash '%s' from '%s' on %s...",
            model_name,
            pack_dir,
            device,
        )
        return ChatterboxFlashTTS.from_local(pack_dir, str(device))