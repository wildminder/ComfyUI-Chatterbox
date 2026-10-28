"""Model acquisition and synthesis for the Chatterbox nodes.

The node layer (B9) calls into this module and nowhere else: give it a pack
name and it returns ``(patcher, tts_model, vc_model)``; give that a text and it
returns ``(tensor, sample_rate)``.

**Ordering is load-bearing.** :func:`load_chatterbox_model` performs, in this
order and no other:

1. resolve the attention mode against this machine's capabilities;
2. resolve the dtype and the devices;
3. compute the cache key from the *resolved* values;
4. **evict the previous model, before anything is allocated**;
5. build the handler;
6. build the patcher;
7. hand it to ``load_to_device``.

Steps 1-3 before step 4 are what make step 4 mean anything: a key built from
the *requested* attention mode would differ from the key the same pack already
holds under its *resolved* one, so a sage -> sdpa fallback would evict a
perfectly good model and reload it (or evict nothing and hold two copies of the
same weights). Step 4 before step 5 is plain eviction discipline -- on a
machine that can hold either of two packs but never both, freeing before
allocating is the difference between working and a CUDA OOM.

**``client_id`` is not part of the cache key.** It names the websocket progress
events go to, not the model. It is stamped onto the patcher per call, so a
second browser tab asking for a loaded pack reuses it *and* gets its own id for
progress. VibeVoice's ``generation.py:78`` stomped the id onto a cached
patcher at construction time and lost the second client's id entirely.

**Interrupt handling.** ``InterruptProcessingException`` subclasses
``BaseException``, not ``Exception``. :func:`generate_audio` names it *before*
the broad ``except Exception`` that logs and re-raises, so a cancelled run
always cancels. The reverse order would let a cancellation be reported as a
generation failure -- and, in the V1 nodes, as one second of silence.

Borrowed from VibeVoice's ``modules/generation.py``; see the plan's §3.1.8 for
the ``client_id`` correction.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Optional

import torch

from . import model_info
from .attention_utils import get_attn_implementation_for_load, resolve_attention_mode
from .audio_utils import (
    S3GEN_SAMPLE_RATE,
    S3_TOKENIZER_SAMPLE_RATE,
    downmix_to_mono,
    interrupt_exception_class,
    preprocess_comfy_audio,
    set_seed,
)
from .device_utils import (
    DEVICE_CPU,
    _get_model_management,
    get_offload_device,
    get_torch_device,
)
from .dtype_utils import DTYPE_AUTO, get_dtype_str, resolve_dtype
from .loader import ChatterboxLoader, ChatterboxModelHandler
from .vendor_isolation import vendored_chatterbox
from .model_registry import FAMILY_TTS, evict_if_changed
from .text_processing import plan_text
from .patcher import (
    CHATTERBOX_PATCHER_CACHE as _SHARED_PATCHER_CACHE,
)
from .patcher import (
    ChatterboxPatcher,
    load_to_device,
    select_patcher_class,
)

__all__ = [
    "CHATTERBOX_PATCHER_CACHE",
    "FAMILY_TTS",
    "FAMILY_MTL",
    "FAMILY_TURBO",
    "REFINEMENT_CACHE_DEPTH",
    "InterruptProgressBar",
    "resolve_generation_family",
    "resolve_attn_implementation",
    "build_cache_key",
    "variant_cache_stem",
    "load_chatterbox_model",
    "load_chatterbox_entry_point",
    "generate_audio",
    "generate_mtl_audio",
    "generate_turbo_audio",
    "convert_voice_audio",
    "iterative_convert_voice",
    "RefinementCache",
    "split_audio_for_conversion",
    "combine_audio_chunks",
    "force_offload_model",
]


# The one cache, re-exported rather than re-created: two dicts means every
# eviction sweeps one of them and leaves the other holding the weights it just
# claimed to free.
CHATTERBOX_PATCHER_CACHE: dict[str, ChatterboxPatcher] = _SHARED_PATCHER_CACHE

#: Model-registry families. One per eviction class: a multilingual pack and a
#: Turbo pack are as mutually exclusive in VRAM as two classic packs are.
FAMILY_MTL = "mtl"
FAMILY_TURBO = "turbo"
FAMILY_FLASH = "flash"

#: How many intermediate iterations :class:`RefinementCache` keeps. The reference
#: implementation caps at 5 and its own tooltip says "recommended max 5 passes --
#: more causes distortions", so holding more would be memory nobody benefits from.
REFINEMENT_CACHE_DEPTH = 5


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


class InterruptProgressBar:
    """A ComfyUI progress bar that polls for cancellation on every tick.

    Chatterbox's sampling loops call ``pbar.update(1)`` -- thousands of times
    over a CPU run -- and the vendored code has no interrupt check of its own.
    This is therefore the hook point for "the user pressed stop": the poll
    raises ComfyUI's interrupt, which :func:`generate_audio` lets through.

    Args:
        total: The widget's token (or timestep) budget.
        inner: A ``comfy.utils.ProgressBar`` to drive, or ``None`` to run
            without one.
    """

    def __init__(self, total: int, inner: Optional[Any] = None) -> None:
        self.total = int(total)
        self._inner = inner
        self._current = 0
        self._model_management = _get_model_management()

    def update(self, value: int = 1) -> None:
        """Advance by ``value`` steps. Raises on a cancelled run."""
        self._current += int(value)
        self._model_management.throw_exception_if_processing_interrupted()
        if self._inner is not None:
            self._inner.update(value)

    def update_absolute(self, value: int, total: Optional[int] = None) -> None:
        """Jump to ``value``. Raises on a cancelled run."""
        if total is not None:
            self.total = int(total)
        self._current = int(value)
        self._model_management.throw_exception_if_processing_interrupted()
        if self._inner is not None:
            self._inner.update_absolute(value, total=total)

    def finish(self) -> None:
        """Close the bar out. Never polls, so it is safe from a ``finally``.

        Polling here would re-raise the interrupt that the ``try`` block just
        propagated, replacing a clean cancellation with one raised from a
        cleanup path.
        """
        if self._inner is not None:
            try:
                self._inner.update_absolute(self.total, total=self.total)
            except Exception as exc:  # pragma: no cover - cosmetic
                logging.debug("[Chatterbox TTS] Could not close the progress bar: %s", exc)


def _build_progress_bar(total: int) -> InterruptProgressBar:
    """An :class:`InterruptProgressBar`, with a core ``ProgressBar`` if available."""
    inner = None
    try:
        from comfy.utils import ProgressBar

        inner = ProgressBar(total)
    except Exception as exc:  # pragma: no cover - core is present under ComfyUI
        logging.debug("[Chatterbox TTS] ProgressBar unavailable: %s", exc)
    return InterruptProgressBar(total, inner)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def resolve_attn_implementation(attention_mode: str) -> str:
    """The ``attn_implementation`` string a pack would be built with.

    ``sage`` loads as ``sdpa``: transformers has no sage implementation to ask
    for, and the kernel is patched in afterwards (see
    :func:`modules.attention_utils.get_attn_implementation_for_load`).
    """
    return get_attn_implementation_for_load(attention_mode)


def resolve_generation_family(
    model_name: str,
    external_model: Optional[dict] = None,
) -> str:
    """Which generation family a pack belongs to.

    Args:
        model_name: Pack name from the dropdown.
        external_model: An externally-loaded pack descriptor, when the caller
            has one. Wins over ``model_name``.

    Returns:
        ``"tts"`` for a synthesis pack, ``"vc"`` for a conversion-only one.

    Raises:
        ValueError: No model was named, or the external descriptor declares a
            family this node pack does not implement.
    """
    if external_model is not None:
        family = external_model.get("family") or external_model.get("type")
        if family not in ("tts", "vc"):
            raise ValueError(
                f"An external Chatterbox model declares family '{family}'; this "
                "node pack implements 'tts' and 'vc' only."
            )
        return str(family)

    if not model_name:
        raise ValueError(
            "No Chatterbox model was selected. Pick a voice pack from the model "
            "dropdown."
        )
    return "tts"


def build_cache_key(
    model_name: str,
    attention_mode: str,
    dtype_str: str,
    device: Optional[torch.device],
) -> str:
    """The process-wide identity of one loaded pack.

    Built from the **resolved** attention mode and dtype, never from the
    requested ones, so a workflow asking for ``sage`` on a machine that falls
    back to ``sdpa`` shares the cache entry of a workflow that asked for
    ``sdpa`` directly instead of loading the same weights twice under two keys.

    ``client_id`` is deliberately absent (see the module docstring).
    """
    device_key = str(device) if device is not None else "auto"
    return f"{model_name}_attn_{attention_mode}_dtype_{dtype_str}_dev_{device_key}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _resolve_turbo_variant(model_name: str) -> Optional[str]:
    """``"turbo"``/``"nano"`` when ``model_name`` names a Turbo-family pack.

    The answer comes from the registry entry when one exists -- an official
    Turbo-family entry, or a local one the scan registered per variant -- and
    from the pack directory's t3 checkpoint otherwise, for community folders
    the scan has not seen. Registry first is what makes the shared-family
    folder work: a directory holding both t3 checkpoints cannot answer "which
    member" from its contents, but each dropdown entry names its own.

    An alias is resolved to its canonical name **before** the registry lookup,
    so the old spelling of a renamed pack answers with the entry that
    describes it -- and never degrades to what the directory happens to hold.
    A folder carrying only Turbo's weights still answers "nano" for the Nano
    alias, because the registry knows which member the name means.
    """
    canonical = model_info.canonical_pack_name(model_name)
    try:
        pack_dir = ChatterboxLoader.resolve_pack_dir(canonical)
    except ValueError:
        return None

    entry = model_info.AVAILABLE_CHATTERBOX_MODELS.get(canonical) or {}
    architecture = entry.get("architecture")
    if architecture not in ("chatterbox_turbo", "chatterbox_nano") and os.path.isdir(pack_dir):
        architecture = model_info.infer_pack_architecture(pack_dir)
    if architecture == "chatterbox_nano":
        return "nano"
    if architecture == "chatterbox_turbo":
        return "turbo"
    return None


def _resolve_flash_pack(model_name: str) -> bool:
    """True when ``model_name`` names a Chatterbox Flash pack.

    Same precedence as the Turbo helper: the registry entry's architecture
    answers first (official entry, or the local one the scan registered),
    the pack directory's t3 checkpoint second -- a flash dir is only ever its
    own pack, so contents alone are unambiguous.
    """
    try:
        pack_dir = ChatterboxLoader.resolve_pack_dir(model_name)
    except ValueError:
        return False

    entry = model_info.AVAILABLE_CHATTERBOX_MODELS.get(model_name) or {}
    if entry.get("architecture") == "chatterbox_flash":
        return True
    return os.path.isdir(pack_dir) and (
        model_info.infer_pack_architecture(pack_dir) == "chatterbox_flash"
    )


def load_chatterbox_model(
    model_name: str,
    device: str = "auto",
    dtype: str = DTYPE_AUTO,
    attention_mode: str = "sdpa",
    force_reload: bool = False,
    client_id: Optional[str] = None,
) -> tuple[ChatterboxPatcher, Any, Optional[Any]]:
    """Load, or reuse, the pack ``model_name``.

    Args:
        model_name: Pack name from the model dropdown.
        device: Device option value (see
            :func:`modules.device_utils.get_device_options`).
        dtype: Dtype widget value; ``"auto"`` lets ComfyUI decide.
        attention_mode: Attention widget value, resolved against this machine.
        force_reload: Build a fresh pack even when one is cached.
        client_id: Websocket id for progress. Stamped onto the patcher per call;
            **not** part of the cache key.

    Returns:
        ``(patcher, tts_model, vc_model)``.

    Raises:
        RuntimeError: The pack could not be loaded.
    """
    # An alias is the pack it now names. The cache identity, the handler's
    # name and the directory have to be the canonical ones, or the same pack
    # would be loaded twice under two spellings.
    model_name = model_info.canonical_pack_name(model_name)

    # 1. Resolve the attention mode -- before the cache key. A key built from
    #    the unresolved request would not match the key this same pack already
    #    holds under its resolved mode.
    actual_attention_mode = resolve_attention_mode(attention_mode, False)

    # 2. Resolve devices and dtype.
    if device == DEVICE_CPU:
        load_device = torch.device(DEVICE_CPU)
        offload_device = torch.device(DEVICE_CPU)
    else:
        load_device = get_torch_device(device)
        offload_device = get_offload_device()
    target_dtype = resolve_dtype(dtype, load_device)

    # 2b. Turbo-family packs (Turbo and its Nano member) load through their
    #     own entry point: no VC half, no classic checkpoint checks, and a
    #     cache entry of their own. The TTS node routes here so one node
    #     serves the whole family. The entry point's own eviction sweep
    #     clears the classic pack this node may still hold, the same way the
    #     classic path below clears a Turbo one.
    if _resolve_flash_pack(model_name):
        patcher, tts_model = load_chatterbox_entry_point(
            model_name,
            entry_point="flash",
            variant="flash",
            family=FAMILY_FLASH,
            device=device,
            dtype=dtype,
            attention_mode=attention_mode,
            force_reload=force_reload,
            client_id=client_id,
        )
        return patcher, tts_model, None

    turbo_variant = _resolve_turbo_variant(model_name)
    if turbo_variant is not None:
        patcher, tts_model = load_chatterbox_entry_point(
            model_name,
            entry_point="turbo",
            variant=turbo_variant,
            family=FAMILY_TURBO,
            device=device,
            dtype=dtype,
            attention_mode=attention_mode,
            force_reload=force_reload,
            client_id=client_id,
        )
        return patcher, tts_model, None

    # 3. The cache key.
    cache_key = build_cache_key(
        model_name, actual_attention_mode, get_dtype_str(target_dtype), load_device
    )

    # 4. Evict the previous pack -- BEFORE any allocation below. On a machine
    #    that holds either pack but not both, this ordering is the difference
    #    between a working load and a CUDA OOM.
    evict_if_changed(FAMILY_TTS, cache_key, (CHATTERBOX_PATCHER_CACHE,))

    # 5-6. Build the handler and the patcher.
    if cache_key not in CHATTERBOX_PATCHER_CACHE or force_reload:
        pack_dir = ChatterboxLoader.resolve_pack_dir(model_name)
        model_handler = ChatterboxModelHandler(
            model_name,
            attention_mode=actual_attention_mode,
            dtype_str=dtype,
            model_path=pack_dir,
        )
        model_handler.cache_key = cache_key

        patcher_cls = select_patcher_class(
            None, load_device, legacy_cls=ChatterboxPatcher
        )
        patcher = patcher_cls(
            model_handler,
            attention_mode=actual_attention_mode,
            load_device=load_device,
            offload_device=offload_device,
            size=model_handler.size,
            dtype=target_dtype,
        )
        CHATTERBOX_PATCHER_CACHE[cache_key] = patcher
        logging.debug(
            "[Chatterbox TTS] Created a patcher for '%s' (attn=%s, dtype=%s, "
            "estimated %d bytes)",
            model_name,
            actual_attention_mode,
            get_dtype_str(target_dtype),
            model_handler.size,
        )

    patcher = CHATTERBOX_PATCHER_CACHE[cache_key]

    # Per-call, never at construction: a cached patcher must carry the *current*
    # caller's id, not the first one's.
    patcher.client_id = client_id

    # 7. Place it.
    load_to_device(patcher)

    tts_model = getattr(patcher.model, "tts_model", None)
    vc_model = getattr(patcher.model, "vc_model", None)
    if tts_model is None and vc_model is None:
        raise RuntimeError(
            f"The Chatterbox voice pack '{model_name}' could not be loaded. "
            "Check the log above for the download or build failure."
        )

    return patcher, tts_model, vc_model


def force_offload_model(
    patcher: ChatterboxPatcher,
    model_name: str,
    warm: bool = False,
) -> None:
    """Take a pack out of VRAM on demand.

    Args:
        patcher: The live patcher.
        model_name: Pack name, for the log line.
        warm: Keep the tensors on the intermediate device for a fast reload
            instead of dropping them.
    """
    logging.info(
        "[Chatterbox TTS] Force offloading voice pack '%s'%s...",
        model_name,
        " (warm)" if warm else "",
    )
    if patcher.is_loaded:
        if warm:
            patcher.unpatch_model(unpatch_weights=True, warm=True)
        else:
            patcher.unpatch_model(unpatch_weights=True, destroy=True)

    model_management = _get_model_management()
    try:
        model_management.unload_all_models()
    except Exception as exc:
        logging.debug("[Chatterbox TTS] unload_all_models failed: %s", exc)
    gc.collect()
    try:
        model_management.soft_empty_cache()
    except Exception as exc:
        logging.debug("[Chatterbox TTS] soft_empty_cache failed: %s", exc)
    logging.info("[Chatterbox TTS] Force offload of '%s' complete.", model_name)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def _tts_family(model: Any) -> str:
    """Which TTS family ``model`` belongs to: ``"flash"``, ``"turbo"`` or
    ``"classic"``.

    The dispatch key is the class name, exactly as it has always been: the
    vendored package is never imported by this module, and the family decides
    the sampler, the chunk semantics and whether the square-bracket tags are
    the model's own vocabulary.
    """
    name = type(model).__name__
    if name == "ChatterboxFlashTTS":
        return "flash"
    if name == "ChatterboxTurboTTS":
        return "turbo"
    return "classic"


def _chunk_seed(seed: int, index: int) -> int:
    """The seed for chunk ``index`` of a run.

    ``0`` is the widget's "random" convention and stays ``0``, so every chunk
    is sampled from fresh noise. Any other seed is re-derived per chunk
    (``seed + index``): one noise sample replayed for every chunk gives every
    chunk the same prosody -- the audible artefact chunking is supposed to
    avoid -- while the run as a whole stays reproducible, because the same
    script and the same seed still produce the same chunk waveforms.
    """
    if not seed:
        return 0
    return (int(seed) + int(index)) % (2 ** 64)


def _log_plan_notices(notices: tuple[str, ...]) -> None:
    """Say once per run what the engine could not honour.

    A notice is never an error and never changes the audio: it is the engine
    admitting that a tag in the script is not going to do what it looks like
    it does -- a Turbo tag on a pack without the Turbo vocabulary, a pause
    that is parsed but never rendered, a language that is parsed but never
    switched. Dropping them on the floor is what made the feature look
    broken, so each one is logged as a warning, once per run.
    """
    for notice in notices:
        logging.warning("[Chatterbox TTS] %s", notice)


def _generate_one_chunk(
    model: Any,
    chunk: str,
    *,
    family: str,
    audio_prompt: Optional[dict],
    exaggeration: float,
    temperature: float,
    cfg_weight: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
    seed: int,
    max_new_tokens: int,
    flow_cfg_scale: float,
    use_watermark: bool,
    top_k: int,
    norm_loudness: bool,
    num_steps: int,
    cfg_scale: float,
) -> tuple[torch.Tensor, int]:
    """Synthesise one chunk with the sampler its family uses.

    The family dispatch that used to sit at the top of :func:`generate_audio`
    lives here now, so the chunk loop above it can call the same code once
    per chunk. A Flash pack is a block-diffusion decoder with no token budget
    to count; a Turbo pack has no ``max_new_tokens`` either, takes the top-k
    sampler and keeps its square-bracket tags in the text for the tokenizer
    to speak; everything else is the classic sampler.
    """
    if family == "flash":
        return generate_flash_audio(
            model,
            chunk,
            audio_prompt=audio_prompt,
            exaggeration=exaggeration,
            temperature=temperature,
            seed=seed,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
        )

    if family == "turbo":
        return generate_turbo_audio(
            model,
            chunk,
            audio_prompt=audio_prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            # Turbo ignores exaggeration for sampling -- ``hp.emotion_adv`` is
            # False on this family (``src/chatterbox/tts_turbo.py:259``) -- and
            # upstream warns about every non-zero value, once per chunk.
            exaggeration=0.0,
            norm_loudness=norm_loudness,
            seed=seed,
            use_watermark=use_watermark,
        )

    interrupt_error = interrupt_exception_class()
    set_seed(seed)
    progress_bar = _build_progress_bar(max_new_tokens)

    try:
        # The voice prompt is handed over in memory: preprocess_comfy_audio
        # downmixes, scrubs and resamples it to the 24 kHz the model
        # conditions on, and the vendored loader takes a waveform as-is
        # (``load_audio_at``, ``src/chatterbox/models/dsp.py``). ``None`` when
        # no clip is connected, which the vendor answers with the pack's
        # built-in voice. Inside the ``try`` so a malformed clip still closes
        # the progress bar, the way the temp-file write it replaces did.
        prompt = preprocess_comfy_audio(audio_prompt, target_sr=S3GEN_SAMPLE_RATE)
        try:
            with torch.no_grad():
                waveform = model.generate(
                    chunk,
                    audio_prompt_path=prompt,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                    pbar=progress_bar,
                    max_new_tokens=max_new_tokens,
                    flow_cfg_scale=flow_cfg_scale,
                    watermark=bool(use_watermark),
                )
        except interrupt_error:
            # FIRST, and before `except Exception`. The interrupt derives
            # from BaseException, so this clause is unreachable from the
            # generic handler -- and a cancelled run is not a failed run:
            # re-raising here is what actually cancels it.
            logging.info(
                "[Chatterbox TTS] Generation was interrupted; cancelling the run."
            )
            raise
        except Exception:
            logging.exception(
                "[Chatterbox TTS] Generation failed for a %d-character text.",
                len(chunk),
            )
            raise
    finally:
        progress_bar.finish()

    if waveform is None or waveform.numel() == 0:
        raise ValueError(
            "Chatterbox produced no audio for this text. Shorten it, raise "
            "'max_new_tokens', or pick a different voice pack."
        )

    result = waveform.detach().to(device="cpu", dtype=torch.float32)
    if result.dim() == 1:
        result = result.unsqueeze(0)
    return result, S3GEN_SAMPLE_RATE


@vendored_chatterbox
def generate_audio(
    model: Any,
    text: str,
    audio_prompt: Optional[dict] = None,
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
    repetition_penalty: float = 1.2,
    min_p: float = 0.05,
    top_p: float = 1.0,
    seed: int = 0,
    max_new_tokens: int = 1000,
    flow_cfg_scale: float = 0.7,
    use_watermark: bool = False,
    top_k: int = 1000,
    norm_loudness: bool = True,
    num_steps: int = 10,
    cfg_scale: float = 1.0,
    *,
    enable_chunking: bool = True,
    max_chars_per_chunk: int = 400,
    chunk_combination_method: str = "auto",
    silence_between_chunks_ms: int = 100,
    crash_protection_template: str = "",
) -> tuple[torch.Tensor, int]:
    """Synthesise ``text`` with a loaded Chatterbox TTS model.

    Turbo-family packs (``ChatterboxTurboTTS`` -- Turbo and Nano) are
    dispatched to :func:`generate_turbo_audio`: their sampler is a top-k /
    top-p pair with no CFG, so ``cfg_weight`` and ``min_p`` are ignored for
    them, and ``top_k``/``norm_loudness`` only mean anything there. The
    dispatch is by model class name -- the vendored package is never imported
    by this module.

    **Every family goes through the same text engine and the same chunk
    loop.** The script is parsed once by :func:`modules.text_processing.plan_text`
    -- which is what resolves ``[Alice]``, keeps Turbo's ``[laugh]`` tags for
    the tokenizer and cuts the script into chunks -- and each chunk is then
    synthesised with its own model call and joined with
    :func:`combine_audio_chunks`. The chunks are what make a long script
    speakable: one call per chunk means one token budget per chunk, so a
    script of any length is spoken in full instead of running out of tokens
    half way through.

    Args:
        model: The ``ChatterboxTTS`` instance from :func:`load_chatterbox_model`.
        text: The text to speak. Must be non-blank.
        audio_prompt: Optional ComfyUI ``AUDIO`` dict to clone the voice from.
            Downmixed, scrubbed and resampled to 24 kHz in memory, then handed
            to the model as a waveform.
        exaggeration: Expressiveness.
        temperature: Sampling temperature.
        cfg_weight: Classifier-free guidance weight.
        repetition_penalty: Token repetition penalty.
        min_p: Nucleus-sampling floor.
        top_p: Nucleus-sampling ceiling.
        seed: Seed; ``0`` means random. A non-zero seed is re-derived per
            chunk (``seed + chunk_index``), so the run stays reproducible
            while the chunks do not all share one noise sample.
        max_new_tokens: Token budget **per chunk**. 25 tokens is roughly one
            second.
        flow_cfg_scale: Flow-matching CFG scale for the mel decoder.
        use_watermark: Whether to apply the PERTH watermark. Off by default
            because ``perth`` is an optional dependency; the vendored models
            take ``watermark=False`` rather than needing a stub object.
        enable_chunking: Whether to cut the script into chunks at all. Off
            means one model call for the whole script.
        max_chars_per_chunk: Soft ceiling on one chunk, in characters.
        chunk_combination_method: ``auto`` / ``concatenate`` /
            ``silence_padding`` / ``crossfade``. ``auto`` resolves by the
            length of the whole script.
        silence_between_chunks_ms: The gap between chunks when the method is
            ``silence_padding``.
        crash_protection_template: Padding template for short segments.
            Empty by default: the TTS node has no such widget, and turning it
            on here would change every existing user's audio.

    Returns:
        ``(waveform[1, T] float32 on CPU, sample_rate)``.

    Raises:
        ValueError: ``text`` is blank, the script held nothing but tags, or
            the model returned no audio.
    """
    if not text or not text.strip():
        raise ValueError(
            "Chatterbox needs some text to speak. The text input was empty."
        )

    family = _tts_family(model)

    # One plan for every family. On a Turbo pack the square-bracket tags are
    # kept verbatim for the tokenizer; on every other pack they stay literal
    # text and are reported as a notice, so nothing is swallowed silently.
    plan = plan_text(
        text,
        turbo_tags=(family == "turbo"),
        crash_protection_template=crash_protection_template,
        enable_chunking=enable_chunking,
        max_chars_per_chunk=max_chars_per_chunk,
        chunk_combination_method=chunk_combination_method,
        silence_between_chunks_ms=silence_between_chunks_ms,
    )
    _log_plan_notices(plan.notices)

    if not plan.text.strip():
        raise ValueError(
            "Chatterbox needs some text to speak. The script held nothing but "
            "tags, so there is nothing left to say."
        )

    chunks = plan.chunks or (plan.text,)
    logging.info(
        "[Chatterbox TTS] Synthesising %d chunk(s) of %d characters, joined "
        "with '%s'.",
        len(chunks),
        plan.char_count,
        plan.chunk_method,
    )

    waveforms: list[torch.Tensor] = []
    sample_rate = S3GEN_SAMPLE_RATE
    for index, chunk in enumerate(chunks):
        waveform, sample_rate = _generate_one_chunk(
            model,
            chunk,
            family=family,
            audio_prompt=audio_prompt,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            seed=_chunk_seed(seed, index),
            max_new_tokens=max_new_tokens,
            flow_cfg_scale=flow_cfg_scale,
            use_watermark=use_watermark,
            top_k=top_k,
            norm_loudness=norm_loudness,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
        )
        waveforms.append(waveform)

    joined = combine_audio_chunks(
        waveforms,
        sample_rate,
        method=plan.chunk_method,
        silence_between_chunks_ms=silence_between_chunks_ms,
    )

    logging.info(
        "[Chatterbox TTS] Generated %.2f s of audio at %d Hz.",
        joined.shape[-1] / float(sample_rate),
        sample_rate,
    )
    return joined, sample_rate


@vendored_chatterbox
def convert_voice_audio(
    model: Any,
    audio: dict,
    target_voice_audio: Optional[dict] = None,
    n_timesteps: int = 10,
    temperature: float = 1.0,
    flow_cfg_scale: float = 0.7,
    use_watermark: bool = False,
) -> tuple[torch.Tensor, int]:
    """Convert ``audio`` into the voice described by ``target_voice_audio``.

    Args:
        model: The ``ChatterboxVC`` instance from :func:`load_chatterbox_model`.
        audio: The source speech, as a ComfyUI ``AUDIO`` dict.
        target_voice_audio: The target voice's timbre. ``None`` uses the pack's
            built-in reference (``conds.pt``).
        n_timesteps: Diffusion steps.
        temperature: Temperature of the initial latent noise.
        flow_cfg_scale: Flow-matching CFG scale.
        use_watermark: Whether to apply the PERTH watermark.

    Returns:
        ``(waveform[1, T] float32 on CPU, sample_rate)``.

    Raises:
        ValueError: The source audio is empty, or the model returned nothing.
    """
    if not audio or audio.get("waveform") is None or audio["waveform"].numel() == 0:
        raise ValueError(
            "Chatterbox voice conversion needs source audio. Connect an AUDIO "
            "input carrying the speech to convert."
        )

    interrupt_error = interrupt_exception_class()
    progress_bar = _build_progress_bar(n_timesteps)

    try:
        # Two rates, on purpose. The vendored ``generate`` loads the *source*
        # at the tokenizer's 16 kHz (``load_audio_at(audio, S3_SR)``,
        # ``vc.py``), and the *target* voice through ``set_target_voice`` at
        # S3Gen's 24 kHz -- the only dual-rate call site in the tree.
        # ``target`` is ``None`` when no target clip is connected, which the
        # vendor answers with the pack's own ``conds.pt``. Both are prepped
        # inside the ``try`` so the progress bar is closed on a malformed
        # clip, the way the temp-file writes they replace were.
        source = preprocess_comfy_audio(
            audio, target_sr=S3_TOKENIZER_SAMPLE_RATE, name="source audio"
        )
        target = preprocess_comfy_audio(
            target_voice_audio, target_sr=S3GEN_SAMPLE_RATE, name="target voice"
        )
        try:
            with torch.no_grad():
                waveform = model.generate(
                    source,
                    target_voice_path=target,
                    n_timesteps=n_timesteps,
                    pbar=progress_bar,
                    temperature=temperature,
                    flow_cfg_scale=flow_cfg_scale,
                    watermark=bool(use_watermark),
                )
        except interrupt_error:
            logging.info(
                "[Chatterbox TTS] Voice conversion was interrupted; "
                "cancelling the run."
            )
            raise
        except Exception:
            logging.exception("[Chatterbox TTS] Voice conversion failed.")
            raise
    finally:
        progress_bar.finish()

    if waveform is None or waveform.numel() == 0:
        raise ValueError(
            "Chatterbox voice conversion produced no audio. Check that the "
            "source clip actually contains speech."
        )

    result = waveform.detach().to(device="cpu", dtype=torch.float32)
    if result.dim() == 1:
        result = result.unsqueeze(0)
    return result, S3GEN_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Other entry points: multilingual and Turbo (S8.1 / S8.2)
# ---------------------------------------------------------------------------


def variant_cache_stem(model_name: str, variant: str) -> str:
    """The cache-key stem for one *variant* of an entry point.

    Two collisions this exists to prevent:

    * **Turbo against Nano** -- the same module, the same directory layout, two
      disjoint weight sets. Without the variant in the key the second load is
      served the first one's weights, and the user gets the wrong model with no
      error anywhere.
    * **multilingual v2 against v3** -- one directory, three T3 checkpoints.

    ``build_cache_key`` keeps its four-argument signature on purpose (a test
    pins it), so the variant is folded into the *name* rather than passed as a
    fifth key input.
    """
    if not variant:
        return model_name
    return f"{model_name}__{variant}"


def load_chatterbox_entry_point(
    model_name: str,
    *,
    entry_point: str,
    variant: str,
    family: str,
    device: str = "auto",
    dtype: str = DTYPE_AUTO,
    attention_mode: str = "sdpa",
    force_reload: bool = False,
    client_id: Optional[str] = None,
) -> tuple[ChatterboxPatcher, Any]:
    """Load, or reuse, one multilingual / Turbo pack.

    :func:`load_chatterbox_model` states its seven ordering steps in prose and
    then performs them. They are not restated here: the three entry points differ
    only in which loader runs at step 5 and which family is evicted at step 4,
    so this function performs the same seven steps against the other two.

    Args:
        model_name: Pack name from the dropdown.
        entry_point: ``"multilingual"`` or ``"turbo"``.
        variant: The resolved T3 checkpoint filename, or ``"nano"`` / ``"turbo"``.
        family: Registry family to evict against (:data:`FAMILY_MTL` /
            :data:`FAMILY_TURBO`).
        device: Device option value.
        dtype: Dtype widget value.
        attention_mode: Attention widget value, resolved against this machine.
        force_reload: Build a fresh pack even when one is cached.
        client_id: Websocket id for progress; stamped per call, never keyed.

    Returns:
        ``(patcher, model)``.

    Raises:
        RuntimeError: The pack could not be loaded.
    """
    # An alias is the pack it now names -- and the cache key below is the
    # pack's identity, so it has to be the canonical spelling.
    model_name = model_info.canonical_pack_name(model_name)

    actual_attention_mode = resolve_attention_mode(attention_mode, False)

    if device == DEVICE_CPU:
        load_device = torch.device(DEVICE_CPU)
        offload_device = torch.device(DEVICE_CPU)
    else:
        load_device = get_torch_device(device)
        offload_device = get_offload_device()
    target_dtype = resolve_dtype(dtype, load_device)

    stem = variant_cache_stem(model_name, variant)
    cache_key = build_cache_key(
        stem, actual_attention_mode, get_dtype_str(target_dtype), load_device
    )
    evict_if_changed(family, cache_key, (CHATTERBOX_PATCHER_CACHE,))

    if cache_key not in CHATTERBOX_PATCHER_CACHE or force_reload:
        model_handler = ChatterboxModelHandler(
            model_name,
            attention_mode=actual_attention_mode,
            dtype_str=dtype,
            # The pack directory is the user's selection; the variant-folded
            # stem is a cache key and must never name a folder. Deriving the
            # path from the stem sent a Turbo download into 'X__nano' next to
            # the pack the user actually picked.
            model_path=ChatterboxLoader.resolve_pack_dir(model_name),
            load_vc=False,
            entry_point=entry_point,
            variant=variant,
        )
        model_handler.cache_key = cache_key

        patcher_cls = select_patcher_class(
            None, load_device, legacy_cls=ChatterboxPatcher
        )
        patcher = patcher_cls(
            model_handler,
            attention_mode=actual_attention_mode,
            load_device=load_device,
            offload_device=offload_device,
            size=model_handler.size,
            dtype=target_dtype,
        )
        CHATTERBOX_PATCHER_CACHE[cache_key] = patcher

    patcher = CHATTERBOX_PATCHER_CACHE[cache_key]
    patcher.client_id = client_id
    load_to_device(patcher)

    model = getattr(patcher.model, "tts_model", None)
    if model is None:
        raise RuntimeError(
            f"The Chatterbox '{entry_point}' pack '{model_name}' could not be "
            "loaded. Check the log above for the download or build failure."
        )
    return patcher, model


def _normalise_waveform(waveform: Any) -> torch.Tensor:
    """``[1, T]`` float32 on CPU, whatever shape the model returned.

    Raises:
        ValueError: The model returned nothing, or an empty tensor. Both mean
            the same thing to a user -- the text was not spoken -- and both are
            worth a message rather than a downstream shape error.
    """
    if waveform is None:
        raise ValueError(
            "Chatterbox produced no audio for this text. Shorten it, raise "
            "'max_new_tokens', or pick a different voice pack."
        )
    result = waveform.detach().to(device="cpu", dtype=torch.float32)
    if result.dim() == 1:
        result = result.unsqueeze(0)
    if result.numel() == 0:
        raise ValueError(
            "Chatterbox produced no audio for this text. Shorten it, raise "
            "'max_new_tokens', or pick a different voice pack."
        )
    return result


@vendored_chatterbox
def generate_mtl_audio(
    model: Any,
    text: str,
    language_id: str,
    audio_prompt: Optional[dict] = None,
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    cfg_weight: float = 0.5,
    repetition_penalty: float = 1.2,
    min_p: float = 0.05,
    top_p: float = 1.0,
    seed: int = 0,
    use_watermark: bool = False,
) -> tuple[torch.Tensor, int]:
    """Synthesise ``text`` with a loaded ``ChatterboxMultilingualTTS``.

    **The language id is passed as a keyword, never spliced into ``text``.** The
    vendored tokenizer prepends it itself (``MTLTokenizer.text_to_tokens`` does
    ``clean_text + lang_id`` before encoding), so a node that also inserted it
    would produce ``<de><de>Hallo`` -- a doubled prefix that is not a valid token
    sequence, and which fails deep inside the T3 with a message about tokens
    rather than about the text box.

    Args:
        model: The ``ChatterboxMultilingualTTS`` instance.
        text: The text to speak, already crash-protected and tag-resolved.
        language_id: One of the 23 ISO codes in
            :data:`modules.constants.MTL_LANGUAGES`.
        audio_prompt: Optional ComfyUI ``AUDIO`` dict to clone the voice from.
        exaggeration: Expressiveness. Inert on v3 weights -- see the node tooltip.
        temperature: Sampling temperature.
        cfg_weight: Classifier-free guidance weight.
        repetition_penalty: Token repetition penalty.
        min_p: Nucleus-sampling floor.
        top_p: Nucleus-sampling ceiling.
        seed: Seed; ``0`` means random.
        use_watermark: Whether to apply the PERTH watermark.

    Returns:
        ``(waveform[1, T] float32 on CPU, sample_rate)``.
    """
    if not text or not text.strip():
        raise ValueError(
            "Chatterbox needs some text to speak. The text input was empty."
        )
    # Upstream would hit `assert self.conds is not None` deep inside its
    # generate() -- a bare AssertionError that names no fix. Five of the six
    # Single Language Packs ship no conds.pt, so this is the common case for
    # them, not an edge case.
    if audio_prompt is None and getattr(model, "conds", None) is None:
        raise ValueError(
            "This Chatterbox pack ships no built-in voice -- connect a "
            "reference audio clip to the node's audio prompt input."
        )

    set_seed(seed)
    prompt = preprocess_comfy_audio(audio_prompt, target_sr=S3GEN_SAMPLE_RATE)
    with torch.no_grad():
        waveform = model.generate(
            text,
            language_id=language_id,
            audio_prompt_path=prompt,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            watermark=bool(use_watermark),
        )
    return _normalise_waveform(waveform), S3GEN_SAMPLE_RATE


@vendored_chatterbox
def generate_turbo_audio(
    model: Any,
    text: str,
    audio_prompt: Optional[dict] = None,
    temperature: float = 0.8,
    top_k: int = 1000,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    exaggeration: float = 0.0,
    norm_loudness: bool = True,
    seed: int = 0,
    use_watermark: bool = False,
) -> tuple[torch.Tensor, int]:
    """Synthesise ``text`` with a loaded ``ChatterboxTurboTTS`` (Turbo or Nano).

    Turbo's sampler is a top-k / top-p pair with no CFG: ``cfg_weight`` and
    ``min_p`` do not exist on this family, and upstream logs a warning whenever
    either is non-zero. This function therefore does not accept them at all --
    offering a widget the model ignores is how a user spends an afternoon on a
    knob with no effect.

    Args:
        model: The ``ChatterboxTurboTTS`` instance.
        text: The text to speak, with Turbo's square-bracket tags left in place.
        audio_prompt: Optional ComfyUI ``AUDIO`` dict to clone the voice from.
        temperature: Sampling temperature (0.05 - 2.0 on this family).
        top_k: Candidate cut-off (1 - 5000).
        top_p: Nucleus-sampling ceiling.
        repetition_penalty: Token repetition penalty.
        exaggeration: Reference-conditioning intensity. Turbo ignores it for
            sampling and upstream warns when it is non-zero.
        norm_loudness: Normalise the reference clip. Skipped with the vendored
            warning when ``pyloudnorm`` is absent -- see ``src/PATCHES.md`` P-026.
        seed: Seed; ``0`` means random.
        use_watermark: Whether to apply the PERTH watermark.

    Returns:
        ``(waveform[1, T] float32 on CPU, sample_rate)``.
    """
    if not text or not text.strip():
        raise ValueError(
            "Chatterbox needs some text to speak. The text input was empty."
        )

    set_seed(seed)
    prompt = preprocess_comfy_audio(audio_prompt, target_sr=S3GEN_SAMPLE_RATE)
    with torch.no_grad():
        waveform = model.generate(
            text,
            audio_prompt_path=prompt,
            temperature=temperature,
            top_k=int(top_k),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            exaggeration=exaggeration,
            norm_loudness=bool(norm_loudness),
            watermark=bool(use_watermark),
        )
    return _normalise_waveform(waveform), S3GEN_SAMPLE_RATE


@vendored_chatterbox
def generate_flash_audio(
    model: Any,
    text: str,
    audio_prompt: Optional[dict] = None,
    exaggeration: float = 0.5,
    temperature: float = 0.8,
    seed: int = 0,
    num_steps: int = 10,
    cfg_scale: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """Synthesise ``text`` with a loaded ``ChatterboxFlashTTS``.

    Flash is a block-diffusion decoder: there is no top-k / min-p surface and
    no ``conds.pt`` ships in its repo -- the reference clip is the only voice
    source, so a prompt is mandatory and conditioning is rebuilt per call
    exactly like the classic entry point does. ``num_steps`` trades speed for
    quality (upstream default 10); ``cfg_scale`` is the block-diffusion
    guidance weight (upstream default 1.0, i.e. no guidance). The remaining
    upstream kwargs (``backend``, ``time_shift_tau``, ...) stay at their
    paper defaults; the engine backend can still be forced with the
    ``CHATTERBOX_FLASH_ENGINE`` environment variable.
    """
    if audio_prompt is None:
        raise ValueError(
            "Chatterbox Flash ships no built-in voice -- connect a reference "
            "audio clip to the node's audio prompt input."
        )

    set_seed(seed)
    prompt = preprocess_comfy_audio(audio_prompt, target_sr=S3GEN_SAMPLE_RATE)
    model.prepare_conditionals(prompt, exaggeration=exaggeration)
    waveform = model.generate(
        text,
        exaggeration=exaggeration,
        temperature=temperature,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
    )
    return _normalise_waveform(waveform), S3GEN_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Iterative and chunked voice conversion (S8.6)
# ---------------------------------------------------------------------------


class RefinementCache:
    """The intermediate iterations of one conversion, capped at five.

    Iterative conversion is a fold: pass *n* is pass *n-1* run again. Asking for
    six passes when five are already computed should cost one conversion, not
    six; asking for three again should cost nothing.

    The cap is not a memory optimisation, it is the documented behaviour: the
    reference implementation keeps five and its own tooltip warns that more
    passes distort the result, so retaining a sixth would be holding audio nobody
    should be able to select.

    Attributes:
        depth: How many iterations are kept per job.
    """

    def __init__(self, depth: int = REFINEMENT_CACHE_DEPTH) -> None:
        self.depth = int(depth)
        self._iterations: dict[str, list[dict]] = {}

    def get(self, key: str) -> list[dict]:
        """The stored iterations for ``key`` (a copy, oldest first)."""
        return list(self._iterations.get(key, ()))

    def put(self, key: str, iterations: list[dict]) -> None:
        """Store ``iterations``, keeping only the newest :attr:`depth`."""
        self._iterations[key] = list(iterations[-self.depth :])

    def clear(self) -> None:
        """Forget everything. Exposed for tests and for an explicit reset."""
        self._iterations.clear()

    def iteration_counts(self) -> dict[str, int]:
        """How many iterations are held, per job.

        The composed cache key is an implementation detail callers do not hold --
        it folds in the audio shape and the sampling settings -- so this is how a
        caller (or a test) checks that the cap is actually being applied rather
        than assuming the key it passed in is the key used.
        """
        return {key: len(value) for key, value in self._iterations.items()}

    def __len__(self) -> int:
        return len(self._iterations)


def _refinement_cache_key(
    cache_key: Optional[str],
    audio: Optional[dict],
    n_timesteps: int,
    temperature: float,
    flow_cfg_scale: float,
) -> str:
    """A stable identity for one conversion job.

    The source audio's *shape* stands in for its content. Hashing megabytes of
    waveform on every run would cost more than the conversion the cache saves,
    and the guarantee that matters -- within one run, a repeated request returns
    the cached iteration -- does not depend on the bytes being identical.
    """
    shape = "none"
    if isinstance(audio, dict) and audio.get("waveform") is not None:
        shape = str(tuple(audio["waveform"].shape))
    return (
        f"{cache_key or 'adhoc'}|{shape}|{int(n_timesteps)}|{temperature}|"
        f"{flow_cfg_scale}"
    )


def iterative_convert_voice(
    model: Any,
    audio: dict,
    *,
    target_voice_audio: Optional[dict] = None,
    n_timesteps: int = 10,
    temperature: float = 1.0,
    flow_cfg_scale: float = 0.7,
    use_watermark: bool = False,
    refinement_passes: int = 1,
    cache: Optional[RefinementCache] = None,
    cache_key: Optional[str] = None,
    single_pass: Optional[Any] = None,
) -> tuple[torch.Tensor, int]:
    """Convert ``audio`` into the target voice, optionally refining it N times.

    Pass 1 is the ordinary single conversion. Pass *n* feeds pass *n-1*'s output
    back in, so each pass moves the audio a little further toward the target
    timbre -- and a little further from the source, which is why the widget stops
    at 30 while its own tooltip recommends 5.

    The passes are **deterministic**: nothing is re-seeded between them, each one
    depends only on the previous one, and the cache key carries every input that
    could change the result. That is what makes returning a cached iteration
    safe rather than merely convenient.

    The watermark is applied on the final pass only. Watermarking every pass and
    then re-converting the watermarked audio is how a long run degrades into an
    unintelligible buzz.

    Args:
        model: The ``ChatterboxVC`` instance.
        audio: The source speech, as a ComfyUI ``AUDIO`` dict.
        target_voice_audio: The target voice's timbre. ``None`` uses the pack's
            built-in reference.
        n_timesteps: Diffusion steps, applied to **every** pass.
        temperature: Temperature of the initial latent noise.
        flow_cfg_scale: Flow-matching CFG scale.
        use_watermark: Whether to apply the PERTH watermark.
        refinement_passes: How many passes to run. Clamped to at least 1.
        cache: An optional :class:`RefinementCache` to reuse iterations in.
        cache_key: Identity of this job within ``cache``.
        single_pass: The one-pass conversion to call each time.
            :func:`convert_voice_audio` unless a caller injects its own -- the
            node passes the name it imported, so a test that substitutes one
            conversion in the node module reaches the refinement path too
            instead of silently bypassing it.

    Returns:
        ``(waveform[1, T] float32 on CPU, sample_rate)``.

    Raises:
        ValueError: The source audio is empty, or every pass produced nothing.
    """
    passes = max(1, int(refinement_passes))
    convert = single_pass or convert_voice_audio
    key = _refinement_cache_key(
        cache_key, audio, n_timesteps, temperature, flow_cfg_scale
    )
    stored = cache.get(key) if cache is not None else []

    current_audio = audio
    result: Optional[torch.Tensor] = None
    completed = 0

    for iteration in stored:
        result = iteration["audio"]
        completed += 1
        if completed >= passes:
            break
        current_audio = iteration["source"]

    while completed < passes:
        waveform, sample_rate = convert(
            model,
            current_audio,
            target_voice_audio=target_voice_audio,
            n_timesteps=n_timesteps,
            temperature=temperature,
            flow_cfg_scale=flow_cfg_scale,
            use_watermark=use_watermark,
        )
        result = waveform
        current_audio = {
            "waveform": waveform.unsqueeze(0).to(dtype=torch.float32),
            "sample_rate": sample_rate,
        }
        completed += 1
        if cache is not None:
            stored.append({"audio": result, "source": current_audio})
            cache.put(key, stored)

    if result is None:
        raise ValueError(
            "Chatterbox voice conversion produced no audio. Check that the "
            "source clip actually contains speech."
        )
    return result, S3GEN_SAMPLE_RATE


def split_audio_for_conversion(
    waveform: torch.Tensor,
    sample_rate: int,
    max_chunk_duration: float,
    chunk_method: str = "smart",
    search_window: float = 0.25,
    frame_ms: int = 20,
) -> list[torch.Tensor]:
    """Cut ``waveform`` into chunks of at most ``max_chunk_duration`` seconds.

    OOM protection: a long recording converts as one diffusion problem, and the
    activation memory grows with the mel length. Cutting the source first bounds
    it. ``0`` disables chunking and the clip comes back whole.

    Args:
        waveform: ``[C, T]`` or ``[B, C, T]`` audio.
        sample_rate: Hz.
        max_chunk_duration: The ceiling, in seconds. ``0`` disables.
        chunk_method: ``"smart"`` cuts as close to silence as it can near the
            ideal boundary -- a boundary in a pause is inaudible, a boundary
            through a vowel is a click. ``"fixed"`` cuts at exact intervals,
            which is cheaper and which the reference implementation documents as
            making the joins more audible.
        search_window: How far from the ideal boundary ``"smart"`` may look, as
            a fraction of the chunk length.
        frame_ms: Analysis frame size for the silence search.

    Returns:
        The chunks, in order. Always at least one; never empty.
    """
    mono = downmix_to_mono(waveform)
    total = int(mono.shape[-1])

    if not max_chunk_duration or float(max_chunk_duration) <= 0:
        return [mono]
    limit = int(float(max_chunk_duration) * float(sample_rate))
    if limit < 1 or total <= limit:
        return [mono]

    frame = max(1, int(sample_rate) * int(frame_ms) // 1000)
    usable = (total // frame) * frame
    flat = mono.reshape(-1)[:usable]
    # Per-frame RMS, computed by hand: the mel spectrogram's own machinery is an
    # order of magnitude more than choosing a cut location needs.
    energy = (
        flat.float().reshape(-1, frame).pow(2).mean(dim=1).sqrt()
        if flat.numel()
        else torch.zeros(1)
    )

    chunks: list[torch.Tensor] = []
    start = 0
    while total - start > limit:
        cut = start + limit
        if chunk_method == "smart" and energy.numel() > 1:
            radius = max(1, int(limit * float(search_window) // frame))
            first = max(0, int(cut // frame) - radius)
            last = min(int(energy.shape[-1]), int(cut // frame) + radius + 1)
            if last > first:
                cut = int((first + int(torch.argmin(energy[first:last]))) * frame)
        cut = max(start + 1, min(cut, total - 1))
        chunks.append(mono[..., start:cut])
        start = cut
    chunks.append(mono[..., start:])
    return [chunk for chunk in chunks if chunk.numel() > 0] or [mono]


def combine_audio_chunks(
    chunks: list[torch.Tensor],
    sample_rate: int,
    method: str = "concatenate",
    silence_between_chunks_ms: int = 100,
    crossfade_ms: int = 10,
) -> torch.Tensor:
    """Join synthesised chunks with the mode ``chunk_combination_method`` names.

    Args:
        chunks: The chunk waveforms, in order.
        sample_rate: Hz.
        method: ``concatenate`` (hard join), ``silence_padding`` (a real gap, so
            the boundary reads as a pause rather than a cut), or ``crossfade``
            (overlap the tail and head with a linear ramp, so the join is
            continuous).
        silence_between_chunks_ms: The gap for ``silence_padding``.
        crossfade_ms: The overlap for ``crossfade``.

    Returns:
        One ``[C, T]`` waveform. A single chunk is returned untouched.

    Raises:
        ValueError: ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("There are no audio chunks to join.")
    if len(chunks) == 1:
        return chunks[0]

    if method == "silence_padding":
        gap_samples = max(0, int(sample_rate * int(silence_between_chunks_ms) / 1000))
        gap = torch.zeros(
            chunks[0].shape[:-1] + (gap_samples,),
            dtype=chunks[0].dtype,
            device=chunks[0].device,
        )
        parts: list[torch.Tensor] = []
        for position, chunk in enumerate(chunks):
            parts.append(chunk)
            if position != len(chunks) - 1:
                parts.append(gap)
        return torch.cat(parts, dim=-1)

    if method == "crossfade":
        overlap = max(
            1,
            min(
                int(sample_rate * int(crossfade_ms) / 1000),
                *(int(c.shape[-1]) for c in chunks),
            ),
        )
        joined = chunks[0]
        for chunk in chunks[1:]:
            tail = joined[..., -overlap:]
            head = chunk[..., :overlap]
            ramp = torch.linspace(
                0.0, 1.0, overlap, dtype=joined.dtype, device=joined.device
            )
            blended = tail * (1.0 - ramp) + head * ramp
            joined = torch.cat(
                [joined[..., :-overlap], blended, chunk[..., overlap:]], dim=-1
            )
        return joined

    return torch.cat(chunks, dim=-1)
