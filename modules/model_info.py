"""Chatterbox model registry, filesystem discovery and feature flags.

Ported from ``ComfyUI-VoxCPM/modules/model_info.py`` (plan S4.6), keeping its
shape -- a static ``MODEL_CONFIGS`` table, an ``AVAILABLE_*`` registry with
``type``/``repo_id``/``architecture``/``sample_rate``/``size_gb`` metadata, and
``read_model_config`` / ``scan_*`` helpers -- and swapping the contents over to
the Chatterbox model family.

Two deliberate differences from the borrowed file:

* **Detection is file-based, not ``architecture``-based.** VoxCPM models carry
  a ``config.json`` with an ``architecture`` field; Chatterbox packs do not --
  a pack is a tokenizer plus safetensors blobs. So
  :func:`is_chatterbox_model_dir` looks for the tokenizer and the weights.
* **Local packs are registered, not baked in.** The registry starts with the
  official entries and gains local ones through :func:`refresh_local_models`,
  which delegates the filesystem walk to
  :mod:`modules.model_discovery` so there is exactly one cached scanner.

Nothing here imports the ComfyUI node-API package, ``comfy.*``, ``folder_paths`` or ``server``
at module scope, and no probe below ever imports the model stack: importing
``chatterbox.tts_turbo`` would drag in torch, librosa and transformers.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .constants import CHATTERBOX_FILES_TO_DOWNLOAD, CHATTERBOX_REPO_ID


__all__ = [
    "MODEL_CONFIGS",
    "AVAILABLE_CHATTERBOX_MODELS",
    "VALID_CHATTERBOX_ARCHITECTURES",
    "CHATTERBOX_WEIGHT_FILES",
    "TEXT_NORMALIZATION_AVAILABLE",
    "TURBO_AVAILABLE",
    "read_model_config",
    "describe_model_dir",
    "infer_pack_architecture",
    "pack_dir_entries",
    "official_pack_names",
    "PACK_ALIASES",
    "canonical_pack_name",
    "pack_spellings",
    "NANO_ENTRY_SUFFIX",
    "is_chatterbox_model_dir",
    "scan_chatterbox_models",
    "scan_custom_model_path",
    "register_local_model",
    "unregister_local_model",
    "refresh_local_models",
    "get_available_models",
    "refresh_feature_flags",
]

# ── Official model configurations ─────────────────────────────────────
#
# ``repo_id``/``architecture``/``sample_rate``/``size_gb`` are what the
# download UI renders and what the loader keys off. ``files`` is the pack's file
# list, used when downloading that specific model rather than the default pack.
#
# Sample rates are not a guess: every Chatterbox model synthesises through
# S3Gen at 24 kHz (``src/chatterbox/models/s3gen/const.py::S3GEN_SR``).

# The multilingual family (and the Single Language Pack finetunes) ship the
# grapheme map the vendored ``MTLTokenizer`` loads instead of a fast tokenizer.
MTL_TOKENIZER_FILE = "grapheme_mtl_merged_expanded_v1.json"

# ── Shared assets ─────────────────────────────────────────────────────
# Byte-identical across every repo that ships them (verified 2026-10-07 by
# sha256 and HF LFS OIDs), so they are fetched at most once into the
# reserved `_shared` store and hardlinked into each pack that needs them.
# Downloads never fetch them into a pack directly.
SHARED_ASSET_DIR_NAME = "_shared"
SHARED_ASSET_FILES = frozenset(
    {
        "ve.safetensors",
        "s3gen.safetensors",
        "s3gen_v3.safetensors",
        MTL_TOKENIZER_FILE,
    }
)


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # The pack a saved workflow names by default. Monolingual English, voice
    # cloning, the source of every other official pack. Named after the
    # upstream repository it comes from -- it used to be keyed by this
    # repository's own internal id, and the old spelling is an alias below.
    "chatterbox": {
        "repo_id": CHATTERBOX_REPO_ID,
        "architecture": "chatterbox_tts",
        "sample_rate": 24000,
        "size_gb": 2.7,
        "files": list(CHATTERBOX_FILES_TO_DOWNLOAD),
        "description": "Original Chatterbox, English, voice cloning.",
    },
    # The multilingual family: same repo, same voice encoder, different T3
    # weights. v2 is upstream's default; v3 is the newer, more stable one.
    #
    # These three are **checkpoints, not packs**: they are T3 variants that
    # live inside the classic repository next to the default pack's weights,
    # and the multilingual loader picks one of them out of a pack directory
    # rather than downloading a directory of its own. So they are never
    # offered in the pack dropdown -- ``is_pack: False`` says so, and
    # :func:`official_pack_names` skips them. Every other entry is a pack.
    "T3_v1": {
        "repo_id": CHATTERBOX_REPO_ID,
        "is_pack": False,
        "architecture": "chatterbox_multilingual",
        "sample_rate": 24000,
        "size_gb": 2.7,
        "files": ["t3_cfg.safetensors", "ve.safetensors", "s3gen.safetensors", MTL_TOKENIZER_FILE, "conds.pt"],
        "description": "Original single-language T3 weights, used by the multilingual loader as its v1.",
    },
    "T3_v2": {
        "repo_id": CHATTERBOX_REPO_ID,
        "is_pack": False,
        "architecture": "chatterbox_multilingual",
        "sample_rate": 24000,
        "size_gb": 2.7,
        "files": ["t3_mtl23ls_v2.safetensors", "ve.safetensors", "s3gen.safetensors", MTL_TOKENIZER_FILE, "conds.pt"],
        "description": "23-language multilingual T3 (upstream default).",
    },
    "T3_v3": {
        "repo_id": CHATTERBOX_REPO_ID,
        "is_pack": False,
        "architecture": "chatterbox_multilingual",
        "sample_rate": 24000,
        "size_gb": 2.7,
        "files": ["t3_mtl23ls_v3.safetensors", "ve.safetensors", "s3gen.safetensors", MTL_TOKENIZER_FILE, "conds.pt"],
        "description": "23-language multilingual T3, newer checkpoint: better similarity, fewer hallucinations.",
    },
    # The Single Language Pack: six dedicated finetunes of the multilingual
    # V3 family, one per priority language (2026-06 release). Each ships its
    # own per-language t3 checkpoint ("t3_checkpoint"); the s3gen_v3 decoder,
    # grapheme tokenizer and voice encoder are byte-identical across all six
    # and come from the shared store (SHARED_ASSET_FILES). Only zh-cmn
    # bundles a conds.pt (built-in voice); the rest have none, so a reference
    # clip is required.
    "chatterbox-multilingual-zh-cmn": {
        "repo_id": "ResembleAI/Chatterbox-Multilingual-zh-cmn",
        "architecture": "chatterbox_multilingual",
        "sample_rate": 24000,
        "size_gb": 2.8,
        "files": ["t3_zh_cmn.safetensors", "s3gen_v3.safetensors", MTL_TOKENIZER_FILE, "conds.pt"],
        "t3_checkpoint": "t3_zh_cmn.safetensors",
        "language": "zh",
        "description": "Single Language Pack: Chinese (zh). Ships a built-in voice.",
    },
    "chatterbox-multilingual-es-mx-latam": {
        "repo_id": "ResembleAI/Chatterbox-Multilingual-es-mx-latam",
        "architecture": "chatterbox_multilingual",
        "sample_rate": 24000,
        "size_gb": 2.6,
        "files": ["t3_es_mx_latam.safetensors", "s3gen_v3.safetensors", MTL_TOKENIZER_FILE],
        "t3_checkpoint": "t3_es_mx_latam.safetensors",
        "language": "es",
        "description": "Single Language Pack: Latin-American Spanish (es). Needs a reference clip.",
    },
    "chatterbox-multilingual-es-es": {
        "repo_id": "ResembleAI/Chatterbox-Multilingual-es-es",
        "architecture": "chatterbox_multilingual",
        "sample_rate": 24000,
        "size_gb": 2.6,
        "files": ["t3_es_es.safetensors", "s3gen_v3.safetensors", MTL_TOKENIZER_FILE],
        "t3_checkpoint": "t3_es_es.safetensors",
        "language": "es",
        "description": "Single Language Pack: Spain Spanish (es). Needs a reference clip.",
    },
    "chatterbox-multilingual-pt-br": {
        "repo_id": "ResembleAI/Chatterbox-Multilingual-pt-br",
        "architecture": "chatterbox_multilingual",
        "sample_rate": 24000,
        "size_gb": 2.6,
        "files": ["t3_pt_br.safetensors", "s3gen_v3.safetensors", MTL_TOKENIZER_FILE],
        "t3_checkpoint": "t3_pt_br.safetensors",
        "language": "pt",
        "description": "Single Language Pack: Brazilian Portuguese (pt). Needs a reference clip.",
    },
    "chatterbox-multilingual-pt-pt": {
        "repo_id": "ResembleAI/Chatterbox-Multilingual-pt-pt",
        "architecture": "chatterbox_multilingual",
        "sample_rate": 24000,
        "size_gb": 2.6,
        "files": ["t3_pt_pt.safetensors", "s3gen_v3.safetensors", MTL_TOKENIZER_FILE],
        "t3_checkpoint": "t3_pt_pt.safetensors",
        "language": "pt",
        "description": "Single Language Pack: Portugal Portuguese (pt). Needs a reference clip.",
    },
    "chatterbox-multilingual-hi": {
        "repo_id": "ResembleAI/Chatterbox-Multilingual-hi",
        "architecture": "chatterbox_multilingual",
        "sample_rate": 24000,
        "size_gb": 2.6,
        "files": ["t3_hi.safetensors", "s3gen_v3.safetensors", MTL_TOKENIZER_FILE],
        "t3_checkpoint": "t3_hi.safetensors",
        "language": "hi",
        "description": "Single Language Pack: Hindi (hi). Needs a reference clip.",
    },
    # The Turbo family, in ONE folder. Turbo is the family's name upstream
    # (ResembleAI/chatterbox-turbo); Nano is its smaller member
    # (ResembleAI/chatterbox-nano). The two repos publish the same file list
    # apart from the t3 checkpoint, so both members live in the same
    # ``chatterbox-turbo`` directory and the dropdown entry -- not the
    # folder -- picks the weights. ``folder`` is that download target; it
    # differs from the entry name for Nano.
    #
    # Neither repo ships a ``tokenizer.json``: both publish the GPT2-style
    # HuggingFace layout (``vocab.json`` + ``merges.txt`` plus the config
    # files), which is what the lists below ask for. Declaring a file
    # upstream does not publish makes the pack report itself as never
    # downloaded, and the missing-file message name a file that is not
    # supposed to be there.
    "chatterbox-turbo": {
        "repo_id": "ResembleAI/chatterbox-turbo",
        "folder": "chatterbox-turbo",
        "architecture": "chatterbox_turbo",
        "sample_rate": 24000,
        "size_gb": 1.4,
        "files": ["t3_turbo_v1.safetensors", "t3_turbo_v1.yaml", "ve.safetensors", "s3gen.safetensors", "s3gen_meanflow.safetensors", "vocab.json", "merges.txt", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json", "conds.pt"],
        "description": "350M English model with paralinguistic tags; lower VRAM.",
    },
    # A quarter of Turbo again; runs on CPU at realtime. Shares the Turbo
    # folder: only the t3 checkpoint is Nano's own.
    "chatterbox-nano": {
        "repo_id": "ResembleAI/chatterbox-nano",
        "folder": "chatterbox-turbo",
        "architecture": "chatterbox_nano",
        "sample_rate": 24000,
        "size_gb": 2.9,
        "files": [
            "t3_nano_v1.safetensors",
            "t3_nano_v1.yaml",
            "ve.safetensors",
            "s3gen.safetensors",
            "s3gen_meanflow.safetensors",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "added_tokens.json",
            "special_tokens_map.json",
            "conds.pt",
        ],
        "description": "Turbo-family 110M model for CPU and tight latency budgets. "
        "Downloads into the shared chatterbox-turbo folder; ships an HF tokenizer "
        "(vocab.json + merges.txt), not a tokenizer.json.",
    },
    # Block-diffusion member (Fast-dLLM-style masked decoder over the same
    # S3Gen / voice-encoder stack). Not runnable by the vendored 0.1.7 code:
    # the sampler is vendored at src/chatterbox_flash/ and imports S3Gen /
    # VoiceEncoder / T3Cond from the vendored tree. No conds.pt ships in the
    # repo, so a reference clip is required.
    "chatterbox-flash": {
        "repo_id": "ResembleAI/chatterbox-flash",
        "folder": "chatterbox-flash",
        "architecture": "chatterbox_flash",
        "sample_rate": 24000,
        "size_gb": 3.1,
        "files": ["t3_flash.safetensors", "s3gen.safetensors", "ve.safetensors", "tokenizer.json"],
        # Flash's s3gen.safetensors is a DIFFERENT decoder than the one the
        # other families share despite the same filename -- it must never be
        # resolved from, or linked into, the shared store.
        "unique_files": ["s3gen.safetensors"],
        "description": "Block-diffusion Chatterbox: fast parallel decoding. Needs a "
        "reference audio clip.",
    },
}

# Official entries, tagged so the API and the loader can tell them from local
# packs. Local entries are added by refresh_local_models().
AVAILABLE_CHATTERBOX_MODELS: Dict[str, Dict[str, Any]] = {
    name: {"type": "official", **config} for name, config in MODEL_CONFIGS.items()
}


# ── Pack names ───────────────────────────────────────────────────────
#
# A pack's name changed under us twice, and both times the old spelling has
# to stay readable or a saved workflow stops queueing: a graph carries the
# name it was saved with. Upstream renamed one Turbo-family member, and the
# classic pack's dropdown entry moved from this repository's internal id to
# the name upstream publishes. Each old spelling is therefore kept alive as
# an alias, and an alias is accepted everywhere a pack name is read -- and
# offered nowhere: the dropdown lists the current name only, and a directory
# on disk that still carries an alias is offered under the name that alias
# maps to.
PACK_ALIASES: Dict[str, str] = {
    "chatterbox-turbo-nano": "chatterbox-nano",
    "resembleai_default_voice": "chatterbox",
}


def canonical_pack_name(name: str) -> str:
    """The current name for ``name`` -- the alias table's answer, or the name.

    Call it before a name reaches the registry, a path or the offered list:
    an alias must never be looked up, resolved or offered as if it were a
    pack of its own.
    """
    return PACK_ALIASES.get(name, name)


def pack_spellings(canonical: str) -> set:
    """``canonical`` plus every alias still mapping to it.

    Used to answer "is this directory the one the canonical pack lives in?"
    across a rename, where the folder on disk can still carry the old id -- and
    to read from rather than hard-code a second copy of the legacy spelling.
    """
    spellings = {canonical}
    spellings.update(
        alias for alias, target in PACK_ALIASES.items() if target == canonical
    )
    return spellings


def official_pack_names() -> List[str]:
    """Every registry entry that is a downloadable pack, in registry order.

    This is the answer to "what can the user download?", and it is read from
    the registry rather than from a second, hand-kept list -- so a pack added
    to :data:`MODEL_CONFIGS` is offered by the dropdown without anyone
    remembering to list it in a second place.

    An entry is a pack unless it says otherwise: ``is_pack: False`` marks the
    entries that are **not** directories of their own. The ``T3_v*`` entries
    are the case in point -- T3 checkpoints that live inside the classic
    repository and are picked out of a pack directory by the multilingual
    loader, never downloaded as a pack. A missing ``is_pack`` key means "is a
    pack", so a new entry is downloadable by default.

    Only entries with a ``repo_id`` are listed: a pack without a repository
    cannot be fetched, and offering it would be a promise the loader cannot
    keep.
    """
    return [
        name
        for name, config in MODEL_CONFIGS.items()
        if config.get("is_pack", True) and config.get("repo_id")
    ]


# ── Recognised architectures ─────────────────────────────────────────

VALID_CHATTERBOX_ARCHITECTURES = (
    "chatterbox",
    "chatterbox_tts",
    "chatterbox_multilingual",
    "chatterbox_turbo",
    "chatterbox_nano",
    "chatterbox_flash",
)

# A file from this set makes a directory a model. ``conds.pt`` is deliberately
# absent: it is the smallest file in a pack and alone proves nothing.
CHATTERBOX_WEIGHT_FILES = (
    "t3_cfg.safetensors",
    "t3_mtl23ls_v2.safetensors",
    "t3_mtl23ls_v3.safetensors",
    "t3_turbo_v1.safetensors",
    "t3_nano_v1.safetensors",
    "t3_flash.safetensors",
    "ve.safetensors",
    "s3gen.safetensors",
    "s3gen_v3.safetensors",  # Single Language Pack decoder (P-036)
    "s3gen_v3.pt",  # legacy serialization, still loaded when present
    "s3gen.pt",
)

# Every Chatterbox entry point needs a tokenizer to build its T3 conditions.
CHATTERBOX_TOKENIZER_FILE = "tokenizer.json"



# ── Feature flags ────────────────────────────────────────────────────

def _probe_text_normalization() -> bool:
    """True when at least one CJK segmenter is installed.

    Chatterbox does not ship a normaliser; it delegates word segmentation for
    Japanese and Chinese to optional packages. The nodes disable text
    normalisation when neither is present, and the frontend greys the widget
    out from ``config["normalization_available"]``.
    """
    for module_name in ("pykakasi", "spacy_pkuseg"):
        try:
            if importlib.util.find_spec(module_name) is not None:
                return True
        except (ImportError, ValueError, AttributeError):
            # ValueError: a parent package is mocked and has no __spec__.
            logging.debug("[Chatterbox TTS] Could not probe optional package '%s'", module_name)
    return False


def _probe_turbo() -> bool:
    """True when the vendored source ships the Turbo entry point.

    Checked on disk rather than by import: the module imports the full model
    stack, and a presence check must cost nothing at node-load time.
    """
    package_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
        "chatterbox",
    )
    return os.path.isfile(os.path.join(package_dir, "tts_turbo.py"))


def refresh_feature_flags() -> None:
    """Recompute the feature flags in this module's namespace."""
    global TEXT_NORMALIZATION_AVAILABLE, TURBO_AVAILABLE

    TEXT_NORMALIZATION_AVAILABLE = _probe_text_normalization()
    TURBO_AVAILABLE = _probe_turbo()
    logging.debug(
        "[Chatterbox TTS] Chatterbox feature flags: normalization=%s turbo=%s",
        TEXT_NORMALIZATION_AVAILABLE, TURBO_AVAILABLE,
    )


TEXT_NORMALIZATION_AVAILABLE: bool = False
TURBO_AVAILABLE: bool = False
refresh_feature_flags()


# ── Model scanning ───────────────────────────────────────────────────

def read_model_config(model_dir: str) -> Optional[dict]:
    """Parse ``<model_dir>/config.json``.

    Returns ``None`` when the file is absent or unreadable -- Chatterbox packs
    normally ship no ``config.json`` at all, so ``None`` is the common answer
    and not a failure. Use :func:`describe_model_dir` to characterise a pack
    that has none.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logging.debug("[Chatterbox TTS] Could not read config.json in %s: %s", model_dir, exc)
        return None


def _list_entries(path: str) -> List[str]:
    try:
        return sorted(os.listdir(path))
    except OSError as exc:
        logging.debug("[Chatterbox TTS] Could not list '%s': %s", path, exc)
        return []


def _infer_architecture(entries: List[str]) -> Optional[str]:
    """Guess a pack's architecture from the weights it carries."""
    if "t3_nano_v1.safetensors" in entries:
        return "chatterbox_nano"
    if "t3_turbo_v1.safetensors" in entries:
        return "chatterbox_turbo"
    if "t3_mtl23ls_v2.safetensors" in entries or "t3_mtl23ls_v3.safetensors" in entries:
        return "chatterbox_multilingual"
    # Single Language Pack finetunes (and community refinetunes of them) keep
    # the multilingual tokenizer and s3gen but rename the t3 blob per language
    # (t3_zh_cmn.safetensors, t3_es_mx_latam.safetensors, ...). The grapheme
    # map is the family marker no other architecture carries.
    if MTL_TOKENIZER_FILE in entries:
        return "chatterbox_multilingual"
    if "t3_flash.safetensors" in entries:
        return "chatterbox_flash"
    if "t3_cfg.safetensors" in entries:
        return "chatterbox_tts"
    return None


def infer_pack_architecture(pack_dir: str) -> Optional[str]:
    """A pack directory's architecture from the t3 checkpoint it carries.

    ``"chatterbox_tts"`` / ``"chatterbox_turbo"`` / ``"chatterbox_nano"`` /
    ``"chatterbox_multilingual"``, or ``None`` when the directory is missing
    or holds no recognised t3 checkpoint -- a community rename or a half
    download answers ``None`` and the caller falls back to whatever else it
    knows. This is the loader-facing half of :func:`_infer_architecture`: the
    t3 filename is the pack's identity, never the folder it sits in.

    A folder carrying **both** Turbo-family t3 checkpoints answers
    ``chatterbox_nano`` here (the same nano-first rule
    :func:`_infer_architecture` applies) -- callers that need to tell the two
    apart must use :func:`pack_dir_entries` instead.
    """
    if not pack_dir or not os.path.isdir(pack_dir):
        return None
    return _infer_architecture(_list_entries(pack_dir))


NANO_ENTRY_SUFFIX = "-nano"


def pack_dir_entries(pack_dir: str) -> List[tuple]:
    """``(entry_name, architecture)`` pairs a pack directory offers.

    One directory is normally one pack named after itself. The exception is
    the Turbo family: the two repos share every weight except the t3
    checkpoint, so a folder holding **both** ``t3_turbo_v1.safetensors`` and
    ``t3_nano_v1.safetensors`` serves both members and is offered twice --
    under its own name (Turbo) and under the canonical Nano name. The
    synthesized name is the user's handle for the variant; the folder stays
    a single copy of the shared weights.

    The base name is canonicalised for the same reason the synthesized one
    is: both are names this module mints for the dropdown, and the dropdown
    must never show a spelling that resolves to another pack. A directory
    still carrying a pack's old id is therefore offered under the name that
    pack carries now -- which is what keeps an existing install working
    rather than showing a pack nobody else in the repository knows. The
    name/path pairing still holds because :func:`canonical_pack_name` only
    rewrites the *name*: every entry minted here points at the very
    directory it was minted from, alias-named or not.
    """
    if not pack_dir or not os.path.isdir(pack_dir):
        return []
    base = canonical_pack_name(os.path.basename(os.path.normpath(pack_dir)))
    entries = set(_list_entries(pack_dir))
    has_turbo = "t3_turbo_v1.safetensors" in entries
    has_nano = "t3_nano_v1.safetensors" in entries
    if has_turbo and has_nano:
        return [
            (base, "chatterbox_turbo"),
            (canonical_pack_name(base + NANO_ENTRY_SUFFIX), "chatterbox_nano"),
        ]
    return [(base, _infer_architecture(sorted(entries)) or "unknown")]


def describe_model_dir(model_dir: str) -> Dict[str, Any]:
    """Summarise a pack directory.

    Prefers an explicit ``config.json`` (community checkpoints carry one), and
    otherwise infers the architecture from the weight filenames. Returns a dict
    with ``name``, ``architecture`` and ``path``; ``architecture`` is
    ``"unknown"`` when neither source says anything.

    ``name`` is the canonical name, because it becomes a registry key
    (:func:`register_local_model`) and a dropdown entry, and neither may carry
    a spelling that resolves to a *different* pack.
    """
    entries = _list_entries(model_dir)
    config = read_model_config(model_dir) or {}
    architecture = config.get("architecture") or _infer_architecture(entries) or "unknown"
    description: Dict[str, Any] = {
        "name": canonical_pack_name(os.path.basename(os.path.normpath(model_dir))),
        "architecture": architecture,
        "path": model_dir,
    }
    if config:
        description["config"] = config
    return description


def is_chatterbox_model_dir(path: str) -> bool:
    """True when ``path`` is a usable Chatterbox pack directory.

    Requires both halves of a pack: the tokenizer that builds the T3
    conditions, and at least one weight blob. Either alone is not a model -- an
    empty directory, a partially cancelled download, or an unrelated
    HuggingFace checkpoint all fail one of the two checks.

    Any ``.safetensors`` counts as a weight, not only the names in
    :data:`CHATTERBOX_WEIGHT_FILES`: community checkpoints rename their blobs,
    and refusing to recognise them would hide a model the loader can actually
    load.
    """
    if not path or not os.path.isdir(path):
        return False

    config = read_model_config(path)
    if config is not None:
        architecture = config.get("architecture", "")
        if architecture and architecture not in VALID_CHATTERBOX_ARCHITECTURES:
            # An explicit config.json naming a foreign architecture settles it.
            return False

    entries = set(_list_entries(path))
    if not _has_usable_tokenizer(entries):
        return False
    return any(weight in entries for weight in CHATTERBOX_WEIGHT_FILES) or any(
        entry.endswith(".safetensors") for entry in entries
    )


def _has_usable_tokenizer(entries: set) -> bool:
    """``tokenizer.json``, the HF fast pair Nano ships, or the MTL grapheme map.

    The Nano repo has no ``tokenizer.json``: its tokenizer is the GPT2-style
    HF layout (``vocab.json`` + ``merges.txt``), which ``AutoTokenizer`` reads
    just as well. Demanding ``tokenizer.json`` outright made every Nano pack
    fail the sniff before any loader could look at it. The multilingual family
    (and the Single Language Pack finetunes) ship
    ``grapheme_mtl_merged_expanded_v1.json`` instead, which the vendored
    ``MTLTokenizer`` loads.
    """
    if CHATTERBOX_TOKENIZER_FILE in entries:
        return True
    if MTL_TOKENIZER_FILE in entries:
        return True
    return {"vocab.json", "merges.txt"}.issubset(entries)


def scan_chatterbox_models(scan_dir: str) -> List[dict]:
    """Every valid Chatterbox pack directly under ``scan_dir``.

    Returns a list of ``{"name", "architecture", "path"}`` dicts, sorted by
    name. A missing directory yields ``[]`` rather than raising: the caller is
    usually a frontend poll that must not 500 because the user has no models
    yet.
    """
    models: List[dict] = []
    if not scan_dir or not os.path.isdir(scan_dir):
        return models

    for item in _list_entries(scan_dir):
        item_path = os.path.join(scan_dir, item)
        if not os.path.isdir(item_path):
            continue
        if not is_chatterbox_model_dir(item_path):
            continue
        description = describe_model_dir(item_path)
        models.append(description)
        logging.debug(
            "[Chatterbox TTS] Found Chatterbox pack %s (architecture=%s) at %s",
            item, description["architecture"], item_path,
        )

    return models


def scan_custom_model_path(path: str) -> List[dict]:
    """Scan a user-supplied model directory.

    Warns rather than raises when the path does not exist: this is the answer
    to an interactive "browse for your models" dialog, and a wrong path is the
    user correcting themselves, not a server fault.
    """
    if not path or not os.path.isdir(path):
        logging.warning("[Chatterbox TTS] Custom model path does not exist: %s", path)
        return []

    if is_chatterbox_model_dir(path):
        # The user pointed straight at a pack rather than at its parent.
        return [describe_model_dir(path)]

    return scan_chatterbox_models(path)


# ── Local model registry ─────────────────────────────────────────────

def register_local_model(name: str, path: str, architecture: Optional[str] = None) -> None:
    """Add or replace a local pack in the registry.

    ``architecture`` overrides what the directory would infer on its own: a
    Turbo-family folder serving both members registers twice against the same
    path, and the t3-first sniff cannot tell those two entries apart.
    """
    description = describe_model_dir(path)
    description["name"] = name
    AVAILABLE_CHATTERBOX_MODELS[name] = {
        "type": "local",
        "repo_id": "",
        "architecture": architecture or description["architecture"],
        "sample_rate": 24000,
        "size_gb": 0.0,
        "path": path,
    }
    logging.debug("[Chatterbox TTS] Registered local Chatterbox pack '%s' at %s", name, path)


def unregister_local_model(name: str) -> bool:
    """Drop a local pack. Official entries are never removed."""
    entry = AVAILABLE_CHATTERBOX_MODELS.get(name)
    if entry is None or entry.get("type") != "local":
        return False
    del AVAILABLE_CHATTERBOX_MODELS[name]
    return True


def _is_official_variant_of(path: str, entry_name: str) -> bool:
    """True when ``entry_name`` is the official pack that lives in ``path``.

    The Turbo family shares one folder, so the scan mints a second entry for
    the folder's other member -- and that member has an official registry
    entry of its own, naming the same folder. Registering the local copy
    would replace that official entry (dropping its ``repo_id`` and its
    download target), and the next sweep would delete the name outright.

    A directory still carrying a pack's **old** spelling is the same case,
    which is why the folder is matched against every spelling of the entry
    and not just its current one: the folder named after the old id *is* the
    canonical pack's directory, and the official entry is the one that knows
    the repository and the download target.
    """
    config = MODEL_CONFIGS.get(entry_name)
    if not config:
        return False
    folder = config.get("folder")
    if folder:
        return folder == os.path.basename(os.path.normpath(path))
    return os.path.basename(os.path.normpath(path)) in pack_spellings(entry_name)


def refresh_local_models() -> List[dict]:
    """Re-scan the models root and mirror its packs into the registry.

    The filesystem walk is delegated to :mod:`modules.model_discovery`, which
    caches on the models root's ``st_mtime_ns`` -- so calling this on every
    frontend refresh is cheap. Official entries are preserved; local entries
    for packs that have disappeared are dropped. A Turbo-family folder serving
    both members registers under each of the names
    :func:`pack_dir_entries` offers -- except the member that has an official
    entry of its own for that very folder: there the official entry stays,
    because it is the one that knows the repository and the download target.
    The same holds for a folder still named after a pack's old id: it is the
    canonical pack's directory, so the official entry survives it too.
    """
    from . import model_discovery

    found_paths = {path for _name, path in model_discovery.get_model_pack_entries()}
    current = {
        name for name, entry in AVAILABLE_CHATTERBOX_MODELS.items()
        if entry.get("type") == "local"
    }

    found_names: set = set()
    local_names: set = set()
    for path in sorted(found_paths):
        base = os.path.basename(os.path.normpath(path))
        for entry_name, architecture in pack_dir_entries(path):
            found_names.add(entry_name)
            if entry_name != base and _is_official_variant_of(path, entry_name):
                # The synthesized name of an official pack that lives in this
                # folder: the official entry already describes it.
                continue
            register_local_model(entry_name, path, architecture=architecture)
            local_names.add(entry_name)

    for name in current - found_names:
        unregister_local_model(name)

    return [
        {"name": name, "path": str(AVAILABLE_CHATTERBOX_MODELS[name].get("path", "")), "type": "local"}
        for name in sorted(local_names & set(AVAILABLE_CHATTERBOX_MODELS))
    ]


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """The registry with local packs freshly scanned in."""
    refresh_local_models()
    return AVAILABLE_CHATTERBOX_MODELS