"""Frozen constants for the Chatterbox node pack.

These four values were defined in ``modules/chatterbox_handler.py``, deleted by
S3.2, and are recorded in ``tests/frozen_widget_snapshot.json``. They are the
compatibility contract: saved workflows were built against the on-disk layout
under ``models/tts/chatterbox`` and against the widget defaults recorded there,
so neither may drift without that record being updated in the same commit.
``tests/test_move_parity.py::TestCarriedForwardConstants`` compares this
module against that record.

The default pack *name* is the one entry that changed on purpose (2026-10-29):
the combo used to offer an internal id, ``resembleai_default_voice``, where
every other entry showed the name upstream publishes. It now reads
``chatterbox``, and the old spelling is carried by
:data:`modules.model_info.PACK_ALIASES`, which is what keeps a workflow saved
with it resolving to the very same directory instead of failing to queue. The
directory layout -- the half of the contract a user actually paid 2.7 GB for
-- did not move.

Nothing here imports ``folder_paths``, ``comfy.*`` or ``server``: it is a leaf
module every other module may import from freely.
"""

from __future__ import annotations

import os

__all__ = [
    "CHATTERBOX_MODEL_SUBDIR",
    "CHATTERBOX_REPO_ID",
    "CHATTERBOX_FILES_TO_DOWNLOAD",
    "DEFAULT_MODEL_PACK_NAME",
    "MTL_LANGUAGES",
    "MTL_LANGUAGE_CODES",
    "MTL_LANGUAGE_LOOKUP",
    "MTL_T3_MODELS",
    "MTL_DEFAULT_T3_MODEL",
]

# Where Chatterbox packs live, relative to a ComfyUI models root. Built with
# ``os.path.join`` so it renders with the platform separator; the snapshot
# records it platform-neutrally as the two parts below.
CHATTERBOX_MODEL_SUBDIR = os.path.join("tts", "chatterbox")

# The upstream Hugging Face repository the default voice pack is fetched from.
CHATTERBOX_REPO_ID = "ResembleAI/chatterbox"

# The complete file list of a default pack. Order is the order
# ``HFProgressTracker`` downloads them in, so it is load-bearing: the voice
# encoder is the slowest piece and the frontend renders progress per file.
CHATTERBOX_FILES_TO_DOWNLOAD = [
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt",
]

# The pack a saved workflow names when its combo widget carries no value.
# Spelled the way upstream spells the repository it comes from, so the
# dropdown reads like the rest of the pack list. The old internal id is an
# alias in ``modules.model_info.PACK_ALIASES``: an existing install keeps the
# folder it has, and a workflow saved with the old spelling still resolves.
DEFAULT_MODEL_PACK_NAME = "chatterbox"

# ---------------------------------------------------------------------------
# Multilingual (S8.1)
# ---------------------------------------------------------------------------
#
# A copy of the vendored ``chatterbox.mtl_tts.SUPPORTED_LANGUAGES``, and it has to
# be a copy: the vendored module imports the whole model stack (librosa,
# transformers, safetensors, huggingface_hub), so importing it to build a Combo's
# option list would cost a ComfyUI startup. It also has to match *exactly* --
# a language the dropdown offers but the tokenizer rejects is a node that fails
# at generation time with a message the user cannot act on. The two are pinned
# against each other by ``tests/test_mtl_node.py``, which reads the real
# vendored dict in a subprocess and compares it with the table here.
#
# 23 codes. There is no Vietnamese: the Vietnamese and Egyptian-Arabic models are
# community finetunes, surfaced as ordinary local directories (S8.8) rather than
# as languages upstream does not define.
MTL_LANGUAGES: dict[str, str] = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}

#: The 23 codes in the order the Combo offers them.
MTL_LANGUAGE_CODES: tuple[str, ...] = tuple(sorted(MTL_LANGUAGES))

#: Lower-cased code *or* display name -> code, so ``[German:Alice]`` and
#: ``[de:Alice]`` resolve to the same thing.
MTL_LANGUAGE_LOOKUP: dict[str, str] = {
    **{code: code for code in MTL_LANGUAGES},
    **{name.lower(): code for code, name in MTL_LANGUAGES.items()},
}

#: ``model_version`` Combo value -> T3 checkpoint filename.
#:
#: The Combo shows the *resolved filename*, not a bare ``v3`` label, because the
#: third-party implementations disagree about what "v3" even is and a user
#: comparing a graph against a Hugging Face listing needs the file name. Upstream
#: still resolves ``None`` to v2 (``DEFAULT_MULTILINGUAL_T3_MODEL``); we default
#: to **v3**, and the docstring on the node says why that differs.
MTL_T3_MODELS: dict[str, str] = {
    "v3": "t3_mtl23ls_v3.safetensors",
    "v2": "t3_mtl23ls_v2.safetensors",
    "v1": "t3_23lang.safetensors",
}

#: The version the multilingual node starts on. v3, deliberately not upstream's
#: v2: the reference corpus records v3 as the recommended checkpoint and notes
#: the two reference implementations disagree on the default.
MTL_DEFAULT_T3_MODEL = "v3"
