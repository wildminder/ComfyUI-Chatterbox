"""S8.1 -- ``ChatterboxMTLTTS``: 23 languages, a v3 default, and an additive id.

Three things here can only be asserted against the real vendored source, and each
is worth the subprocess:

* **The language table is a copy, and a copy is a bug waiting to happen.**
  ``modules/constants.py`` cannot import ``chatterbox.mtl_tts`` -- that module
  drags in librosa, transformers and the whole model stack -- so the 23 codes are
  duplicated on purpose. This file reads the real ``SUPPORTED_LANGUAGES`` in a
  child interpreter and compares it with the copy, so the two cannot drift.
* **``language_id`` must be prepended exactly once.** The vendored tokenizer
  prepends it itself; a node that also inserted it emits ``<de><de>Hallo``, which
  is not a token sequence and dies inside the T3. The assertion is on what the
  model was actually handed.
* **The new node id is additive.** ``tests/frozen_widget_snapshot.json`` is the
  V1 compatibility record and is never written to; this file asserts the frozen
  entries are still exactly what they were, and that the new ids are new.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from ComfyUI_Chatterbox.modules.constants import (
    MTL_DEFAULT_T3_MODEL,
    MTL_LANGUAGES,
    MTL_LANGUAGE_CODES,
    MTL_LANGUAGE_LOOKUP,
    MTL_T3_MODELS,
)
from ComfyUI_Chatterbox.modules.text_processing import CRASH_PROTECTION_TEMPLATE
from ComfyUI_Chatterbox.nodes.mtl_node import (
    MTL_MODEL_VERSION_DEFAULT,
    MTL_MODEL_VERSION_OPTIONS,
    ChatterboxMTLTTSNode,
)

from conftest import PKG_ALIAS, ROOT_DIR

SRC_DIR = os.path.join(ROOT_DIR, "src")

MTL_NODE_ID = "ChatterboxMTLTTS"
FROZEN_NODE_IDS = ("ChatterboxTTS", "ChatterboxVC")

#: The 23 ISO 639-1 codes upstream v0.1.7 ships. Spelled out here rather than
#: derived, so a table that loses an entry fails instead of shrinking quietly.
EXPECTED_CODES = frozenset(
    {
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it",
        "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
    }
)


class _RecordingModel:
    """A stand-in for ``ChatterboxMultilingualTTS`` that records its call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        # A pack with a built-in voice: without this the no-conds guard in
        # generate_mtl_audio refuses the call before the model is reached.
        self.conds = object()

    def generate(self, text, language_id, **kwargs):
        import torch

        self.calls.append({"text": text, "language_id": language_id, **kwargs})
        return torch.zeros((1, 2400), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_the_node_id_is_new_and_does_not_replace_anything(self) -> None:
        schema = ChatterboxMTLTTSNode.GET_SCHEMA()
        assert schema.node_id == MTL_NODE_ID
        assert MTL_NODE_ID not in FROZEN_NODE_IDS
        assert schema.outputs[0].display_name == "audio"
        assert schema.is_output_node is False

    def test_the_frozen_snapshot_is_untouched_and_still_complete(self) -> None:
        """``ChatterboxMTLTTS`` is added to the surface, never carved out of it."""
        with open(
            os.path.join(ROOT_DIR, "tests", "frozen_widget_snapshot.json"),
            encoding="utf-8",
        ) as handle:
            snapshot = json.load(handle)
        assert set(snapshot["nodes"]) == set(FROZEN_NODE_IDS), (
            "the frozen record holds exactly ChatterboxTTS and ChatterboxVC; a "
            "new node id belongs in the node list, not in this file"
        )
        assert MTL_NODE_ID not in snapshot["nodes"]

    def test_the_frozen_tts_and_vc_still_render_unchanged(self) -> None:
        from ComfyUI_Chatterbox.nodes import ChatterboxTTSNode, ChatterboxVCNode

        assert ChatterboxTTSNode.GET_SCHEMA().node_id == "ChatterboxTTS"
        assert ChatterboxVCNode.GET_SCHEMA().node_id == "ChatterboxVC"

    def test_the_schema_finalizes(self) -> None:
        schema = ChatterboxMTLTTSNode.FINALIZE_SCHEMA()
        assert schema.outputs[0].id

    def test_the_sampling_widgets_are_the_multilingual_ones(self) -> None:
        """Wider ``exaggeration``, a zero-floor ``cfg_weight``, and no ``top_k``."""
        required = ChatterboxMTLTTSNode.GET_NODE_INFO_V1()["input"]["required"]
        assert (required["exaggeration"][1]["min"], required["exaggeration"][1]["max"]) == (
            0.0,
            5.0,
        )
        assert (required["cfg_weight"][1]["min"], required["cfg_weight"][1]["max"]) == (
            0.0,
            1.0,
        )
        assert required["repetition_penalty"][1]["default"] == 1.2
        assert "min_p" in required and "top_p" in required

    def test_top_k_is_rejected_on_the_multilingual_node(self) -> None:
        """Turbo-only knob. The multilingual sampler has no top-k at all.

        A widget the model ignores is how a user spends an afternoon on a knob
        with no effect, so it is not offered rather than offered and refused.
        """
        info = ChatterboxMTLTTSNode.GET_NODE_INFO_V1()["input"]
        names = set(info["required"]) | set(info.get("optional", {}))
        assert "top_k" not in names

        import inspect

        assert "top_k" not in inspect.signature(
            ChatterboxMTLTTSNode.execute
        ).parameters, "the multilingual sampler has no top-k to pass"


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------


class TestLanguages:
    def test_there_are_exactly_twenty_three(self) -> None:
        assert len(MTL_LANGUAGES) == 23
        assert len(MTL_LANGUAGE_CODES) == 23

    def test_the_codes_are_the_documented_set(self) -> None:
        assert set(MTL_LANGUAGE_CODES) == EXPECTED_CODES
        assert set(MTL_LANGUAGES) == EXPECTED_CODES

    def test_there_is_no_vietnamese(self) -> None:
        """Vietnamese is a community finetune, not an upstream language."""
        assert "vi" not in MTL_LANGUAGES
        assert "Vietnamese" not in MTL_LANGUAGES.values()

    def test_the_dropdown_offers_every_code_and_defaults_to_english(self) -> None:
        info = ChatterboxMTLTTSNode.GET_NODE_INFO_V1()["input"]
        config = info["required"]["language"]
        assert set(config[1]["options"]) == EXPECTED_CODES
        assert config[1]["options"] == sorted(MTL_LANGUAGE_CODES)
        assert config[1]["default"] == "en"

    @pytest.mark.parametrize("code", sorted(EXPECTED_CODES))
    def test_every_language_id_is_accepted(self, code: str) -> None:
        assert ChatterboxMTLTTSNode.validate_inputs(language=code) is True

    def test_an_unknown_language_is_refused_with_a_message(self) -> None:
        result = ChatterboxMTLTTSNode.validate_inputs(language="xx")
        assert isinstance(result, str)
        assert "xx" in result

    @pytest.mark.parametrize(
        ("tag", "expected"),
        [("de", "de"), ("German", "de"), ("german", "de"), ("zh", "zh"), ("Chinese", "zh")],
    )
    def test_both_a_code_and_a_display_name_resolve(self, tag: str, expected: str) -> None:
        assert MTL_LANGUAGE_LOOKUP[tag.lower()] == expected


class TestLanguageTableMatchesTheVendoredSource:
    """The copy in ``modules/constants.py`` is pinned to the real dictionary."""

    def test_the_23_codes_are_the_ones_the_tokenizer_accepts(self) -> None:
        probe = textwrap.dedent(
            """
            import importlib.machinery, json, sys, types

            # The node reaches its vendored tree package-relatively, and the
            # checkout directory carries a hyphen -- so the child registers the
            # same alias the harness uses and imports through it.
            alias = {alias!r}
            package = types.ModuleType(alias)
            package.__path__ = [{root!r}]
            package.__package__ = alias
            package.__spec__ = importlib.machinery.ModuleSpec(alias, None, is_package=True)
            package.__spec__.submodule_search_locations = [{root!r}]
            sys.modules[alias] = package

            from {alias}.src.chatterbox.mtl_tts import SUPPORTED_LANGUAGES
            from {alias}.src.chatterbox.mtl_tts import _resolve_multilingual_t3_model
            print("__PROBE__" + json.dumps({{
                "languages": SUPPORTED_LANGUAGES,
                "default_t3": _resolve_multilingual_t3_model(None),
                "v3": _resolve_multilingual_t3_model("v3"),
            }}))
            """
        ).format(alias=PKG_ALIAS, root=ROOT_DIR)
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            cwd=ROOT_DIR,
            timeout=600,
        )
        line = next(
            (l for l in result.stdout.splitlines() if l.startswith("__PROBE__")), None
        )
        assert line is not None, (
            "the vendored multilingual entry point did not import in a child "
            f"interpreter; stdout={result.stdout[-800:]} stderr={result.stderr[-800:]}"
        )
        payload = json.loads(line[len("__PROBE__") :])
        assert payload["languages"] == MTL_LANGUAGES, (
            "modules/constants.MTL_LANGUAGES has drifted from "
            "chatterbox.mtl_tts.SUPPORTED_LANGUAGES"
        )
        # And the vendor's own default is v2 -- which is why ours is not.
        assert payload["default_t3"] == "t3_mtl23ls_v2.safetensors"
        assert payload["v3"] == "t3_mtl23ls_v3.safetensors"
        assert MTL_DEFAULT_T3_MODEL == "v3"


# ---------------------------------------------------------------------------
# model_version
# ---------------------------------------------------------------------------


class TestModelVersion:
    def test_the_dropdown_shows_resolved_filenames(self) -> None:
        assert MTL_MODEL_VERSION_OPTIONS == (
            "t3_mtl23ls_v3.safetensors",
            "t3_mtl23ls_v2.safetensors",
            "t3_23lang.safetensors",
        )
        options = ChatterboxMTLTTSNode.GET_NODE_INFO_V1()["input"]["required"][
            "model_version"
        ][1]["options"]
        assert tuple(options) == MTL_MODEL_VERSION_OPTIONS

    def test_the_default_is_v3_not_upstreams_v2(self) -> None:
        """The single most important assertion in this file.

        Upstream still resolves ``None`` to v2. A node that shipped v2 as its
        default would be silently giving users the checkpoint upstream itself
        treats as the older one, while looking identical in every screenshot.
        """
        config = ChatterboxMTLTTSNode.GET_NODE_INFO_V1()["input"]["required"][
            "model_version"
        ]
        assert config[1]["default"] == "t3_mtl23ls_v3.safetensors"
        assert MTL_MODEL_VERSION_DEFAULT == "t3_mtl23ls_v3.safetensors"
        assert config[1]["default"] == MTL_T3_MODELS["v3"]
        assert MTL_T3_MODELS["v3"] != MTL_T3_MODELS["v2"]

    @pytest.mark.parametrize("filename", MTL_MODEL_VERSION_OPTIONS)
    def test_every_offered_version_is_accepted(self, filename: str) -> None:
        assert ChatterboxMTLTTSNode.validate_inputs(model_version=filename) is True

    def test_an_unknown_version_is_refused_with_a_message(self) -> None:
        result = ChatterboxMTLTTSNode.validate_inputs(model_version="t3_nope.safetensors")
        assert isinstance(result, str)
        assert "t3_mtl23ls_v3.safetensors" in result

    def test_the_tooltip_says_that_v3_exaggeration_is_inert(self) -> None:
        tooltip = ChatterboxMTLTTSNode.GET_NODE_INFO_V1()["input"]["required"][
            "exaggeration"
        ][1].get("tooltip", "")
        assert "v3" in tooltip
        assert "no audible response" in tooltip


# ---------------------------------------------------------------------------
# Language id is prepended exactly once
# ---------------------------------------------------------------------------


class TestLanguageIdIsPrependedExactlyOnce:
    def test_the_text_reaches_the_model_without_a_language_prefix(self, monkeypatch) -> None:
        """The vendored tokenizer adds the prefix; the node must not add a second.

        ``MTLTokenizer.text_to_tokens`` does ``clean_text + lang_id`` before
        encoding. A node that also spliced ``<de>`` into the string would emit
        ``<de><de>Hallo``, which is not a token sequence and fails inside the T3
        with a message about tokens rather than about the text box.
        """
        model = _RecordingModel()
        monkeypatch.setattr(
            "ComfyUI_Chatterbox.nodes.mtl_node.load_chatterbox_entry_point",
            lambda *a, **k: (type("P", (), {"cache_key": "k"})(), model),
        )
        ChatterboxMTLTTSNode.execute(
            model_pack_name="resembleai_default_voice",
            model_version="t3_mtl23ls_v3.safetensors",
            language="de",
            text="Hallo, wie geht es dir?",
            exaggeration=0.5,
            temperature=0.8,
            cfg_weight=0.5,
            repetition_penalty=1.2,
            min_p=0.05,
            top_p=1.0,
            seed=1,
            use_watermark=False,
        )

        assert len(model.calls) == 1
        call = model.calls[0]
        assert call["language_id"] == "de"
        assert call["text"] == "Hallo, wie geht es dir?"
        assert "<de>" not in call["text"]
        assert call["text"].lower().count("hallo") == 1

    def test_the_language_id_is_a_keyword_not_part_of_the_text(self) -> None:
        import inspect

        from ComfyUI_Chatterbox.modules.generation import generate_mtl_audio

        assert "language_id" in inspect.signature(generate_mtl_audio).parameters

    def test_empty_text_returns_silence_without_loading_a_model(self, monkeypatch) -> None:
        """Blank text is silence, and it costs nothing: no load, no download.

        The loader is swapped for a recorder rather than a raiser, so what this
        asserts is the *count* -- the model was never built -- rather than "it did
        not explode". The output is then checked to be real silence, which is the
        one legitimate silence in the node contract (the other, cancellation, is
        a BaseException and never reaches here).
        """
        loads: list[dict] = []

        monkeypatch.setattr(
            "ComfyUI_Chatterbox.nodes.mtl_node.load_chatterbox_entry_point",
            lambda *a, **k: loads.append(k),
        )
        output = ChatterboxMTLTTSNode.execute(
            model_pack_name="resembleai_default_voice",
            model_version="t3_mtl23ls_v3.safetensors",
            language="en",
            text="   ",
            exaggeration=0.5,
            temperature=0.8,
            cfg_weight=0.5,
            repetition_penalty=1.2,
            min_p=0.05,
            top_p=1.0,
            seed=1,
            use_watermark=False,
        )
        assert loads == [], "an empty text box triggered a model load"
        waveform = output.result[0]["waveform"]
        assert waveform.numel() > 0
        assert float(waveform.abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------


class TestOptionalTokenizers:
    """``pykakasi`` / ``spacy-pkuseg`` are optional, and the node does not assume them.

    The degradation is upstream's: both are imported *inside* the tokenizer's
    functions and warn when absent. What this repository owns is that it does not
    break that contract -- so the assertions are about the import being lazy and
    guarded, not about a code path the mocked vendor tree cannot execute.
    """

    def test_the_normaliser_is_genuinely_absent_here(self) -> None:
        """This is the situation being guarded: neither package is installed."""
        import importlib.util

        assert importlib.util.find_spec("pykakasi") is None
        assert importlib.util.find_spec("spacy_pkuseg") is None

    def test_no_shipped_module_imports_a_normaliser_at_module_scope(self) -> None:
        import ast

        for directory in ("modules", "nodes"):
            for name in sorted(os.listdir(os.path.join(ROOT_DIR, directory))):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(ROOT_DIR, directory, name)
                with open(path, encoding="utf-8") as handle:
                    tree = ast.parse(handle.read(), filename=path)
                for node in tree.body:  # module scope only
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            assert "pykakasi" not in alias.name, path
                            assert "spacy" not in alias.name, path
                    elif isinstance(node, ast.ImportFrom):
                        assert "pykakasi" not in (node.module or ""), path

    def test_the_vendored_tokenizer_imports_the_normaliser_lazily(self) -> None:
        """The upstream guarantee the node relies on, asserted on the source.

        A module-scope ``import pykakasi`` would make
        ``chatterbox.mtl_tts`` -- and therefore the whole multilingual entry
        point -- unimportable on a machine without it, which is the failure the
        corpus says must not happen.
        """
        import ast

        path = os.path.join(SRC_DIR, "chatterbox", "models", "tokenizers", "tokenizer.py")
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)

        targets = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if "pykakasi" in alias.name or "spacy" in alias.name
        ]
        assert targets, "the tokenizer no longer mentions the normalisers at all"
        for node in targets:
            assert not any(node is child for child in tree.body), (
                f"line {node.lineno}: the optional normaliser is imported at "
                f"module scope, which makes the whole entry point unimportable "
                f"without it"
            )

    def test_the_feature_flag_reports_the_absent_normaliser_honestly(self) -> None:
        """``_probe_text_normalization`` is the live answer; the module constant is not.

        ``TEXT_NORMALIZATION_AVAILABLE`` is recomputed only by
        ``refresh_feature_flags()``, which other tests in this suite call with a
        stubbed probe, so asserting on it here would be asserting on whatever the
        last caller left behind. The probe is a pure function of what is
        installed, which is the thing worth pinning.
        """
        from ComfyUI_Chatterbox.modules.model_info import _probe_text_normalization

        assert _probe_text_normalization() is False, (
            "neither pykakasi nor spacy-pkuseg is installed, so the probe must be "
            "False; a True here would grey-out nothing and promise a normaliser"
        )

    def test_the_nodes_offer_no_normalisation_widget_that_does_nothing(self) -> None:
        """No ``use_normalization`` switch: there is no switch to make true."""
        info = ChatterboxMTLTTSNode.GET_NODE_INFO_V1()["input"]
        names = set(info["required"]) | set(info.get("optional", {}))
        assert not any("normal" in name for name in names)


# ---------------------------------------------------------------------------
# Runtime widgets and validation
# ---------------------------------------------------------------------------


class TestSharedWidgets:
    def test_the_runtime_widgets_are_present_and_follow_the_model_specific_ones(self) -> None:
        info = ChatterboxMTLTTSNode.GET_NODE_INFO_V1()
        required = list(info["input_order"]["required"])
        assert required[-4:] == [
            "device",
            "dtype",
            "attention_mode",
            "crash_protection_template",
        ]
        assert list(info["input"]["optional"]) == ["audio_prompt"], (
            "only the voice reference is a socket; every other input is a widget"
        )

    def test_auto_is_the_last_device_option(self) -> None:
        options = ChatterboxMTLTTSNode.GET_NODE_INFO_V1()["input"]["required"][
            "device"
        ][1]["options"]
        assert options[-1] == "auto"

    def test_the_crash_protection_default_is_the_documented_one(self) -> None:
        config = ChatterboxMTLTTSNode.GET_NODE_INFO_V1()["input"]["required"][
            "crash_protection_template"
        ]
        assert config[1]["default"] == CRASH_PROTECTION_TEMPLATE
        assert "{seg}" in config[1]["tooltip"]

    @pytest.mark.parametrize(
        ("dtype", "attention", "accepted"),
        [("fp32", "sage", False), ("auto", "sdpa", True), ("bf16", "sage", True)],
    )
    def test_the_shared_dtype_attention_cross_check_applies(
        self, dtype: str, attention: str, accepted: bool
    ) -> None:
        result = ChatterboxMTLTTSNode.validate_inputs(
            dtype=dtype, attention_mode=attention
        )
        assert (result is True) is accepted
