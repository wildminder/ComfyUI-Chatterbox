"""Bidirectional signature and compatibility checks on the vendored tree.

The ComfyUI node passes a set of kwargs that upstream v0.1.7 does not have
(``pbar``, ``flow_cfg_scale``, ``max_new_tokens``, ``n_timesteps``,
``temperature``), and upstream v0.1.7 added a set the node does not know about
(``n_cfm_timesteps``, ``noised_mels``, ``skip_vocoder``, ``speech_token_lens``,
``drop_invalid_tokens``). Both directions have to hold: re-syncing must neither
break the node nor shadow what upstream added.

**Why AST and not ``import``.** ``conftest.py`` registers a ``MagicMock`` for
every dotted name under ``src/chatterbox`` at collection time, so the vendor can
never be imported here -- and weakening that mock to make these tests work would
drag ``perth`` (absent) and the whole model stack into collection. Instead each
module is parsed and its ``def`` nodes are executed in isolation to recover a real
signature: a stub module supplies the decorators and globals the ``def`` line
touches, so ``inspect.signature`` reports exactly what a caller would see. No
third-party import ever happens.

The ``diffusers``-import checks that used to live here (which pinned the *remaining*
surface) moved to ``tests/test_diffusers_compat.py`` once the dependency was
removed outright rather than trimmed.
"""

from __future__ import annotations

import ast
import copy
import inspect
import os
import sys
from types import ModuleType
from typing import Any

import pytest

from conftest import PKG_ALIAS, ROOT_DIR

SRC_DIR = os.path.join(ROOT_DIR, "src")
VENDOR_DIR = os.path.join(SRC_DIR, "chatterbox")

#: Kwargs the ComfyUI nodes pass that upstream does not have.
COMFY_ONLY_KWARGS: dict[str, tuple[str, ...]] = {
    "chatterbox/tts.py::ChatterboxTTS.generate": ("pbar", "max_new_tokens", "flow_cfg_scale"),
    "chatterbox/vc.py::ChatterboxVC.generate": (
        "n_timesteps",
        "pbar",
        "temperature",
        "flow_cfg_scale",
    ),
    "chatterbox/models/t3/t3.py::T3.inference": ("pbar", "max_new_tokens"),
    "chatterbox/models/s3gen/flow.py::CausalMaskedDiffWithXvec.inference": (
        "pbar",
        "temperature",
        "flow_cfg_scale",
    ),
    "chatterbox/models/s3gen/flow_matching.py::ConditionalCFM.forward": (
        "pbar",
        "flow_cfg_scale",
    ),
    "chatterbox/models/s3gen/flow_matching.py::ConditionalCFM.solve_euler": ("pbar",),
    "chatterbox/models/s3gen/flow_matching.py::CausalConditionalCFM.forward": (
        "pbar",
        "flow_cfg_scale",
    ),
    "chatterbox/models/s3gen/s3gen.py::S3Token2Mel.forward": (
        "pbar",
        "temperature",
        "flow_cfg_scale",
    ),
    "chatterbox/models/s3gen/s3gen.py::S3Token2Wav.forward": (
        "pbar",
        "temperature",
        "flow_cfg_scale",
    ),
    "chatterbox/models/s3gen/s3gen.py::S3Token2Wav.flow_inference": (
        "pbar",
        "temperature",
        "flow_cfg_scale",
    ),
    "chatterbox/models/s3gen/s3gen.py::S3Token2Wav.inference": (
        "pbar",
        "temperature",
        "flow_cfg_scale",
    ),
}

#: Kwargs upstream v0.1.7 added that the patch layer must not shadow.
UPSTREAM_ONLY_KWARGS: dict[str, tuple[str, ...]] = {
    "chatterbox/models/s3gen/s3gen.py::S3Token2Mel.forward": (
        "n_cfm_timesteps",
        "speech_token_lens",
        "noised_mels",
    ),
    "chatterbox/models/s3gen/s3gen.py::S3Token2Wav.forward": (
        "n_cfm_timesteps",
        "noised_mels",
        "skip_vocoder",
        "speech_token_lens",
    ),
    "chatterbox/models/s3gen/s3gen.py::S3Token2Wav.flow_inference": (
        "n_cfm_timesteps",
        "speech_token_lens",
    ),
    "chatterbox/models/s3gen/s3gen.py::S3Token2Wav.inference": (
        "n_cfm_timesteps",
        "speech_token_lens",
        "drop_invalid_tokens",
    ),
    "chatterbox/models/t3/t3.py::T3.inference": ("stop_on_eos", "num_return_sequences"),
}

COMFY_TARGETS = sorted(COMFY_ONLY_KWARGS)
UPSTREAM_TARGETS = sorted(UPSTREAM_ONLY_KWARGS)
ALL_TARGETS = sorted(set(COMFY_ONLY_KWARGS) | set(UPSTREAM_ONLY_KWARGS))


def _split_target(target: str) -> tuple[str, str]:
    rel_file, _, qualified = target.partition("::")
    return rel_file, qualified


# ---------------------------------------------------------------------------
# Signature recovery from source
# ---------------------------------------------------------------------------


def read_source(rel_file: str) -> str:
    path = os.path.join(SRC_DIR, rel_file.replace("/", os.sep))
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


class _Stub:
    """Callable that absorbs any decoration and reports its argument names."""

    def __call__(self, *args: Any, **kwargs: Any) -> "_Stub":  # noqa: D401
        return self

    def __getattr__(self, name: str) -> "_Stub":
        return _Stub()

    def __iter__(self):
        return iter(())


def _stub_module(name: str) -> ModuleType:
    """A module object whose every attribute is a permissive :class:`_Stub`."""
    module = ModuleType(name)
    module.__dict__["__name__"] = name
    module.__getattr__ = lambda attr: _Stub()  # PEP 562, module level
    return module


def find_def(rel_file: str, qualified: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Return the ``def`` node for ``Class.attr`` or ``func`` in one module."""
    tree = ast.parse(read_source(rel_file), filename=rel_file)
    class_name, _, attr = qualified.partition(".")
    if not attr:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == class_name:
                return node
        raise AssertionError(f"{rel_file}: no module-level function {class_name!r}")

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == attr
                ):
                    return child
            raise AssertionError(f"{rel_file}: {class_name} has no method {attr!r}")
    raise AssertionError(f"{rel_file}: no class {class_name!r}")


def _blank_annotations(args: ast.arguments) -> ast.arguments:
    """Strip every annotation so the stub never evaluates a real type name.

    ``prompt_feat: torch.Tensor`` would otherwise try to resolve ``torch`` in the
    probe namespace; the annotations carry no signature information we need.
    ``ast.unparse`` is not an option here -- an argument list is not a statement
    and does not round-trip through the parser.
    """
    args = copy.deepcopy(args)
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        arg.annotation = None
    if args.vararg is not None:
        args.vararg.annotation = None
    if args.kwarg is not None:
        args.kwarg.annotation = None
    return args


def _default_globals(node: ast.FunctionDef) -> dict[str, _Stub]:
    """Stub globals for every free name a default expression references.

    ``flow_cache=torch.zeros(1, 80, 0, 2)`` would otherwise need the real
    ``torch``. A permissive stand-in is enough: the default's *value* is never
    used, only the fact that a default exists.
    """
    names: set[str] = set()
    defaults = [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]
    for default in defaults:
        for sub in ast.walk(default):
            if isinstance(sub, ast.Name) and sub.id not in dir(__builtins__) + ["None"]:
                names.add(sub.id)
    return {name: _Stub() for name in names}


def signature_for(rel_file: str, qualified: str) -> inspect.Signature:
    """The signature a caller would see, recovered without importing anything.

    Only the ``def`` statement is compiled and executed: annotations are made
    inert, decorators are stripped, and free names in default expressions are
    bound to permissive stubs. The result is the bare Python signature --
    exactly the argument names, kinds and defaults ``inspect.signature`` would
    report on the real function, with no third-party import anywhere.
    """
    node = find_def(rel_file, qualified)
    stripped = ast.FunctionDef(
        name=node.name,
        args=_blank_annotations(node.args),
        body=[ast.Pass()],
        decorator_list=[],
        returns=None,
        type_comment=None,
        type_params=[],
    )
    ast.fix_missing_locations(stripped)
    module = _stub_module("__vendor_sig_probe__")
    module.__dict__.update(_default_globals(node))
    exec(compile(ast.Module(body=[stripped], type_ignores=[]), rel_file, "exec"), module.__dict__)
    return inspect.signature(getattr(module, node.name))


def rejects(signature: inspect.Signature, name: str) -> bool:
    """True when the parameter exists but cannot be passed by keyword."""
    param = signature.parameters.get(name)
    if param is None:
        return False
    return param.kind is inspect.Parameter.POSITIONAL_ONLY


# ---------------------------------------------------------------------------
# Bidirectional signatures
# ---------------------------------------------------------------------------


class TestComfySignatures:
    @pytest.mark.parametrize("target", ALL_TARGETS)
    def test_target_resolves(self, target: str) -> None:
        rel_file, qualified = _split_target(target)
        signature = signature_for(rel_file, qualified)
        assert signature.parameters, f"{target} has an empty signature"

    @pytest.mark.parametrize("target", COMFY_TARGETS)
    def test_comfy_kwargs_accepted(self, target: str) -> None:
        rel_file, qualified = _split_target(target)
        expected = COMFY_ONLY_KWARGS[target]
        signature = signature_for(rel_file, qualified)
        missing = [name for name in expected if name not in signature.parameters]
        assert not missing, f"{target} lost our kwargs: {missing}"

    @pytest.mark.parametrize("target", COMFY_TARGETS)
    def test_comfy_kwargs_are_passable_by_keyword(self, target: str) -> None:
        """The nodes pass everything by keyword; a positional-only param breaks it."""
        rel_file, qualified = _split_target(target)
        signature = signature_for(rel_file, qualified)
        positional_only = [
            name for name in COMFY_ONLY_KWARGS[target] if rejects(signature, name)
        ]
        assert not positional_only, (
            f"{target} declares {positional_only} positional-only; the nodes "
            f"pass everything by keyword"
        )

    @pytest.mark.parametrize("target", COMFY_TARGETS)
    def test_pbar_defaults_to_none(self, target: str) -> None:
        """Omitting pbar must stay valid for upstream callers."""
        if "pbar" not in COMFY_ONLY_KWARGS[target]:
            pytest.skip(f"{target} has no pbar kwarg")
        rel_file, qualified = _split_target(target)
        signature = signature_for(rel_file, qualified)
        assert signature.parameters["pbar"].default is None

    @pytest.mark.parametrize("target", COMFY_TARGETS)
    def test_pbar_defaults_to_none(self, target: str) -> None:
        """Omitting pbar must stay valid for upstream callers."""
        if "pbar" not in COMFY_ONLY_KWARGS[target]:
            pytest.skip(f"{target} has no pbar kwarg")
        rel_file, qualified = _split_target(target)
        signature = signature_for(rel_file, qualified)
        assert signature.parameters["pbar"].default is None

    @pytest.mark.parametrize("target", UPSTREAM_TARGETS)
    def test_upstream_kwargs_accepted(self, target: str) -> None:
        rel_file, qualified = _split_target(target)
        expected = UPSTREAM_ONLY_KWARGS[target]
        signature = signature_for(rel_file, qualified)
        missing = [name for name in expected if name not in signature.parameters]
        assert not missing, (
            f"{target} shadows upstream v0.1.7 kwargs {missing}; a re-sync must "
            f"keep upstream's surface intact"
        )

    def test_generate_signatures_overlap_in_the_right_way(self) -> None:
        """`temperature` exists on both, and ours keeps upstream's meaning.

        ``ChatterboxTTS.generate`` already carried ``temperature`` upstream; our
        addition must not move or retype it.
        """
        tts = signature_for("chatterbox/tts.py", "ChatterboxTTS.generate")
        assert tts.parameters["temperature"].default == 0.8
        assert tts.parameters["max_new_tokens"].default == 1000

        vc = signature_for("chatterbox/vc.py", "ChatterboxVC.generate")
        assert vc.parameters["temperature"].default == 1.0
        assert vc.parameters["n_timesteps"].default == 10

    def test_max_new_tokens_is_optional_on_t3(self) -> None:
        """Our nodes may omit it; upstream made it required."""
        t3 = signature_for("chatterbox/models/t3/t3.py", "T3.inference")
        assert t3.parameters["max_new_tokens"].default is None

    def test_pbar_defaults_to_none_everywhere(self) -> None:
        targets = [t for t in ALL_TARGETS if "pbar" in COMFY_ONLY_KWARGS.get(t, ())]
        assert len(targets) >= 8, "pbar threading regressed across the layer stack"
        for target in targets:
            rel_file, qualified = _split_target(target)
            signature = signature_for(rel_file, qualified)
            assert signature.parameters["pbar"].default is None, (
                f"{target}: pbar must default to None so upstream callers are unaffected"
            )

    def test_patch_threading_reaches_every_layer(self) -> None:
        """`pbar` must reach the innermost loop, not stop at an intermediate API."""
        layers = [
            ("chatterbox/tts.py", "ChatterboxTTS.generate"),
            ("chatterbox/models/t3/t3.py", "T3.inference"),
            ("chatterbox/models/s3gen/s3gen.py", "S3Token2Wav.inference"),
            ("chatterbox/models/s3gen/flow.py", "CausalMaskedDiffWithXvec.inference"),
            ("chatterbox/models/s3gen/flow_matching.py", "CausalConditionalCFM.forward"),
            ("chatterbox/models/s3gen/flow_matching.py", "ConditionalCFM.solve_euler"),
        ]
        for rel_file, qualified in layers:
            assert "pbar" in signature_for(rel_file, qualified).parameters, (
                f"{rel_file}::{qualified} does not thread pbar"
            )

    def test_vc_generate_threads_its_extra_kwargs(self) -> None:
        vc_source = read_source("chatterbox/vc.py")
        # Drop comment lines first: a comment may contain parens, and the
        # block is delimited by the first real closing paren.
        call_block = chr(10).join(
            line
            for line in vc_source.split("self.s3gen.inference(", 1)[1].splitlines()
            if not line.strip().startswith("#")
        )
        s3gen_call = call_block.split(")", 1)[0]
        # Upstream 0.1.7 renamed the kwarg; the node surface keeps the 0.1.2
        # ``n_timesteps`` name and maps it at the call site (see P-009).
        assert "n_cfm_timesteps=n_timesteps" in s3gen_call, (
            "vc.py must map the node-surface n_timesteps to upstream's "
            "n_cfm_timesteps at the S3Gen.inference call"
        )
        for name in ("pbar", "temperature", "flow_cfg_scale"):
            assert f"{name}={name}" in s3gen_call, (
                f"vc.py no longer forwards {name} to S3Gen.inference"
            )


# ---------------------------------------------------------------------------
# Compatibility patches
# ---------------------------------------------------------------------------


def imported_modules(rel_file: str) -> set[str]:
    """Every top-level module a file imports, from the AST (comments excluded)."""
    tree = ast.parse(read_source(rel_file), filename=rel_file)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def called_names(rel_file: str) -> set[str]:
    """Every ``<something>.<attr>(...)`` constructor call in a file."""
    tree = ast.parse(read_source(rel_file), filename=rel_file)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr:
                names.add(func.attr)
            elif isinstance(func, ast.Name):
                names.add(func.id)
    return names


class TestCompatPatches:
    """One assertion per local fix, anchored on the exact upstream text replaced."""

    def test_device_comparison_is_stringified_in_tts(self) -> None:
        source = read_source("chatterbox/tts.py")
        assert 'if str(device) in ["cpu", "mps"]:' in source
        assert 'if device in ["cpu", "mps"]:' not in source

    def test_device_comparison_is_stringified_in_vc(self) -> None:
        source = read_source("chatterbox/vc.py")
        assert 'if str(device) in ["cpu", "mps"]:' in source
        assert 'if device in ["cpu", "mps"]:' not in source

    def test_reference_clip_guards_use_is_not_not_none(self) -> None:
        """A reference clip handed in as audio must not hit a truth test.

        ``if audio_prompt_path:`` raises on a multi-element ndarray
        (``ValueError: The truth value of an array ... is ambiguous``), so
        every entry point that takes one in memory -- the four entry points
        P-039..P-042 teach to accept one -- has to test identity instead.
        The mocked unit suite never exercises this: the vendor is a
        ``MagicMock`` here, so both spellings pass every other check.

        An empty-string path stays falsy under both spellings, and the
        ``else`` branch still catches the case where nothing was passed.
        """
        audio_guards = (
            "chatterbox/tts.py",
            "chatterbox/mtl_tts.py",
            "chatterbox/tts_turbo.py",
        )
        for rel_file in audio_guards:
            source = read_source(rel_file)
            assert "if audio_prompt_path is not None:" in source, (
                f"{rel_file}: the reference-clip guard must be an identity test"
            )
            assert "if audio_prompt_path:" not in source, (
                f"{rel_file}: an ndarray reference clip raises ValueError under a truth test"
            )

        vc = read_source("chatterbox/vc.py")
        assert "if target_voice_path is not None:" in vc
        assert "if target_voice_path:" not in vc

    def test_perceiver_uses_the_modern_sdpa_api(self) -> None:
        source = read_source("chatterbox/models/t3/modules/perceiver.py")
        assert "from torch.nn.attention import SDPBackend, sdpa_kernel" in source
        assert "sdpa_kernel(backends=backends)" in source
        for backend in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"):
            assert f"SDPBackend.{backend}" in source, f"{backend} backend not enabled"

    def test_perceiver_keeps_a_pre_2_2_fallback(self) -> None:
        source = read_source("chatterbox/models/t3/modules/perceiver.py")
        assert "SUPPORTS_SDPA_KERNEL" in source
        assert "except ImportError:" in source
        assert "torch.backends.cuda.sdp_kernel(**config)" in source, (
            "the torch < 2.2 fallback was removed"
        )

    def test_perceiver_flash_config_none_is_handled(self) -> None:
        """`flash_config` is None when flash is off; ``None.get`` would crash."""
        source = read_source("chatterbox/models/t3/modules/perceiver.py")
        assert "self.flash_config or {}" in source
        flash = source.split("def flash_attention", 1)[1]
        assert ".get('enable_flash'" in flash, (
            "flash_attention must read the backend flags defensively"
        )

    def test_matcha_transformer_drops_the_lora_linear(self) -> None:
        """AST, not text: the patch rationale names the symbol it removes."""
        rel_file = "chatterbox/models/s3gen/matcha/transformer.py"
        assert "diffusers.models.lora" not in imported_modules(rel_file)
        assert "LoRACompatibleLinear" not in called_names(rel_file)

    def test_matcha_transformer_uses_plain_linear_twice(self) -> None:
        """Both upstream construction sites became ``nn.Linear``."""
        rel_file = "chatterbox/models/s3gen/matcha/transformer.py"
        source = read_source(rel_file)
        assert source.count("nn.Linear(in_features, out_features)") == 1
        assert source.count("nn.Linear(inner_dim, dim_out)") == 1

    def test_vendor_init_survives_missing_distribution_metadata(self) -> None:
        """S1.5: the package is vendored, so ``version()`` must not be fatal."""
        source = read_source("chatterbox/__init__.py")
        tree = ast.parse(source)
        guarded = False
        assigned = False
        for node in tree.body:
            if not isinstance(node, ast.Try):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and any(
                    getattr(target, "id", None) == "__version__" for target in sub.targets
                ):
                    assigned = True
                    guarded = True
                if isinstance(sub, ast.ExceptHandler):
                    guarded = True
        assert assigned, "__version__ is no longer assigned inside a try/except"
        assert guarded, "the version lookup is not guarded; it will raise on import"
        assert 'version("chatterbox-tts")' in source

    def test_fp16_flag_is_a_plain_attribute(self) -> None:
        source = read_source("chatterbox/models/s3gen/flow.py")
        assert "self.fp16 = False" in source
        assert "if self.fp16:" in source

    def test_trt_estimator_path_exists_and_is_locked(self) -> None:
        source = read_source("chatterbox/models/s3gen/flow_matching.py")
        assert "import threading" in source
        assert "self.lock = threading.Lock()" in source
        assert "def forward_estimator(" in source
        assert "execute_v2(" in source
        assert "with self.lock:" in source

    def test_solve_euler_routes_through_the_estimator_wrapper(self) -> None:
        source = read_source("chatterbox/models/s3gen/flow_matching.py")
        solve = source.split("def solve_euler", 1)[1].split("def forward_estimator", 1)[0]
        assert "self.forward_estimator(" in solve
        assert "self.estimator.forward(" not in solve, (
            "solve_euler bypasses forward_estimator, so a TensorRT estimator "
            "would never be used"
        )
        assert "pbar.update(1)" in solve

    def test_flow_cfg_scale_overrides_the_configured_rate(self) -> None:
        source = read_source("chatterbox/models/s3gen/flow_matching.py")
        causal = source.split("class CausalConditionalCFM", 1)[1]
        assert "self.inference_cfg_rate = flow_cfg_scale" in causal, (
            "the causal solver must apply the per-call CFG override"
        )

    def test_40ms_padding_helper_exists_in_both_entry_points(self) -> None:
        for rel_file in ("chatterbox/tts.py", "chatterbox/vc.py"):
            source = read_source(rel_file)
            assert "def _pad_wav_to_40ms_multiple(" in source
            assert "S3_TOKEN_RATE" in source
            assert source.count("_pad_wav_to_40ms_multiple(") == 2, (
                f"{rel_file}: helper defined but not called"
            )


class TestUpstreamFixes:
    """Fixes that arrive with v0.1.7 and must not be reverted by a re-sync.

    ``VoiceEncoder.embeds_from_wavs`` is deliberately absent: P-049 replaced
    that file's librosa resampler, and with it the soxr ``kaiser_fast`` choice
    the upstream row recorded -- a file the patch table claims as locally
    edited cannot also be adopted verbatim, and the resampler the row qualified
    no longer exists in the tree. Recorded in ``src/PATCHES.md`` instead.
    """

    def test_campplus_disables_memory_efficient_checkpointing(self) -> None:
        source = read_source("chatterbox/models/s3gen/s3gen.py")
        assert "CAMPPlus(" in source and "memory_efficient=False" in source

    def test_s3gen_silence_token_constant_exists(self) -> None:
        source = read_source("chatterbox/models/s3gen/const.py")
        assert "S3GEN_SIL = 4299" in source

    def test_make_pad_mask_casts_lengths_to_long(self) -> None:
        source = read_source("chatterbox/models/s3gen/utils/mask.py")
        body = source.split("def make_pad_mask", 1)[1]
        assert "lengths = lengths.long()" in body

    def test_llama_config_uses_sdpa(self) -> None:
        source = read_source("chatterbox/models/t3/llama_configs.py")
        assert 'attn_implementation="sdpa"' in source
        assert "eager" not in source

    def test_intmeanflow_helper_shipped(self) -> None:
        assert os.path.isfile(
            os.path.join(VENDOR_DIR, "models", "s3gen", "utils", "intmeanflow.py")
        )


class TestVendorIsNeverImportedForReal:
    def test_mock_is_still_in_place(self) -> None:
        from unittest.mock import MagicMock

        for name in (
            "chatterbox",
            "chatterbox.tts",
            "chatterbox.vc",
            "chatterbox.models.t3.t3",
        ):
            aliased = f"{PKG_ALIAS}.src.{name}"
            assert isinstance(sys.modules.get(aliased), MagicMock), (
                f"{aliased} was imported for real; the harness must keep the "
                "vendor mocked"
            )

    def test_sig_probe_leaves_no_trace(self) -> None:
        """The signature probe must not pollute ``sys.modules``."""
        assert "__vendor_sig_probe__" not in sys.modules


def test_heavy_deps_are_absent_unless_a_test_asked_for_them() -> None:
    """``diffusers`` and the real ``perth`` must never enter the process.

    ``s3tokenizer``/``conformer`` are deliberately imported by
    ``test_baseline_env.py`` to prove the three additions resolve, so their
    presence is not evidence of anything; what matters is that the *vendor*never
    reached for them, which ``test_mock_is_still_in_place`` covers.
    """
    assert "diffusers" not in sys.modules, (
        "diffusers was imported; it is installed but broken against "
        "huggingface_hub 1.x and nothing may pull it in"
    )

    perth = sys.modules.get("perth")
    assert perth is None, (
        "the real 'perth' distribution was imported; it is an optional "
        "dependency and the repo entry point must never fabricate a stand-in"
    )

class TestVcToS3genKwargContract:
    """The cross-file contract the mock suite could not see.

    ``ChatterboxVC.generate`` exposes the 0.1.2 ``n_timesteps`` (the frozen
    widget name) but upstream 0.1.7's ``S3Token2Wav.inference`` renamed it to
    ``n_cfm_timesteps``. B2's patch kept the old spelling at the call site,
    which only a real-weight run caught (TypeError). Pinned by AST here.
    """

    def test_generate_maps_n_timesteps_to_the_upstream_name(self) -> None:
        import ast

        source = open(os.path.join(VENDOR_DIR, "vc.py"), encoding="utf-8").read()
        assert "n_cfm_timesteps=n_timesteps" in source
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "inference"
        ]
        assert calls, "no s3gen.inference call found in vc.py"
        for call in calls:
            keywords = {kw.arg for kw in call.keywords}
            assert "n_timesteps" not in keywords, (
                "vc.py passes the removed 0.1.2 kwarg 'n_timesteps' to "
                "s3gen.inference; map it to 'n_cfm_timesteps'"
            )


class TestHalfDtypeContracts:
    """The bf16/fp16 contracts P-031/P-032 pin against a re-sync.

    A half-dtype cast must never reach the fp32-only frontends, and the fp32
    tensors they produce must be aligned with the half weights that consume
    them. The first half was found only by a real-weight bf16 run
    (``RuntimeError: Unsupported dtype BFloat16`` in ``Kaldi.fbank``); the
    second would have been the next field failure (a Linear dtype mismatch in
    ``T3CondEnc``). Pinned by AST here so a re-sync cannot quietly revert them.
    """

    @staticmethod
    def _function_source(rel_path: str, func_name: str) -> str:
        source = open(os.path.join(VENDOR_DIR, rel_path), encoding="utf-8").read()
        tree = ast.parse(source)
        segments = [
            ast.get_source_segment(source, node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        ]
        segments = [segment for segment in segments if segment]
        assert len(segments) == 1, (
            f"{func_name!r} in {rel_path}: expected exactly one definition, "
            f"found {len(segments)}"
        )
        return segments[0]

    @staticmethod
    def _class_method_source(rel_path: str, class_name: str, func_name: str) -> str:
        """Source of one method of one class, when the module has several
        definitions of the same method name (``forward`` twice in s3gen.py)."""
        source = open(os.path.join(VENDOR_DIR, rel_path), encoding="utf-8").read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in node.body:
                    if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name == func_name
                    ):
                        segment = ast.get_source_segment(source, child)
                        assert segment is not None
                        return segment
        raise AssertionError(f"{class_name}.{func_name} not found in {rel_path}")

    def test_embed_ref_feeds_the_speaker_encoder_float32(self) -> None:
        block = self._function_source("models/s3gen/s3gen.py", "embed_ref")
        assert "self.speaker_encoder.inference(ref_wav_16.float())" in block, (
            "P-031: the 16 kHz clip must reach CAMPPlus uncast -- Kaldi.fbank's "
            "FFT has no half-dtype kernel"
        )
        assert "ref_wav_16.to(dtype=self.dtype)" not in block

    def test_embed_ref_casts_the_x_vector_to_the_flow_dtype(self) -> None:
        block = self._function_source("models/s3gen/s3gen.py", "embed_ref")
        assert "embedding=ref_x_vector.to(dtype=self.dtype)" in block, (
            "P-031: the fp32 x-vector must match the flow/hift weights' dtype "
            "before conditioning consumes it"
        )

    def test_inference_hands_the_vocoder_float32(self) -> None:
        block = self._function_source("models/s3gen/s3gen.py", "inference")
        assert "output_mels = output_mels.float()" in block, (
            "P-033: the HiFT vocoder is float32-only; the mel must reach it "
            "uncast instead of being re-cast to self.dtype"
        )
        assert "output_mels.to(dtype=self.dtype)" not in block

    def test_ref_dict_casting_skips_integer_tensors(self) -> None:
        block = self._class_method_source(
            "models/s3gen/s3gen.py", "S3Token2Mel", "forward"
        )
        assert "if ref_dict[rk].is_floating_point():" in block, (
            "P-034: token ids and lengths in the ref dict are integers -- they "
            "follow the device, never the dtype (bf16 rounds 635 to 632 and "
            "cannot represent every token id)"
        )

    def test_turbo_from_local_accepts_the_nano_tokenizer_layout(self) -> None:
        block = self._class_method_source(
            "tts_turbo.py", "ChatterboxTurboTTS", "from_local"
        )
        assert 'if (ckpt_dir / "tokenizer.json").is_file()' in block, (
            "P-035: the tokenizer file must follow the pack layout -- the Nano "
            "repo ships vocab.json + merges.txt and no tokenizer.json"
        )
        assert "(*REQUIRED_CHECKPOINT_FILES, tokenizer_file," in block, (
            "P-035: the completeness check must include the layout-chosen "
            "tokenizer file next to the variant T3 checkpoint"
        )

    def test_t3_cond_enc_aligns_fp32_conditioning_to_the_projectors(self) -> None:
        block = self._function_source("models/t3/modules/cond_enc.py", "forward")
        assert (
            "cond.speaker_emb.view(-1, self.hp.speaker_embed_size).to(proj_dtype)"
            in block
        ), "P-032: the fp32 speaker embedding must be aligned with spkr_enc"
        assert "self.perceiver(cond_prompt_speech_emb.to(proj_dtype))" in block, (
            "P-032: the cond-prompt embedding must be aligned with the perceiver"
        )
        assert (
            "cond.emotion_adv.view(-1, 1, 1).to(self.emotion_adv_fc.weight.dtype)"
            in block
        ), "P-032: the fp32 emotion scalar must be aligned with emotion_adv_fc"
