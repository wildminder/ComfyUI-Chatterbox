"""S5.6 -- ``modules/patcher.py`` and the handler contract in ``modules/loader.py``.

Two of the assertions here are named regressions, and both have to *fail*
against the code they replace:

**D3 (the VC-model leak).** V1's ``nodes.py:73`` wrote ``self.vc_model = None``
inside ``unpatch_model``, where ``self`` is the *patcher*. The attributes live
on the wrapper (``patcher.model``), so that assignment created a new attribute
on the patcher, freed nothing, and the VC weights stayed resident for the rest
of the session. :class:`TestD3WrapperRelease` asserts the wrapper is cleared --
and :meth:`TestD3WrapperRelease.test_nulling_on_the_patcher_cannot_pass_the_check`
demonstrates that writing to the patcher provably does *not* satisfy it.

**D4 (the hardcoded size).** V1 passed ``size=int(1.5 * 1024**3)`` for a pack
that is ~2.7 GB on disk. :class:`TestD4RealSize` asserts the handler measures
the real safetensors bytes, that ``ModelPatcher._load_list()`` over the handler
finds real parameters, and that ``model_size()`` reports what was measured.

Nothing here loads real weights or touches the network: the "packs" are three
``torch.nn.Linear`` layers in a class with Chatterbox's attribute names.
"""

from __future__ import annotations

import gc
import inspect
import logging
import os

import pytest
import torch

from modules import loader as loader_module
from modules import patcher as patcher_module
from modules.loader import ChatterboxLoader, ChatterboxModelHandler, pack_size_on_disk
from modules.patcher import (
    ACTIVATION_HEADROOM_BYTES,
    CHATTERBOX_PATCHER_CACHE,
    ChatterboxPatcher,
    dynamic_vram_available,
    load_to_device,
    release_wrapper_references,
    resolve_core_patcher_class,
    select_patcher_class,
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class StubPack:
    """Stands in for ``ChatterboxTTS`` / ``ChatterboxVC``.

    The real classes are plain objects holding ``nn.Module`` attributes; that
    shape is the whole reason the handler has an ``attach_weights`` step, so the
    double reproduces it rather than being an ``nn.Module`` itself.
    """

    def __init__(self, out_features: int = 8) -> None:
        self.t3 = torch.nn.Linear(8, out_features)
        self.s3gen = torch.nn.Linear(out_features, out_features)
        self.ve = torch.nn.Linear(4, 4)
        self.sr = 24000
        self.conds = None
        self.ref_dict = None


class RecordingModelManagement:
    """A ``comfy.model_management`` stand-in that records what was asked of it."""

    def __init__(self, free_bytes: int = 0) -> None:
        self.free_bytes = free_bytes
        self.calls: list[tuple] = []
        self.intermediate_device_value = torch.device("cpu")
        self.load_model_gpu_calls: list = []
        self.load_models_gpu_calls: list[dict] = []

    def get_free_memory(self, device) -> int:
        self.calls.append(("get_free_memory", str(device)))
        return self.free_bytes

    def intermediate_device(self) -> torch.device:
        self.calls.append(("intermediate_device",))
        return self.intermediate_device_value

    def load_model_gpu(self, patcher) -> None:
        self.calls.append(("load_model_gpu",))
        self.load_model_gpu_calls.append(patcher)

    def load_models_gpu(self, patchers, **kwargs) -> None:
        self.calls.append(("load_models_gpu", tuple(kwargs)))
        self.load_models_gpu_calls.append({"patchers": list(patchers), **kwargs})

    def soft_empty_cache(self, force: bool = False) -> None:
        self.calls.append(("soft_empty_cache",))

    def unload_all_models(self) -> None:
        self.calls.append(("unload_all_models",))


@pytest.fixture
def pack_dir(tmp_path):
    """A pack directory with known weight-file sizes."""
    directory = tmp_path / "resembleai_default_voice"
    directory.mkdir()
    (directory / "t3_cfg.safetensors").write_bytes(b"a" * 12345)
    (directory / "s3gen.safetensors").write_bytes(b"b" * 6789)
    (directory / "ve.safetensors").write_bytes(b"c" * 100)
    # Not a weight file: must not be counted.
    (directory / "tokenizer.json").write_text("{}")
    (directory / "conds.pt").write_bytes(b"d" * 50)
    return str(directory)


def make_handler(pack_dir: str, name: str = "resembleai_default_voice") -> ChatterboxModelHandler:
    """A handler with a stub pack attached, as if it had just loaded."""
    handler = ChatterboxModelHandler(name, model_path=pack_dir)
    handler.tts_model = StubPack()
    handler.vc_model = StubPack(out_features=4)
    handler.attach_weights()
    handler._refine_size()
    return handler


def make_patcher(handler: torch.nn.Module, **kwargs) -> ChatterboxPatcher:
    return ChatterboxPatcher(
        handler,
        torch.device("cpu"),
        torch.device("cpu"),
        kwargs.pop("size", handler.size),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _clean_cache():
    CHATTERBOX_PATCHER_CACHE.clear()
    yield
    CHATTERBOX_PATCHER_CACHE.clear()


# ---------------------------------------------------------------------------
# D3 -- the VC-model leak
# ---------------------------------------------------------------------------


def wrapper_is_released(patcher: ChatterboxPatcher) -> bool:
    """The check D3 is about: the *wrapper* holds no pack any more."""
    wrapper = patcher.model
    return (
        getattr(wrapper, "tts_model", None) is None
        and getattr(wrapper, "vc_model", None) is None
    )


class TestD3WrapperRelease:
    def test_destroy_nulls_both_fields_on_the_wrapper(self, pack_dir) -> None:
        patcher = make_patcher(make_handler(pack_dir))
        assert patcher.model.tts_model is not None
        assert patcher.model.vc_model is not None

        patcher.unpatch_model(unpatch_weights=True, destroy=True)

        assert patcher.model.tts_model is None
        assert patcher.model.vc_model is None
        assert wrapper_is_released(patcher)

    def test_nulling_on_the_patcher_cannot_pass_the_check(self, pack_dir) -> None:
        """The V1 defect, executed: ``nodes.py:73``'s exact assignment.

        This is the evidence that the D3 assertion discriminates. It performs
        the two writes V1 did and shows the wrapper check still fails -- so a
        patcher that nulls the *patcher* cannot satisfy
        :meth:`test_destroy_nulls_both_fields_on_the_wrapper`, let alone pass
        it by accident.
        """
        patcher = make_patcher(make_handler(pack_dir))
        wrapper = patcher.model

        # V1: `self.vc_model = None` / `self.tts_model = None` on the patcher.
        patcher.vc_model = None
        patcher.tts_model = None

        assert patcher.vc_model is None and patcher.tts_model is None
        assert wrapper.vc_model is not None, (
            "the V1 assignment landed on the patcher and freed nothing; the VC "
            "model is still reachable through the wrapper"
        )
        assert wrapper_is_released(patcher) is False

        # And the real fix does clear it.
        release_wrapper_references(patcher)
        assert wrapper_is_released(patcher) is True

    def test_destroy_also_detaches_the_registered_submodules(self, pack_dir) -> None:
        """The weight modules are held twice; clearing one is not enough.

        ``attach_weights`` registers ``t3`` / ``s3gen`` / ``ve`` on the handler.
        A module-registry entry is as strong a reference as the plain attribute,
        so a handler whose attributes are nulled but whose ``_modules`` are not
        still owns every byte.
        """
        handler = make_handler(pack_dir)
        assert dict(handler.named_children()), "the fixture registered no weights"

        patcher = make_patcher(handler)
        patcher.unpatch_model(unpatch_weights=True, destroy=True)

        assert patcher.model.tts_model is None
        assert list(patcher.model.named_children()) == []
        assert list(patcher.model.parameters()) == []

    def test_destroy_drops_the_cache_entry(self, pack_dir) -> None:
        patcher = make_patcher(make_handler(pack_dir))
        patcher.cache_key = "some_key"
        CHATTERBOX_PATCHER_CACHE["some_key"] = patcher

        patcher.unpatch_model(unpatch_weights=True, destroy=True)

        assert "some_key" not in CHATTERBOX_PATCHER_CACHE

    def test_release_is_safe_on_an_already_empty_wrapper(self, pack_dir) -> None:
        handler = make_handler(pack_dir)
        patcher = make_patcher(handler)
        patcher.unpatch_model(unpatch_weights=True, destroy=True)
        assert release_wrapper_references(patcher) == []


# ---------------------------------------------------------------------------
# D4 -- the hardcoded size
# ---------------------------------------------------------------------------


class TestD4RealSize:
    def test_handler_measures_the_real_on_disk_weights(self, pack_dir) -> None:
        expected = os.path.getsize(os.path.join(pack_dir, "t3_cfg.safetensors"))
        expected += os.path.getsize(os.path.join(pack_dir, "s3gen.safetensors"))
        expected += os.path.getsize(os.path.join(pack_dir, "ve.safetensors"))
        assert expected == 12345 + 6789 + 100

        handler = ChatterboxModelHandler("resembleai_default_voice", model_path=pack_dir)
        assert handler.size == expected, (
            "the handler must size itself from the safetensors on disk; "
            f"got {handler.size}, expected {expected}"
        )
        assert handler.size != int(1.5 * 1024**3), (
            "the handler is back to the V1 hardcoded 1.5 GB constant"
        )

    def test_the_tokenizer_and_conds_are_not_counted_as_weights(self, pack_dir) -> None:
        # conds.pt is a 50-byte pickle, tokenizer.json is 2 bytes; a pack whose
        # reported size moved by 52 would be reporting the wrong thing.
        assert pack_size_on_disk(pack_dir) == 12345 + 6789 + 100

    def test_model_size_reports_the_measured_bytes(self, pack_dir) -> None:
        handler = ChatterboxModelHandler("resembleai_default_voice", model_path=pack_dir)
        patcher = make_patcher(handler)

        assert patcher.model_size() == handler.size == 19234
        assert patcher.size == 19234

    def test_load_list_over_the_handler_is_not_empty(self, pack_dir) -> None:
        handler = make_handler(pack_dir)
        patcher = make_patcher(handler)

        loading = patcher._load_list()
        assert loading, (
            "ModelPatcher._load_list() found no parameters in the handler; the "
            "pack's weight submodules are not registered, so ComfyUI would "
            "manage this model as if it were free"
        )

        enumerated = sum(entry[1] for entry in loading)
        measured = ChatterboxModelHandler._estimate_size(handler)
        assert measured > 0
        assert enumerated == measured, (
            "the enumerated load list does not account for every parameter "
            f"({enumerated} != {measured}); a module is being skipped"
        )

    def test_size_is_refined_from_the_measured_parameters_after_load(self, pack_dir) -> None:
        handler = ChatterboxModelHandler("resembleai_default_voice", model_path=pack_dir)
        assert handler.size == 19234

        handler.tts_model = StubPack(out_features=64)
        handler.vc_model = None
        handler.attach_weights()
        handler._refine_size()

        measured = ChatterboxModelHandler._estimate_size(handler)
        assert measured > 0
        assert handler.size == measured
        assert handler.size != 19234, (
            "the post-load refinement did not replace the on-disk estimate"
        )

    def test_refine_ignores_an_unmeasurable_model(self, pack_dir) -> None:
        handler = ChatterboxModelHandler("resembleai_default_voice", model_path=pack_dir)
        handler.tts_model = object()  # no .parameters()
        handler._refine_size()
        assert handler.size == 19234

    def test_a_missing_pack_reports_zero_not_a_guess(self, tmp_path) -> None:
        assert pack_size_on_disk(str(tmp_path / "nope")) == 0
        assert pack_size_on_disk(None) == 0


# ---------------------------------------------------------------------------
# Keyword-only construction
# ---------------------------------------------------------------------------


class TestKeywordOnlyInit:
    def test_attention_mode_and_dtype_are_keyword_only(self) -> None:
        signature = inspect.signature(ChatterboxPatcher.__init__)
        for name in ("attention_mode", "dtype"):
            parameter = signature.parameters[name]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
                f"{name} must be keyword-only: core's ModelPatcher.clone() "
                "reconstructs this class positionally"
            )

    def test_clone_reconstructs_positionally(self, pack_dir) -> None:
        """The real failure this guards: ``comfy/model_patcher.py:446``.

        ``clone()`` calls ``class_(model, load_device, offload_device,
        model_size(), weight_inplace_update=..., fast_disk=...)``. A parameter
        declared before ``*args`` would bind ``load_device`` to
        ``attention_mode`` and raise ``TypeError: missing 1 required positional
        argument: 'offload_device'``.
        """
        patcher = make_patcher(make_handler(pack_dir), attention_mode="sdpa")
        clone = patcher.clone()

        assert clone.load_device == patcher.load_device
        assert clone.offload_device == patcher.offload_device
        assert clone.model_size() == patcher.model_size()

    def test_positional_construction_is_accepted(self, pack_dir) -> None:
        patcher = ChatterboxPatcher(
            make_handler(pack_dir), torch.device("cpu"), torch.device("cpu"), 4242
        )
        assert patcher.size == 4242
        assert patcher.attention_mode == "eager"
        assert patcher.target_dtype is None


# ---------------------------------------------------------------------------
# is_loaded
# ---------------------------------------------------------------------------


class TestIsLoaded:
    def test_true_when_the_pack_is_on_the_load_device(self, pack_dir) -> None:
        patcher = make_patcher(make_handler(pack_dir))
        patcher.load_device = torch.device("cpu")
        assert patcher.is_loaded is True

    def test_false_when_the_model_sits_on_the_offload_device(self, pack_dir) -> None:
        """A pack parked in RAM is "loaded" but not runnable.

        The load device is set to CUDA so the resident (CPU) tensors genuinely
        are not on it -- this test machine does not need a GPU to exercise the
        comparison.
        """
        patcher = make_patcher(make_handler(pack_dir))
        patcher.load_device = torch.device("cuda")

        assert next(patcher.model.parameters()).device.type == "cpu"
        assert patcher.is_loaded is False

    def test_false_when_nothing_is_loaded(self, pack_dir) -> None:
        handler = ChatterboxModelHandler("resembleai_default_voice", model_path=pack_dir)
        patcher = make_patcher(handler)
        assert patcher.is_loaded is False

    def test_false_while_warm_offloaded(self, pack_dir) -> None:
        patcher = make_patcher(make_handler(pack_dir))
        patcher.load_device = torch.device("cpu")
        assert patcher.is_loaded is True

        patcher._warm_offloaded = True
        assert patcher.is_loaded is False


# ---------------------------------------------------------------------------
# Offload contract
# ---------------------------------------------------------------------------


class TestOffloadContract:
    def test_cold_offload_keeps_the_pack_in_ram(self, pack_dir, monkeypatch) -> None:
        handler = make_handler(pack_dir)
        patcher = make_patcher(handler)
        fake_mm = RecordingModelManagement()
        monkeypatch.setattr(patcher_module, "_get_model_management", lambda: fake_mm)

        patcher.unpatch_model(device_to=torch.device("cpu"), unpatch_weights=True)

        assert patcher.model.tts_model is not None, (
            "a cold offload must keep the pack; dropping it forces a full disk "
            "reload on the next ComfyUI-initiated offload"
        )
        assert patcher.model.vc_model is not None
        assert patcher._warm_offloaded is False
        assert patcher.is_loaded is True

    def test_warm_offload_parks_tensors_on_the_intermediate_device(
        self, pack_dir, monkeypatch
    ) -> None:
        handler = make_handler(pack_dir)
        patcher = make_patcher(handler)
        fake_mm = RecordingModelManagement()
        monkeypatch.setattr(patcher_module, "_get_model_management", lambda: fake_mm)

        patcher.unpatch_model(unpatch_weights=True, warm=True)

        assert patcher._warm_offloaded is True
        assert patcher.is_loaded is False
        assert patcher.model.device == fake_mm.intermediate_device_value
        assert patcher.model.tts_model is not None, (
            "a warm offload keeps every reference; only the device changes"
        )
        assert fake_mm.calls.count(("intermediate_device",)) == 1

    def test_destroy_releases_everything(self, pack_dir, monkeypatch) -> None:
        handler = make_handler(pack_dir)
        patcher = make_patcher(handler)
        fake_mm = RecordingModelManagement()
        monkeypatch.setattr(patcher_module, "_get_model_management", lambda: fake_mm)
        monkeypatch.setattr(
            patcher_module, "unregister_from_comfy", lambda _p: [], raising=False
        )

        patcher.unpatch_model(unpatch_weights=True, destroy=True)

        assert patcher.model.tts_model is None
        assert patcher.model.vc_model is None
        assert patcher.is_loaded is False
        assert ("soft_empty_cache",) in fake_mm.calls

    def test_warm_and_destroy_together_mean_destroy(self, pack_dir, monkeypatch) -> None:
        handler = make_handler(pack_dir)
        patcher = make_patcher(handler)
        monkeypatch.setattr(
            patcher_module, "_get_model_management", RecordingModelManagement()
        )

        patcher.unpatch_model(unpatch_weights=True, warm=True, destroy=True)

        assert patcher.model.tts_model is None, (
            "a caller that asked for destroy must not be satisfied by tensors "
            "parked on the intermediate device"
        )
        assert patcher._warm_offloaded is False

    def test_unpatch_without_weights_changes_nothing(self, pack_dir) -> None:
        patcher = make_patcher(make_handler(pack_dir))
        patcher.load_device = torch.device("cpu")

        patcher.unpatch_model(unpatch_weights=False, destroy=True)

        assert patcher.model.tts_model is not None
        assert patcher.model.vc_model is not None

    def test_the_registry_eviction_keywords_are_accepted(self, pack_dir) -> None:
        """``model_registry.evict_patcher`` calls exactly this signature."""
        import inspect as _inspect

        from modules import model_registry

        signature = _inspect.signature(ChatterboxPatcher.unpatch_model)
        for name in ("unpatch_weights", "warm", "destroy"):
            assert name in signature.parameters

        assert callable(model_registry.evict_patcher)
        patcher = make_patcher(make_handler(pack_dir))
        patcher.cache_key = "k"
        CHATTERBOX_PATCHER_CACHE["k"] = patcher
        errors = model_registry.evict_patcher(patcher, CHATTERBOX_PATCHER_CACHE, "k")
        assert errors == [], f"the registry eviction reported: {errors}"
        assert patcher.model.tts_model is None
        assert "k" not in CHATTERBOX_PATCHER_CACHE


# ---------------------------------------------------------------------------
# load_to_device
# ---------------------------------------------------------------------------


class TestLoadToDevice:
    def _prepare(self, pack_dir, headroom: int, load_device, monkeypatch):
        """A patcher whose free VRAM is ``model_size() + headroom``."""
        handler = make_handler(pack_dir)
        patcher = make_patcher(handler)
        patcher.load_device = load_device
        fake_mm = RecordingModelManagement(
            free_bytes=patcher.model_size() + headroom
        )
        monkeypatch.setattr(patcher_module, "_get_model_management", lambda: fake_mm)
        patched: list[dict] = []

        def record_patch_model(**kwargs):
            patched.append(kwargs)

        patcher.patch_model = record_patch_model  # type: ignore[method-assign]
        return patcher, fake_mm, patched

    def test_shortcut_fires_with_enough_activation_headroom(
        self, pack_dir, monkeypatch
    ) -> None:
        patcher, fake_mm, patched = self._prepare(
            pack_dir,
            headroom=ACTIVATION_HEADROOM_BYTES + 1,
            load_device=torch.device("cuda"),
            monkeypatch=monkeypatch,
        )

        result = load_to_device(patcher)

        assert result is patcher
        assert len(patched) == 1, "the full-load shortcut did not patch the model"
        assert patched[0]["lowvram_model_memory"] == 0
        assert patched[0]["load_weights"] is True
        assert len(fake_mm.load_models_gpu_calls) == 1
        assert fake_mm.load_models_gpu_calls[0]["force_full_load"] is True
        assert fake_mm.load_model_gpu_calls == []

    def test_shortcut_does_not_fire_at_exactly_the_headroom(
        self, pack_dir, monkeypatch
    ) -> None:
        """1.5 GB of headroom is the floor; exactly that is not enough.

        The comparison is strict: a model that fills the card to the last byte
        leaves no room for a sampling step's activations, which is precisely the
        CUDA OOM the headroom exists to avoid.
        """
        patcher, fake_mm, patched = self._prepare(
            pack_dir,
            headroom=ACTIVATION_HEADROOM_BYTES,
            load_device=torch.device("cuda"),
            monkeypatch=monkeypatch,
        )

        load_to_device(patcher)

        assert patched == []
        assert fake_mm.load_models_gpu_calls == []
        assert len(fake_mm.load_model_gpu_calls) == 1

    def test_shortcut_does_not_fire_without_the_headroom(
        self, pack_dir, monkeypatch
    ) -> None:
        patcher, fake_mm, patched = self._prepare(
            pack_dir,
            headroom=ACTIVATION_HEADROOM_BYTES - 1,
            load_device=torch.device("cuda"),
            monkeypatch=monkeypatch,
        )

        load_to_device(patcher)

        assert patched == []
        assert len(fake_mm.load_model_gpu_calls) == 1

    def test_cpu_never_takes_the_cuda_shortcut(self, pack_dir, monkeypatch) -> None:
        patcher, fake_mm, patched = self._prepare(
            pack_dir,
            headroom=1 << 40,
            load_device=torch.device("cpu"),
            monkeypatch=monkeypatch,
        )

        load_to_device(patcher)

        assert patched == []
        assert fake_mm.calls == [("load_model_gpu",)]

    def test_memory_required_reaches_the_dynamic_branch(self, pack_dir, monkeypatch) -> None:
        patcher = make_patcher(make_handler(pack_dir))
        fake_mm = RecordingModelManagement(free_bytes=1 << 40)
        monkeypatch.setattr(patcher_module, "_get_model_management", lambda: fake_mm)
        monkeypatch.setattr(
            ChatterboxPatcher, "is_dynamic", lambda self: True, raising=False
        )

        load_to_device(patcher, memory_required=1234)

        assert fake_mm.load_models_gpu_calls[0]["memory_required"] == 1234
        assert fake_mm.load_model_gpu_calls == []


# ---------------------------------------------------------------------------
# Patcher-class selection
# ---------------------------------------------------------------------------


class TestPatcherClassSelection:
    def test_select_returns_the_legacy_class(self) -> None:
        assert select_patcher_class(None, torch.device("cpu")) is ChatterboxPatcher
        assert select_patcher_class("chatterbox_tts", torch.device("cuda")) is (
            ChatterboxPatcher
        )

    def test_select_respects_an_explicit_legacy_class(self) -> None:
        class Other(ChatterboxPatcher):
            pass

        assert select_patcher_class(None, torch.device("cpu"), legacy_cls=Other) is Other

    def test_core_patcher_class_resolves(self) -> None:
        import comfy.model_patcher

        resolved = resolve_core_patcher_class()
        assert isinstance(resolved, type)
        assert issubclass(resolved, comfy.model_patcher.ModelPatcher)

    def test_dynamic_probe_is_false_off_cuda(self) -> None:
        assert dynamic_vram_available(ChatterboxPatcher, torch.device("cpu")) is False

    def test_dynamic_probe_is_false_for_an_unusable_device(self) -> None:
        assert dynamic_vram_available(ChatterboxPatcher, "not a device") is False

    def test_dynamic_probe_is_false_for_a_class_without_the_hook(self) -> None:
        class Bare:
            pass

        assert dynamic_vram_available(Bare, torch.device("cuda")) is False

    def test_dynamic_probe_reports_a_class_that_claims_dynamic(self) -> None:
        class Claims:
            @staticmethod
            def is_dynamic(_patcher):
                return True

        assert dynamic_vram_available(Claims, torch.device("cuda")) is True


# ---------------------------------------------------------------------------
# Loader surface
# ---------------------------------------------------------------------------


class TestLoaderSurface:
    def test_ensure_downloaded_skips_a_complete_pack(self, pack_dir, monkeypatch) -> None:
        calls: list[dict] = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            return pack_dir

        monkeypatch.setitem(
            __import__("sys").modules, "huggingface_hub", type("m", (), {"snapshot_download": staticmethod(fake_snapshot_download)})
        )

        ChatterboxLoader._ensure_downloaded("ResembleAI/chatterbox", pack_dir, "pack")
        assert calls == [], "a complete pack must not be re-downloaded"

    def test_ensure_downloaded_probes_for_the_removed_symlink_kwarg(
        self, pack_dir, monkeypatch
    ) -> None:
        """``local_dir_use_symlinks`` is gone in huggingface_hub 1.x.

        Passing it raises ``TypeError`` on the *first* download of a model, so
        the kwarg is only passed when the installed signature declares it.
        """
        seen: list[dict] = []

        def make_fake(accepts: bool):
            def fake(**kwargs):
                seen.append(kwargs)

            if accepts:
                import inspect

                def fake_with_symlink(
                    repo_id=None,
                    local_dir=None,
                    local_dir_use_symlinks=None,
                ):
                    seen.append(
                        {
                            "repo_id": repo_id,
                            "local_dir": local_dir,
                            "local_dir_use_symlinks": local_dir_use_symlinks,
                        }
                    )

                del inspect
                return fake_with_symlink
            return fake

        import sys

        empty_dir = str(os.path.join(pack_dir, "..", "empty_pack"))
        os.makedirs(empty_dir, exist_ok=True)

        monkeypatch.setitem(
            sys.modules,
            "huggingface_hub",
            type("m", (), {"snapshot_download": staticmethod(make_fake(False))}),
        )
        ChatterboxLoader._ensure_downloaded("repo", empty_dir, "pack")
        assert seen[-1] == {"repo_id": "repo", "local_dir": empty_dir}

        monkeypatch.setitem(
            sys.modules,
            "huggingface_hub",
            type("m", (), {"snapshot_download": staticmethod(make_fake(True))}),
        )
        ChatterboxLoader._ensure_downloaded("repo", empty_dir, "pack")
        assert seen[-1]["local_dir_use_symlinks"] is False

    def test_ensure_downloaded_ignores_an_empty_repo_id(self, pack_dir) -> None:
        # No repo id and a complete pack: nothing to do, nothing raised.
        ChatterboxLoader._ensure_downloaded("", pack_dir, "pack")

    def test_resolve_pack_dir_rejects_a_blank_name(self) -> None:
        with pytest.raises(ValueError, match="No Chatterbox model"):
            ChatterboxLoader.resolve_pack_dir("")

    def test_resolve_pack_dir_is_absolute_for_the_default_pack(self) -> None:
        from modules.constants import DEFAULT_MODEL_PACK_NAME

        resolved = ChatterboxLoader.resolve_pack_dir(DEFAULT_MODEL_PACK_NAME)
        assert os.path.isabs(resolved)
        assert resolved.endswith(DEFAULT_MODEL_PACK_NAME)

    def test_load_model_reports_a_missing_pack_clearly(self, tmp_path, monkeypatch) -> None:
        empty = tmp_path / "not_a_pack"
        empty.mkdir()
        attempts: list[tuple] = []

        patterns_seen: list = []

        def record_download_pack(
            model_name, repo_id, local_dir, client_id=None, allow_patterns=None
        ):
            attempts.append((model_name, repo_id, local_dir, client_id))
            patterns_seen.append(allow_patterns)

        monkeypatch.setattr(
            ChatterboxLoader, "download_pack", staticmethod(record_download_pack)
        )

        with pytest.raises(FileNotFoundError, match="not available at"):
            ChatterboxLoader.load_model(
                "resembleai_default_voice",
                torch.device("cpu"),
                model_path=str(empty),
                load_vc=False,
            )

        assert attempts, (
            "load_model must attempt a download before giving up; the failure "
            "message is only truthful if it did"
        )
        # The combo value is a legacy spelling, but the download step reads the
        # registry by the *canonical* name: an alias must still find the
        # repository the classic pack comes from, or a saved workflow would be
        # handed an empty repo and then told the pack was missing.
        assert attempts[0][1] == "ResembleAI/chatterbox", (
            f"the fetch targeted {attempts[0][1]!r}, not the classic repository"
        )
        # The classic fetch is filtered: shared weights come from the store.
        assert patterns_seen == [
            ["t3_cfg.safetensors", "tokenizer.json", "conds.pt"]
        ]

    def test_download_pack_is_skipped_for_a_complete_pack(self, pack_dir) -> None:
        called: list[tuple] = []
        original = ChatterboxLoader._ensure_downloaded
        try:
            ChatterboxLoader._ensure_downloaded = staticmethod(
                lambda *args, **kwargs: called.append(args)
            )
            ChatterboxLoader.download_pack("pack", "repo", pack_dir, None)
        finally:
            ChatterboxLoader._ensure_downloaded = original
        assert called == []

    def test_pack_completeness_uses_the_pack_contract(self, pack_dir, tmp_path) -> None:
        assert ChatterboxLoader.is_pack_complete(pack_dir) is True
        assert ChatterboxLoader.is_pack_complete(str(tmp_path)) is False
        assert ChatterboxLoader.is_pack_complete(None) is False

    def test_weight_suffixes_cover_what_a_pack_ships(self) -> None:
        assert ".safetensors" in loader_module.WEIGHT_SUFFIXES


# ---------------------------------------------------------------------------
# Logging discipline
# ---------------------------------------------------------------------------


class TestLoggingDiscipline:
    def test_no_logger_objects_or_handlers(self) -> None:
        with open(patcher_module.__file__, "r", encoding="utf-8") as handle:
            source = handle.read()
        for forbidden in (
            "getLogger(",
            "logging.Formatter(",
            "LoggerAdapter(",
            ".addHandler(",
            ".setLevel(",
        ):
            assert forbidden not in source, f"patcher.py must not use {forbidden}"

    def test_records_go_through_the_root_logger(self, caplog, tmp_path) -> None:
        patcher = make_patcher(make_handler(str(tmp_path)))
        patcher.cache_key = "log_key"
        with caplog.at_level(logging.DEBUG, logger=""):
            patcher.unpatch_model(unpatch_weights=True, destroy=True)

        messages = [record.getMessage() for record in caplog.records]
        assert any("Destroying voice pack" in message for message in messages)
        assert all(
            record.name == "root" for record in caplog.records if record.name
        ), "a named logger was used instead of the root logger"

    def test_a_failed_release_is_logged_not_raised(self, caplog) -> None:
        """A wrapper that refuses the write must not take the eviction down."""

        class Hostile:
            def __init__(self) -> None:
                object.__setattr__(self, "tts_model", object())
                object.__setattr__(self, "vc_model", object())

            def __setattr__(self, name, value) -> None:
                raise RuntimeError("read-only wrapper")

        class Patcher:
            model = Hostile()

        with caplog.at_level(logging.WARNING, logger=""):
            assert release_wrapper_references(Patcher()) == []

        messages = [record.getMessage() for record in caplog.records]
        assert any("tts_model" in message for message in messages), (
            f"a failed wrapper release was swallowed: {messages}"
        )

    def test_release_reports_a_wrapper_with_no_pack(self) -> None:
        class Empty:
            model = object()

        assert release_wrapper_references(Empty()) == []


# ---------------------------------------------------------------------------
# Memory hygiene
# ---------------------------------------------------------------------------


def test_a_destroyed_patcher_releases_its_weights(pack_dir) -> None:
    """The end-to-end shape of D3: after an eviction the pack is collectable."""
    patcher = make_patcher(make_handler(pack_dir))
    patcher.cache_key = "evict_me"
    CHATTERBOX_PATCHER_CACHE["evict_me"] = patcher

    release_wrapper_references(patcher)
    CHATTERBOX_PATCHER_CACHE.pop("evict_me")
    gc.collect()

    assert list(patcher.model.named_children()) == []