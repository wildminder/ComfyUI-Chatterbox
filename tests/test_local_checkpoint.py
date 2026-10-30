"""S8.8 -- community and arbitrary checkpoints: load them like any other pack.

Three claims, three different kinds of evidence:

* **A CUDA-saved checkpoint loads on CPU.** No CUDA device is required to prove
  it, because the guarantee lives entirely in one argument: ``from_local`` must
  pass ``map_location=torch.device("cpu")`` when the device is cpu or mps. The
  upstream bug is ``device in ["cpu", "mps"]`` against a ``torch.device``, which
  is False for every device -- so the load would try to restore CUDA-saved
  tensors on a CPU. This records the actual ``map_location`` each entry point
  passes to ``torch.load``.
* **An arbitrary directory is discovered.** Viterbox, Egyptian Arabic, a
  finetune from last Tuesday -- all of them are just a directory under the
  models root with weights in it, and nothing in the code may treat a known one
  differently from an unknown one. The scan is run against a real temporary
  directory holding a pack with a name nobody has heard of.
* **A directory missing weights is refused with a message that names them.**
  Upstream dies on the first ``load_file`` with one filename in the exception
  and the other three left to guesswork.

No network, no real weights: every file here is a byte the test wrote.
"""

from __future__ import annotations

import inspect
import os
import textwrap

import pytest
import torch

from ComfyUI_Chatterbox.modules.loader import (
    MTL_CHECKPOINT_FILES,
    MTL_S3GEN_CHECKPOINTS,
    TURBO_CHECKPOINT_FILES,
    ChatterboxLoader,
)
from ComfyUI_Chatterbox.modules.model_info import (
    describe_model_dir,
    is_chatterbox_model_dir,
    scan_chatterbox_models,
)

from conftest import PKG_ALIAS, ROOT_DIR

#: The vendored Flash package is reached package-relatively, under the alias
#: this checkout is registered under -- not under the top-level name.
FLASH_PACKAGE = f"{PKG_ALIAS}.src.chatterbox_flash"


# ---------------------------------------------------------------------------
# map_location
# ---------------------------------------------------------------------------

_MAP_LOCATION_PROBE = textwrap.dedent(
    """
    import tempfile
    from pathlib import Path

    from ComfyUI_Chatterbox.src.chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from ComfyUI_Chatterbox.src.chatterbox.vc import ChatterboxVC

    def _pack(*names):
        directory = Path(tempfile.mkdtemp())
        for name in names:
            (directory / name).write_bytes(b"placeholder")
        return directory

    def _run(cls, device, files, **kwargs):
        recorded_load.calls.clear()
        try:
            cls.from_local(_pack(*files), device, **kwargs)
        except _Stop:
            pass
        except Exception as exc:
            recorded_load.calls.append("raised:%s" % type(exc).__name__)
        return list(recorded_load.calls)

    # `ChatterboxVC` and `ChatterboxMultilingualTTS` reach `torch.load` before
    # any weight-shaped work, so the recorded kwarg is the real one. The other
    # two entry points read safetensors first and pass `map_location` to
    # `Conditionals.load`; those are covered by the AST check below, which
    # proves the variable is threaded into a load call rather than assigned and
    # forgotten.
    _PROBE = {
        "vc_device_cpu": _run(ChatterboxVC, torch.device("cpu"), ["s3gen.safetensors", "conds.pt"]),
        "vc_cpu_string": _run(ChatterboxVC, "cpu", ["s3gen.safetensors", "conds.pt"]),
        "vc_mps": _run(ChatterboxVC, "mps", ["s3gen.safetensors", "conds.pt"]),
        "vc_cuda": _run(ChatterboxVC, "cuda", ["s3gen.safetensors", "conds.pt"]),
        "mtl_device_cpu": _run(
            ChatterboxMultilingualTTS,
            torch.device("cpu"),
            ["ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
             "t3_mtl23ls_v3.safetensors"],
            t3_model="v3",
        ),
        "mtl_mps": _run(
            ChatterboxMultilingualTTS,
            "mps",
            ["ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json",
             "t3_mtl23ls_v2.safetensors"],
            t3_model="v2",
        ),
    }
    """
)


class _Stop(Exception):
    """Ends a ``from_local`` at the first ``torch.load``, on purpose."""


def _recording_load(*_args, **kwargs):
    """A ``torch.load`` stand-in that records ``map_location`` and stops."""
    _recording_load.calls.append(kwargs.get("map_location", "<absent>"))
    raise _Stop


_recording_load.calls = []


@pytest.fixture(scope="module")
def real_vendor():
    """The real vendored entry points, with the harness mocks put back afterwards.

    These tests are about the ``from_local`` guards, which exist only in the real
    files. :func:`conftest_helpers.real_vendor_modules` restores the aliased
    prefix the harness registers on teardown, so every later test in the suite
    still finds a mocked tree.
    """
    from .conftest_helpers import real_vendor_modules

    with real_vendor_modules(["tts", "vc", "mtl_tts", "tts_turbo"]) as modules:
        yield modules


def _map_location_probe(real_vendor: dict) -> dict:
    """Run the vendored ``from_local`` implementations with ``torch.load`` observed.

    The probe stops at the first ``torch.load``, which is exactly the call site
    the CPU-remap guarantee is about; everything before it (the directory
    completeness check, which is patched out of the way below) is not what is
    under test here.
    """
    real_load = torch.load
    torch.load = _recording_load
    namespace: dict = {
        "torch": torch,
        "_Stop": _Stop,
        "recorded_load": _recording_load,
    }
    try:
        exec(compile(_MAP_LOCATION_PROBE, "probe", "exec"), namespace)
    finally:
        # ``torch.load`` is a global; a recorder left installed would break every
        # later test that saves and loads a checkpoint -- which is exactly what
        # ``tests/test_watermark.py`` does.
        torch.load = real_load
    return namespace["_PROBE"]


class TestMapLocationForcesCpu:
    """``map_location=torch.device('cpu')`` whenever the device is cpu or mps.

    The upstream defect is ``device in ["cpu", "mps"]`` evaluated against a
    ``torch.device``, which is False for *every* device -- so a CPU load selected
    ``map_location=None`` and tried to restore CUDA-saved tensors. A CUDA-saved
    checkpoint therefore could not be loaded on a CPU, which is most of the
    machines that would want a community finetune.
    """

    @pytest.fixture(scope="class")
    def probe(self, real_vendor) -> dict:
        return _map_location_probe(real_vendor)

    def test_the_probe_actually_reached_a_torch_load(self, probe: dict) -> None:
        """A probe that stopped before the call site would assert nothing."""
        assert probe["vc_device_cpu"], "the vendored from_local never reached torch.load"
        assert all(entry != "<absent>" for entry in probe["vc_device_cpu"]), (
            "torch.load was called without a map_location at all"
        )

    @pytest.mark.parametrize(
        "key", ["vc_device_cpu", "vc_cpu_string", "vc_mps", "mtl_device_cpu", "mtl_mps"]
    )
    def test_cpu_and_mps_load_through_cpu(self, probe: dict, key: str) -> None:
        for entry in probe[key]:
            assert str(entry) == "cpu", (
                f"{key} passed map_location={entry!r}; a CUDA-saved checkpoint "
                f"would fail to restore there"
            )

    def test_a_cuda_device_still_maps_to_the_default(self, probe: dict) -> None:
        """Forcing CPU everywhere would make a GPU load pay for a host round trip."""
        assert probe["vc_cuda"], probe["vc_cuda"]
        assert all(entry is None for entry in probe["vc_cuda"]), probe["vc_cuda"]

    @pytest.mark.parametrize("name", ["tts.py", "vc.py", "mtl_tts.py", "tts_turbo.py"])
    def test_every_entry_point_compares_str_device(self, name: str) -> None:
        """The guard itself, in all four files.

        ``torch.device('cpu') in ["cpu", "mps"]`` is False; ``str(torch.device(
        'cpu'))`` is ``"cpu"``. The second spelling is the one that works, and
        all four entry points have to use it.
        """
        with open(
            os.path.join(ROOT_DIR, "src", "chatterbox", name), encoding="utf-8"
        ) as handle:
            source = handle.read()
        assert 'if str(device) in ["cpu", "mps"]' in source, name
        assert "map_location = torch.device('cpu')" in source, name

    @pytest.mark.parametrize("name", ["tts.py", "vc.py", "mtl_tts.py", "tts_turbo.py"])
    def test_map_location_reaches_a_load_call(self, name: str) -> None:
        """Assigned is not enough -- it has to be threaded into the load.

        A ``map_location`` that is computed and then passed to nothing would pass
        the guard above and fix nothing, which is exactly the shape of a bug an
        AST check catches and a text search does not.
        """
        import ast

        with open(
            os.path.join(ROOT_DIR, "src", "chatterbox", name), encoding="utf-8"
        ) as handle:
            tree = ast.parse(handle.read(), filename=name)

        from_local = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "from_local"
        )
        threaded = [
            node
            for node in ast.walk(from_local)
            if isinstance(node, ast.Call)
            and any(
                keyword.arg == "map_location"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "map_location"
                for keyword in node.keywords
            )
        ]
        assert threaded, (
            f"{name}: `from_local` computes map_location but never passes it to "
            f"a load"
        )


# ---------------------------------------------------------------------------
# Discovery of an arbitrary directory
# ---------------------------------------------------------------------------


def _write_pack(directory, t3_name: str = "t3_cfg.safetensors") -> str:
    """A minimal *shape* of a Chatterbox pack: the files, not the weights."""
    os.makedirs(directory, exist_ok=True)
    for name in ("tokenizer.json", "ve.safetensors", "s3gen.safetensors", t3_name):
        with open(os.path.join(directory, name), "wb") as handle:
            handle.write(b"\x00" * 16)
    with open(os.path.join(directory, "conds.pt"), "wb") as handle:
        handle.write(b"\x00" * 16)
    return directory


class TestArbitraryDirectories:
    def test_a_pack_nobody_has_heard_of_is_recognised(self, tmp_path) -> None:
        root = tmp_path / "models"
        _write_pack(str(root / "viterbox_vietnamese_mystery_weights"))
        found = scan_chatterbox_models(str(root))
        assert [entry["name"] for entry in found] == [
            "viterbox_vietnamese_mystery_weights"
        ]
        assert is_chatterbox_model_dir(str(root / "viterbox_vietnamese_mystery_weights"))

    def test_a_community_finetune_gets_no_special_treatment(self, tmp_path) -> None:
        """Viterbox and Egyptian Arabic are ordinary local directories.

        Nothing in the discovery path may name them, so a finetune released
        tomorrow is as loadable as one released last year.
        """
        root = tmp_path / "models"
        _write_pack(str(root / "viterbox"))
        _write_pack(str(root / "oddadmix_chatterbox_egyptian"))
        _write_pack(str(root / "something_nobody_wrote_about"))
        found = {entry["name"] for entry in scan_chatterbox_models(str(root))}
        assert found == {
            "viterbox",
            "oddadmix_chatterbox_egyptian",
            "something_nobody_wrote_about",
        }

    def test_a_directory_of_unknown_blobs_is_still_a_pack(self, tmp_path) -> None:
        """Community checkpoints rename their weights; that is not a reason to refuse."""
        pack = tmp_path / "renamed_everything"
        pack.mkdir()
        for name in ("tokenizer.json", "model-t3-v2.safetensors", "model-s3gen.safetensors"):
            (pack / name).write_bytes(b"\x00" * 8)
        assert is_chatterbox_model_dir(str(pack))
        assert describe_model_dir(str(pack))["architecture"] == "unknown"

    def test_an_empty_or_partial_directory_is_not_a_pack(self, tmp_path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert not is_chatterbox_model_dir(str(empty))

        weights_only = tmp_path / "weights_only"
        weights_only.mkdir()
        (weights_only / "ve.safetensors").write_bytes(b"\x00")
        assert not is_chatterbox_model_dir(str(weights_only)), (
            "a directory with weights and no tokenizer cannot build a T3 condition"
        )

        tokenizer_only = tmp_path / "tokenizer_only"
        tokenizer_only.mkdir()
        (tokenizer_only / "tokenizer.json").write_text("{}")
        assert not is_chatterbox_model_dir(str(tokenizer_only))

    def test_a_scan_of_a_missing_directory_is_empty_not_an_error(self, tmp_path) -> None:
        assert scan_chatterbox_models(str(tmp_path / "nope")) == []


# ---------------------------------------------------------------------------
# Actionable refusals
# ---------------------------------------------------------------------------


class TestIncompleteDirectoryIsRefused:
    def test_the_message_names_the_missing_files(self, real_vendor) -> None:
        vendored_tts = real_vendor["tts"]

        with pytest.raises(FileNotFoundError) as caught:
            vendored_tts._require_checkpoint_files(
                "models/nowhere", vendored_tts.REQUIRED_CHECKPOINT_FILES
            )
        message = str(caught.value)
        for name in vendored_tts.REQUIRED_CHECKPOINT_FILES:
            assert name in message
        assert "Missing" in message and "complete pack holds" in message

    def test_a_complete_directory_passes(self, tmp_path, real_vendor) -> None:
        vendored_tts = real_vendor["tts"]

        _write_pack(str(tmp_path / "pack"))
        vendored_tts._require_checkpoint_files(
            str(tmp_path / "pack"), vendored_tts.REQUIRED_CHECKPOINT_FILES
        )

    def test_the_vc_pack_needs_far_less(self, real_vendor) -> None:
        vendored_vc = real_vendor["vc"]

        assert vendored_vc.REQUIRED_CHECKPOINT_FILES == ("s3gen.safetensors",), (
            "a conversion loads no T3 weights, so the message names one file; "
            "conds.pt is optional and is not in the required set"
        )

    def test_all_four_entry_points_carry_the_check(self, real_vendor) -> None:
        entry_points = {
            "tts": real_vendor["tts"].ChatterboxTTS,
            "vc": real_vendor["vc"].ChatterboxVC,
            "mtl_tts": real_vendor["mtl_tts"].ChatterboxMultilingualTTS,
            "tts_turbo": real_vendor["tts_turbo"].ChatterboxTurboTTS,
        }
        for key, module in real_vendor.items():
            assert hasattr(module, "_require_checkpoint_files"), key
            assert module.REQUIRED_CHECKPOINT_FILES, key
        for key, entry_point in entry_points.items():
            source = inspect.getsource(entry_point.from_local)
            assert "_require_checkpoint_files(" in source, key

    def test_the_multilingual_set_is_the_pt_layout(self, real_vendor) -> None:
        """The grapheme tokenizer is fixed; ve + s3gen + t3 resolve per pack (P-036).

        The resolution order prefers the safetensors serializations -- the
        ``.pt`` twins hold the same weights and are never fetched.
        """
        assert MTL_CHECKPOINT_FILES == ("grapheme_mtl_merged_expanded_v1.json",)
        assert MTL_S3GEN_CHECKPOINTS == (
            "s3gen_v3.safetensors",
            "s3gen_v3.pt",
            "s3gen.safetensors",
            "s3gen.pt",
        )
        assert real_vendor["mtl_tts"].REQUIRED_CHECKPOINT_FILES == MTL_CHECKPOINT_FILES

    def test_the_turbo_set_is_shared_by_turbo_and_nano(self, real_vendor) -> None:
        """One module, two weight sets -- so one required list, plus two names.

        The tokenizer is not in the constant: Turbo ships ``tokenizer.json``
        while Nano ships ``vocab.json`` + ``merges.txt``, so the loaders pick
        the file per pack layout (P-035). ``s3gen_meanflow.safetensors`` is,
        because the vendored loader builds ``S3Gen(meanflow=True)`` from it
        unconditionally.
        """
        assert TURBO_CHECKPOINT_FILES == (
            "ve.safetensors",
            "s3gen.safetensors",
            "s3gen_meanflow.safetensors",
        )
        assert real_vendor["tts_turbo"].REQUIRED_CHECKPOINT_FILES == TURBO_CHECKPOINT_FILES
        # The T3 checkpoint is the one name that is not fixed, so the caller adds
        # it rather than the constant claiming to know it.
        assert "t3_turbo_v1.safetensors" not in TURBO_CHECKPOINT_FILES
        assert "t3_nano_v1.safetensors" not in TURBO_CHECKPOINT_FILES
        assert "tokenizer.json" not in TURBO_CHECKPOINT_FILES

    def test_the_loader_refusal_names_the_missing_files_too(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            ChatterboxLoader,
            "_ensure_entry_point_dir",
            staticmethod(lambda *a, **k: None),
        )
        empty = tmp_path / "half_a_pack"
        empty.mkdir()
        with pytest.raises(FileNotFoundError) as caught:
            ChatterboxLoader.load_multilingual_model(
                "half_a_pack", "t3_mtl23ls_v3.safetensors", torch.device("cpu"),
                model_path=str(empty),
            )
        message = str(caught.value)
        assert "t3_mtl23ls_v3.safetensors" in message
        assert "ve.safetensors" in message and "s3gen_v3.safetensors" in message
        assert str(empty) in message

    def test_the_turbo_loader_refusal_names_the_missing_files(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            ChatterboxLoader,
            "_ensure_entry_point_dir",
            staticmethod(lambda *a, **k: None),
        )
        empty = tmp_path / "half_a_turbo"
        empty.mkdir()
        with pytest.raises(FileNotFoundError) as caught:
            ChatterboxLoader.load_turbo_model(
                "half_a_turbo", torch.device("cpu"), nano=True, model_path=str(empty)
            )
        message = str(caught.value)
        assert "Nano" in message
        assert "t3_nano_v1.safetensors" in message
        # P-035: the dynamic tokenizer file, not the Turbo-only name.
        assert "vocab.json" in message
        assert "tokenizer.json" not in message

    def test_a_nano_shaped_pack_passes_the_turbo_completeness_check(self, tmp_path) -> None:
        """The real Nano repo layout: no tokenizer.json, HF tokenizer instead."""
        pack = tmp_path / "nano"
        pack.mkdir()
        for name in (
            "t3_nano_v1.safetensors",
            "ve.safetensors",
            "s3gen.safetensors",
            "s3gen_meanflow.safetensors",
            "vocab.json",
            "merges.txt",
        ):
            (pack / name).write_bytes(b"\x00" * 4)

        missing = ChatterboxLoader._missing_checkpoint_files(
            str(pack),
            (*TURBO_CHECKPOINT_FILES, "vocab.json", "t3_nano_v1.safetensors"),
        )
        assert missing == [], "the real Nano layout must not be called incomplete"
        # And the generic sniff (which the classic loader gates downloads on)
        # recognises it too, so the pack stays selectable and downloadable.
        from modules.model_info import is_chatterbox_model_dir

        assert is_chatterbox_model_dir(str(pack))

    def test_the_classic_loader_refuses_a_turbo_pack_with_a_pointer(self, tmp_path) -> None:
        """A Turbo pack in the classic node pointed at the Turbo node, by name."""
        pack = tmp_path / "chatterbox-nano"
        pack.mkdir()
        for name in ("t3_nano_v1.safetensors", "ve.safetensors", "vocab.json"):
            (pack / name).write_bytes(b"\x00" * 4)
        with pytest.raises(FileNotFoundError) as caught:
            ChatterboxLoader.load_model(
                "chatterbox-nano", torch.device("cpu"), model_path=str(pack)
            )
        message = str(caught.value)
        assert "Turbo" in message and "nano" in message
        assert "t3_cfg.safetensors" not in message, (
            "the classic missing-file list must not imply the pack is broken"
        )

    def test_a_flash_pack_is_refused_by_name_before_any_download(self, tmp_path, monkeypatch) -> None:
        """Flash ships in its own (vendored) package; the refusal must not trigger one.

        The guard runs before ``_ensure_entry_point_dir``, so selecting a Flash
        pack can never start downloading the Turbo repo into the Flash
        directory.
        """
        downloads: list[tuple] = []
        monkeypatch.setattr(
            ChatterboxLoader,
            "_ensure_entry_point_dir",
            staticmethod(lambda *a, **k: downloads.append(a)),
        )
        pack = tmp_path / "chatterbox-flash"
        pack.mkdir()
        for name in ("t3_flash.safetensors", "ve.safetensors", "s3gen.safetensors", "tokenizer.json"):
            (pack / name).write_bytes(b"\x00" * 4)

        with pytest.raises(FileNotFoundError) as caught:
            ChatterboxLoader.load_turbo_model(
                "chatterbox-flash", torch.device("cpu"), model_path=str(pack)
            )
        assert "chatterbox-flash" in str(caught.value)
        assert "block-diffusion" in str(caught.value)
        assert downloads == []
        with pytest.raises(FileNotFoundError) as caught:
            ChatterboxLoader.load_model(
                "chatterbox-flash", torch.device("cpu"), model_path=str(pack)
            )
        assert "chatterbox-flash" in str(caught.value)

    def test_a_complete_pack_is_not_re_downloaded(self, tmp_path, monkeypatch) -> None:
        """The completeness test is per-entry-point, not the classic one.

        ``is_pack_complete`` looks for ``tokenizer.json`` plus a ``.safetensors``,
        which a *multilingual* pack does not have. Reusing it would re-download a
        perfectly good 4 GB checkpoint on every single run.
        """
        calls: list[str] = []
        monkeypatch.setattr(
            ChatterboxLoader,
            "download_pack",
            staticmethod(lambda name, *a, **k: calls.append(name)),
        )
        pack = tmp_path / "mlt"
        pack.mkdir()
        for name in (*MTL_CHECKPOINT_FILES, "t3_mtl23ls_v3.safetensors"):
            (pack / name).write_bytes(b"\x00" * 4)

        ChatterboxLoader._ensure_entry_point_dir(
            str(pack),
            "ResembleAI/chatterbox",
            (*MTL_CHECKPOINT_FILES, "t3_mtl23ls_v3.safetensors"),
            "mlt",
        )
        assert calls == [], "a complete multilingual pack was re-downloaded"


    def test_a_flash_pack_loads_through_the_flash_entry_point(
        self, tmp_path, monkeypatch
    ) -> None:
        """The flash loader checks its own file set, then builds the pack."""
        pack = tmp_path / "chatterbox-flash"
        pack.mkdir()
        for name in ("t3_flash.safetensors", "ve.safetensors", "s3gen.safetensors", "tokenizer.json"):
            (pack / name).write_bytes(b"\x00" * 4)
        seen: dict = {}

        class FakeFlashTTS:
            @classmethod
            def from_local(cls, ckpt_dir, device):
                seen["ckpt_dir"] = str(ckpt_dir)
                return object()

        import sys as _sys
        import types as _types

        monkeypatch.setattr(
            ChatterboxLoader, "download_pack",
            staticmethod(lambda *a, **k: None),
        )
        stub = _types.ModuleType(FLASH_PACKAGE)
        stub.ChatterboxFlashTTS = FakeFlashTTS
        monkeypatch.setitem(_sys.modules, FLASH_PACKAGE, stub)

        ChatterboxLoader.load_flash_model(
            "chatterbox-flash", torch.device("cpu"), model_path=str(pack)
        )
        assert seen["ckpt_dir"] == str(pack)

    def test_a_failed_flash_import_names_the_vendored_tree_and_bans_pip(
        self, tmp_path, monkeypatch
    ) -> None:
        """A dead ``chatterbox_flash`` import blames the checkout, not pip.

        The package is vendored, so the message must point at
        ``src/chatterbox_flash`` and forbid a pip install (whose dependency
        block would shadow the vendored Chatterbox) rather than name one.
        """
        monkeypatch.setattr(
            ChatterboxLoader, "download_pack",
            staticmethod(lambda *a, **k: None),
        )
        pack = tmp_path / "chatterbox-flash"
        pack.mkdir()
        for name in ("t3_flash.safetensors", "ve.safetensors", "s3gen.safetensors", "tokenizer.json"):
            (pack / name).write_bytes(b"\x00" * 4)
        import sys as _sys

        saved = _sys.modules.pop(FLASH_PACKAGE, None)
        _sys.modules[FLASH_PACKAGE] = None  # import raises ImportError
        try:
            with pytest.raises(FileNotFoundError) as caught:
                ChatterboxLoader.load_flash_model(
                    "chatterbox-flash", torch.device("cpu"), model_path=str(pack)
                )
        finally:
            if saved is not None:
                _sys.modules[FLASH_PACKAGE] = saved
            else:
                del _sys.modules[FLASH_PACKAGE]
        message = str(caught.value)
        assert "src/chatterbox_flash" in message
        assert "pip-install" in message and "no-deps" not in message

    def test_an_incomplete_flash_pack_lists_its_own_files(
        self, tmp_path, monkeypatch
    ) -> None:
        pack = tmp_path / "chatterbox-flash"
        pack.mkdir()
        (pack / "t3_flash.safetensors").write_bytes(b"\x00" * 4)
        # The stub keeps the incompleteness refusal from "fixing" the pack
        # with a real download of the flash repo.
        monkeypatch.setattr(
            ChatterboxLoader, "download_pack",
            staticmethod(lambda *a, **k: None),
        )
        with pytest.raises(FileNotFoundError) as caught:
            ChatterboxLoader.load_flash_model(
                "chatterbox-flash", torch.device("cpu"), model_path=str(pack)
            )
        message = str(caught.value)
        assert "ve.safetensors" in message and "tokenizer.json" in message
        assert "t3_cfg.safetensors" not in message
