"""S2.3 -- the watermark opt-out upstream does not have.

Three claims, checked against the shipped code rather than a paraphrase of it:

* the default (``watermark=True``) still routes audio through the watermarker, so
  every existing caller keeps upstream's behaviour byte for byte;
* ``watermark=False`` returns clean, un-watermarked audio and touches neither the
  watermarker nor the log;
* the ``Conditionals`` save/load round-trip is untouched by any of it -- the
  watermark flag lives on the call, not in the serialised conditionals.

``conftest.py`` mocks the vendor tree, so the modules are never imported. The
statements each check needs are extracted from the source and executed in a
fresh namespace; that is the shipped code, running for real.
"""

from __future__ import annotations

import ast
import inspect
import logging
import os

import pytest
import torch

from conftest import ROOT_DIR
from tests.test_vendor_perth_patch import (
    ENTRY_POINTS,
    Recorder,
    load_watermark_helper,
    read,
    vendor_path,
)

CLASS_NAME = "ChatterboxTTS"


def parse(rel_file: str) -> ast.Module:
    return ast.parse(read(rel_file), filename=vendor_path(rel_file))


@pytest.fixture(params=ENTRY_POINTS, ids=lambda p: os.path.basename(p))
def entry_point(request) -> str:
    return request.param


class TestDefaultPreservesUpstreamBehaviour:
    def test_signature_default_is_true(self, entry_point: str) -> None:
        """``generate(..., watermark=True)`` -- old callers are unaffected."""
        helper = load_watermark_helper(entry_point)["_apply_watermark"]
        assert inspect.signature(helper).parameters["watermark"].default is True

    def test_default_call_reaches_the_watermarker(self, entry_point: str) -> None:
        apply = load_watermark_helper(entry_point)["_apply_watermark"]
        recorder = Recorder()
        wav = object()
        result = apply(wav, 24000, recorder)
        assert result == "watermarked"
        assert recorder.calls == [(wav, 24000)], (
            "the default path must still call apply_watermark with the audio and "
            "the model sample rate, exactly as upstream did"
        )

    def test_explicit_true_behaves_like_the_default(self, entry_point: str) -> None:
        apply = load_watermark_helper(entry_point)["_apply_watermark"]
        recorder = Recorder()
        apply(object(), 24000, recorder, watermark=True)
        assert len(recorder.calls) == 1


class TestOptOutReturnsCleanAudio:
    def test_false_returns_the_same_object(self, entry_point: str) -> None:
        apply = load_watermark_helper(entry_point)["_apply_watermark"]
        recorder = Recorder()
        wav = torch.zeros(4)
        out = apply(wav, 24000, recorder, watermark=False)
        assert out is wav, "watermark=False must hand back the untouched waveform"
        assert recorder.calls == []

    def test_false_never_consults_a_missing_library(self, entry_point: str) -> None:
        """With no watermarker object *and* no opt-out, upstream would have crashed."""
        apply = load_watermark_helper(entry_point)["_apply_watermark"]
        wav = torch.zeros(4)
        assert apply(wav, 24000, None, watermark=False) is wav

    def test_false_does_not_warn(self, entry_point: str, caplog) -> None:
        apply = load_watermark_helper(entry_point)["_apply_watermark"]
        with caplog.at_level(logging.WARNING, logger="chatterbox.watermark"):
            apply(torch.zeros(4), 24000, None, watermark=False)
        assert caplog.records == []


def _conditionals_class(rel_file: str) -> ast.ClassDef:
    module = parse(rel_file)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "Conditionals":
            return node
    raise AssertionError(f"{rel_file} no longer defines Conditionals")


class _StubT3Cond:
    """Stand-in for the real ``T3Cond``, which needs the model stack to import."""

    def __init__(self, **fields) -> None:
        self.__dict__.update(fields)

    def __eq__(self, other) -> bool:
        if not isinstance(other, _StubT3Cond):
            return NotImplemented
        if set(other.__dict__) != set(self.__dict__):
            return False
        return all(
            torch.equal(left, right) if torch.is_tensor(left) else left == right
            for left, right in zip(self.__dict__.values(), other.__dict__.values())
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_StubT3Cond({sorted(self.__dict__)})"


def load_conditionals(rel_file: str) -> type:
    """Execute just the ``Conditionals`` dataclass in isolation.

    Only ``import torch``, ``from dataclasses import dataclass``, ``from pathlib
    import Path`` and the class body run. ``T3Cond`` is supplied as a stub so no
    model code is imported, and the round-trip below is a real ``torch.save`` /
    ``torch.load``.
    """
    keep: list[ast.stmt] = []
    for node in parse(rel_file).body:
        if isinstance(node, ast.Import) and any(
            alias.name == "torch" for alias in node.names
        ):
            keep.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module in {
            "dataclasses",
            "pathlib",
        }:
            keep.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "Conditionals":
            keep.append(node)

    namespace: dict = {"T3Cond": _StubT3Cond}
    exec(  # noqa: S102 -- executing the shipped source, by design
        compile(ast.Module(body=keep, type_ignores=[]), vendor_path(rel_file), "exec"),
        namespace,
    )
    assert "Conditionals" in namespace, f"{rel_file} lost Conditionals"
    return namespace["Conditionals"]


def _conditionals_files() -> list[str]:
    """The entry points that define ``Conditionals`` (``vc.py`` does not)."""
    found: list[str] = []
    for rel_file in ENTRY_POINTS:
        module = parse(rel_file)
        if any(
            isinstance(node, ast.ClassDef) and node.name == "Conditionals"
            for node in module.body
        ):
            found.append(rel_file)
    assert found, "no entry point defines Conditionals any more"
    return found


@pytest.fixture(params=_conditionals_files(), ids=lambda p: os.path.basename(p))
def conditionals_file(request) -> str:
    return request.param


class TestConditionalsRoundTripIsUnaffected:
    """The watermark flag is a call argument; it must not leak into the cache."""

    def test_save_load_round_trip(self, conditionals_file: str, tmp_path) -> None:
        conditionals = load_conditionals(conditionals_file)
        original = conditionals(
            t3=_StubT3Cond(
                speaker_emb=torch.arange(6, dtype=torch.float32),
                emotion_adv=torch.tensor([[[0.5]]]),
            ),
            gen={"prompt_feat": torch.arange(4, dtype=torch.float32), "embedding": None},
        )
        target = tmp_path / "conds.pt"
        original.save(target)
        restored = conditionals.load(target)

        assert restored.t3 == original.t3, "t3 conditionals did not survive the trip"
        assert torch.equal(restored.gen["prompt_feat"], original.gen["prompt_feat"])
        assert restored.gen["embedding"] is None

    def test_the_flag_is_not_part_of_the_serialised_state(
        self, conditionals_file: str, tmp_path
    ) -> None:
        conditionals = load_conditionals(conditionals_file)
        instance = conditionals(t3=_StubT3Cond(a=1), gen={})
        assert not hasattr(instance, "watermark"), (
            "watermark must be a generate() argument, not instance state; "
            "anything else would be written into the model pack's conds.pt"
        )
        target = tmp_path / "conds.pt"
        instance.save(target)
        payload = torch.load(target, map_location="cpu", weights_only=True)
        assert set(payload) == {"t3", "gen"}, (
            f"the saved payload changed shape: {sorted(payload)}"
        )

    def test_conditionals_still_exposes_save_and_load(self, conditionals_file: str) -> None:
        cls = _conditionals_class(conditionals_file)
        members = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}
        assert {"save", "load", "to"} <= members, (
            f"{conditionals_file}: Conditionals lost "
            f"{sorted({'save', 'load', 'to'} - members)}"
        )


class TestNoWatermarkStateLeaksIntoTheConstructor:
    def test_init_still_takes_the_same_arguments(self, entry_point: str) -> None:
        """The patch must not have widened the constructor's public signature."""
        module = parse(entry_point)
        classes = [
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and any(
                isinstance(child, ast.FunctionDef) and child.name == "__init__"
                for child in node.body
            )
        ]
        assert classes, f"{entry_point} has no class with an __init__"
        for cls in classes:
            init = next(
                child
                for child in cls.body
                if isinstance(child, ast.FunctionDef) and child.name == "__init__"
            )
            names = [arg.arg for arg in init.args.args]
            assert names[0] == "self"
            assert "watermark" not in names, (
                f"{entry_point}.{cls.name}.__init__ grew a `watermark` argument; "
                "watermarking is chosen per call"
            )


def test_repo_entry_point_no_longer_injects_a_perth_stub() -> None:
    """S2.4: the repo ``__init__.py`` used to install a fake ``perth`` module."""
    repo_init = os.path.join(ROOT_DIR, "__init__.py")
    with open(repo_init, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert "sys.modules['perth']" not in source and 'sys.modules["perth"]' not in source, (
        "the perth stub is still injected into sys.modules; patch the source "
        "instead (src/PATCHES.md P-023..P-026)"
    )
    assert "ensure_vendor_on_path" not in source and "vendor_loader" not in source, (
        "the entry point must not publish the vendored src/ as a top-level "
        "import root; the tree is imported package-relatively"
    )


def test_manifest_records_every_entry_point(entry_point: str) -> None:
    """Each optional-dependency patch is on the record, by name."""
    manifest = os.path.join(ROOT_DIR, "src", "PATCHES.md")
    with open(manifest, "r", encoding="utf-8") as handle:
        text = handle.read()
    module_name = entry_point.split("/")[-1]
    assert f"`chatterbox/{module_name}`" in text, (
        f"{module_name} is patched but src/PATCHES.md does not mention it"
    )