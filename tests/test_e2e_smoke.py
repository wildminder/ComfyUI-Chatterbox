"""Env-gated real generation smoke test.

Everything else in the suite mocks the vendor tree, which means nothing here
proves the model still produces audible speech. This file does, but only when a
real checkpoint is pointed at explicitly:

    RUN_CHATTERBOX_E2E=1 CHATTERBOX_MODEL_DIR=/path/to/chatterbox pytest tests/

Both variables are required. **No machine path is ever committed** -- the model
directory arrives through the environment, never a constant in this file -- and
with the variables unset every test skips with a named reason, which is the
correct and expected outcome for a developer machine and for CI.

The checks are deliberately about *signal*, not fidelity: a finite, non-silent
waveform of plausible length, and a monotonic relationship between script length
and audio length. Those are exactly the properties a bad re-vendor or a dropped
patch breaks first.
"""

from __future__ import annotations

import math
import os

import pytest

#: Set to ``1`` to opt in. Anything else leaves the suite skipped.
ENV_ENABLE = "RUN_CHATTERBOX_E2E"
#: Absolute path to a downloaded chatterbox checkpoint directory.
ENV_MODEL_DIR = "CHATTERBOX_MODEL_DIR"

SAMPLE_RATE = 24_000

#: A waveform quieter than this is silence, not speech. Chatterbox output sits
#: well above 0.01 RMS for normal text.
RMS_FLOOR = 0.01
#: Shorter than this and the model produced a click, not an utterance.
MIN_SECONDS = 0.5
#: Doubling the script must measurably lengthen the audio. A regression that
#: truncates generation, or a `max_new_tokens` that no longer reaches `T3`,
#: shows up here as a ratio near 1.
MIN_LENGTH_RATIO = 1.2

SHORT_SCRIPT = "The quick brown fox jumps over the lazy dog."
LONG_SCRIPT = " ".join([SHORT_SCRIPT] * 4)


def _e2e_enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "").strip() == "1"


def _model_dir() -> str | None:
    value = os.environ.get(ENV_MODEL_DIR, "").strip()
    return value or None


pytestmark = pytest.mark.skipif(
    not _e2e_enabled(),
    reason=(
        f"{ENV_ENABLE}=1 is not set: real generation is opt-in because it needs "
        f"a multi-gigabyte checkpoint and a GPU"
    ),
)


@pytest.fixture(scope="module")
def model_dir() -> str:
    directory = _model_dir()
    if directory is None:
        pytest.skip(f"{ENV_MODEL_DIR} is not set: no checkpoint to load")
    if not os.path.isdir(directory):
        pytest.skip(f"{ENV_MODEL_DIR} points at {directory!r}, which does not exist")
    return directory


@pytest.fixture(scope="module")
def reference_wav(model_dir: str) -> str:
    """A reference clip for voice conditioning.

    Either an audio file already in the checkpoint directory, or one named by
    ``CHATTERBOX_REF_WAV``. The path is supplied through the environment so no
    machine-specific location lands in this repository.
    """
    explicit = os.environ.get("CHATTERBOX_REF_WAV", "").strip()
    if explicit:
        if not os.path.isfile(explicit):
            pytest.skip(f"CHATTERBOX_REF_WAV points at {explicit!r}, which does not exist")
        return explicit

    for name in ("reference.wav", "ref.wav", "conds.wav", "speaker.wav"):
        candidate = os.path.join(model_dir, name)
        if os.path.isfile(candidate):
            return candidate

    pytest.skip(
        "no reference clip found: put a .wav in the checkpoint directory or set "
        "CHATTERBOX_REF_WAV"
    )


@pytest.fixture(scope="module")
def model(model_dir: str, reference_wav: str):
    """A loaded ``ChatterboxTTS``, moved off the mock tree for this process only.

    The vendored package is mocked for the rest of the suite; this fixture is the
    one place it is imported for real, which is why it only runs under the opt-in.
    """
    import torch

    from .conftest_helpers import load_real_vendor  # noqa: PLC0415 -- opt-in only

    load_real_vendor()
    from ComfyUI_Chatterbox.src.chatterbox.tts import ChatterboxTTS  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("no CUDA device: real generation on CPU is too slow to be useful")

    tts = ChatterboxTTS.from_local(model_dir, device=device)
    tts.prepare_conditionals(reference_wav)
    return tts


def rms(wave) -> float:
    """Root-mean-square level of a 1-D waveform."""
    import torch

    tensor = torch.as_tensor(wave).float().reshape(-1)
    return float(torch.sqrt(torch.mean(tensor**2)).item())


def duration_seconds(wave, sample_rate: int = SAMPLE_RATE) -> float:
    import torch

    return float(torch.as_tensor(wave).reshape(-1).numel()) / sample_rate


def synthesize(model, text: str):
    """Generate with a fixed seed so the length comparison is reproducible."""
    import torch

    torch.manual_seed(0)
    return model.generate(
        text=text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        max_new_tokens=1000,
        flow_cfg_scale=0.7,
    )


class TestRealGeneration:
    def test_waveform_is_finite(self, model) -> None:
        wav = synthesize(model, SHORT_SCRIPT)
        assert wav.numel() > 0
        assert bool(wav.isfinite().all()), "the generated waveform contains NaN or inf"

    def test_waveform_is_not_silent(self, model) -> None:
        wav = synthesize(model, SHORT_SCRIPT)
        level = rms(wav)
        assert level > RMS_FLOOR, (
            f"RMS {level:.5f} is at or below the silence floor {RMS_FLOOR}; "
            f"the model produced audio of the right length but no signal"
        )

    def test_duration_is_plausible(self, model) -> None:
        wav = synthesize(model, SHORT_SCRIPT)
        seconds = duration_seconds(wav)
        assert seconds > MIN_SECONDS, f"only {seconds:.3f}s of audio was produced"

    def test_duration_scales_with_script_length(self, model) -> None:
        """The strongest end-to-end check on the ``max_new_tokens`` patch.

        If the node's length knob stopped reaching ``T3.inference``, or the
        sampling loop stopped honouring it, a four-times-longer script would
        produce roughly the same audio.
        """
        short = synthesize(model, SHORT_SCRIPT)
        long = synthesize(model, LONG_SCRIPT)

        short_seconds = duration_seconds(short)
        long_seconds = duration_seconds(long)
        ratio = long_seconds / short_seconds

        assert math.isfinite(ratio)
        assert ratio > MIN_LENGTH_RATIO, (
            f"4x the script produced only {ratio:.2f}x the audio "
            f"({short_seconds:.2f}s -> {long_seconds:.2f}s)"
        )

    def test_generation_is_deterministic_under_a_fixed_seed(self, model) -> None:
        first = synthesize(model, SHORT_SCRIPT)
        second = synthesize(model, SHORT_SCRIPT)
        assert duration_seconds(first) == duration_seconds(second)

    def test_max_new_tokens_reaches_the_model(self, model) -> None:
        """A very small budget must truncate; proves the kwarg is not ignored."""
        import torch

        torch.manual_seed(0)
        short_budget = model.generate(
            text=SHORT_SCRIPT,
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.8,
            repetition_penalty=1.2,
            min_p=0.05,
            top_p=1.0,
            max_new_tokens=8,
            flow_cfg_scale=0.7,
        )
        torch.manual_seed(0)
        full_budget = model.generate(
            text=SHORT_SCRIPT,
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.8,
            repetition_penalty=1.2,
            min_p=0.05,
            top_p=1.0,
            max_new_tokens=1000,
            flow_cfg_scale=0.7,
        )
        assert duration_seconds(short_budget) < duration_seconds(full_budget), (
            "max_new_tokens had no effect on the generated length; the "
            "T3.inference patch (P-019) is not wired through"
        )


class TestVoiceConversion:
    def test_vc_produces_audio(self, model_dir: str, reference_wav: str) -> None:
        import torch

        from .conftest_helpers import load_real_vendor  # noqa: PLC0415

        load_real_vendor()
        from ComfyUI_Chatterbox.src.chatterbox.vc import ChatterboxVC  # noqa: PLC0415

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")

        vc = ChatterboxVC.from_local(model_dir, device="cuda")
        torch.manual_seed(0)
        wav = vc.generate(
            audio=reference_wav,
            target_voice_path=reference_wav,
            n_timesteps=10,
            flow_cfg_scale=0.7,
        )
        assert wav.numel() > 0
        assert bool(wav.isfinite().all())
        assert rms(wav) > RMS_FLOOR
        assert duration_seconds(wav) > MIN_SECONDS

#: Long enough to be cut into more than one chunk at the default limit.
CHUNKED_SCRIPT = " ".join([SHORT_SCRIPT] * 12)


class TestChunkedRealGeneration:
    """The chunked path, against the real model.

    The same property as the unchunked test -- a longer script has to produce
    longer audio -- but through the chunk loop and the join, so it also proves
    that real waveforms survive being cut and put back together.
    """

    def test_a_long_script_grows_with_the_chunked_path(self, model) -> None:
        from ComfyUI_Chatterbox.modules.generation import (  # noqa: PLC0415
            generate_audio,
        )

        short, rate = generate_audio(
            model, SHORT_SCRIPT, seed=1, max_chars_per_chunk=400
        )
        long, _ = generate_audio(
            model, CHUNKED_SCRIPT, seed=1, max_chars_per_chunk=400
        )

        short_seconds = duration_seconds(short, rate)
        long_seconds = duration_seconds(long, rate)
        ratio = long_seconds / short_seconds

        assert math.isfinite(ratio)
        assert ratio > MIN_LENGTH_RATIO, (
            f"12x the script produced only {ratio:.2f}x the audio "
            f"({short_seconds:.2f}s -> {long_seconds:.2f}s) through the "
            "chunked path"
        )

    def test_the_chunked_audio_is_not_silent(self, model) -> None:
        from ComfyUI_Chatterbox.modules.generation import (  # noqa: PLC0415
            generate_audio,
        )

        wave, _ = generate_audio(
            model, CHUNKED_SCRIPT, seed=1, max_chars_per_chunk=400
        )
        level = rms(wave)
        assert level > RMS_FLOOR, (
            f"RMS {level:.5f} is at or below the silence floor {RMS_FLOOR}; "
            "the chunked path produced audio of the right length but no signal"
        )
