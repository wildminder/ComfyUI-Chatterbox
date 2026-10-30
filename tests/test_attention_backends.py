"""S8.9b -- attention backends: same shape, finite output, and a refusal not a crash.

The vendored ``AttentionQKV`` is the module this project actually spends its
attention through, so it is what runs here -- two head configurations, a real
forward, and the real ``sdpa_kernel`` context P-010 introduced.

Three properties are asserted, and the third is the one that has historically
been a crash rather than a message:

* **Every available backend yields the same shape and a finite result.** A
  backend that silently produced NaN would otherwise look exactly like one that
  worked until a user listened to it.
* **``"auto"`` equals the default backend.** "Let the machine decide" has to be
  the same answer as naming the default explicitly; if it resolved to something
  else, a saved graph would run differently from the one it was saved from.
* **Sage on sm100/sm103 is refused, not crashed.** SageAttention has no kernel
  for Blackwell datacenter parts and raises ``ValueError: Unsupported CUDA
  architecture`` deep inside the attention call. The project catches that at
  queue time and at load time by checking membership of
  ``SAGE_SUPPORTED_ARCHS``, so the refusal is a ``False`` and a log line, and the
  mode falls back to sdpa. These tests pin that.
"""

from __future__ import annotations

import inspect
import logging
from unittest.mock import patch

import pytest
import torch

from ComfyUI_Chatterbox.modules import attention_utils
from ComfyUI_Chatterbox.modules.attention_utils import (
    ATTENTION_AUTO,
    ATTENTION_MODES,
    DEFAULT_ATTENTION_MODE,
    SAGE_SUPPORTED_ARCHS,
    check_sage_attention_compatible,
    get_attn_implementation_for_load,
    get_available_attention_modes,
    resolve_attention_mode,
)

from conftest import ROOT_DIR


@pytest.fixture(scope="module")
def attention_qkv():
    """The real vendored ``AttentionQKV``, with the harness mocks restored after.

    ``conftest.py`` mocks every name under the vendored tree -- under the node
    package's alias, never under the top-level ``chatterbox`` -- so collection
    never loads the model stack. This one class is cheap -- it imports torch and
    einops only -- and it is the module whose behaviour the attention tests are
    about, so the mocks are lifted for this fixture and put back on teardown by
    :func:`conftest_helpers.real_vendor_modules`, which restores the one prefix
    the harness registers.
    """
    from .conftest_helpers import real_vendor_modules

    with real_vendor_modules(["models.t3.modules.perceiver"]) as modules:
        yield modules["models.t3.modules.perceiver"].AttentionQKV


def _inputs(batch: int = 1, length: int = 8, dim: int = 64, dtype=torch.float32):
    generator = torch.Generator().manual_seed(20261004)
    return tuple(
        torch.randn(batch, length, dim, generator=generator, dtype=dtype)
        for _ in range(3)
    )


# ---------------------------------------------------------------------------
# Every available backend
# ---------------------------------------------------------------------------


class TestEveryAvailableBackend:
    def test_the_fixture_reached_the_real_module(self, attention_qkv) -> None:
        source = inspect.getsource(attention_qkv)
        assert "class AttentionQKV" in source
        assert "scaled_dot_product_attention" in source
        assert "sdpa_kernel" in source, (
            "the vendored attention no longer carries the P-010 fix"
        )

    def test_eager_and_fused_agree_on_shape_and_are_finite(self, attention_qkv) -> None:
        eager = attention_qkv(4, 16, dropout_rate=0.0, flash=False)
        fused = attention_qkv(4, 16, dropout_rate=0.0, flash=True)
        eager.eval()
        fused.eval()

        q, k, v = _inputs()
        with torch.no_grad():
            out_eager = eager(q, k, v)
            out_fused = fused(q, k, v)

        assert out_eager.shape == q.shape == out_fused.shape
        assert torch.isfinite(out_eager).all()
        assert torch.isfinite(out_fused).all()

    def test_the_fused_path_is_correct_attention(self, attention_qkv) -> None:
        """The default path -- ``flash=True``, and the only one upstream builds.

        ``AttentionBlock2`` constructs ``AttentionQKV(..., flash_attention=True)``
        with no argument, so this is the code every real forward runs. It is
        checked against a reference attention written out here, so the assertion
        is about the mathematics rather than about agreeing with itself.
        """
        fused = attention_qkv(4, 16, dropout_rate=0.0, flash=True).eval()
        q, k, v = _inputs()
        qh, kh, vh = (t.view(1, 8, 4, 16).permute(0, 2, 1, 3) for t in (q, k, v))
        scores = torch.einsum("bhid,bhjd->bhij", qh, kh) * (16**-0.5)
        reference = torch.einsum(
            "bhij,bhjd->bhid", torch.softmax(scores, dim=-1), vh
        ).permute(0, 2, 1, 3).reshape(q.shape)
        with torch.no_grad():
            assert torch.allclose(fused(q, k, v), reference, atol=1e-5, rtol=1e-4)

    def test_the_unused_eager_path_computes_something_different(self, attention_qkv) -> None:
        """A recorded upstream quirk, not a fix.

        ``AttentionQKV.scaled_dot_product_attention`` contracts
        ``bhlt,bhls->bhts`` -- over the **sequence** axis ``l`` rather than the
        head dimension -- and then indexes its own output by the head dimension,
        so it is not the same function as the fused path it is the fallback for.
        Nothing constructs it: the only ``AttentionQKV`` call site passes the
        default ``flash_attention=True``.

        This test exists so the divergence is a recorded fact rather than a
        surprise. It is *not* patched -- a vendored maths change in a dead branch
        is not this batch's job, and a patch would need its own entry in
        ``src/PATCHES.md`` and a decision about whether upstream would take it.
        """
        eager = attention_qkv(4, 16, dropout_rate=0.0, flash=False).eval()
        fused = attention_qkv(4, 16, dropout_rate=0.0, flash=True).eval()
        q, k, v = _inputs()
        with torch.no_grad():
            assert not torch.allclose(eager(q, k, v), fused(q, k, v), atol=1e-3)

    def test_the_only_construction_site_uses_the_fused_path(self) -> None:
        """So the quirk above stays a quirk and never becomes a live bug."""
        import os

        with open(
            os.path.join(
                ROOT_DIR,
                "src",
                "chatterbox",
                "models",
                "t3",
                "modules",
                "perceiver.py",
            ),
            encoding="utf-8",
        ) as handle:
            source = handle.read()

        # Two mentions: the class definition and the one construction site.
        assert source.count("AttentionQKV(") == 2, (
            "AttentionQKV gained a second construction site; check whether it "
            "passes flash=False before trusting the divergence test above"
        )
        assert "flash=flash_attention" in source.split("AttentionQKV(")[2]
        assert "flash_attention=True" in source, (
            "AttentionBlock2's flash_attention default changed; if it is now "
            "False the unused eager path has become the live one"
        )

    @pytest.mark.parametrize("batch,length,heads,head_dim", [
        (1, 8, 4, 16), (2, 32, 8, 32), (1, 128, 2, 64),
    ])
    def test_shape_and_finiteness_hold_across_shapes(self, attention_qkv,
                                                     batch, length, heads, head_dim) -> None:
        module = attention_qkv(heads, head_dim, dropout_rate=0.0, flash=True).eval()
        q, k, v = _inputs(batch=batch, length=length, dim=heads * head_dim)
        with torch.no_grad():
            out = module(q, k, v)
        assert out.shape == (batch, length, heads * head_dim)
        assert torch.isfinite(out).all()

    def test_a_mask_is_honoured_and_the_output_stays_finite(self, attention_qkv) -> None:
        module = attention_qkv(4, 16, dropout_rate=0.0, flash=True).eval()
        q, k, v = _inputs()
        mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        mask[..., 4:] = False
        with torch.no_grad():
            out = module(q, k, v, mask=mask)
        assert torch.isfinite(out).all()

    def test_the_dropout_rate_does_not_leak_into_inference(self, attention_qkv) -> None:
        """A non-zero ``dropout_rate`` must not randomise an eval-mode forward."""
        module = attention_qkv(4, 16, dropout_rate=0.5, flash=True).eval()
        q, k, v = _inputs()
        with torch.no_grad():
            first = module(q, k, v)
            second = module(q, k, v)
        assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# "auto" is the default backend
# ---------------------------------------------------------------------------


class TestAutoEqualsTheDefault:
    def test_auto_resolves_to_the_declared_default(self) -> None:
        assert DEFAULT_ATTENTION_MODE == "sdpa"
        assert resolve_attention_mode(ATTENTION_AUTO) == DEFAULT_ATTENTION_MODE

    def test_auto_and_sdpa_produce_the_same_load_implementation(self) -> None:
        auto_impl = get_attn_implementation_for_load(
            resolve_attention_mode(ATTENTION_AUTO)
        )
        default_impl = get_attn_implementation_for_load(DEFAULT_ATTENTION_MODE)
        assert auto_impl == default_impl == "sdpa"

    def test_auto_is_resolved_before_the_unknown_mode_branch(self) -> None:
        """Otherwise ``"auto"`` falls through to ``eager`` and costs performance.

        The unknown-mode fallback is the right answer for a *typo*; it is the
        wrong answer for a deliberate "the machine decides".
        """
        assert ATTENTION_AUTO not in ATTENTION_MODES, (
            "'auto' is a request, not a backend, and must never appear in the "
            "dropdown"
        )
        assert resolve_attention_mode(ATTENTION_AUTO) != "eager"
        assert resolve_attention_mode("flash-attn-9") == "eager", (
            "a genuine typo should still fall back to the hardware-free backend"
        )

    def test_the_widget_default_and_auto_agree(self) -> None:
        """What the dropdown starts on is what ``"auto"`` means."""
        from ComfyUI_Chatterbox.nodes.common import runtime_widgets

        widgets = {widget.id: widget for widget in runtime_widgets()}
        assert widgets["attention_mode"].default == DEFAULT_ATTENTION_MODE


# ---------------------------------------------------------------------------
# sage is refused, not crashed
# ---------------------------------------------------------------------------


def _pretend_gpu(capability, available: bool = True):
    """Patch the three things ``check_sage_attention_compatible`` asks about."""
    return (
        patch.object(attention_utils, "SAGE_ATTENTION_AVAILABLE", True),
        patch.object(torch.cuda, "is_available", lambda: available),
        patch.object(torch.cuda, "get_device_capability", lambda *a, **k: capability),
    )


class TestSageIsRefusedNotCrashed:
    @pytest.mark.parametrize(
        "capability", [(10, 0), (10, 3)]
    )
    def test_blackwell_datacenter_is_refused(self, capability, caplog) -> None:
        """CC 10.x has no sage kernel and raises ``ValueError`` inside the call.

        A plain "compute capability major >= 8" test lets these through, and an
        ``arch_code >= 90`` branch then hands an SM90 kernel to silicon it was
        not built for. The set membership test is what prevents the crash.
        """
        library, cuda, device = _pretend_gpu(capability)
        with library, cuda, device, caplog.at_level(logging.WARNING):
            assert check_sage_attention_compatible() is False
            assert resolve_attention_mode("sage") == "sdpa"
        assert any("architectures SageAttention ships kernels for" in r.message
                   for r in caplog.records)

    @pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
    def test_the_fallback_is_a_warning_not_an_exception(
        self, capability, caplog
    ) -> None:
        """Nothing is raised: a saved graph naming sage has to keep opening."""
        library, cuda, device = _pretend_gpu((10, 0))
        with library, cuda, device, caplog.at_level(logging.WARNING):
            resolved = resolve_attention_mode("sage")
        assert resolved == "sdpa"
        assert not any(r.exc_info for r in caplog.records)

    @pytest.mark.parametrize("capability", [(10, 3)])
    def test_an_unsupported_arch_is_removed_from_the_dropdown(self, capability) -> None:
        library, cuda, device = _pretend_gpu((10, 3))
        with library, cuda, device:
            assert "sage" not in get_available_attention_modes()

    @pytest.mark.parametrize(
        "capability", [(8, 0), (8, 6), (8, 9), (9, 0), (12, 0)]
    )
    def test_a_supported_arch_still_works(self, capability) -> None:
        library, cuda, device = _pretend_gpu(capability)
        with library, cuda, device:
            assert check_sage_attention_compatible() is True
            assert resolve_attention_mode("sage") == "sage"
            assert "sage" in get_available_attention_modes()

    def test_blackwell_is_not_in_the_supported_set(self) -> None:
        assert "sm100" not in SAGE_SUPPORTED_ARCHS
        assert "sm103" not in SAGE_SUPPORTED_ARCHS
        assert "sm90" in SAGE_SUPPORTED_ARCHS

    def test_the_check_never_raises_without_cuda(self) -> None:
        library, cuda, device = _pretend_gpu((9, 0), available=False)
        with library, cuda, device:
            assert check_sage_attention_compatible() is False
            assert resolve_attention_mode("sage") == "sdpa"

    def test_sage_loads_as_sdpa_and_stays_sdpa(self) -> None:
        """Transformers has no sage implementation to ask for.

        The weights are built with ``"sdpa"`` and the kernel is patched in
        afterwards, so asking ``from_pretrained`` for sage would fail at the
        constructor rather than at the patch site.
        """
        assert get_attn_implementation_for_load("sage") == "sdpa"
        assert get_attn_implementation_for_load("sdpa") == "sdpa"

    def test_sage_needs_a_half_dtype(self) -> None:
        from ComfyUI_Chatterbox.modules.attention_utils import (
            check_dtype_attention_compatible,
        )

        message = check_dtype_attention_compatible("fp32", "sage")
        assert isinstance(message, str) and "fp32" in message and "sage" in message
        assert check_dtype_attention_compatible("bf16", "sage") is None
        assert check_dtype_attention_compatible("fp16", "sage") is None


# ---------------------------------------------------------------------------
# The node surface
# ---------------------------------------------------------------------------


class TestTheNodeSurface:
    def test_no_node_offers_an_unavailable_backend(self) -> None:
        """A dropdown is a promise; core range-checks it before ``execute()``."""
        from ComfyUI_Chatterbox.nodes import ChatterboxTTSNode, ChatterboxVCNode

        with patch.object(
            attention_utils, "get_available_attention_modes", lambda: ["eager", "sdpa"]
        ):
            import ComfyUI_Chatterbox.nodes.common as common

            with patch.object(common, "get_available_attention_modes",
                              return_value=["eager", "sdpa"]):
                for node in (ChatterboxTTSNode, ChatterboxVCNode):
                    options = node.GET_NODE_INFO_V1()["input"]["required"][
                        "attention_mode"
                    ][1]["options"]
                    assert options == ["eager", "sdpa"]
                    assert "sage" not in options
                    assert "flash_attention_2" not in options

    def test_every_available_mode_survives_resolution(self) -> None:
        for mode in get_available_attention_modes():
            assert resolve_attention_mode(mode) == mode

    def test_the_modes_are_unique(self) -> None:
        modes = get_available_attention_modes()
        assert len(modes) == len(set(modes))
        assert all(mode in ATTENTION_MODES for mode in modes)
