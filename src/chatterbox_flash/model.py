"""``ChatterboxFlashT3`` — block-diffusion T3 decoder for Chatterbox-Flash.

We extend Chatterbox-TTS's :class:`~chatterbox.models.t3.t3.T3` decoder with a
single extra ``[MASK]`` token (``id = V``) added to the speech embedding, and
replace the autoregressive ``inference()`` path with a block-diffusion
``generate()`` method.

The decoder is otherwise unchanged: same Llama backbone, same conditioning
encoder, same text / speech heads.  This keeps the released checkpoint
compatible with the public ``chatterbox-tts`` weights for everything but the
expanded ``speech_emb`` (which is initialized by copying the V rows from the
pretrained checkpoint and adding one normal-initialized row for ``[MASK]``).

The ``generate()`` method here implements the production configuration
described in the paper:

* dual-stream-free single-stream KV cache (``[prefix | speech]``);
* FlashInfer paged KV cache + CUDA-graph capture of the per-block forward;
* zero-text-batch classifier-free guidance with ``zero_all`` null prefix
  (text and conditioning zeroed, speech-x_t duplicated); the only
  PMI/CFG combination is ``pmi_cfg`` — ``(1+w)·pmi_c − w·pmi_u`` on the
  PMI scale, no other mode exposed;
* **pmi_count_early** outlier rule — OmniVoice's r_n count schedule
  (controlled by ``omnivoice_schedule_t_shift``) as the per-step
  **floor**, plus the paper's time-shifted PMI-quantile
  ``q_k = max(0, 1 - τ · (k+1)/K)`` as an early-decoding **ceiling**.
  When ``time_shift_tau == 0`` the schedule reduces to plain pmi_count
  (fixed-K, no early decoding);
* prior-calibrated PMI scoring
  ``s_i = log p_i(\\hat x_i) - log p̄(\\hat x_i)`` with the **precomputed
  unconditional block prior** ``p̄`` — a forward on
  ``[SOS, MASK × block_size]`` with no conditioning, lazy-cached on the
  model object so it's evaluated at most once per
  ``(block_size, dtype, device)`` triple over the lifetime of the model;
* optional Gumbel ``position_temperature`` perturbing the PMI ranking
  for stochastic top-k position selection (OmniVoice §3.4 uses T=5).

Training / experimental code paths (full block-diffusion forward, dual-stream,
speaker probe, magi attention, sdpa/flex inference backends, alternative CFG
modes and outlier methods, remasking, …) are intentionally omitted.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..chatterbox.models.t3.t3 import T3 as _ChatterboxT3
from ..chatterbox.models.t3.modules.cond_enc import T3Cond
from ..chatterbox.models.t3.modules.t3_config import T3Config

from .cfg_guidance import apply_zero_text_cfg_from_logits, pmi_cfg_combine
from .engines import FLASHINFER_AVAILABLE, InferenceEngine, build_engine
from .calibration import compute_pmi


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #


def _gumbel_sample(logits: Tensor, temperature: float) -> Tensor:
    scaled = logits / temperature
    u = torch.rand_like(scaled)
    gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return scaled + gumbel


def _sample_at_temperature(
    logits: Tensor,
    temperature: float,
    *,
    sampling: Literal["multinomial", "gumbel"] = "multinomial",
) -> Tensor:
    if sampling == "gumbel":
        return _gumbel_sample(logits, temperature).argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# --------------------------------------------------------------------------- #
# Prefix-shaping helpers (CFG ``zero_all`` null branch)
# --------------------------------------------------------------------------- #


def _cond_emb_zero_all(ce: Tensor) -> Tensor:
    """Zero the entire conditioning prefix (paper's ``zero_all`` null branch)."""
    return torch.zeros_like(ce)


def _zero_text_content_keep_pad(
    text_emb: Tensor,
    text_tokens: Tensor,
    text_token_lens: Tensor | None,
) -> Tensor:
    """Null-text replacement: non-pad text positions go to 0, pad positions copy.

    Pad positions are detected from ``text_tokens == -100`` and / or
    ``text_token_lens`` when supplied. This keeps the sequence length identical
    so the CFG forward can be batched 2x.
    """
    squeeze_batch = text_tokens.ndim == 1
    toks = text_tokens.unsqueeze(0) if squeeze_batch else text_tokens
    emb = text_emb.unsqueeze(0) if text_emb.ndim == 2 else text_emb
    Btxt, Tt = toks.shape[0], toks.size(1)
    device = toks.device
    is_pad = toks.eq(-100)
    if text_token_lens is not None:
        ttl = text_token_lens.to(device=device, dtype=torch.long).reshape(-1)
        if ttl.numel() == 1 and Btxt > 1:
            ttl = ttl.expand(Btxt)
        is_pad = is_pad | (torch.arange(Tt, device=device)[None, :] >= ttl[:, None])
    out = emb.clone()
    out.masked_fill_((~is_pad).unsqueeze(-1), 0.0)
    return out.squeeze(0) if squeeze_batch else out


# --------------------------------------------------------------------------- #
# OmniVoice §3.4 r_n count schedule
# --------------------------------------------------------------------------- #


def _omnivoice_unmask_schedule(
    n_total_mask: int,
    num_steps: int,
    t_shift: float,
) -> list[int]:
    """Per-step unmask **counts** under OmniVoice's time-shifted ``r_n`` curve.

    Cumulative fraction unmasked by the end of step ``s`` is
    ``t_s = r_n(s/K, τ)`` where
    ``r_n(s, τ) = τ * s / (1 + (τ - 1) * s)`` (eq. (3) of OmniVoice).
    We then convert the cumulative target into per-step integer counts by
    cumulative rounding (``target = round(N * t_{s+1})``,
    ``num_s = max(0, target - cum)``), capped at the remaining masks.
    The last step always closes out whatever is left.

    Returns a list of length ``num_steps`` summing to ``n_total_mask``.
    """
    K = max(1, int(num_steps))
    N = max(0, int(n_total_mask))
    if N == 0:
        return [0] * K
    if K == 1:
        return [N]
    tau = float(t_shift)
    if tau <= 0:
        # Degenerate: split as evenly as possible, last step takes remainder.
        base = N // K
        out = [base] * K
        out[-1] += N - sum(out)
        return out
    ts = [tau * (s / K) / (1.0 + (tau - 1.0) * (s / K)) for s in range(K + 1)]
    counts: list[int] = []
    cum = 0
    rem = N
    for s in range(K):
        if s == K - 1:
            num = rem
        else:
            target = int(round(N * ts[s + 1]))
            num = max(0, target - cum)
            num = min(num, rem)
        counts.append(num)
        cum += num
        rem -= num
    return counts


# --------------------------------------------------------------------------- #
# Speech-block step (block-diffusion inner loop)
# --------------------------------------------------------------------------- #


def _pmi_count_early_step_unmask(
    blk_logits_2d: Tensor,
    blk_toks_1d: Tensor,
    *,
    bl: int,
    k: int,
    K: int,
    MASK: int,
    device: torch.device,
    temperature: float,
    temperature_sampling: Literal["multinomial", "gumbel"],
    omnivoice_unmask_schedule_k: int,
    time_shift_tau: float,
    prior_override_v: Tensor | None,
    block_marginal_prior: Tensor | None,
    pmi_cfg_probs_c_bl: Tensor | None,
    pmi_cfg_probs_u_bl: Tensor | None,
    cfg_scale_for_pmi: float,
    position_temperature: float,
) -> dict:
    """One denoising step for a single sample block — ``pmi_count_early``.

    Per-step unmask count combines OmniVoice's count schedule (floor) with
    the paper's PMI-quantile (ceiling):

        n_sched  = omnivoice_unmask_schedule_k  (OmniVoice r_n at step k)
        q_k      = max(0, 1 - time_shift_tau * (k+1) / K)
        tau_k    = Quantile({pmi_i}_{i in M}, q_k)        (if time_shift_tau>0)
        n_quant  = |{i in M : pmi_i >= tau_k}|             (else 0)
        n_unmask = min(max(n_sched, n_quant), m_cur)

    The top ``n_unmask`` masked positions by PMI get unmasked. Properties:
      * ``time_shift_tau == 0`` reduces to plain pmi_count (fixed-K, no
        early decoding); per-step count is exactly ``n_sched``.
      * ``time_shift_tau > 0`` may unmask more than ``n_sched`` when PMI
        is concentrated, allowing the block to early-finish (n_mask==0)
        before step ``K-1`` — the *only* source of step savings above the
        baseline.

    PMI scoring:
      * Numerator: probabilities from the conditional softmax of the
        outer block logits (``probs_bl``).
      * Denominator: ``prior_override_v`` (precomputed unconditional
        block prior) when provided, otherwise the running per-block
        marginal frozen at step 0 (``block_marginal_prior``).
      * Under CFG (``cfg_scale > 0``) the caller passes
        ``pmi_cfg_probs_{c,u}_bl`` and the CFG-combined PMI
        ``(1+w) * pmi_c - w * pmi_u`` is used (this is the only PMI/CFG
        combination supported — no other ``cfg_prior_mode`` is exposed).

    Gumbel position scoring (OmniVoice §3.4 T=5): when
    ``position_temperature > 0`` we add Gumbel noise scaled by ``T`` to
    masked positions' PMI before ranking, turning deterministic top-k
    into stochastic top-k without replacement.
    """
    probs_raw = F.softmax(blk_logits_2d, dim=-1)
    if temperature > 0:
        sampled = _sample_at_temperature(
            blk_logits_2d, temperature, sampling=temperature_sampling,
        )
    else:
        sampled = blk_logits_2d.argmax(dim=-1)

    ar = torch.arange(bl, device=device)
    is_mask = blk_toks_1d == MASK

    # ---- PMI scoring ----
    probs_bl = probs_raw
    if k == 0 and block_marginal_prior is None:
        block_marginal_prior = probs_bl.mean(dim=0)

    if pmi_cfg_probs_c_bl is not None and pmi_cfg_probs_u_bl is not None:
        mp_c = (
            prior_override_v
            if prior_override_v is not None
            else pmi_cfg_probs_c_bl.mean(dim=0)
        )
        mp_u = (
            prior_override_v
            if prior_override_v is not None
            else pmi_cfg_probs_u_bl.mean(dim=0)
        )
        pmi_c = compute_pmi(pmi_cfg_probs_c_bl, sampled, marginal_prior=mp_c)
        pmi_u = compute_pmi(pmi_cfg_probs_u_bl, sampled, marginal_prior=mp_u)
        pmi = pmi_cfg_combine(pmi_c, pmi_u, cfg_scale_for_pmi)
    else:
        _mp = (
            prior_override_v
            if prior_override_v is not None
            else block_marginal_prior
        )
        pmi = compute_pmi(probs_bl, sampled, marginal_prior=_mp)

    if position_temperature > 0.0:
        noise = _gumbel_sample(torch.zeros_like(pmi), position_temperature)
        pmi = torch.where(is_mask, pmi + noise, pmi)

    # ---- Outlier rule (pmi_count_early): max(count-schedule, quantile) ----
    if k == K - 1:
        do_unmask = is_mask
    else:
        masked_idx = ar[is_mask]
        m_cur = int(masked_idx.numel())
        do_unmask = torch.zeros_like(is_mask, dtype=torch.bool)
        if m_cur > 0:
            pmi_m = pmi[masked_idx]
            n_sched = max(0, min(int(omnivoice_unmask_schedule_k), m_cur))
            if time_shift_tau > 0.0:
                q_shifted = max(0.0, 1.0 - time_shift_tau * (k + 1) / K)
                tau_k = torch.quantile(pmi_m.float(), q_shifted).to(dtype=pmi_m.dtype)
                n_quant = int((pmi_m >= tau_k).sum().item())
            else:
                n_quant = 0
            n_unmask = min(max(n_sched, n_quant), m_cur)
            if n_unmask >= m_cur:
                do_unmask[masked_idx] = True
            elif n_unmask > 0:
                _, _topi = pmi_m.topk(n_unmask)
                do_unmask[masked_idx[_topi]] = True

    xt_step = torch.where(do_unmask, sampled, blk_toks_1d)
    n_mask = int((xt_step == MASK).sum().item())
    return dict(
        xt_step=xt_step,
        n_mask=n_mask,
        block_marginal_prior=block_marginal_prior,
    )


# --------------------------------------------------------------------------- #
# ChatterboxFlashT3
# --------------------------------------------------------------------------- #


class ChatterboxFlashT3(_ChatterboxT3):
    """Chatterbox-TTS T3 decoder + [MASK] token + block-diffusion generate().

    The Llama backbone, conditioning encoder, text / speech embeddings and
    heads come straight from the parent ``chatterbox.models.t3.t3.T3``.  We
    only add the extra mask-token row in :attr:`speech_emb` and override
    :meth:`generate` with the block-diffusion inference loop.
    """

    def __init__(
        self,
        hp: T3Config | None = None,
        *,
        drf_block_size: int = 16,
    ) -> None:
        super().__init__(hp=hp)
        if self.is_gpt:
            raise NotImplementedError(
                "ChatterboxFlashT3 currently supports the Llama backbone only "
                "(hp.llama_config_name='Llama_520M').",
            )
        self._mask_token_id: int = self.hp.speech_tokens_dict_size
        self.drf_block_size = int(drf_block_size)

        # Extend speech_emb by one row for the [MASK] token (id = V).
        old_speech_emb = self.speech_emb
        self.speech_emb = nn.Embedding(self.hp.speech_tokens_dict_size + 1, self.dim)
        with torch.no_grad():
            self.speech_emb.weight[: self.hp.speech_tokens_dict_size].copy_(
                old_speech_emb.weight,
            )
            nn.init.normal_(
                self.speech_emb.weight[self.hp.speech_tokens_dict_size :], std=0.02,
            )

        # Backward-compat alias so :class:`FlashInferEngine` can read
        # ``model.tfmr.config`` (chatterbox T3 only exposes ``.cfg``).
        self.config = self.cfg
        # Inference engine cache (re-used between calls when shape compatible).
        self._cached_engine: InferenceEngine | None = None

    # ------------------------------------------------------------------ #
    #  Properties / helpers
    # ------------------------------------------------------------------ #

    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id

    @property
    def is_llama(self) -> bool:
        return not self.is_gpt

    def set_block_size(self, block_size: int) -> None:
        self.drf_block_size = int(block_size)

    def prime_uncond_block_prior(
        self,
        dtype: torch.dtype,
        device: torch.device | str,
        block_size: int | None = None,
    ) -> None:
        """Eagerly compute & cache the PMI unconditional block prior.

        Called at load time so the first ``generate()`` does not pay for the
        one-shot prior forward — keeping it out of the timed / RTF loop. Uses
        ``self.drf_block_size`` unless ``block_size`` is given; a later
        ``generate()`` with a different block size recomputes automatically
        (the cache is keyed on ``(block_size, dtype, device)``).
        """
        device = torch.device(device) if isinstance(device, str) else device
        bs = int(block_size) if block_size is not None else int(self.drf_block_size)
        with torch.no_grad():
            self._compute_uncond_block_prior(
                bs, int(self.hp.start_speech_token), int(self.mask_token_id),
                dtype, device,
            )

    def _embed_speech_tokens(self, tokens: Tensor) -> Tensor:
        """Embed speech tokens (including the extra ``[MASK]`` id)."""
        emb = self.speech_emb(tokens)
        if self.hp.input_pos_emb == "learned":
            emb = emb + self.speech_pos_emb(tokens)
        return emb

    # ------------------------------------------------------------------ #
    #  Precomputed unconditional block prior (PMI denominator)
    # ------------------------------------------------------------------ #

    def _compute_uncond_block_prior(
        self,
        block_size: int,
        SOS: int,
        MASK: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """PMI prior from a forward on ``[SOS, MASK × block_size]`` with **no
        conditioning at all** — no speaker, no text, no prompt speech, not
        even zero-filled embeddings of those.  Captures the model's intrinsic
        marginal over speech tokens at the very start of an utterance.

        The result depends only on ``(block_size, SOS, MASK, dtype, device)``
        and the (frozen, inference-time) model weights, so we **cache** it
        on ``self._uncond_block_prior_cache`` keyed by that tuple.  The actual
        forward runs at most once per ``(block_size, dtype, device)`` triple
        across the entire lifetime of the model object — subsequent
        ``generate()`` calls reuse the cached ``(V,)`` tensor.  Invalidated
        implicitly when the model is recreated (e.g. on checkpoint reload).
        """
        cache: dict = getattr(self, "_uncond_block_prior_cache", None)
        if cache is None:
            cache = {}
            self._uncond_block_prior_cache = cache
        key = (int(block_size), int(SOS), int(MASK), str(dtype), str(device))
        if key in cache:
            return cache[key]

        n_tok = 1 + int(block_size)
        tok_seq = torch.cat(
            [
                torch.full((1, 1), int(SOS), device=device, dtype=torch.long),
                torch.full(
                    (1, int(block_size)), int(MASK),
                    device=device, dtype=torch.long,
                ),
            ],
            dim=1,
        )
        seq_emb = self._embed_speech_tokens(tok_seq).to(dtype)
        position_ids = torch.arange(n_tok, device=device).unsqueeze(0)

        # The FlashInfer engine swaps the LlamaAttention forward in-place;
        # we explicitly route this one-shot prior pass through HuggingFace's
        # SDPA implementation so it never touches the FlashInfer paged cache.
        cfg = self.tfmr.config
        old_impl = getattr(cfg, "_attn_implementation", "sdpa")
        cfg._attn_implementation = "sdpa"
        try:
            out = self.tfmr(
                input_ids=None,
                inputs_embeds=seq_emb,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
        finally:
            cfg._attn_implementation = old_impl

        # Standard token-shift: logits for mask position i come from hidden
        # at position i (SOS → mask 0, mask k-1 → mask k).
        shift_hidden = out.last_hidden_state[:, : int(block_size), :]
        logits = self.speech_head(shift_hidden)
        probs = F.softmax(logits.float(), dim=-1)
        prior = probs.mean(dim=1).squeeze(0).detach()
        cache[key] = prior
        return prior

    # ------------------------------------------------------------------ #
    #  State-dict surgery (lets the model accept either a pure chatterbox
    #  checkpoint of size V or a Chatterbox-Flash checkpoint of size V+1)
    # ------------------------------------------------------------------ #

    def _load_from_state_dict(
        self, state_dict, prefix, *args, **kwargs,
    ):  # noqa: D401
        key = prefix + "speech_emb.weight"
        if key in state_dict:
            ckpt_w = state_dict[key]
            model_v = self.speech_emb.weight.shape[0]
            if ckpt_w.shape[0] == model_v - 1:
                logger.info(
                    "Expanding speech_emb from %d to %d (adding [MASK] token).",
                    ckpt_w.shape[0], model_v,
                )
                expanded = self.speech_emb.weight.detach().clone()
                expanded[: ckpt_w.shape[0]] = ckpt_w
                state_dict[key] = expanded
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # ------------------------------------------------------------------ #
    #  Prefix forward (run once per generation, cached in FlashInfer)
    # ------------------------------------------------------------------ #

    def _build_prefix_emb(
        self,
        cond_emb: Tensor,
        text_tokens: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Build ``[cond | text | <start_speech>]`` embedding."""
        text_emb = self.text_emb(text_tokens)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
        sos = torch.full(
            (1, 1), self.hp.start_speech_token, device=device, dtype=torch.long,
        )
        sos_emb = self._embed_speech_tokens(sos)
        return torch.cat([cond_emb, text_emb, sos_emb], dim=1).to(dtype)

    # ------------------------------------------------------------------ #
    #  Block-diffusion generate
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def generate(
        self,
        *,
        t3_cond: "T3Cond | list[T3Cond] | tuple[T3Cond, ...]",
        text_tokens: "torch.Tensor | list[torch.Tensor]",
        text_token_lens: "torch.Tensor | None" = None,
        total_speech_len: "int | list[int]" = 0,
        num_steps: int = 10,
        temperature: float = 0.6,
        temperature_sampling: Literal["multinomial", "gumbel"] = "multinomial",
        time_shift_tau: float = 0.1,
        omnivoice_schedule_t_shift: float = 0.5,
        cfg_scale: float = 1.0,
        position_temperature: float = 5.0,
        pmi_uncond_prior_precompute: bool = True,
        batch_size: int = 1,
        page_size: int = 16,
        flashinfer_reserve_max_seq: int | None = None,
        use_cuda_graph: bool = True,
        backend: Literal["auto", "flashinfer", "torch"] = "auto",
    ) -> Tensor:
        """Block-wise denoising generation matching the paper's best config.

        Inputs accept either a single sample (``text_tokens`` is a tensor and
        ``t3_cond`` is a :class:`T3Cond`) or a heterogeneous batch
        (``text_tokens`` is a list, ``t3_cond`` is a list of :class:`T3Cond`).
        Both modes are dispatched through the same FlashInfer block-decoding
        loop. Heterogeneous batches pad to ``max(total_speech_len)`` and stop
        each row once it emits ``hp.stop_speech_token``.

        Decoding-relevant parameters (paper defaults):

        ``num_steps``
            Maximum denoising iterations per block (``K``). Default 10.
        ``time_shift_tau``
            Early-decoding aggressiveness in
            ``q_k = max(0, 1 - τ·(k+1)/K)``. ``0.0`` disables early decoding
            (every block uses exactly the OmniVoice count schedule). Default
            ``0.1`` (paper).
        ``omnivoice_schedule_t_shift``
            OmniVoice r_n schedule parameter that controls the per-step
            unmask count floor. Default ``0.5``.
        ``cfg_scale``
            CFG strength. When ``> 0`` the forward batch is doubled with a
            null branch (``zero_text_batch`` + ``zero_all`` null prefix +
            text/cond zeroed + speech-x_t duplicated). Token sampling uses
            the guided distribution; PMI scoring is combined on the PMI
            scale via ``(1+w)·pmi_c − w·pmi_u`` (``pmi_cfg``) — this is the
            only supported PMI/CFG combination, no other mode is exposed.
        ``position_temperature``
            Gumbel temperature added to PMI for stochastic top-k position
            ranking. ``0`` = deterministic argmax-top-k; default ``5.0``
            (OmniVoice §3.4 T=5).
        ``pmi_uncond_prior_precompute``
            When ``True`` (default), the PMI denominator is the
            unconditional block prior computed once via
            :meth:`_compute_uncond_block_prior` and cached on the model.
            When ``False``, falls back to the running per-block marginal
            from step 0.

        Returns
        -------
        torch.LongTensor
            Speech token ids of shape ``(B, T)`` with
            ``T <= max(total_speech_len)``.
        """
        device = self.device
        BS = self.drf_block_size
        K = num_steps
        MASK = self.mask_token_id
        dtype = next(self.parameters()).dtype
        stop_tok = self.hp.stop_speech_token

        # ---- cross-text vs same-text batching ----
        _cross_text = isinstance(text_tokens, list)
        if _cross_text:
            B = len(text_tokens)
            _speech_lens: list[int] = (
                list(total_speech_len)
                if isinstance(total_speech_len, list)
                else [total_speech_len] * B
            )
            N = max(_speech_lens) if _speech_lens else 0
            _per_sample_num_blocks = [math.ceil(sl / BS) for sl in _speech_lens]
            num_blocks = max(_per_sample_num_blocks) if _per_sample_num_blocks else 0
        else:
            B = max(1, batch_size)
            _speech_lens = []
            N = total_speech_len if isinstance(total_speech_len, int) else total_speech_len[0]
            num_blocks = math.ceil(N / BS) if N > 0 else 0
            _per_sample_num_blocks = []

        if N <= 0:
            return torch.empty((B, 0), device=device, dtype=torch.long)

        B_usr = B
        _ztb = cfg_scale > 0
        B_fwd = (2 * B_usr) if _ztb else B_usr

        # ---- prefix(es) ----
        t3_is_seq = isinstance(t3_cond, (list, tuple))
        if t3_is_seq and not _cross_text:
            raise ValueError(
                "t3_cond as a list/tuple is only supported when text_tokens is a list.",
            )

        cond_embs: list[Tensor] | None = None
        cond_emb_single: Tensor | None = None
        if _cross_text:
            if t3_is_seq:
                if len(t3_cond) != len(text_tokens):
                    raise ValueError(
                        f"len(t3_cond) ({len(t3_cond)}) must match "
                        f"len(text_tokens) ({len(text_tokens)}).",
                    )
                cond_embs = [self.prepare_conditioning(c) for c in t3_cond]
            else:
                cond_emb_single = self.prepare_conditioning(t3_cond)
        else:
            cond_emb_single = self.prepare_conditioning(t3_cond)

        prefix_emb_cond: Tensor | list[Tensor]
        prefix_emb_null: Tensor | list[Tensor] | None = None
        _prefix_lens: list[int]

        if _cross_text:
            cond_list: list[Tensor] = []
            null_list: list[Tensor] = []
            lens: list[int] = []
            for i, tt in enumerate(text_tokens):
                ce = cond_embs[i] if cond_embs is not None else cond_emb_single
                p = self._build_prefix_emb(ce, tt, device, dtype)
                cond_list.append(p)
                lens.append(p.size(1))
                if _ztb:
                    ce_null = _cond_emb_zero_all(ce)
                    text_emb = self.text_emb(tt)
                    if self.hp.input_pos_emb == "learned":
                        text_emb = text_emb + self.text_pos_emb(tt)
                    text_emb_zero = _zero_text_content_keep_pad(
                        text_emb, tt, None,
                    )
                    sos = torch.full(
                        (1, 1), self.hp.start_speech_token,
                        device=device, dtype=torch.long,
                    )
                    sos_emb = self._embed_speech_tokens(sos)
                    p_null = torch.cat(
                        [ce_null, text_emb_zero, sos_emb], dim=1,
                    ).to(dtype)
                    null_list.append(p_null)
            prefix_emb_cond = cond_list
            lp = max(lens)
            _prefix_lens = list(lens)
            if _ztb:
                prefix_emb_null = null_list
                _prefix_lens = _prefix_lens + [p.size(1) for p in null_list]
        else:
            assert cond_emb_single is not None
            prefix_emb = self._build_prefix_emb(
                cond_emb_single, text_tokens, device, dtype,
            )
            lp = prefix_emb.size(1)
            if B_usr > 1:
                prefix_emb = prefix_emb.expand(B_usr, -1, -1).contiguous()
            prefix_emb_cond = prefix_emb
            _prefix_lens = [lp] * B_fwd
            if _ztb:
                cond_for_null = _cond_emb_zero_all(cond_emb_single)
                text_emb = self.text_emb(text_tokens)
                if self.hp.input_pos_emb == "learned":
                    text_emb = text_emb + self.text_pos_emb(text_tokens)
                text_emb_zero = _zero_text_content_keep_pad(
                    text_emb, text_tokens, text_token_lens,
                )
                sos = torch.full(
                    (1, 1), self.hp.start_speech_token,
                    device=device, dtype=torch.long,
                )
                sos_emb = self._embed_speech_tokens(sos)
                p_null = torch.cat(
                    [cond_for_null, text_emb_zero, sos_emb], dim=1,
                ).to(dtype)
                if B_usr > 1:
                    p_null = p_null.expand(B_usr, -1, -1).contiguous()
                prefix_emb_null = p_null

        # ---- speech buffer (mask everywhere; we fill it block by block) ----
        xt = torch.full((B_fwd, N), MASK, device=device, dtype=torch.long)

        # ---- precomputed unconditional block prior (PMI denominator) ----
        # Single forward on [SOS, MASK × BS] with no conditioning, lazy-cached
        # on the model. Reused as PMI denominator for every step of every
        # block in this generate() call (and across subsequent calls).
        SOS = int(self.hp.start_speech_token)
        prior_override_v: Tensor | None = None
        if pmi_uncond_prior_precompute:
            prior_override_v = self._compute_uncond_block_prior(
                BS, SOS, MASK, dtype, device,
            )

        # ---- inference engine (FlashInfer if available, else torch SDPA) ----
        engine_max_seq = max(lp + N + 64, flashinfer_reserve_max_seq or 0)
        cached = self._cached_engine
        if (
            cached is not None
            and cached.can_reuse(
                engine_max_seq, dtype, batch_size=B_fwd, page_size=page_size,
            )
        ):
            engine = cached
            engine.reset()
        else:
            if (
                use_cuda_graph
                and cached is not None
                and getattr(cached, "_max_seq_len", 0) < engine_max_seq
                and getattr(cached, "_batch_size", -1) == B_fwd
                and getattr(cached, "_dtype", None) == dtype
            ):
                # Grow allocation in 512-token buckets so we don't re-instantiate
                # the engine every time the sequence length nudges up by one.
                engine_max_seq = max(
                    ((engine_max_seq + 511) // 512) * 512,
                    int(getattr(cached, "_max_seq_len", 0) * 1.5),
                )
            engine = build_engine(
                self, engine_max_seq, dtype,
                backend=backend,
                batch_size=B_fwd, page_size=page_size,
            )

        if _ztb:
            assert prefix_emb_null is not None
            if isinstance(prefix_emb_cond, list):
                assert isinstance(prefix_emb_null, list)
                pfx_for_engine = [*prefix_emb_cond, *prefix_emb_null]
            else:
                assert isinstance(prefix_emb_null, torch.Tensor)
                pfx_for_engine = [
                    *[prefix_emb_cond[bi : bi + 1] for bi in range(B_usr)],
                    *[prefix_emb_null[bi : bi + 1] for bi in range(B_usr)],
                ]
            shift_ctx = engine.prefix_forward(pfx_for_engine)
        else:
            shift_ctx = engine.prefix_forward(prefix_emb_cond)

        if use_cuda_graph:
            # ``capture_cuda_graph`` is a no-op on the torch SDPA fallback.
            engine.capture_cuda_graph(BS, speech_head=self.speech_head)
        self._cached_engine = engine

        # ---- block-by-block decoding ----
        # ``_row_eos[bi]`` — row has emitted EOS, skip remaining blocks.
        # ``_block_done[bi]`` — inner k-loop finished for this row in the
        # current block (reset at every block boundary; not a sequence-level
        # flag).
        _row_eos = (
            torch.zeros(B_usr, dtype=torch.bool, device=device)
            if B_usr > 1
            else None
        )
        _block_done = (
            torch.zeros(B_usr, dtype=torch.bool, device=device)
            if B_usr > 1
            else None
        )
        block_marginal_priors: list[Tensor | None] = [None] * B_usr

        for b_idx in range(num_blocks):
            bs_ = b_idx * BS
            be_ = min(bs_ + BS, N)
            bl = be_ - bs_

            # OmniVoice r_n count schedule for this block's K steps. Used as
            # the per-step unmask **floor** by the pmi_count_early outlier
            # rule. Computed once per block since ``bl`` only changes at the
            # tail.
            _omnivoice_sched = _omnivoice_unmask_schedule(
                n_total_mask=bl,
                num_steps=K,
                t_shift=float(omnivoice_schedule_t_shift),
            )

            if _block_done is not None:
                _block_done.zero_()
                if _cross_text:
                    for bi in range(B_usr):
                        if (
                            b_idx >= _per_sample_num_blocks[bi]
                            or (_row_eos is not None and bool(_row_eos[bi]))
                        ):
                            _block_done[bi] = True
                elif _row_eos is not None:
                    for bi in range(B_usr):
                        if bool(_row_eos[bi]):
                            _block_done[bi] = True

            for k in range(K):
                # Per-row cache-start positions (cross-text & null-prefix paths
                # may differ between rows; same-text reuses a scalar).
                if _cross_text:
                    cache_start: int | list[int] = [
                        _prefix_lens[bi] + bs_ for bi in range(B_fwd)
                    ]
                else:
                    cache_start = lp + bs_

                sp_blk = self._embed_speech_tokens(xt[:, bs_:be_]).to(dtype)

                use_graph = engine.has_cuda_graph and bl == BS
                if use_graph:
                    if k == 0:
                        engine.set_shift_ctx(shift_ctx)
                    blk_hidden, graph_logits = engine.block_forward_graph(
                        sp_blk, cache_start,
                    )
                    if graph_logits is not None:
                        blk_logits = graph_logits
                    else:
                        shift_hidden = torch.cat(
                            [shift_ctx, blk_hidden[:, : bl - 1]], dim=1,
                        )
                        blk_logits = self.speech_head(shift_hidden)
                else:
                    blk_hidden = engine.block_forward(sp_blk, cache_start)
                    shift_hidden = torch.cat(
                        [shift_ctx, blk_hidden[:, : bl - 1]], dim=1,
                    )
                    blk_logits = self.speech_head(shift_hidden)

                pmi_cfg_probs_c: Tensor | None = None
                pmi_cfg_probs_u: Tensor | None = None
                if _ztb:
                    logits_cond = blk_logits[:B_usr]
                    logits_uncond = blk_logits[B_usr : 2 * B_usr]
                    # Always pmi_cfg: PMI scores are combined on the PMI
                    # scale via (1+w)·pmi_c − w·pmi_u. No other CFG/PMI
                    # combination is exposed.
                    pmi_cfg_probs_c = F.softmax(logits_cond, dim=-1)
                    pmi_cfg_probs_u = F.softmax(logits_uncond, dim=-1)
                    blk_logits = apply_zero_text_cfg_from_logits(
                        logits_cond, logits_uncond, cfg_scale,
                    )

                step_break = False
                for bi in range(B_usr):
                    if _block_done is not None and _block_done[bi]:
                        continue
                    pmi_c_bi = None if pmi_cfg_probs_c is None else pmi_cfg_probs_c[bi]
                    pmi_u_bi = None if pmi_cfg_probs_u is None else pmi_cfg_probs_u[bi]
                    r = _pmi_count_early_step_unmask(
                        blk_logits[bi],
                        xt[bi, bs_:be_],
                        bl=bl,
                        k=k, K=K,
                        MASK=MASK, device=device,
                        temperature=temperature,
                        temperature_sampling=temperature_sampling,
                        omnivoice_unmask_schedule_k=_omnivoice_sched[k],
                        time_shift_tau=time_shift_tau,
                        prior_override_v=prior_override_v,
                        block_marginal_prior=block_marginal_priors[bi],
                        pmi_cfg_probs_c_bl=pmi_c_bi,
                        pmi_cfg_probs_u_bl=pmi_u_bi,
                        cfg_scale_for_pmi=cfg_scale,
                        position_temperature=position_temperature,
                    )
                    xt[bi, bs_:be_] = r["xt_step"]
                    block_marginal_priors[bi] = r["block_marginal_prior"]
                    if _ztb:
                        xt[B_usr + bi, bs_:be_] = r["xt_step"]
                    row_hit_eos = bool(
                        (xt[bi, bs_:be_] == stop_tok).any().item(),
                    )
                    if r["n_mask"] == 0 or row_hit_eos:
                        if _block_done is not None:
                            _block_done[bi] = True
                        if row_hit_eos and _row_eos is not None:
                            _row_eos[bi] = True
                        if B_usr == 1 and row_hit_eos:
                            step_break = True
                        elif B_usr == 1 and r["n_mask"] == 0:
                            step_break = True
                if step_break:
                    break
                if _block_done is not None and bool(_block_done.all().item()):
                    break

            # Mid-block finalize: write the committed (possibly partial) tokens
            # of this block to the KV cache so the next block sees clean context.
            if b_idx < num_blocks - 1:
                fin_emb = self._embed_speech_tokens(xt[:, bs_:be_]).to(dtype)
                if _cross_text:
                    fin_cs: int | list[int] = [
                        _prefix_lens[bi] + bs_ for bi in range(B_fwd)
                    ]
                else:
                    fin_cs = lp + bs_
                fin_hidden = engine.block_forward(fin_emb, fin_cs)
                engine.advance_cache(bl)
                shift_ctx = fin_hidden[:, bl - 1 : bl, :].clone()

            # Early termination if all (same-text) rows have finished.
            if B_usr == 1:
                eos_pos = (xt[0, bs_:be_] == stop_tok).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    return xt[:B_usr, : bs_ + eos_pos[0].item()]
            elif _row_eos is not None and bool(_row_eos.all().item()):
                break

        return xt[:B_usr]
