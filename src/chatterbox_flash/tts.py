"""User-facing API for Chatterbox-Flash."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from ..chatterbox.models.dsp import load_audio_at, resample
from ..chatterbox.models.s3gen import S3GEN_SR, S3Gen
from ..chatterbox.models.s3tokenizer import S3_SR
from ..chatterbox.models.t3.modules.cond_enc import T3Cond
from ..chatterbox.models.tokenizers import EnTokenizer
from ..chatterbox.models.voice_encoder import VoiceEncoder

from .model import ChatterboxFlashT3
from .text_norm import en_us_cleaner

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "ResembleAI/chatterbox-flash"

# Per-file names expected inside a checkpoint dir / HF repo.
T3_FILENAME = "t3_flash.safetensors"
S3GEN_FILENAME = "s3gen.safetensors"
VE_FILENAME = "ve.safetensors"
TOKENIZER_FILENAME = "tokenizer.json"

# Conditioning durations (mirrored from chatterbox-tts).
ENC_COND_LEN = 6 * S3_SR
DEC_COND_LEN = 10 * S3GEN_SR


@dataclass
class Conditionals:
    """Prepared reference-speech conditioning for one speaker.

    Bundles the T3 conditioning (speaker embedding + prompt speech tokens)
    and the S3Gen reference dict produced by :meth:`S3Gen.embed_ref`. Use
    :meth:`save` / :meth:`load` to cache these for repeated generations.
    """

    t3: T3Cond
    gen: dict

    def to(self, device) -> "Conditionals":
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path) -> None:
        torch.save({"t3": self.t3.__dict__, "gen": self.gen}, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu") -> "Conditionals":
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs["t3"]), kwargs["gen"])


def _speech_len_for_text_tokens(n_text_tokens: int) -> int:
    """Heuristic max-speech-token budget per utterance (matches paper config)."""
    return max(n_text_tokens * 6, 300)


class ChatterboxFlashTTS:
    """Zero-shot TTS pipeline matching the configuration in the paper.

    Stage 1 is the :class:`ChatterboxFlashT3` block-diffusion decoder.
    Stage 2 is the unmodified Chatterbox-TTS S3Gen flow-matching vocoder.
    Reference encoding uses the GE2E-trained voice encoder from Chatterbox.
    """

    def __init__(
        self,
        t3: ChatterboxFlashT3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str | torch.device,
        conds: Conditionals | None = None,
    ) -> None:
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.conds = conds

    # -------- construction ------------------------------------------------

    @classmethod
    def from_local(
        cls,
        ckpt_dir: str | Path,
        device: str | torch.device,
        *,
        dtype: torch.dtype = torch.bfloat16,
        drf_block_size: int = 16,
    ) -> "ChatterboxFlashTTS":
        """Load a Chatterbox-Flash checkpoint from a local directory.

        Expects ``t3_flash.safetensors`` (with the expanded MASK row),
        ``s3gen.safetensors``, ``ve.safetensors`` and ``tokenizer.json``.
        """
        ckpt_dir = Path(ckpt_dir)
        device = torch.device(device) if isinstance(device, str) else device

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(str(ckpt_dir / VE_FILENAME)))
        ve.to(device).eval()

        t3 = ChatterboxFlashT3(drf_block_size=drf_block_size)
        t3_state = load_file(str(ckpt_dir / T3_FILENAME))
        if "model" in t3_state:  # legacy layout
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device=device, dtype=dtype).eval()
        # Prime the PMI unconditional block prior now (at load) so the first
        # generate() doesn't fold the one-shot prior forward into the timed /
        # RTF loop. A generate() with a different block size recomputes it
        # automatically (cache keyed on block_size/dtype/device).
        t3.prime_uncond_block_prior(dtype, device)

        # ``meanflow=True``: the released ``s3gen.safetensors`` is the
        # meanflow-distilled vocoder, sampled with the meanflow ODE path at 2
        # CFM steps (matches the internal ``S3GenStreamer`` defaults). On a
        # standard-CFM checkpoint this would degrade quality — flip to False.
        s3gen = S3Gen(meanflow=True)
        s3gen.load_state_dict(
            load_file(str(ckpt_dir / S3GEN_FILENAME)), strict=False,
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(str(ckpt_dir / TOKENIZER_FILENAME))

        return cls(t3, s3gen, ve, tokenizer, device)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        device: str | torch.device = "cuda",
        *,
        dtype: torch.dtype = torch.bfloat16,
        drf_block_size: int = 16,
    ) -> "ChatterboxFlashTTS":
        """Fetch a Chatterbox-Flash release from the Hugging Face Hub."""
        last = None
        for fname in (
            VE_FILENAME, T3_FILENAME, S3GEN_FILENAME, TOKENIZER_FILENAME,
        ):
            last = hf_hub_download(repo_id=repo_id, filename=fname)
        return cls.from_local(
            Path(last).parent, device,
            dtype=dtype, drf_block_size=drf_block_size,
        )

    # -------- conditioning -----------------------------------------------

    def prepare_conditionals(
        self,
        wav_fpath: str | Path,
        exaggeration: float = 0.5,
    ) -> Conditionals:
        """Build the speaker / prompt-speech conditioning for a reference wav.

        The reference is resampled and segmented identically to chatterbox-tts:
        the first 6 s feed S3's speech tokenizer to produce prompt speech
        tokens, the first 10 s feed S3Gen's :meth:`embed_ref`, and the GE2E
        voice encoder produces the global speaker embedding.
        """
        s3gen_ref_wav = load_audio_at(wav_fpath, S3GEN_SR)
        ref_16k_wav = resample(s3gen_ref_wav, S3GEN_SR, S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(
            s3gen_ref_wav, S3GEN_SR, device=self.device,
        )

        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            tokens, _ = self.s3gen.tokenizer.forward(
                [ref_16k_wav[:ENC_COND_LEN]], max_len=plen,
            )
            t3_cond_prompt_tokens = torch.atleast_2d(tokens).to(self.device)

        t3_dtype = next(self.t3.parameters()).dtype
        ve_embed = torch.from_numpy(
            self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR),
        )
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(device=self.device, dtype=t3_dtype)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=t3_dtype),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)
        return self.conds

    # -------- tokenization -----------------------------------------------

    def _encode_text(
        self, text: str, normalize_text: bool = True,
    ) -> torch.Tensor:
        if normalize_text:
            text = en_us_cleaner(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)
        text_tokens = F.pad(text_tokens, (1, 0), value=self.t3.hp.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=self.t3.hp.stop_text_token)
        return text_tokens

    # -------- inference --------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        *,
        audio_prompt_path: str | Path | None = None,
        exaggeration: float = 0.5,
        normalize_text: bool = True,
        num_steps: int = 10,
        temperature: float = 0.6,
        time_shift_tau: float = 0.1,
        omnivoice_schedule_t_shift: float = 0.5,
        cfg_scale: float = 1.0,
        position_temperature: float = 5.0,
        pmi_uncond_prior_precompute: bool = True,
        use_cuda_graph: bool = True,
        backend: Literal["auto", "flashinfer", "torch", "mlx"] = "auto",
        max_speech_tokens: int | None = None,
        n_cfm_timesteps: int = 2,
    ) -> torch.Tensor:
        """Synthesize a single utterance.

        Parameters mirror the inference defaults from the paper (best
        configuration). Returns the synthesized waveform as a 1xT float
        tensor at :attr:`sr` Hz. See
        :meth:`ChatterboxFlashT3.generate` for the full parameter spec.
        """
        if audio_prompt_path is not None:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        if self.conds is None:
            raise ValueError(
                "Conditioning not prepared — pass audio_prompt_path or call "
                "prepare_conditionals() first.",
            )

        text_tokens = self._encode_text(text, normalize_text=normalize_text)
        n_tokens = int(text_tokens.size(1))
        total_speech_len = _speech_len_for_text_tokens(n_tokens)
        if max_speech_tokens is not None:
            total_speech_len = min(total_speech_len, max_speech_tokens)

        speech_tokens = self.t3.generate(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            text_token_lens=torch.tensor([n_tokens], device=self.device),
            total_speech_len=total_speech_len,
            num_steps=num_steps,
            temperature=temperature,
            time_shift_tau=time_shift_tau,
            omnivoice_schedule_t_shift=omnivoice_schedule_t_shift,
            cfg_scale=cfg_scale,
            position_temperature=position_temperature,
            pmi_uncond_prior_precompute=pmi_uncond_prior_precompute,
            use_cuda_graph=use_cuda_graph,
            backend=backend,
            batch_size=1,
        )
        speech_tokens = self._trim_to_eos(speech_tokens[0])
        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens.to(self.device),
            ref_dict=self.conds.gen,
            n_cfm_timesteps=n_cfm_timesteps,
        )
        return wav.detach().cpu().squeeze(0)

    @torch.inference_mode()
    def generate_batch(
        self,
        texts: Iterable[str],
        *,
        audio_prompt_paths: Iterable[str | Path] | None = None,
        conds_list: Iterable[Conditionals] | None = None,
        exaggeration: float = 0.5,
        normalize_text: bool = True,
        num_steps: int = 10,
        temperature: float = 0.6,
        time_shift_tau: float = 0.1,
        omnivoice_schedule_t_shift: float = 0.5,
        cfg_scale: float = 1.0,
        position_temperature: float = 5.0,
        pmi_uncond_prior_precompute: bool = True,
        use_cuda_graph: bool = True,
        backend: Literal["auto", "flashinfer", "torch", "mlx"] = "auto",
        max_speech_tokens: int | None = None,
        n_cfm_timesteps: int = 2,
    ) -> list[torch.Tensor]:
        """Synthesize a heterogeneous batch (one prompt + text per row).

        Exactly one of ``audio_prompt_paths`` and ``conds_list`` must be
        provided. Both ``texts`` and the conditioning iterable must have the
        same length. Returns a list of 1D waveforms (one per row).
        """
        texts = list(texts)
        if audio_prompt_paths is not None and conds_list is not None:
            raise ValueError("Pass exactly one of audio_prompt_paths / conds_list.")
        if audio_prompt_paths is None and conds_list is None:
            raise ValueError("Pass either audio_prompt_paths or conds_list.")

        if conds_list is None:
            conds_list = [
                dataclasses.replace(
                    self.prepare_conditionals(p, exaggeration=exaggeration),
                ) if False else self.prepare_conditionals(p, exaggeration=exaggeration)
                for p in audio_prompt_paths
            ]
        else:
            conds_list = list(conds_list)
        if len(conds_list) != len(texts):
            raise ValueError(
                f"texts ({len(texts)}) and conditioning ({len(conds_list)}) "
                "must have equal length.",
            )

        text_tokens_list: list[torch.Tensor] = []
        speech_lens: list[int] = []
        for t in texts:
            tt = self._encode_text(t, normalize_text=normalize_text)
            text_tokens_list.append(tt)
            sl = _speech_len_for_text_tokens(int(tt.size(1)))
            if max_speech_tokens is not None:
                sl = min(sl, max_speech_tokens)
            speech_lens.append(sl)

        speech_tokens_batch = self.t3.generate(
            t3_cond=[c.t3 for c in conds_list],
            text_tokens=text_tokens_list,
            text_token_lens=None,
            total_speech_len=speech_lens,
            num_steps=num_steps,
            temperature=temperature,
            time_shift_tau=time_shift_tau,
            omnivoice_schedule_t_shift=omnivoice_schedule_t_shift,
            cfg_scale=cfg_scale,
            position_temperature=position_temperature,
            pmi_uncond_prior_precompute=pmi_uncond_prior_precompute,
            use_cuda_graph=use_cuda_graph,
            backend=backend,
        )

        wavs: list[torch.Tensor] = []
        for i, sl in enumerate(speech_lens):
            row = speech_tokens_batch[i : i + 1, :sl]
            row = self._trim_to_eos(row[0]).unsqueeze(0)
            wav, _ = self.s3gen.inference(
                speech_tokens=row.to(self.device),
                ref_dict=conds_list[i].gen,
                n_cfm_timesteps=n_cfm_timesteps,
            )
            wavs.append(wav.detach().cpu().squeeze(0))
        return wavs

    # -------- internal ---------------------------------------------------

    def _trim_to_eos(self, speech_tokens: torch.Tensor) -> torch.Tensor:
        """Truncate at the first stop-speech token and drop non-codec ids.

        T3 generates ids in ``[0, hp.speech_tokens_dict_size)``, but S3Gen's
        flow embedding is sized to the raw S3 codec vocab
        (``hp.start_speech_token``). Anything ``>= start_speech_token`` —
        the SOS / EOS specials, the ``[MASK]`` row, or never-unmasked slots —
        must be removed before vocoding.
        """
        if speech_tokens.numel() == 0:
            return speech_tokens.long()
        stop = self.t3.hp.stop_speech_token
        eos = (speech_tokens == stop).nonzero(as_tuple=True)[0]
        if len(eos) > 0:
            speech_tokens = speech_tokens[: eos[0].item()]
        codec_vocab = int(self.t3.hp.start_speech_token)  # 6561 for chatterbox
        speech_tokens = speech_tokens[speech_tokens < codec_vocab]
        return speech_tokens.long()
