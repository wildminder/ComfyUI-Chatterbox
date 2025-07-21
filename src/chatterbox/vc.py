import logging
from pathlib import Path

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR, S3_TOKEN_RATE
from .models.s3gen import S3GEN_SR, S3Gen

logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox"

def _pad_wav_to_40ms_multiple(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """
    Pads a waveform to be a multiple of 40ms to prevent rounding errors between
    the mel spectrogram (20ms hop) and the speech tokenizer (40ms hop).
    """
    S3_TOKEN_DURATION_S = 1 / S3_TOKEN_RATE  # 0.04 seconds
    samples_per_token = int(sr * S3_TOKEN_DURATION_S)
    current_samples = wav.shape[-1]
    remainder = current_samples % samples_per_token
    if remainder != 0:
        padding_needed = samples_per_token - remainder
        padded_wav = F.pad(wav, (0, padding_needed))
        return padded_wav
    return wav

class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict=None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self.watermarker = perth.PerthImplicitWatermarker()
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        ckpt_dir = Path(ckpt_dir)
        
        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None
            
        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states['gen']

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxVC':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logger.warning("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                logger.warning("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"
            
        for fpath in ["s3gen.safetensors", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def set_target_voice(self, wav_fpath):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        # Convert to tensor and pad to a 40ms boundary
        s3gen_ref_wav = torch.from_numpy(s3gen_ref_wav).float().unsqueeze(0)
        s3gen_ref_wav = _pad_wav_to_40ms_multiple(s3gen_ref_wav, S3GEN_SR)
        s3gen_ref_wav_np = s3gen_ref_wav.squeeze(0).numpy()

        s3gen_ref_wav_np = s3gen_ref_wav_np[:self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav_np, S3GEN_SR, device=self.device)

    def generate(
        self,
        audio,
        target_voice_path=None,
        n_timesteps=10,
        pbar=None,
        temperature=1.0,
        flow_cfg_scale=0.7
    ):
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        else:
            assert self.ref_dict is not None, "Please `set_target_voice` first or specify `target_voice_path`"

        with torch.inference_mode():
            audio_16, _ = librosa.load(audio, sr=S3_SR)
            audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None, ]

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
                n_timesteps=n_timesteps,
                pbar=pbar,
                temperature=temperature,
                flow_cfg_scale=flow_cfg_scale
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
