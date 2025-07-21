import os
import torch
import tempfile
import soundfile as sf
import numpy as np
import logging
import perth

import comfy.model_management as mm
import comfy.model_patcher
from comfy.utils import ProgressBar

from .modules.chatterbox_handler import (
    get_chatterbox_model_pack_names,
    load_chatterbox_models,
    set_chatterbox_seed,
    DEFAULT_MODEL_PACK_NAME
)

logger = logging.getLogger(__name__)

CHATTERBOX_PATCHER_CACHE = {}

class ChatterboxModelWrapper(torch.nn.Module):
    """
    A simple torch.nn.Module wrapper for the Chatterbox models.
    This allows ComfyUI's model management to treat our custom models like any other
    torch module, enabling device placement (.to()) and other standard operations.
    """
    def __init__(self, model_pack_name):
        super().__init__()
        self.model_pack_name = model_pack_name
        self.tts_model = None
        self.vc_model = None

    def load_model(self, device):
        self.tts_model, self.vc_model = load_chatterbox_models(self.model_pack_name, device)

class ChatterboxPatcher(comfy.model_patcher.ModelPatcher):
    """
    Custom ModelPatcher for Chatterbox. This class hooks into ComfyUI's
    model management system (loading, offloading) to handle our non-standard models.
    """
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def patch_model(self, device_to=None, *args, **kwargs):
        """
        This method is called by ComfyUI's model manager when it's time to load
        the model onto the target device (usually the GPU). Our responsibility here
        is to ensure the model weights are loaded from disk if they haven't been already.
        """
        target_device = self.load_device

        # The core loading logic: If the model isn't in memory, load it from disk.
        if self.model.tts_model is None:
            logger.info(f"Loading Chatterbox models for '{self.model.model_pack_name}' to {target_device}...")
            self.model.load_model(target_device)
            self.model.model_loaded_weight_memory = self.size
        else:
            logger.info(f"Chatterbox models for '{self.model.model_pack_name}' already in memory.")

        return super().patch_model(device_to=target_device, *args, **kwargs)

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        """
        This method is called by ComfyUI's model manager to offload the model
        (usually to the CPU) and free up VRAM.
        """
        if unpatch_weights:
            logger.info(f"Offloading Chatterbox models for '{self.model.model_pack_name}' to {device_to}...")
            self.model.tts_model = None
            self.vc_model = None
            # Reset memory footprint
            self.model.model_loaded_weight_memory = 0
            mm.soft_empty_cache()
        return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)

class ChatterboxTTSNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_pack_name": (get_chatterbox_model_pack_names(), {
                "default": DEFAULT_MODEL_PACK_NAME,
                "tooltip": "Select the Chatterbox voice model pack to use for generation."
            }),
            "text": ("STRING", {
                "multiline": True,
                "default": "Hello, this is a test of Chatterbox TTS in ComfyUI.",
                "tooltip": "Text to be synthesized into speech."
            }),
            "max_new_tokens": ("INT", {
                "default": 1000, "min": 16, "max": 4000, "step": 8,
                "tooltip": "Maximum number of audio tokens to generate. 25 tokens ≈ 1 second. The hard limit is 4096 tokens (≈ 163 seconds)."
            }),
            "flow_cfg_scale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "CFG scale for the mel spectrogram decoder (flow matching). Higher values increase adherence to content and timbre but may reduce naturalness."}),
            "exaggeration": ("FLOAT", {
                "default": 0.5, "min": 0.25, "max": 2.0, "step": 0.05,
                "tooltip": "Controls the expressiveness and emotional intensity. Higher values lead to more exaggerated prosody."
            }),
            "temperature": ("FLOAT", {
                "default": 0.8, "min": 0.05, "max": 5.0, "step": 0.05,
                "tooltip": "Controls the randomness of the sampling process. Higher values produce more diverse speech, while lower values are more deterministic."
            }),
            "cfg_weight": ("FLOAT", {
                "default": 0.5, "min": 0.2, "max": 1.0, "step": 0.05,
                "tooltip": "Classifier-Free Guidance (CFG) weight. Controls how strongly the model adheres to the text prompt. Higher values may reduce naturalness."
            }),
            "repetition_penalty": ("FLOAT", {
                "default": 1.2, "min": 1.0, "max": 2.0, "step": 0.1,
                "tooltip": "Penalizes repeated tokens to discourage monotonous or repetitive speech. A value of 1.0 means no penalty."
            }),
            "min_p": ("FLOAT", {
                "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Sets a minimum probability threshold for nucleus sampling (Min-P). Filters out tokens with very low probability."
            }),
            "top_p": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Nucleus sampling (Top-P) parameter. The model samples from the smallest set of tokens whose cumulative probability exceeds this value."
            }),
            "seed": ("INT", {
                "default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                "tooltip": "Seed for random number generation. A value of 0 will use a random seed."
            }),
            "use_watermark": ("BOOLEAN", {
                "default": False,
                "tooltip": "Enable or disable the audio watermark. Requires 'resemble-perth' to be installed."
            }),
        }, "optional": {"audio_prompt": ("AUDIO",),}}

    RETURN_TYPES = ("AUDIO",); RETURN_NAMES = ("audio",); FUNCTION = "synthesize"; CATEGORY = "audio/generation"; OUTPUT_NODE = True

    def synthesize(self, model_pack_name, text, max_new_tokens, flow_cfg_scale, exaggeration, temperature, cfg_weight, repetition_penalty, min_p, top_p, seed, use_watermark, audio_prompt=None):
        if not text.strip():
            logger.info("Empty text provided, returning silent audio.")
            dummy_sr = 24000; silent_waveform = torch.zeros((1, dummy_sr), dtype=torch.float32, device="cpu")
            return ({"waveform": silent_waveform.unsqueeze(0), "sample_rate": dummy_sr},)

        cache_key = model_pack_name
        if cache_key not in CHATTERBOX_PATCHER_CACHE:
            load_device = mm.get_torch_device()
            logger.info(f"Creating Chatterbox ModelPatcher for {model_pack_name} on device {load_device}")
            model_wrapper = ChatterboxModelWrapper(model_pack_name)
            patcher = ChatterboxPatcher(
                model=model_wrapper,
                load_device=load_device,
                offload_device=mm.unet_offload_device(),
                size=int(1.5 * 1024**3)
            )
            CHATTERBOX_PATCHER_CACHE[cache_key] = patcher

        patcher = CHATTERBOX_PATCHER_CACHE[cache_key]

        mm.load_model_gpu(patcher)
        tts_model = patcher.model.tts_model

        if tts_model is None:
            logger.error("TTS model failed to load. Please check logs for download or loading errors.")
            dummy_sr = 24000; silent_waveform = torch.zeros((1, dummy_sr), dtype=torch.float32, device="cpu")
            return ({"waveform": silent_waveform.unsqueeze(0), "sample_rate": dummy_sr},)

        set_chatterbox_seed(seed)

        is_perth_installed = not getattr(perth, '_is_mock', False)
        if use_watermark and not is_perth_installed:
            logger.warning("Watermarking is enabled, but 'resemble-perth' is not installed. Output will not be watermarked.")

        original_watermarker = tts_model.watermarker
        if not use_watermark:
            class TmpDummyWatermarker:
                def apply_watermark(self, wav, sample_rate): return wav
            tts_model.watermarker = TmpDummyWatermarker()
            if is_perth_installed: logger.info("Watermarking disabled by user.")

        wav_tensor_chatterbox = None; audio_prompt_path_temp = None
        
        pbar = ProgressBar(max_new_tokens)

        try:
            if audio_prompt and audio_prompt.get("waveform") is not None and audio_prompt["waveform"].numel() > 0:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                    audio_prompt_path_temp = tmp_wav.name
                    waveform_in = audio_prompt["waveform"]; sample_rate_in = audio_prompt["sample_rate"]
                    waveform_cpu = waveform_in.cpu()[0]
                    current_waveform = torch.mean(waveform_cpu, dim=0) if waveform_cpu.shape[0] > 1 else waveform_cpu.squeeze(0)
                    sf.write(audio_prompt_path_temp, current_waveform.numpy().astype(np.float32), sample_rate_in)

            wav_tensor_chatterbox = tts_model.generate(
                text,
                audio_prompt_path=audio_prompt_path_temp,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                pbar=pbar,
                max_new_tokens=max_new_tokens,
                flow_cfg_scale=flow_cfg_scale
            )
        except Exception as e:
            logger.error(f"Error during TTS generation: {e}", exc_info=True)
            dummy_sr = 24000; silent_waveform = torch.zeros((1, dummy_sr), dtype=torch.float32, device="cpu")
            return ({"waveform": silent_waveform.unsqueeze(0), "sample_rate": dummy_sr},)
        finally:
            tts_model.watermarker = original_watermarker
            if audio_prompt_path_temp and os.path.exists(audio_prompt_path_temp):
                try: os.remove(audio_prompt_path_temp)
                except Exception as e: logger.error(f"Error removing temp audio prompt file: {e}")

        wav_tensor_comfy = wav_tensor_chatterbox.cpu().unsqueeze(0)
        return ({"waveform": wav_tensor_comfy, "sample_rate": tts_model.sr},)

class ChatterboxVCNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_pack_name": (get_chatterbox_model_pack_names(), {
                "default": DEFAULT_MODEL_PACK_NAME,
                "tooltip": "Select the Chatterbox voice model pack to use for conversion."
            }),
            "source_audio": ("AUDIO", {
                "tooltip": "The audio containing the speech content to be converted."
            }),
            "n_timesteps": ("INT", {
                "default": 10, "min": 2, "max": 50, "step": 1,
                "tooltip": "Number of diffusion steps for the flow matching process. Higher values may improve quality at the cost of speed."
            }),
            "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                      "tooltip": "Controls the randomness of the initial noise. 1.0 is standard. Lower values are more deterministic."}),
            "flow_cfg_scale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                                         "tooltip": "CFG scale for the mel spectrogram decoder. Higher values increase adherence to the target voice but may reduce naturalness."}),
            "use_watermark": ("BOOLEAN", {
                "default": False,
                "tooltip": "Enable or disable the audio watermark. Requires 'resemble-perth' to be installed."
            }),
        },
            "optional": {"target_voice_audio": ("AUDIO", {
            "tooltip": "The audio file containing the target voice timbre. If not provided, the default voice from the model pack will be used."
        }), }}

    RETURN_TYPES = ("AUDIO",); RETURN_NAMES = ("converted_audio",); FUNCTION = "convert_voice"; CATEGORY = "audio/generation"; OUTPUT_NODE = True

    def _save_audio_to_temp_file(self, audio_data, prefix=""):
        if audio_data is None or audio_data.get("waveform") is None or audio_data["waveform"].numel() == 0: return None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            try:
                waveform_in = audio_data["waveform"]; sample_rate_in = audio_data["sample_rate"]
                waveform_cpu = waveform_in.cpu()[0]
                current_waveform = torch.mean(waveform_cpu, dim=0) if waveform_cpu.shape[0] > 1 else waveform_cpu.squeeze(0)
                sf.write(tmp_wav.name, current_waveform.numpy().astype(np.float32), sample_rate_in)
                return tmp_wav.name
            except Exception as e:
                logger.error(f"Error writing temp {prefix}audio file: {e}", exc_info=True)
                return None

    def convert_voice(self, model_pack_name, source_audio, n_timesteps, temperature, flow_cfg_scale, use_watermark, target_voice_audio=None):
        if source_audio is None or source_audio.get("waveform") is None or source_audio["waveform"].numel() == 0:
            logger.warning("No source audio provided, returning silent audio.")
            dummy_sr = 24000; silent_waveform = torch.zeros((1, dummy_sr), dtype=torch.float32, device="cpu")
            return ({"waveform": silent_waveform.unsqueeze(0), "sample_rate": dummy_sr},)

        cache_key = model_pack_name
        if cache_key not in CHATTERBOX_PATCHER_CACHE:
            load_device = mm.get_torch_device()
            logger.info(f"Creating Chatterbox ModelPatcher for {model_pack_name} on device {load_device}")
            model_wrapper = ChatterboxModelWrapper(model_pack_name)
            patcher = ChatterboxPatcher(
                model=model_wrapper,
                load_device=load_device,
                offload_device=mm.unet_offload_device(),
                size=int(1.5 * 1024**3)
            )
            CHATTERBOX_PATCHER_CACHE[cache_key] = patcher
        
        patcher = CHATTERBOX_PATCHER_CACHE[cache_key]

        mm.load_model_gpu(patcher)
        vc_model = patcher.model.vc_model

        if vc_model is None:
            logger.error("VC model failed to load. Please check logs for download or loading errors.")
            dummy_sr = 24000; silent_waveform = torch.zeros((1, dummy_sr), dtype=torch.float32, device="cpu")
            return ({"waveform": silent_waveform.unsqueeze(0), "sample_rate": dummy_sr},)

        is_perth_installed = not getattr(perth, '_is_mock', False)
        if use_watermark and not is_perth_installed:
            logger.warning("Watermarking is enabled, but 'resemble-perth' is not installed. Output will not be watermarked.")
        
        original_watermarker = vc_model.watermarker
        if not use_watermark:
            class TmpDummyWatermarker:
                def apply_watermark(self, wav, sample_rate): return wav
            vc_model.watermarker = TmpDummyWatermarker()
            if is_perth_installed: logger.info("Watermarking disabled by user.")

        source_audio_path_temp = None; target_voice_path_temp = None
        
        pbar = ProgressBar(n_timesteps)

        try:
            source_audio_path_temp = self._save_audio_to_temp_file(source_audio, prefix="Source ")
            if not source_audio_path_temp: raise ValueError("Failed to process source audio.")
            if target_voice_audio and target_voice_audio.get("waveform") is not None and target_voice_audio["waveform"].numel() > 0:
                target_voice_path_temp = self._save_audio_to_temp_file(target_voice_audio, prefix="Target ")
            
            converted_wav_tensor = vc_model.generate(
                audio=source_audio_path_temp,
                target_voice_path=target_voice_path_temp,
                n_timesteps=n_timesteps,
                pbar=pbar,
                temperature=temperature,
                flow_cfg_scale=flow_cfg_scale
            )
        except Exception as e:
            logger.error(f"Error during voice conversion: {e}", exc_info=True)
            dummy_sr = 24000; silent_waveform = torch.zeros((1, dummy_sr), dtype=torch.float32, device="cpu")
            return ({"waveform": silent_waveform.unsqueeze(0), "sample_rate": dummy_sr},)
        finally:
            vc_model.watermarker = original_watermarker
            if source_audio_path_temp and os.path.exists(source_audio_path_temp): os.remove(source_audio_path_temp)
            if target_voice_path_temp and os.path.exists(target_voice_path_temp): os.remove(target_voice_path_temp)
        
        vc_wav_tensor_comfy = converted_wav_tensor.cpu().unsqueeze(0)
        return ({"waveform": vc_wav_tensor_comfy, "sample_rate": vc_model.sr},)
