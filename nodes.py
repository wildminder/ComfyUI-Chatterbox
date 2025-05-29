import os
import torch
import folder_paths
import tempfile
import soundfile as sf
import numpy as np

from .modules.chatterbox_handler import (
    get_chatterbox_model_pack_names,
    get_cached_chatterbox_tts_model,
    get_cached_chatterbox_vc_model,
    set_chatterbox_seed,
    CHATTERBOX_MODEL_SUBDIR,
    DEFAULT_MODEL_PACK_NAME
)

class ChatterboxTTSNode:
    @classmethod
    def INPUT_TYPES(cls):
        available_model_packs = get_chatterbox_model_pack_names()
        displayed_packs = [DEFAULT_MODEL_PACK_NAME] + [p for p in available_model_packs if p != DEFAULT_MODEL_PACK_NAME]
        if not displayed_packs:
            displayed_packs = [DEFAULT_MODEL_PACK_NAME]

        return {
            "required": {
                "model_pack_name": (displayed_packs, {"default": DEFAULT_MODEL_PACK_NAME if DEFAULT_MODEL_PACK_NAME in displayed_packs else (displayed_packs[0] if displayed_packs else DEFAULT_MODEL_PACK_NAME)}),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test of Chatterbox TTS in ComfyUI."}),
                "exaggeration": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 2.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.05, "max": 5.0, "step": 0.05}),
                "cfg_weight": ("FLOAT", {"default": 0.5, "min": 0.2, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "device": (["cuda", "cpu"], {"default": "cuda" if torch.cuda.is_available() else "cpu"}),
            },
            "optional": {
                "audio_prompt": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "synthesize"
    CATEGORY = "audio/generation"
    OUTPUT_NODE = True 

    def synthesize(self, model_pack_name, text, exaggeration, temperature, cfg_weight, seed, device, audio_prompt=None):
        if not text.strip():
            #print("Chatterbox TTS: Empty text provided, returning silent audio.")
            dummy_sr = 24000 
            silent_waveform = torch.zeros((1, dummy_sr), dtype=torch.float32, device="cpu")
            return ({"waveform": silent_waveform.unsqueeze(0), "sample_rate": dummy_sr},)

        try:
            chatterbox_model = get_cached_chatterbox_tts_model(model_pack_name, device_str=device)
        except Exception as e:
            print(f"ChatterboxTTS: Error loading/downloading TTS model pack '{model_pack_name}': {e}")
            dummy_sr = 24000
            silent_waveform = torch.zeros((1, dummy_sr), dtype=torch.float32, device="cpu")
            return ({"waveform": silent_waveform.unsqueeze(0), "sample_rate": dummy_sr},)

        set_chatterbox_seed(seed)
        
        audio_prompt_path_temp = None
        if audio_prompt is not None and \
           audio_prompt.get("waveform") is not None and \
           audio_prompt["waveform"].numel() > 0:
            
            waveform_in = audio_prompt["waveform"] 
            sample_rate_in = audio_prompt["sample_rate"]
            waveform_cpu = waveform_in.cpu()
            if waveform_cpu.shape[0] > 1:
                print(f"ChatterboxTTS: Audio prompt has batch size {waveform_cpu.shape[0]}, using first item.")
            current_waveform = waveform_cpu[0] 
            if current_waveform.shape[0] > 1: 
                current_waveform = torch.mean(current_waveform, dim=0)
            else: 
                current_waveform = current_waveform.squeeze(0)
            
            processed_audio_prompt = current_waveform.numpy().astype(np.float32)
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                    audio_prompt_path_temp = tmp_wav.name
                sf.write(audio_prompt_path_temp, processed_audio_prompt, sample_rate_in)
                #print(f"ChatterboxTTS: Using audio prompt from temp file: {audio_prompt_path_temp}")
            except Exception as e:
                print(f"ChatterboxTTS: Error writing temp audio prompt file: {e}")
                audio_prompt_path_temp = None

        try:
            wav_tensor_chatterbox = chatterbox_model.generate(
                text,
                audio_prompt_path=audio_prompt_path_temp,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight
            ) 
        except Exception as e:
            print(f"ChatterboxTTS: Error during TTS generation: {e}")
            dummy_sr = 24000
            silent_waveform = torch.zeros((1, dummy_sr), dtype=torch.float32, device="cpu")
            return ({"waveform": silent_waveform.unsqueeze(0), "sample_rate": dummy_sr},)
        finally:
            if audio_prompt_path_temp and os.path.exists(audio_prompt_path_temp):
                try:
                    os.remove(audio_prompt_path_temp)
                except Exception as e:
                    print(f"ChatterboxTTS: Error removing temp audio prompt file {audio_prompt_path_temp}: {e}")
        
        wav_tensor_comfy = wav_tensor_chatterbox.cpu().unsqueeze(0) 
        return ({"waveform": wav_tensor_comfy, "sample_rate": chatterbox_model.sr},)


class ChatterboxVCNode:
    @classmethod
    def INPUT_TYPES(cls):
        available_model_packs = get_chatterbox_model_pack_names()
        displayed_packs = [DEFAULT_MODEL_PACK_NAME] + [p for p in available_model_packs if p != DEFAULT_MODEL_PACK_NAME]
        if not displayed_packs:
            displayed_packs = [DEFAULT_MODEL_PACK_NAME]

        return {
            "required": {
                "model_pack_name": (displayed_packs, {"default": DEFAULT_MODEL_PACK_NAME if DEFAULT_MODEL_PACK_NAME in displayed_packs else (displayed_packs[0] if displayed_packs else DEFAULT_MODEL_PACK_NAME)}),
                "source_audio": ("AUDIO",),
                "device": (["cuda", "cpu"], {"default": "cuda" if torch.cuda.is_available() else "cpu"}),
            },
            "optional": {
                "target_voice_audio": ("AUDIO",), # Optional: if not provided, uses default voice from conds.pt
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("converted_audio",)
    FUNCTION = "convert_voice"
    CATEGORY = "audio/generation"
    OUTPUT_NODE = True

    def _save_audio_to_temp_file(self, audio_data, prefix=""):
        """Helper to save ComfyUI AUDIO dict to a temporary WAV file."""
        if audio_data is None or \
           audio_data.get("waveform") is None or \
           audio_data["waveform"].numel() == 0:
            return None

        waveform_in = audio_data["waveform"]
        sample_rate_in = audio_data["sample_rate"]
        waveform_cpu = waveform_in.cpu()

        if waveform_cpu.shape[0] > 1:
            print(f"ChatterboxVC: {prefix}Audio has batch size {waveform_cpu.shape[0]}, using first item.")
        current_waveform = waveform_cpu[0]

        if current_waveform.shape[0] > 1: # If C > 1 (stereo or more)
            print(f"ChatterboxVC: {prefix}Audio has {current_waveform.shape[0]} channels, converting to mono by averaging.")
            current_waveform = torch.mean(current_waveform, dim=0)
        else: # If C == 1
            current_waveform = current_waveform.squeeze(0)
        
        processed_audio = current_waveform.numpy().astype(np.float32)
        
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                temp_file_path = tmp_wav.name
            sf.write(temp_file_path, processed_audio, sample_rate_in)
            #print(f"ChatterboxVC: Saved {prefix}audio to temp file: {temp_file_path}")
            return temp_file_path
        except Exception as e:
            print(f"ChatterboxVC: Error writing temp {prefix}audio file: {e}")
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return None

    def convert_voice(self, model_pack_name, source_audio, device, target_voice_audio=None):
        if source_audio is None or source_audio.get("waveform") is None or source_audio["waveform"].numel() == 0:
            print("ChatterboxVC: No source audio provided, returning silent audio.")
            dummy_sr = 24000
            silent_waveform = torch.zeros((1, dummy_sr), dtype=torch.float32, device="cpu")
            return ({"waveform": silent_waveform.unsqueeze(0), "sample_rate": dummy_sr},)

        try:
            vc_model = get_cached_chatterbox_vc_model(model_pack_name, device_str=device)
        except Exception as e:
            print(f"ChatterboxVC: Error loading/downloading VC model pack '{model_pack_name}': {e}")
            dummy_sr = 24000
            silent_waveform = torch.zeros((1, dummy_sr), dtype=torch.float32, device="cpu")
            return ({"waveform": silent_waveform.unsqueeze(0), "sample_rate": dummy_sr},)

        source_audio_path_temp = None
        target_voice_path_temp = None

        try:
            source_audio_path_temp = self._save_audio_to_temp_file(source_audio, prefix="Source ")
            if not source_audio_path_temp:
                raise ValueError("Failed to process source audio.")

            if target_voice_audio is not None and \
               target_voice_audio.get("waveform") is not None and \
               target_voice_audio["waveform"].numel() > 0:
                target_voice_path_temp = self._save_audio_to_temp_file(target_voice_audio, prefix="Target ")
                #print(f"ChatterboxVC: Using target voice from temp file: {target_voice_path_temp}")
            else:
                print("ChatterboxVC: No target voice audio provided or it's empty. Using default reference from model pack if available.")
            
            # ChatterboxVC.generate expects file paths
            converted_wav_tensor = vc_model.generate(
                audio=source_audio_path_temp,
                target_voice_path=target_voice_path_temp # This will be None if no target_voice_audio was provided or saving it failed
            ) # Expected output: (1, num_samples)

        except Exception as e:
            print(f"ChatterboxVC: Error during voice conversion: {e}")
            dummy_sr = 24000
            silent_waveform = torch.zeros((1, dummy_sr), dtype=torch.float32, device="cpu")
            return ({"waveform": silent_waveform.unsqueeze(0), "sample_rate": dummy_sr},)
        finally:
            if source_audio_path_temp and os.path.exists(source_audio_path_temp):
                try:
                    os.remove(source_audio_path_temp)
                except Exception as e:
                    print(f"ChatterboxVC: Error removing temp source audio file {source_audio_path_temp}: {e}")
            if target_voice_path_temp and os.path.exists(target_voice_path_temp):
                try:
                    os.remove(target_voice_path_temp)
                except Exception as e:
                    print(f"ChatterboxVC: Error removing temp target audio file {target_voice_path_temp}: {e}")
        
        # ComfyUI AUDIO format: {"waveform": tensor (B, C, T), "sample_rate": int}
        # ChatterboxVC output: tensor (1, T)
        vc_wav_tensor_comfy = converted_wav_tensor.cpu().unsqueeze(0) # (1, T) -> (1, 1, T)
        return ({"waveform": vc_wav_tensor_comfy, "sample_rate": vc_model.sr},)