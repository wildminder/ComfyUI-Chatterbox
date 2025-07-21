import os
import sys
import types
import logging
import folder_paths
from importlib import metadata as importlib_metadata

# Configure a logger for the entire custom node package
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Add a handler if none exist to avoid duplicate logs
if not logger.hasHandlers():
    # Use stdout for info, stderr for errors
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(f"[%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# Monkey-Patch for 'chatterbox-tts' version
# original_importlib_version_func = importlib_metadata.version
# def patched_version_lookup(package_name):
#     if package_name == "chatterbox-tts":
#         return "local-vendored"
#     return original_importlib_version_func(package_name)
# importlib_metadata.version = patched_version_lookup


# easily check if the real 'perth' is being used.
try:
    import perth
    # A simple check to ensure it's not our mock from a previous run
    if not hasattr(perth, '_is_mock'):
        logger.info("Found and using 'resemble-perth' library for watermarking.")
except ImportError:
    logger.warning("'resemble-perth' not found. Watermarking will be unavailable.")
    class DummyPerthImplicitWatermarker:
        def apply_watermark(self, wav, sample_rate):
            logger.warning("Watermarking skipped: 'resemble-perth' is not installed.")
            return wav
    perth_mock = types.ModuleType('perth')
    perth_mock.PerthImplicitWatermarker = DummyPerthImplicitWatermarker
    # Flag to identify our mock module
    perth_mock._is_mock = True
    sys.modules['perth'] = perth_mock


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


from .nodes import ChatterboxTTSNode, ChatterboxVCNode
from .modules.chatterbox_handler import CHATTERBOX_MODEL_SUBDIR

NODE_CLASS_MAPPINGS = {
    "ChatterboxTTS": ChatterboxTTSNode,
    "ChatterboxVC": ChatterboxVCNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterboxTTS": "Chatterbox TTS 📢",
    "ChatterboxVC": "Chatterbox Voice Conversion 🗣️",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Model path setup for ComfyUI
chatterbox_models_full_path = os.path.join(folder_paths.models_dir, CHATTERBOX_MODEL_SUBDIR)
if not os.path.exists(chatterbox_models_full_path):
    try:
        os.makedirs(chatterbox_models_full_path, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating models directory {chatterbox_models_full_path}: {e}")

# Register the tts/chatterbox path with ComfyUI
tts_chatterbox_path = os.path.join(folder_paths.models_dir, "tts")
if "tts" not in folder_paths.folder_names_and_paths:
    supported_exts = folder_paths.supported_pt_extensions.union({".safetensors"})
    folder_paths.folder_names_and_paths["tts"] = ([tts_chatterbox_path], supported_exts)
else:
    if tts_chatterbox_path not in folder_paths.folder_names_and_paths["tts"][0]:
        folder_paths.folder_names_and_paths["tts"][0].append(tts_chatterbox_path)