import os
import folder_paths
from .nodes import ChatterboxTTSNode, ChatterboxVCNode
from .modules.chatterbox_handler import CHATTERBOX_MODEL_SUBDIR, DEFAULT_MODEL_PACK_NAME

NODE_CLASS_MAPPINGS = {
    "ChatterboxTTS": ChatterboxTTSNode,
    "ChatterboxVC": ChatterboxVCNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterboxTTS": "Chatterbox TTS 📢",
    "ChatterboxVC": "Chatterbox Voice Conversion 🗣️",
}

# WEB_DIRECTORY = "./js" 
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

chatterbox_models_full_path = os.path.join(folder_paths.models_dir, CHATTERBOX_MODEL_SUBDIR)
if not os.path.exists(chatterbox_models_full_path):
    try:
        os.makedirs(chatterbox_models_full_path, exist_ok=True)
        #print(f"ChatterboxTTS/VC Init: Created models directory at {chatterbox_models_full_path}")
    except OSError as e:
        print(f"ChatterboxTTS/VC Init: Error creating models directory {chatterbox_models_full_path}: {e}")

if CHATTERBOX_MODEL_SUBDIR not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[CHATTERBOX_MODEL_SUBDIR] = (
        [chatterbox_models_full_path],
        folder_paths.supported_pt_extensions
    )
    #print(f"ChatterboxTTS/VC Init: Registered '{CHATTERBOX_MODEL_SUBDIR}' with ComfyUI model paths: {chatterbox_models_full_path}")
else:
    if chatterbox_models_full_path not in folder_paths.folder_names_and_paths[CHATTERBOX_MODEL_SUBDIR][0]:
        folder_paths.folder_names_and_paths[CHATTERBOX_MODEL_SUBDIR][0].append(chatterbox_models_full_path)
        #print(f"ChatterboxTTS/VC Init: Appended path for '{CHATTERBOX_MODEL_SUBDIR}': {chatterbox_models_full_path}")

