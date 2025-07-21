import os
import logging
import torch
import random
import numpy as np
import folder_paths
from huggingface_hub import hf_hub_download

from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC

logger = logging.getLogger(__name__)

CHATTERBOX_MODEL_SUBDIR = os.path.join("tts", "chatterbox")
CHATTERBOX_REPO_ID = "ResembleAI/chatterbox"
CHATTERBOX_FILES_TO_DOWNLOAD = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]
DEFAULT_MODEL_PACK_NAME = "resembleai_default_voice"

def get_chatterbox_model_pack_names():
    chatterbox_models_base_path = os.path.join(folder_paths.models_dir, CHATTERBOX_MODEL_SUBDIR)
    if not os.path.isdir(chatterbox_models_base_path):
        os.makedirs(chatterbox_models_base_path, exist_ok=True)
        return [DEFAULT_MODEL_PACK_NAME]
    packs = [d for d in os.listdir(chatterbox_models_base_path) if os.path.isdir(os.path.join(chatterbox_models_base_path, d))]
    # Ensure default is first if it exists
    if DEFAULT_MODEL_PACK_NAME in packs:
        packs.insert(0, packs.pop(packs.index(DEFAULT_MODEL_PACK_NAME)))
    
    # Return default even if folder doesn't exist yet, to prompt the user to download it.
    return packs if packs else [DEFAULT_MODEL_PACK_NAME]

def get_model_pack_path(model_pack_name):
    # Added check for None or empty string to prevent errors
    if not model_pack_name: 
        return None
    return os.path.join(folder_paths.models_dir, CHATTERBOX_MODEL_SUBDIR, model_pack_name)

def _download_file_from_hf(repo_id, filename, local_dir):
    destination = os.path.join(local_dir, filename)
    if not os.path.exists(destination):
        logger.info(f"Downloading '{filename}' from '{repo_id}'...")
        try:
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, local_dir_use_symlinks=False, resume_download=True)
            logger.info(f"Successfully downloaded '{filename}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to download '{filename}': {e}")
            if os.path.exists(destination + ".incomplete"): os.remove(destination + ".incomplete")
            return False
    return True

def download_chatterbox_model_pack_if_missing(model_pack_name):
    ckpt_dir = get_model_pack_path(model_pack_name)
    if not ckpt_dir:
        logger.warning(f"Invalid model pack name '{model_pack_name}', cannot download.")
        return False
    os.makedirs(ckpt_dir, exist_ok=True)
    all_files_ok = all(_download_file_from_hf(CHATTERBOX_REPO_ID, f, ckpt_dir) for f in CHATTERBOX_FILES_TO_DOWNLOAD)
    if not all_files_ok:
        logger.error(f"Some files failed to download for model pack '{model_pack_name}'. Check logs.")
    return all_files_ok

def load_chatterbox_models(model_pack_name, device):
    """Loads both TTS and VC models for a given pack onto a specified device."""
    ckpt_dir = get_model_pack_path(model_pack_name)
    if not ckpt_dir:
        raise ValueError(f"Invalid model_pack_name: {model_pack_name}")
    
    if not download_chatterbox_model_pack_if_missing(model_pack_name):
        logger.warning(f"Not all model files could be verified for '{model_pack_name}'. Loading may fail.")
        
    if not os.path.isdir(ckpt_dir):
         raise FileNotFoundError(f"Model pack directory '{model_pack_name}' not found at '{ckpt_dir}'.")

    try:
        logger.info(f"Loading Chatterbox TTS model from {ckpt_dir} onto {device}")
        tts_model = ChatterboxTTS.from_local(ckpt_dir, device=device)
    except Exception as e:
        logger.error(f"Error loading ChatterboxTTS from '{ckpt_dir}': {e}", exc_info=True)
        raise
        
    try:
        logger.info(f"Loading Chatterbox VC model from {ckpt_dir} onto {device}")
        vc_model = ChatterboxVC.from_local(ckpt_dir, device=device)
    except Exception as e:
        logger.error(f"Error loading ChatterboxVC from '{ckpt_dir}': {e}", exc_info=True)
        raise

    return tts_model, vc_model

def set_chatterbox_seed(seed: int):
    MAX_NUMPY_SEED = 2**32 - 1
    actual_seed_for_torch_random = random.randint(1, 0xffffffffffffffff) if seed == 0 else seed
    actual_seed_for_numpy = random.randint(1, MAX_NUMPY_SEED) if seed == 0 else (seed % MAX_NUMPY_SEED)
    torch.manual_seed(actual_seed_for_torch_random)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(actual_seed_for_torch_random)
    random.seed(actual_seed_for_torch_random)
    np.random.seed(actual_seed_for_numpy)