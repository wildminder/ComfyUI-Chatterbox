# ComfyUI Chatterbox TTS & Voice Conversion Node

Custom nodes for ComfyUI that integrate the [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) library for Text-to-Speech (TTS) and Voice Conversion (VC).

## Features

*   **Chatterbox TTS Node:**
    *   Synthesize speech from text.
    *   Optional voice cloning using an audio prompt.
    *   Adjustable parameters: exaggeration, temperature, CFG weight, seed.
*   **Chatterbox Voice Conversion Node:**
    *   Convert the voice in a source audio file to sound like a target voice.
    *   Uses a target audio file for voice characteristics or defaults to a built-in voice if no target is provided.
*   **Automatic Model Downloading:** Necessary model files are automatically downloaded from Hugging Face (`ResembleAI/chatterbox`) on first use if not found locally.

## Installation

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/wildminder/ComfyUI-Chatterbox.git ComfyUI/custom_nodes/ComfyUI-Chatterbox
    ```

2.  **Install Dependencies:**
    Navigate to the custom node's directory and install the required packages:
    ```bash
    cd ComfyUI/custom_nodes/ComfyUI-Chatterbox
    pip install -r requirements.txt
    ```

3.  **Model Pack Directory (Automatic Setup):**
    The node will automatically attempt to download the default model pack (`resembleai_default_voice`) into `ComfyUI/models/chatterbox_tts/` when you first use a node that requires it.
    You can also manually create subdirectories in `ComfyUI/models/chatterbox_tts/` and place other Chatterbox model packs there. Each pack should contain:
    *   `ve.pt`
    *   `t3_cfg.pt`
    *   `s3gen.pt`
    *   `tokenizer.json`
    *   `conds.pt` (for default voice capabilities)

4.  **Restart ComfyUI.**

## Usage

After installation and restarting ComfyUI:

*   The **"Chatterbox TTS 📢"** node will be available under the `audio/generation` category.
*   The **"Chatterbox Voice Conversion 🗣️"** node will be available under the `audio/conversion` category.

Load example workflows from the `workflow-examples/` directory in this repository to get started.

## Notes

*   The Chatterbox library is vendored (included) within this custom node's `src/` directory.


## Acknowledgements

*   This node relies on the [Chatterbox](https://github.com/resemble-ai/chatterbox) library by Resemble AI.
*   Inspiration for auto-downloading and node structure from the ComfyUI ecosystem.
