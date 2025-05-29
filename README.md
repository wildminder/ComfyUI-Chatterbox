<div align="center">

<h1>ComfyUI Chatterbox TTS & Voice Conversion Node</h1>

<p align="center">
  <img src="./assets/preview.png" alt="ComfyUI-KEEP Workflow Example">
</p>
    
</div>

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


## 🎭 Chatterbox TTS Demo Samples  
Exampler from the [official demo](https://resemble-ai.github.io/chatterbox_demopage/)

**Text Prompt:**  
_"Everybody be cool. This is a robbery. Any of you fucking pricks move and I'll execute every motherfucking last one of you."_

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Prompt</th>
    <th class="tg-0pky">Exaggeration 0.5</th>
    <th class="tg-0pky">Exaggeration 1.0</th>
    <th class="tg-0pky">Exaggeration 2.0</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky"><audio class="audio-narrow" src="https://storage.googleapis.com/chatterbox-demo-samples/prompts/male_old_movie.flac" controls="" preload=""></audio></td>
    <td class="tg-0pky"><audio src="https://storage.googleapis.com/chatterbox-demo-samples/samples/old_movie_exaggerate_0.5.wav" controls="" preload=""></audio></td>
    <td class="tg-0pky"><audio src="https://storage.googleapis.com/chatterbox-demo-samples/samples/old_movie_exaggerate_1.0.wav" controls="" preload=""></audio></td>
    <td class="tg-0pky"><audio src="https://storage.googleapis.com/chatterbox-demo-samples/samples/old_movie_exaggerate_2.0.wav" controls="" preload=""></audio></td>
  </tr>
   <tr>
    <td class="tg-0pky"><audio class="audio-narrow" src="https://storage.googleapis.com/chatterbox-demo-samples/prompts/male_petergriffin.wav" controls="" preload=""></audio></td>
    <td class="tg-0pky"><audio src="https://storage.googleapis.com/chatterbox-demo-samples/samples/peter_griffin_exag_0.5.wav" controls="" preload=""></audio></td>
    <td class="tg-0pky"><audio src="https://storage.googleapis.com/chatterbox-demo-samples/samples/peter_griffin_exag_1.0.wav" controls="" preload=""></audio></td>
    <td class="tg-0pky"><audio src="https://storage.googleapis.com/chatterbox-demo-samples/samples/peter_griffin_exag_2.0.wav" controls="" preload=""></audio></td>
  </tr>
</tbody>
</table>
<hr />


**Text Prompt:**  
_"My name is Maximus Decimus Meridius, commander of the Armies of the North, General of the Felix Legions and loyal servant to the true emperor, Marcus Aurelius.  
Father to a murdered son, husband to a murdered wife. And I will have my vengeance, in this life or the next."_

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Prompt</th>
    <th class="tg-0pky">Output</th>
  </tr>
</thead>
<tbody>
   <tr>
    <td class="tg-0pky"><audio class="audio-narrow" src="https://storage.googleapis.com/chatterbox-demo-samples/prompts/male_rickmorty.mp3" controls="" preload=""></audio></td>
    <td class="tg-0pky"><audio src="https://storage.googleapis.com/chatterbox-demo-samples/samples/gladiator_rick.wav" controls="" preload=""></audio></td>
  </tr>
  <tr>
    <td class="tg-0pky"><audio class="audio-narrow" src="https://storage.googleapis.com/chatterbox-demo-samples/prompts/male_old_movie.flac" controls="" preload=""></audio></td>
    <td class="tg-0pky"><audio src="https://storage.googleapis.com/chatterbox-demo-samples/samples/gladiator_old_movie.wav" controls="" preload=""></audio></td>
  </tr>
</tbody>
</table>

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

*   The Chatterbox library is included within this custom node's `src/` directory.


## Acknowledgements

*   This node relies on the [Chatterbox](https://github.com/resemble-ai/chatterbox) library by Resemble AI.
