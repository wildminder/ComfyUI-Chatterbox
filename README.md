<a id="readme-top"></a>

<div align="center">

# ◇ ComfyUI-Chatterbox ◆ Revised ◆

<img src="https://github.com/user-attachments/assets/7f5524c3-7c02-46fc-a548-b4157dff17db" />


<p>

Chatterbox, refactored. **Text-to-Speech and Voice Conversion for ComfyUI** on a leaner core.
Supports: **classic, Turbo, Nano, Flash, Multilingual.**

*Less to import. No temp files. Same voices, faster to a result.*

</p>

<!-- PROJECT SHIELDS -->
[![Telegram][telegram-shield]][telegram-url]
[![X][x-shield]][x-url]

</div>

> [!NOTE]
> **ComfyUI-Chatterbox** runs a refactored Chatterbox core. The vendored model
> code reimplements its audio DSP in pure numpy/torch, bit-for-bit with the
> original (verified against librosa 1.0.0 on the trained featurization
> parameters), and drops `librosa`, `scipy`, `omegaconf` and `conformer`.
> Reference clips are handed to the model in memory, never through a temp
> `.wav`. The only package this node adds is `s3tokenizer`. Models download on
> first run into `ComfyUI/models/tts/chatterbox/`.

<details>
<summary><b>Table of Contents</b></summary>

* [Features](#features)
* [Models](#models)
* [Getting started](#getting-started)
* [Usage](#usage)
* [Good to know](#good-to-know)
* [Acknowledgments](#acknowledgments)

</details>

<p id="features" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ❖ Features

| Capability | Detail |
| :--- | :--- |
| **One TTS node, every pack** | classic, Turbo 350M, Nano 110M and Flash are routed automatically from the pack's contents |
| **Multilingual TTS** | 23 languages, v1 / v2 / v3 weights, plus six Single Language Pack finetunes (zh-cmn, es-mx, es-es, pt-br, pt-pt, hi) |
| **Voice cloning** | any `.wav` / `.mp3` as the voice prompt, passed to the model in memory |
| **Voice conversion** | re-voice a source recording into a target timbre |
| **Multi-speaker dialog** | up to four cloned voices in one output, with SRT subtitle alignment |
| **Lean core** | DSP in pure numpy/torch, bit-exact with the original; `s3tokenizer` is the only dependency |
| **Devices** | CUDA, ROCm, XPU, MPS and CPU, under ComfyUI's VRAM manager |

<p id="models" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ⌸ Models

Downloaded on first run into `ComfyUI/models/tts/chatterbox/` (~2.5 GB each).
Shared weights (`ve.safetensors`, `s3gen*`) are stored once and reused across
packs.

| Family | Members | Notes |
| :--- | :--- | :--- |
| **Classic** | `chatterbox` | English TTS, built-in reference voice in `conds.pt` |
| **Turbo** | `chatterbox-turbo`, `chatterbox-nano` | Paralinguistic tags, top-k sampler; 350M / 110M. One family, one folder -- the dropdown entry picks the weights. |
| **Flash** | `chatterbox-flash` | Block-diffusion sampler; a reference clip is required |
| **Multilingual** | `chatterbox-mtl` | 23 languages, v1 / v2 / v3 |
| **Single Language Packs** | zh-cmn, es-mx-latam, es-es, pt-br, pt-pt, hi | MTL node only |

> [!TIP]
> Picks appear in the node's model dropdown; nothing to configure by hand.
> Ready-made graphs for every family live in [`example_workflows/`](./example_workflows).

**Pack names.** The dropdown shows this repository's names; the weights come
from upstream's own packs, fetched on first run:

| Pick | Upstream pack | Folder under `models/tts/chatterbox/` |
| :--- | :--- | :--- |
| `chatterbox-turbo` | [`ResembleAI/chatterbox-turbo`](https://huggingface.co/ResembleAI/chatterbox-turbo) | `chatterbox-turbo` |
| `chatterbox-nano` | [`ResembleAI/chatterbox-nano`](https://huggingface.co/ResembleAI/chatterbox-nano) | `chatterbox-turbo` (shared with Turbo) |

Turbo and Nano are one family with two `t3` checkpoints: the folder is shared,
the dropdown entry -- not the folder -- picks the weights.

The vendored code is private to this node: it is imported under this node's
own package name, so it never collides with another plugin's `chatterbox`.

<details>
<summary><b>Paralinguistic tags -- the 19 exact spellings</b></summary>

`chatterbox-turbo` and `chatterbox-nano` speak these natively: write them into
the text box in square brackets and the model says them.

* **Sound (9):** `[laugh]`, `[sigh]`, `[gasp]`, `[chuckle]`, `[cough]`, `[sniff]`, `[groan]`, `[shush]`, `[clear throat]`
* **Emotion (10):** `[angry]`, `[fear]`, `[surprised]`, `[whispering]`, `[advertisement]`, `[dramatic]`, `[narration]`, `[crying]`, `[happy]`, `[sarcastic]`

Spell them exactly as written -- the tokenizer matches the literal text, so
`[HAPPY]` is not `[happy]`, and `[clear throat]` takes a space, not an
underscore. The v2 spelling `<clear_throat>` is re-spelled for these packs.

No other pack has this vocabulary: on classic, multilingual, Single Language
Pack and Flash the same words are read out as plain text, and the run says so
in the log.

</details>

<p id="getting-started" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ⇩ Getting started

**ComfyUI Manager** — `Manager` → `Install Custom Nodes` → search
`ComfyUI-Chatterbox` → `Install` → restart.

<details>
<summary><b>Manual installation</b></summary>

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/wildminder/ComfyUI-Chatterbox.git
pip install -r ComfyUI-Chatterbox/requirements.txt
```

Restart ComfyUI. Nodes appear under `WMNodes/sound/*`.

| Requirement | Version |
| :--- | :--- |
| Python | `>=3.10` (tested through 3.13) |
| ComfyUI | `>=0.37.0` (V3 node API) |
| Added dependency | `s3tokenizer` only, a pure-Python wheel |

</details>

Optional extras, installed as `pip install ComfyUI-Chatterbox[extra]`:
`watermarking` (resemble-perth), `multilingual` (pykakasi, spacy-pkuseg) and
`turbo` (pyloudnorm). Nothing else is required; the rest of the stack comes
from ComfyUI itself.

<p id="usage" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ❯ Usage

1. Add a node from `WMNodes/sound/*`.
2. Pick the model; connect an `AUDIO` input to clone a voice.
3. Queue the prompt.

| Node | Menu name | Does |
| :--- | :--- | :--- |
| `ChatterboxTTS` | Chatterbox TTS 📢 | Speech from text, optional voice cloning. One node for every TTS pack. |
| `ChatterboxMTLTTS` | Chatterbox Multilingual TTS 🌍 | Speech in 23 languages, plus the Single Language Packs. |
| `ChatterboxVC` | Chatterbox Voice Conversion 🗣️ | Re-voices a source recording into a target voice. |
| `ChatterboxDialog` | Chatterbox Multi-Speaker Dialog 🎭 | Multi-speaker conversation, up to four cloned voices. |
| `ChatterboxSRT` | Chatterbox SRT / Subtitles 📝 | Speaks a timed SRT script and returns adjusted subtitles. |

Every widget is documented in its node tooltip. Load a graph from
[`example_workflows/`](./example_workflows) to see the nodes wired up.

When a node fails at execution — a missing reference clip, a pack with no
conversion model — the message shows up as a toast in the web UI, naming the
node and the error. Add `Chatterbox.ErrorToasts` to the `Comfy.Extension.Disabled`
list in ComfyUI's settings to turn the toasts off.

<p align="right"><a href="#readme-top">⟔ ▲ ⟓ back to top</a></p>

<p id="good-to-know" align="center">◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆</p>

## ⌗ Good to know

| | |
| :--- | :--- |
| **No version pins** | ComfyUI Manager resolves the stack; a pin could force a downgrade of the host's torch build. |
| **Offline** | Packs are fetched by a first-run download and a local pack is never re-downloaded afterwards. |
| **VRAM** | Weights load under ComfyUI's manager and offload between runs; ~2.5 GB per pack on disk as safetensors. |
| **Layout** | Node code lives in `nodes/`, the vendored Chatterbox core in `src/chatterbox/`. |
| **Emotion tags & chunking** | `chatterbox-turbo` and `chatterbox-nano` speak the 19 paralinguistic tags natively: write `[laugh]`, `[happy]`, `[clear throat]` and the rest into the text box and the model says them -- the exact 19 spellings are in the [tag list](#models). Classic, multilingual, Single Language Pack and Flash packs have no such vocabulary, so the same words are read out as plain text and the run says so in the log. `<giggle>`-style v2 tokens are re-spelled for Turbo and left alone elsewhere. Long scripts are cut into chunks, spoken one chunk at a time and joined again -- `auto` picks the join by script length (clean under 500 characters, crossfaded above it, padded with silence above 1000), and the cut never lands inside a tag. |
| **Upgrading from 1.x** | 2.0.0 moved packs from `models/chatterbox_tts/` to `models/tts/chatterbox/`. No compatibility shim is provided, so delete the old directory by hand. Widget order did not change and node ids did not change, so saved workflows still load. |
| **Changelog** | [CHANGELOG.md](./CHANGELOG.md) |

<p align="right"><a href="#readme-top">⟔ ▲ ⟓ back to top</a></p>

<p id="acknowledgments" align="center">◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆</p>

## ❖ Acknowledgments

*   [Resemble AI](https://www.resemble.ai/chatterbox/) for the Chatterbox models
*   [comfyanonymous](https://github.com/comfyanonymous/ComfyUI) for ComfyUI

<p align="right"><a href="#readme-top">⟔ ▲ ⟓ back to top</a></p>


<!-- BADGE DEFINITIONS -->
[stars-shield]: https://img.shields.io/github/stars/wildminder/awesome-ltx2.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/awesome-ltx2/stargazers
[telegram-shield]: https://img.shields.io/badge/TokenDiff-26A5E4?style=for-the-badge&logo=telegram&logoColor=white
[telegram-url]: https://t.me/TokenDiff
[x-shield]: https://img.shields.io/badge/wildmindai-000000?style=for-the-badge&logo=x&logoColor=white
[x-url]: https://x.com/wildmindai

[contributors-shield]: https://img.shields.io/github/contributors/wildminder/ComfyUI-Chatterbox.svg?style=for-the-badge
[contributors-url]: https://github.com/wildminder/ComfyUI-Chatterbox/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/wildminder/ComfyUI-Chatterbox.svg?style=for-the-badge
[forks-url]: https://github.com/wildminder/ComfyUI-Chatterbox/network/members
[stars-shield]: https://img.shields.io/github/stars/wildminder/ComfyUI-Chatterbox.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/ComfyUI-Chatterbox/stargazers
[issues-shield]: https://img.shields.io/github/issues/wildminder/ComfyUI-Chatterbox.svg?style=for-the-badge
[issues-url]: https://github.com/wildminder/ComfyUI-Chatterbox/issues
