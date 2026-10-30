# Changelog

All notable changes to ComfyUI-Chatterbox are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.6] - 2026-10-31

### Fixed

- The classic voice pack is offered as `chatterbox` in the model dropdown; the internal id `resembleai_default_voice` is still accepted, so a workflow saved with it keeps loading.

## [2.0.5] - 2026-10-23

The release in which the models speak like the script asks. Everything below
landed since 2.0.0. **The node surface did not change:** node ids, widget order
and output socket names are the same.

### Added

* **Emotion tags work.** The Turbo/Nano vocabulary is nineteen tags, not nine.
  The ten emotion tags -- `[dramatic]`, `[sarcastic]`, `[happy]`,
  `[surprised]`, `[angry]`, `[fear]`, `[crying]`, `[whispering]`, `[narration]`,
  `[advertisement]` -- were parsed as a voice name and never reached the
  tokenizer, so they did nothing. All nineteen now pass through verbatim and
  land on the model's own token ids (50257-50275).
* **Long-script chunking.** Four widgets on the TTS node:
  `enable_chunking`, `max_chars_per_chunk`, `chunk_combination_method`,
  `silence_between_chunks_ms`. Chunks join by `auto`, `concatenate`,
  `silence_padding` or `crossfade`, and a cut never lands inside a tag.
* **Toasts for node errors.** The web UI shows the node's own error message
  instead of only the console.

### Fixed

* **The first run lists the packs.** With no `models/tts/chatterbox` folder the
  dropdown still offers every downloadable pack, and picking one downloads it.
* **The download asks for the right files.** The multilingual path stopped
  demanding the classic pack's checkpoint.
* **Pack names follow upstream.** `chatterbox-nano`, `chatterbox-turbo`,
  `chatterbox-flash`. `chatterbox-turbo-nano` still resolves everywhere, so
  saved workflows keep working.
* **The notice names the tag's group.** A sound tag and an emotion tag are
  different things, and the log now says which one was typed.

## [2.0.0] - 2026-10-04

A structural release. The internals were rebuilt against the ComfyUI V3 node API
and a much newer upstream Chatterbox; **the node surface did not change.**

### Node ids and widget order did not change

This is the important half of the release, so it comes first:

* **`ChatterboxTTS` and `ChatterboxVC` keep their ids.** Every saved workflow
  that loads today loads in 2.0.0 without being touched.
* **Widget order did not change.** ComfyUI binds a saved graph's widget values
  *by index*, not by name, so a reorder would silently re-bind every existing
  user graph — `use_watermark` onto `seed`, for instance — and nothing would
  complain until someone synthesised weeks later. The required-input order was
  captured before the rewrite in `tests/frozen_widget_snapshot.json` and is now
  asserted by **exact tuple equality**, not a prefix check.
* **Output socket names did not change**: `audio` for TTS, `converted_audio` for
  VC.
* The three new `device` / `dtype` / `attention_mode` widgets are **appended
  after** the frozen block, not inserted into it.

**No user workflow breaks.** If a graph loaded under 1.2.1, it loads under
2.0.0.

### Breaking

* **The vendored `chatterbox` package moves from upstream v0.1.2 to v0.1.7.**
  Every local edit on top of upstream is recorded in `src/PATCHES.md` with a
  rationale, so the patch layer can be re-applied after a re-sync.
* **The V1 registration surface is gone.** `NODE_CLASS_MAPPINGS` and
  `NODE_DISPLAY_NAME_MAPPINGS` no longer exist, and the repo-root `nodes.py` was
  deleted. The package is now a V3 extension: ComfyUI awaits
  `comfy_entrypoint()` and calls `get_node_list()`. Anything that imported
  `NODE_CLASS_MAPPINGS` from this package — a third-party workflow pack, a
  script — will not find it.
* **`diffusers` is no longer a dependency.** The vendored tree no longer imports
  it; a small compatibility shim replaces the three symbols that were used.
  (`diffusers` 0.27 cannot import under `huggingface_hub` 1.x anyway, which is
  what made the old dependency a liability rather than an asset.)
* **The models directory moved** from `ComfyUI/models/chatterbox_tts/` to the
  ComfyUI-standard `ComfyUI/models/tts/chatterbox/`. No shim is provided — see
  [Migration](#migration) below.

### Migration

Delete the old model folder by hand:

```text
ComfyUI/models/chatterbox_tts/
```

It is no longer read. If you want to keep the weights you already downloaded,
move or copy `models/chatterbox_tts/<pack>/` into
`models/tts/chatterbox/<pack>/` before restarting; the node will find it there
and will not download anything.

### Added

* Resumable first-run model download with per-file progress in the ComfyUI
  frontend. Partial files are written as `*.incomplete`, so an interrupted
  transfer can never be mistaken for a finished pack — and a completed
  sibling file survives a later cleanup.
* Model packs are discovered under **every** `tts` root ComfyUI resolves, so a
  pack installed under a secondary root from `extra_model_paths.yaml` is
  visible instead of invisible-but-present.
* `device`, `dtype` and `attention_mode` widgets on both nodes, appended after
  the frozen block. The `attention_mode` list is built at schema time from what
  the machine actually supports, and falls back to `sdpa`.
* A local web API for pack discovery and download progress.
* `tests/` — roughly 1.3k tests over the node schemas, the frozen compatibility
  surface, the download stack, the model patcher and the packaging metadata.
  A CI workflow runs them on every push.

### Changed

* **The model stack is loaded through ComfyUI's model-patching system**, so it
  participates in the same VRAM management as the rest of your workflow. The
  pack's size is summed from the real `*.safetensors` bytes on disk and refined
  after load from the parameters the module actually holds — the previous
  hard-coded 1.5 GB budget is gone.
* **The voice-conversion model is actually released.** The V1 implementation
  cleared its reference on the patcher rather than on the wrapper, so a VC model
  was never released and each run leaked one copy. It is now released on
  completion.
* **Model patches are cached by resolved model identity** — pack name, attention
  mode, dtype, device — so a second browser tab asking for the same model reuses
  the loaded weights.
* `requirements.txt` and `pyproject.toml` declare only what ComfyUI does not
  already bundle, and pin nothing. See the README for why a pin here is an
  instruction to downgrade the user's own working install.
* The example workflow directory is now `example_workflows/`, matching the
  ComfyUI packaging convention. The JSON is byte-identical to the file it
  replaces.

### Fixed

* An interrupt was reported to the frontend as one second of silence produced
  by an ordinary error, because the ComfyUI interrupt exception derives from
  `BaseException` and could not be caught by the handler meant for it. Interrupts
  are now distinguished from failures, and real failures raise with the pack
  name instead of emitting silence.
* `use_watermark` now takes a defensive import of `resemble-perth` inside the
  vendored code, so a machine without the watermarking extra produces plain
  audio rather than failing to load.

## [1.2.1] - 2025-07-21

* Fixed the `tqdm` packaging bug: it was declared in `pyproject.toml` but absent
  from `requirements.txt`, so ComfyUI Manager never installed it.

## [1.2.0] - 2025-07-21

* Refactor for alignment with the ComfyUI codebase, unlocking all generation
  parameters.
* Removed the 40-second generation cap.
* Automatic model download from Hugging Face on first use.
* Model patching integration for load-when-needed / offload-afterwards.

## [1.1.0]

* Chat nodes for single-line and multi-line TTS.

## [1.0.0]

* First public release: `ChatterboxTTS` and `ChatterboxVC`.

[2.0.0]: https://github.com/wildminder/ComfyUI-Chatterbox/releases/tag/v2.0.0
[1.2.1]: https://github.com/wildminder/ComfyUI-Chatterbox/releases/tag/v1.2.1
[1.2.0]: https://github.com/wildminder/ComfyUI-Chatterbox/releases/tag/v1.2.0