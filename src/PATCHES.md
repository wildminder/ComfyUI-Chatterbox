# Vendored `chatterbox` — patch manifest

The `src/chatterbox/` tree is a vendored copy of upstream `resemble-ai/chatterbox`
(MIT). It is re-synced by **taking upstream first, then re-applying this layer** —
never the reverse. This file is the contract: every entry below is a local edit to
vendored source, and `tests/test_vendor_patch_manifest.py` fails if an entry's
symbol is missing from the tree, or if the file leaks a path outside the repository.

Provenance is recorded by rationale, not by pointer: nothing here may name the
scratch directory the upstream checkout was read from.

## Categories

| Category | Meaning |
| --- | --- |
| `ComfyUI-API patch` | The ComfyUI node surface — an argument, progress report or option the node passes that upstream does not have. |
| `compat patch` | Upstream regressed something, or assumes an environment ours does not provide. |
| `feature addition` | New capability with no upstream counterpart. |

## Patch manifest

Paths are relative to `src/`. Symbols are module-level names, `Class.attr` pairs,
or function names, exactly as they appear in the source.

Rows are grouped per file, and a later batch's row is appended to the run it
belongs to rather than at the end of the table, so the IDs do not read in
ascending order.

| ID | File | Symbol | Category | Rationale |
| --- | --- | --- | --- | --- |
| P-001 | `chatterbox/__init__.py` | `__version__` | compat patch | Upstream calls `importlib.metadata.version("chatterbox-tts")` at import time. The package is vendored, never pip-installed, so the lookup raises `PackageNotFoundError` and `import chatterbox` dies before any node registers. The lookup is kept but its failure is non-fatal. |
| P-002 | `chatterbox/tts.py` | `_pad_wav_to_40ms_multiple` | compat patch | The mel spectrogram hops every 20 ms and the S3 tokenizer emits one token every 40 ms. A reference clip that is not an exact multiple of 40 ms produces `mel_len != 2 * token_len`, which `embed_ref` only repairs by dropping tokens. Pad before resampling instead. |
| P-003 | `chatterbox/tts.py` | `ChatterboxTTS.from_local` | compat patch | `device in ["cpu", "mps"]` is False for every `torch.device`, so a CPU load selected `map_location=None` and tried to restore CUDA-saved tensors. Compare `str(device)`. |
| P-004 | `chatterbox/tts.py` | `ChatterboxTTS.prepare_conditionals` | compat patch | Applies P-002: the loaded clip is round-tripped through `torch` so it can be padded onto the token grid before being resampled to 16 kHz. |
| P-005 | `chatterbox/tts.py` | `ChatterboxTTS.generate` | ComfyUI-API patch | Accepts `pbar`, `max_new_tokens` and `flow_cfg_scale`, and forwards them to `T3.inference` and `S3Gen.inference`. The node exposes all three as widgets; upstream hardcodes `max_new_tokens=1000` and takes no progress report. |
| P-023 | `chatterbox/tts.py` | `_apply_watermark` | feature addition | Upstream hard-imports `perth` and constructs `perth.PerthImplicitWatermarker()` in `__init__`, so `import chatterbox.tts` fails outright without an optional watermarking library. The import is now guarded, construction yields `None` when the library is absent, and `generate()` takes `watermark: bool = True` — the opt-out upstream has no equivalent of. Default `True` keeps upstream behaviour exactly; `False` skips `apply_watermark()`; `True` without the library warns once on the named `chatterbox.watermark` logger and returns the tensor unchanged. |
| P-027 | `chatterbox/tts.py` | `_require_checkpoint_files` | compat patch | `ChatterboxTTS.from_local` opens with a completeness check over the four files a classic pack cannot be built without, raising `FileNotFoundError` that names the missing files and then the full set. Upstream reaches its first `load_file` on a path that does not exist and dies with `FileNotFoundError: .../ve.safetensors` — one file named, the other three left to guesswork, which is exactly the failure a community finetune or a half-cancelled download produces. The helper is duplicated per entry point rather than shared through a new vendored module, so each file stays independently vendorable; the vendored file set sits at 52, its single later addition being `chatterbox/models/dsp.py` (P-044), which no `from_local` needs. P-028…P-030 are the same patch against the other three `from_local` implementations, with the file set each of those actually needs. This is also what lets an arbitrary local directory — Viterbox, Egyptian Arabic, anything a user drops under `models/tts/chatterbox` — be an ordinary pack rather than a special case: it clears exactly the same bar as an official one, and is refused in exactly the same words when it does not. |
| P-039 | `chatterbox/tts.py` | `ChatterboxTTS.prepare_conditionals` | compat patch | `librosa.load(wav_fpath, sr=S3GEN_SR)` becomes `load_audio_at(wav_fpath, S3GEN_SR)` and `librosa.resample(...)` becomes `resample(...)`, so the reference clip no longer has to arrive as a `.wav` path: the new loader takes a path, an in-memory waveform or a tensor, reads a path with `soundfile` in float32 under the same channel-mean downmix librosa used, and passes what is already in memory straight through. librosa is a heavyweight import (numba/llvmlite at module load) that this tree reached for a handful of functions, and it leaves the vendored tree entirely with this patch. Resampling is the one deliberate numerical difference — torchaudio's sinc kernel instead of soxr, see P-044 — and it touches only conditioning clips supplied at call time. `generate`'s `if audio_prompt_path:` becomes `is not None` in the same pass: a multi-element ndarray raises `ValueError` under truth testing, so a reference clip handed in as audio would crash on the guard before reaching the model. P-004's token-grid padding still applies to the loaded clip. |
| P-006 | `chatterbox/vc.py` | `_pad_wav_to_40ms_multiple` | compat patch | Same token-grid padding as P-002, for the voice-conversion reference clip. |
| P-007 | `chatterbox/vc.py` | `ChatterboxVC.from_local` | compat patch | Same `str(device)` comparison as P-003. |
| P-008 | `chatterbox/vc.py` | `ChatterboxVC.set_target_voice` | compat patch | Applies P-006: pads the reference clip onto the token grid before truncating to `DEC_COND_LEN`. |
| P-009 | `chatterbox/vc.py` | `ChatterboxVC.generate` | ComfyUI-API patch | Accepts `n_timesteps`, `pbar`, `temperature` and `flow_cfg_scale` and forwards them to `S3Gen.inference`. Upstream exposes only `audio` and `target_voice_path`; the node drives all four as widgets. |
| P-024 | `chatterbox/vc.py` | `_apply_watermark` | feature addition | Same optional-`perth` patch as P-023 for the voice-conversion entry point, including the `watermark` kwarg on `generate()`. |
| P-028 | `chatterbox/vc.py` | `_require_checkpoint_files` | compat patch | Same patch as P-027 for `ChatterboxVC.from_local`, which needs only `s3gen.safetensors` — a conversion-only pack loads no T3 weights, so the message names a one-file set rather than a four-file one. |
| P-042 | `chatterbox/vc.py` | `ChatterboxVC.set_target_voice` | compat patch | Same librosa drop as P-039, for the voice-conversion target clip: `load_audio_at(wav_fpath, S3GEN_SR)` replaces `librosa.load` and the rate it returned was never read, so the target voice may now also be handed in as a waveform. P-008's token-grid padding still applies to the clip it loads. |
| P-043 | `chatterbox/vc.py` | `ChatterboxVC.generate` | compat patch | The conversion source is loaded with `load_audio_at(audio, S3_SR)` at 16 kHz — the only dual-rate call site in the tree, the target voice having already come in at `S3GEN_SR` through P-042 — and `if target_voice_path:` becomes `is not None`, because a multi-element ndarray raises `ValueError` under truth testing and a node-supplied target clip is exactly that. |
| P-010 | `chatterbox/models/t3/modules/perceiver.py` | `AttentionQKV.flash_attention` | compat patch | Upstream calls `torch.backends.cuda.sdp_kernel`, deprecated in torch 2.2 and removed later. Replaced with `torch.nn.attention.sdpa_kernel([FLASH, EFFICIENT, MATH])`, with the old call kept as the torch < 2.2 fallback. |
| P-011 | `chatterbox/models/s3gen/matcha/transformer.py` | `FeedForward` | compat patch | `LoRACompatibleLinear` is replaced by `nn.Linear` at both construction sites, and `diffusers.models.lora` is no longer imported. Modern diffusers no longer exports the class, and the LoRA scaling it added is unused by this inference-only model. |
| P-021 | `chatterbox/models/s3gen/matcha/transformer.py` | `BasicTransformerBlock` | compat patch | The four activations, the two adaptive norms, `Attention` and `maybe_allow_in_graph` are imported from `._diffusers_compat` instead of `diffusers.models.attention`, `diffusers.models.attention_processor` and `diffusers.utils.torch_utils`. Completes P-011: the file no longer imports `diffusers` at all. |
| P-020 | `chatterbox/models/s3gen/matcha/_diffusers_compat.py` | `Attention` | compat patch | New local module holding the eight `diffusers` symbols the Matcha decoder imported (`GEGLU`, `GELU`, `ApproximateGELU`, `AdaLayerNorm`, `AdaLayerNormZero`, `Attention`, `get_activation`, `maybe_allow_in_graph`), copied from `diffusers` (Apache-2.0, notice retained) and reduced to the parameters `models/s3gen/matcha/` actually reaches. `diffusers` is a heavyweight training-only library this inference-only decoder pulled in, and the build otherwise installed cannot be imported against `huggingface_hub` 1.x. The module docstring lists every dropped path, the upstream defects kept verbatim because the code that would hit them is unreachable, and the one deliberate improvement (`scale_qk` is honoured via SDPA's `scale=`). |
| P-022 | `chatterbox/models/s3gen/matcha/decoder.py` | `Decoder` | compat patch | `get_activation` is imported from `._diffusers_compat` instead of `diffusers.models.activations`. |
| P-038 | `chatterbox/models/s3gen/matcha/decoder.py` | `Decoder.get_block` | compat patch | `ConformerWrapper(ConformerBlock)` and the `get_block("conformer")` branch were dead code: the model constructs diffusers-style `BasicTransformerBlock` blocks, every shipped s3gen checkpoint contains zero lucidrains-conformer keys (`to_kv`, `rel_pos_emb`, `ff1`, `conv.net`, `post_norm`), and the module was imported only for its helper classes. Removed, dropping the pip `conformer` dependency; `get_block` still raises `ValueError` for an unknown block type. |
| P-012 | `chatterbox/models/s3gen/flow_matching.py` | `ConditionalCFM.lock` | feature addition | A TensorRT estimator is a stateful single-stream object; two workflows sharing one model instance can enter it concurrently. Guarded by a `threading.Lock`. |
| P-013 | `chatterbox/models/s3gen/flow_matching.py` | `CausalConditionalCFM.forward` | ComfyUI-API patch | Accepts `pbar` and `flow_cfg_scale`. `pbar` is forwarded to the Euler solver; `flow_cfg_scale` overrides the baked-in `inference_cfg_rate` for the duration of the call. Also applied to `ConditionalCFM.forward` and `basic_euler`. |
| P-014 | `chatterbox/models/s3gen/flow_matching.py` | `ConditionalCFM.solve_euler` | ComfyUI-API patch | Accepts `pbar` and advances it once per Euler step, so the node reports progress across the mel-generation loop instead of appearing hung. |
| P-015 | `chatterbox/models/s3gen/flow_matching.py` | `ConditionalCFM.forward_estimator` | feature addition | Lets a TensorRT `IExecutionContext`-style object stand in for the torch estimator module. The TRT path rebinds input shapes, calls `execute_v2` with raw device pointers under :data:`P-012`'s lock, and returns the `x` buffer the engine wrote into. |
| P-016 | `chatterbox/models/s3gen/flow.py` | `CausalMaskedDiffWithXvec.inference` | ComfyUI-API patch | Accepts `pbar`, `temperature` and `flow_cfg_scale` and forwards them to the decoder. Upstream's signature is fixed and offers no progress report. |
| P-017 | `chatterbox/models/s3gen/flow.py` | `CausalMaskedDiffWithXvec.fp16` | ComfyUI-API patch | Half-precision switch. The fp32 checkpoint stays resident and only the conditioning tensors (`prompt_feat`, `embedding`) are downcast; the returned mel is cast back to fp32 for the vocoder. |
| P-037 | `chatterbox/models/s3gen/flow.py` | `CausalMaskedDiffWithXvec.decoder_conf` | compat patch | The only omegaconf use in the vendored tree: the default `decoder_conf` wraps its `cfm_params` (scalar-only) in a `DictConfig`, a type only the omegaconf package provides. Replaced with the tree's own `AttrDict` (`configs.py` builds `CFM_PARAMS` the same way, from identical values), which preserves the attribute access every consumer uses (`cfm_params.t_scheduler`, `hasattr(cfm_params, "sigma_min")`). The dependency is dropped from requirements. |
| P-018 | `chatterbox/models/s3gen/s3gen.py` | `S3Token2Wav.inference` | ComfyUI-API patch | `S3Token2Mel.forward`, `S3Token2Wav.flow_inference`, `S3Token2Wav.forward` and `S3Token2Wav.inference` (exported as `S3Gen`) each accept `pbar`, `temperature` and `flow_cfg_scale` and thread them down to `CausalConditionalCFM`. Every upstream kwarg (`n_cfm_timesteps`, `noised_mels`, `skip_vocoder`, `speech_token_lens`, `drop_invalid_tokens`) is preserved. |
| P-031 | `chatterbox/models/s3gen/s3gen.py` | `S3Token2Mel.embed_ref` | compat patch | The 16 kHz clip is no longer cast to `self.dtype` before `speaker_encoder.inference()`: CAMPPlus is float32-only (its `inference()` hard-casts the fbank features to float32) and `Kaldi.fbank`'s `torch.fft.rfft` has no half-dtype kernel, so the cast turned every bf16/fp16 generation into `RuntimeError: Unsupported dtype BFloat16`. The fp32 x-vector is instead cast to the flow's dtype where the ref dict is built, keeping conditioning aligned with the flow/hift weights. Companion to the fp32-resident cast exclusion in `modules/dtype_utils.py` (`CAMPPlus`, `VoiceEncoder`, `S3Tokenizer`, `HiFTGenerator` stay fp32). |
| P-033 | `chatterbox/models/s3gen/s3gen.py` | `S3Token2Wav.inference` | compat patch | The generated mel is handed to the HiFT vocoder in float32 instead of being re-cast to `self.dtype`: the vocoder is float32-only (SineGen's harmonic math promotes to fp32 via `torch.arange` defaults and `torch.istft` consumes the fp32 window, so half weights crash at the first Linear) and is kept fp32-resident by `modules/dtype_utils.py` alongside the other frontends. `hift_inference`'s zero-length cache now follows the vocoder's own weight dtype rather than the flow's. `S3Token2Wav.forward` already passed fp32 — only `inference` re-cast — which is why the break surfaced one layer past P-031's fix. |
| P-034 | `chatterbox/models/s3gen/s3gen.py` | `S3Token2Mel.forward` | compat patch | The ref-dict casting loop no longer dtype-casts integer tensors: `prompt_token` (token ids) and `prompt_token_len` (lengths) are integers that must move between devices but never between dtypes. Under a bf16 flow the old blanket cast rounded the combined token length 635 down to 632 (bf16's 7-bit mantissa), truncating the padding mask and crashing `flow.inference` with a shape mismatch — and quietly corrupted token ids above 256 that bf16 cannot represent exactly. In fp32 the cast was an exact no-op, so upstream never noticed; device movement is preserved for both branches. |
| P-019 | `chatterbox/models/t3/t3.py` | `T3.inference` | ComfyUI-API patch | Accepts `pbar` and advances it once per sampled token. `max_new_tokens` keeps upstream's default of `None` but now falls back to `self.hp.max_speech_tokens` instead of raising, so the node may omit it. |
| P-025 | `chatterbox/mtl_tts.py` | `_apply_watermark` | feature addition | Same optional-`perth` patch as P-023 for the multilingual entry point. Also applies the `str(device)` comparison of P-003 and P-007 in `ChatterboxMultilingualTTS.from_local`, which upstream still compares raw. |
| P-036 | `chatterbox/mtl_tts.py` | `_resolve_s3gen_checkpoint` | compat patch | The Single Language Pack finetunes (`ResembleAI/Chatterbox-Multilingual-zh-cmn` and its five siblings) ship their S3Gen decoder as `s3gen_v3.safetensors` (plus a legacy `.pt` twin), while the shared multilingual repo ships `s3gen.safetensors` — upstream's hard-coded `s3gen.pt` load made every SLP pack fail at the file check. `_resolve_s3gen_checkpoint` picks whichever serialization the pack holds (safetensors first), `_resolve_ve_checkpoint` does the same for the voice encoder, and `_load_checkpoint` dispatches `load_file`/`torch.load` by extension, so a pack may carry either form. `REQUIRED_CHECKPOINT_FILES` shrinks to the grapheme tokenizer; the resolved ve/s3gen/T3 names join the required set at call time. The s3gen state-dict load is non-strict for a related reason: the SLP checkpoints predate the persistent `tokenizer._mel_filters`/`tokenizer.window` buffers (see `S3Token2Wav.ignore_state_dict_missing`), which are recomputed in `__init__` and never learned — a strict load refuses every SLP pack. The classic entry point already loads s3gen non-strict. `from_pretrained`'s `allow_patterns` fetch the safetensors serializations only — the repo's `.pt` twins are the same weights and are never loaded. |
| P-029 | `chatterbox/mtl_tts.py` | `_require_checkpoint_files` | compat patch | Same patch as P-027 for `ChatterboxMultilingualTTS.from_local`. Its set differs: `ve.pt`, `s3gen.pt` and `grapheme_mtl_merged_expanded_v1.json`, plus the T3 checkpoint for the selected version. That last name is not fixed, so `from_local` appends the already-resolved `t3_model` to the tuple rather than the helper hard-coding it. |
| P-040 | `chatterbox/mtl_tts.py` | `ChatterboxMultilingualTTS.prepare_conditionals` | compat patch | Same librosa drop as P-039, for the multilingual entry point: `load_audio_at(wav_fpath, S3GEN_SR)` for the 24 kHz reference clip and `resample` for the 16 kHz copy the S3 tokenizer consumes. The `_sr` return disappears with the pair — the loader already answers at the rate it was asked for, which is every call site's rate. `generate`'s `if audio_prompt_path:` becomes `is not None` for the same reason as P-039. |
| P-026 | `chatterbox/tts_turbo.py` | `_apply_watermark` | feature addition | Same optional-`perth` patch as P-023 for the turbo entry point. Also applies the `str(device)` comparison of P-003 and P-007 in `ChatterboxTurboTTS.from_local`, which upstream still compares raw, and makes `pyloudnorm` an optional import — it is used only by `norm_loudness`, which already degrades gracefully, so the module-level hard import was making the whole entry point unimportable without it. |
| P-030 | `chatterbox/tts_turbo.py` | `_require_checkpoint_files` | compat patch | Same patch as P-027 for `ChatterboxTurboTTS.from_local`, covering Turbo and Nano together — there is no separate Nano module, so there is no separate `from_local`. The variant's T3 checkpoint (`t3_turbo_v1.safetensors` or `t3_nano_v1.safetensors`) is appended by the caller for the same reason as P-029. |
| P-035 | `chatterbox/tts_turbo.py` | `ChatterboxTurboTTS.from_local` | compat patch | The required-file set matches what the repos actually ship: `s3gen_meanflow.safetensors` is required (the loader builds `S3Gen(meanflow=True)` from it unconditionally — its absence crashed at `load_file`, past every check), and the tokenizer is checked per layout instead of demanding `tokenizer.json` outright — the Nano repo ships the HF fast layout (`vocab.json` + `merges.txt`) and no `tokenizer.json` at all, so every real Nano pack failed the check `AutoTokenizer.from_pretrained` would have passed. `modules/loader.py`'s `TURBO_CHECKPOINT_FILES` and `model_info.is_chatterbox_model_dir` carry the same fix on the node side. |
| P-041 | `chatterbox/tts_turbo.py` | `ChatterboxTurboTTS.prepare_conditionals` | compat patch | Same librosa drop as P-039, for the Turbo/Nano entry point. It is the one that read the rate back twice: the 5-second assertion and `norm_loudness` both took `_sr`, the rate the load had *asked* for, so once the loader answers at `S3GEN_SR` they take that constant directly and lose nothing. `generate`'s `if audio_prompt_path:` becomes `is not None` for the same reason as P-039. |
| P-032 | `chatterbox/models/t3/modules/cond_enc.py` | `T3CondEnc.forward` | compat patch | The voice-encoder speaker embedding and the emotion scalar arrive as float32 tensors (the voice encoder is fp32-resident by design), while t3's weights may be cast to a half dtype; feeding fp32 inputs to half `nn.Linear` weights raises a dtype mismatch. Each conditioning input is now aligned to the dtype of the weights that consume it (`spkr_enc`, `perceiver`, `emotion_adv_fc`). |
| P-044 | `chatterbox/models/dsp.py` | `mel_filterbank` | feature addition | New vendored-only module — the one file in the tree with no upstream counterpart, which is why it carries a row of its own: the Slaney mel filterbank, the voice encoder's STFT, the periodic Hann window, pre-emphasis, silence trimming, resampling and a loader, each reproducing librosa 1.0's order of operations operation-for-operation (the ISC notice is retained in the module) so the training-baked featurization comes out bit-identical on the parameters this package uses, while librosa and scipy — and with librosa, its numba/llvmlite import-time cost — leave the vendored tree's import list. The featurization is trained end-to-end against these exact filters, so a replica that drifted by an ulp would move every mel a model has ever seen; the module says so where the arithmetic is finest (float64 triangles assigned into a float32 array, then the Slaney enorm applied in place). Resampling is the single deliberate difference — torchaudio's sinc kernel instead of soxr — and it only touches conditioning clips supplied at call time. Everything else in the tree imports it rather than repeating it: P-039…P-043, P-045…P-049, plus the two `chatterbox_flash` call sites recorded in that tree's provenance note. |
| P-045 | `chatterbox/models/s3gen/utils/mel.py` | `mel_spectrogram` | compat patch | `librosa.filters.mel` is imported as `mel_filterbank` from the tree's own DSP module and called with the same arguments, so the mel basis the s3gen decoder runs on (24 kHz, n_fft 1920, 80 mels, fmax 8000) is the same float32 array it always was. Only the filterbank moved: the `torch.hann_window` beside it never came from librosa. |
| P-046 | `chatterbox/models/s3gen/hifigan.py` | `HiFTGenerator.__init__` | compat patch | `scipy.signal.get_window("hann", n_fft, fftbins=True)` becomes `hann_window(n_fft)` from the tree's own DSP module — the same periodic Hann, the symmetric formula over M+1 points with the last one dropped — so the ISTFT window the vocoder converges to is unchanged and one of the tree's two scipy imports is gone. |
| P-047 | `chatterbox/models/s3tokenizer/s3tokenizer.py` | `S3Tokenizer.__init__` | compat patch | `librosa.filters.mel` becomes `mel_filterbank` from the tree's own DSP module, and `librosa` leaves the import block. The filterbank is registered as the `_mel_filters` buffer, which `ignore_state_dict_missing` already excludes from every load: it is recomputed here and never learned, so a replica that drifted would shift the tokenizer's front end with nothing in a checkpoint to catch it. |
| P-048 | `chatterbox/models/voice_encoder/melspec.py` | `mel_basis` | compat patch | The voice encoder's three librosa/scipy calls become the DSP module's replicas: the cached 16 kHz / n_fft 400 / 40-mel / fmax 8000 filterbank, `scipy.signal.lfilter` for the pre-emphasis filter, and `librosa.stft` for the spectrum — the last still `center=pad` and still reflect-padded, because that is what the pretrained featurization was trained on. The `lru_cache` key, every hyperparameter and the module's own amplitude/db helpers are untouched. |
| P-049 | `chatterbox/models/voice_encoder/voice_encoder.py` | `VoiceEncoder.embeds_from_wavs` | compat patch | `librosa.resample(..., res_type="kaiser_fast")` becomes `resample(...)` and `librosa.effects.trim` becomes `trim_silence`, so the speaker-embedding front end runs on the tree's own DSP module and the last librosa import in the vendored tree is gone. The per-wav resample still fires only when the input rate differs from the encoder's own, and the trim still takes the caller's `top_db`. The kernel swap of P-044 retires the upstream `kaiser_fast` choice this file used to carry — the resampler it qualified no longer exists here — which is why that upstream row is gone from the table below. |

## Upstream changes adopted verbatim

These arrive with the v0.1.7 sources. No local edit, but each was a deliberate
upgrade over what this repository previously carried, and each is asserted by
`tests/test_vendor_signatures.py::TestUpstreamFixes` so a future re-sync cannot
silently revert it.

`VoiceEncoder.embeds_from_wavs` is deliberately absent: P-049 replaced that
file's librosa resampler, and with it the soxr `res_type="kaiser_fast"` choice
this table used to record — a file the patch table already claims as locally
edited cannot also be adopted verbatim, and the resampler the upstream row
qualified no longer exists in the tree.

| ID | File | Symbol | Rationale |
| --- | --- | --- | --- |
| U-001 | `chatterbox/models/s3gen/s3gen.py` | `S3Token2Mel.__init__` | `CAMPPlus(memory_efficient=False)`. Activation checkpointing is a training optimization that makes `accelerate`'s DDP wrapper crash at inference. |
| U-003 | `chatterbox/models/s3gen/const.py` | `S3GEN_SIL` | The sentinel silence token id, needed by the token-dropping path. |
| U-004 | `chatterbox/models/s3gen/utils/mask.py` | `make_pad_mask` | `lengths.long()` before comparison. An int32 length tensor compared against a bool mask silently takes the wrong branch. |
| U-005 | `chatterbox/models/t3/llama_configs.py` | `LLAMA_520M_CONFIG_DICT` | `attn_implementation="sdpa"`. `"eager"` was needed only to expose attention weights to the removed alignment analyzer. |
| U-006 | `chatterbox/models/t3/inference/t3_hf_backend.py` | `T3HuggingfaceBackend` | The dead `alignment_stream_analyzer` argument is gone upstream, along with the `models/t3/inference/alignment_stream_analyzer.py` module. |

## Known upstream leftovers

None. The two Matcha files that still reached into `diffusers` after P-011 no
longer do: P-020 vendors the eight symbols they used as
`chatterbox/models/s3gen/matcha/_diffusers_compat.py`, and P-021/P-022 point the
imports at it. `tests/test_diffusers_compat.py` asserts that no vendored file
imports `diffusers`, and that the compat module's attention is numerically equal
to a hand-written `scaled_dot_product_attention` reference.

## Repository-side changes

These are this repository's own files, not vendored ones, so they carry no patch
id above; recorded here because they are the other half of the same change.

- `modules/vendor_loader.py` (removed) — the `sys.path.append` of this
  repository's `src/` is gone. The vendored tree is imported
  package-relatively (`from ..src.chatterbox…`), so it no longer occupies the
  top-level `chatterbox` name: a pip-installed or foreign `chatterbox` can
  neither shadow it nor be shadowed by it.
- `modules/__init__.py` (new, empty) — makes `modules` a real package so the
  loader can be imported without re-exports.
- `__init__.py` — dropped the `sys.path` manipulation that published `src/` as
  a top-level import root, and dropped the `sys.modules['perth'] = perth_mock`
  stub injection, which the optional-`perth` patches P-023…P-026 make
  unnecessary. Importing the node package no longer mutates global import state.
- `modules/chatterbox_handler.py` — `chatterbox.tts` / `chatterbox.vc` are
  imported inside `load_chatterbox_models` instead of at module scope, so ComfyUI's
  node load no longer pulls librosa, transformers and the model stack in.
- `chatterbox_nodes.py` — dropped the `ensure_vendor_on_path()` call from the
  extension's `on_load`: the vendored tree is reached package-relatively and
  needs no import-path preparation, so the node's extension entry point no
  longer mutates global import state either.

## Vendored `chatterbox_flash` changes

`src/chatterbox_flash/` is a second vendored tree, and its local edits are
recorded in `src/chatterbox_flash/VENDORED.md` rather than in the table above:
every row of that table must name a file under `chatterbox/`, and this
manifest's coverage walk only looks at `src/chatterbox/`, so a
`chatterbox_flash/` row would be invisible to the very tests that check the
manifest. The entry below is the same change, on this side of the record.
Paths are relative to `src/`, as in the table above.

- `chatterbox_flash/tts.py`, `chatterbox_flash/model.py`,
  `chatterbox_flash/eval/generate.py`, `chatterbox_flash/eval/wer_seedtts.py`
  — the eleven absolute `from chatterbox.models…` imports the flash tree
  shipped with are now package-relative: `..chatterbox.models…` from the two
  files at the package root, `...chatterbox.models…` from the two under
  `eval/`. The flash tree is not a top-level `chatterbox_flash` package any
  more — it is private to this node and is reached under this node's own
  package name — so an absolute import could only ever reach whatever
  `chatterbox` happened to be installed, which is exactly what the tree
  exists to prevent. The rewrite changes which tree the imports reach and
  nothing else: the same names, from the same modules, in the same order.