"""Seed-TTS-style WER scorer for Chatterbox-Flash outputs.

Calls Whisper-large-v3 directly with greedy ``model.generate()`` one clip at
a time (mirrors ``seed-tts-eval/run_wer.py``). The upstream
``omnivoice.eval.wer.seedtts`` uses ``transformers.pipeline`` with
``batch_size>1`` which can produce ASR repetition hallucinations on short
clips and inflate WER by 10–50pp on otherwise-correct audio. This module
keeps the OmniVoice JSONL I/O and the exact summary-line format that
``run_eval.sh`` greps, but swaps the ASR call to the robust path.

Outputs (``--decode-path``):

  Name<TAB>WER<TAB>Truth<TAB>Hypothesis<TAB>Insertions<TAB>Deletions<TAB>Substitutions

then per-utt rows and three trailing summary lines::

  Seed-TTS WER (Avg of WERs): X.XX%
  WER (Weighted):              X.XX%
  Errors: <ins> ins, <del> del, <sub> sub / <words> words

Run as a module::

    python -m chatterbox_flash.eval.wer_seedtts \\
        --wav-path  /path/to/generated_wavs \\
        --test-list /path/to/test.jsonl \\
        --decode-path /path/to/wer.log \\
        --model-dir /path/to/TTS_eval_models
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import string
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from ...chatterbox.models.dsp import resample as dsp_resample
from tqdm import tqdm

try:
    from jiwer import compute_measures
except ImportError as e:  # noqa: F401
    print("ERROR: jiwer not installed (pip install jiwer)", file=sys.stderr)
    raise


# ── per-worker globals (set in process_init) ────────────────────────────
_worker_processor = None
_worker_model = None
_worker_device = None
_worker_dtype = None


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Direct Whisper WER (seed-tts-eval style) over an OmniVoice "
            "JSONL test list."
        ),
    )
    p.add_argument("--wav-path", type=str, required=True,
                   help="Directory containing generated wavs ({id}.{ext}).")
    p.add_argument("--extension", type=str, default="wav")
    p.add_argument("--test-list", type=str, required=True,
                   help="JSONL with fields: id, text, …")
    p.add_argument("--decode-path", type=str, required=True,
                   help="Output WER log path.")
    p.add_argument("--model-dir", type=str, required=True,
                   help="Local path of TTS_eval_models. Whisper at "
                   "<model-dir>/wer/whisper-large-v3/.")
    p.add_argument("--lang", type=str, default="en", choices=["en"])
    p.add_argument("--nj-per-gpu", type=int, default=1)
    p.add_argument("--text-norm", type=str, default="omnivoice",
                   choices=["omnivoice", "whisper"],
                   help="omnivoice = lowercase + punct removal; "
                   "whisper = whisper_normalizer.english.EnglishTextNormalizer.")
    p.add_argument("--dtype", type=str, default="fp16",
                   choices=["fp16", "fp32"])
    return p


# ── normalization ──────────────────────────────────────────────────────


def _post_process_omnivoice(text: str) -> str:
    """Mirror ``omnivoice.eval.wer.seedtts.post_process`` for English."""
    try:
        from zhon.hanzi import punctuation as zhon_punct
    except ImportError:
        zhon_punct = ""
    punctuation_all = zhon_punct + string.punctuation
    for ch in punctuation_all:
        if ch == "'":
            continue
        text = text.replace(ch, "")
    text = text.replace("  ", " ")
    return text.lower()


def _post_process_whisper_norm(text: str) -> str:
    """Mirror ``seed-tts-eval/run_wer.py`` for English."""
    from whisper_normalizer.english import EnglishTextNormalizer
    norm = EnglishTextNormalizer()
    text = norm(text).lower().replace("-", "")
    return " ".join(text.split())


# ── ASR ────────────────────────────────────────────────────────────────


def _resample(wav: np.ndarray, src_sr: int, target_sr: int = 16000) -> np.ndarray:
    if src_sr == target_sr:
        return wav
    return dsp_resample(wav, src_sr, target_sr)


def _transcribe(wav_path: str) -> str:
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = _resample(wav, sr, 16000)

    proc, mdl, dev, dtype = (
        _worker_processor, _worker_model, _worker_device, _worker_dtype,
    )
    feats = proc(wav, sampling_rate=16000, return_tensors="pt").input_features
    feats = feats.to(device=dev, dtype=dtype)
    forced_decoder_ids = proc.get_decoder_prompt_ids(
        language="english", task="transcribe",
    )
    with torch.inference_mode():
        out_ids = mdl.generate(feats, forced_decoder_ids=forced_decoder_ids)
    return proc.batch_decode(out_ids, skip_special_tokens=True)[0]


def _process_one(hypo: str, truth: str, normalize) -> dict:
    raw_truth, raw_hypo = truth, hypo
    truth = normalize(truth)
    hypo = normalize(hypo)
    m = compute_measures(truth, hypo)
    ref_words = max(1, len(truth.split()))
    return {
        "raw_truth": raw_truth,
        "raw_hypo": raw_hypo,
        "wer": float(m["wer"]),
        "ins": float(m["insertions"]),
        "del": float(m["deletions"]),
        "sub": float(m["substitutions"]),
        "word_num": ref_words,
    }


def _process_init(rank_queue, model_dir: str, dtype_str: str) -> None:
    global _worker_processor, _worker_model, _worker_device, _worker_dtype
    torch.set_num_threads(2)
    rank = rank_queue.get(timeout=10)
    assert torch.cuda.is_available(), "CUDA required."
    _worker_device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] "
               "[Worker %(process)d] %(message)s",
        level=logging.INFO, force=True,
    )
    whisper_dir = os.path.join(model_dir, "wer/whisper-large-v3/")
    if not os.path.isdir(whisper_dir):
        raise FileNotFoundError(f"Whisper not at {whisper_dir!r}")

    from transformers import (
        WhisperForConditionalGeneration, WhisperProcessor,
    )
    _worker_processor = WhisperProcessor.from_pretrained(whisper_dir)
    _worker_dtype = torch.float16 if dtype_str == "fp16" else torch.float32
    _worker_model = WhisperForConditionalGeneration.from_pretrained(
        whisper_dir, torch_dtype=_worker_dtype,
    ).to(_worker_device).eval()
    logging.info(
        "Whisper-large-v3 loaded on %s (dtype=%s)",
        _worker_device, dtype_str,
    )


def _run_chunk(items, text_norm: str) -> list[dict]:
    normalize = (
        _post_process_whisper_norm
        if text_norm == "whisper"
        else _post_process_omnivoice
    )
    results = []
    for it in items:
        wav_path = it["wav_path"]
        try:
            hypo = _transcribe(wav_path).strip()
        except Exception:
            logging.error(
                "transcribe failed for %s:\n%s",
                wav_path, traceback.format_exc(),
            )
            continue
        m = _process_one(hypo, it["truth_text"], normalize)
        m["wav_path"] = wav_path
        results.append(m)
    return results


# ── main ───────────────────────────────────────────────────────────────


def _read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("id") and isinstance(obj.get("text"), str):
                out.append({"id": str(obj["id"]), "text": obj["text"]})
    return out


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO, force=True,
    )

    samples = _read_jsonl(args.test_list)
    data_list = []
    for s in samples:
        wav_path = str(Path(args.wav_path) / f"{s['id']}.{args.extension}")
        if not os.path.isfile(wav_path):
            logging.warning("missing wav: %s", wav_path)
            continue
        data_list.append({"wav_path": wav_path, "truth_text": s["text"]})
    total = len(data_list)
    logging.info(
        "Direct-Whisper WER (seed-tts-eval style): %d files; "
        "text_norm=%s; dtype=%s",
        total, args.text_norm, args.dtype,
    )

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "CUDA required."
    total_workers = num_gpus * args.nj_per_gpu

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    rank_queue = manager.Queue()
    for _ in range(args.nj_per_gpu):
        for r in range(num_gpus):
            rank_queue.put(r)

    chunk_size = 8
    chunks = [data_list[i : i + chunk_size] for i in range(0, total, chunk_size)]
    logging.info(
        "Split into %d chunk(s) (~%d/chunk); spawning %d worker(s) across %d GPU(s).",
        len(chunks), chunk_size, total_workers, num_gpus,
    )

    results = []
    with ProcessPoolExecutor(
        max_workers=total_workers,
        initializer=_process_init,
        initargs=(rank_queue, args.model_dir, args.dtype),
    ) as ex:
        futures = [ex.submit(_run_chunk, c, args.text_norm) for c in chunks]
        with tqdm(total=total, desc="WER (direct Whisper)", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                try:
                    chunk_metrics = fut.result()
                    results.extend(chunk_metrics)
                    pbar.update(len(chunk_metrics))
                except Exception:
                    logging.error("chunk failed:\n%s", traceback.format_exc())

    wers = [r["wer"] for r in results]
    inses = [r["ins"] for r in results]
    deles = [r["del"] for r in results]
    subses = [r["sub"] for r in results]
    word_nums = sum(r["word_num"] for r in results)

    wer_avg = round(np.mean(wers) * 100, 2) if wers else float("nan")
    wer_weighted = (
        round((sum(subses) + sum(deles) + sum(inses)) / word_nums * 100, 2)
        if word_nums > 0 else float("nan")
    )

    os.makedirs(os.path.dirname(args.decode_path) or ".", exist_ok=True)
    with open(args.decode_path, "w", encoding="utf-8") as fout:
        fout.write(
            "Name\tWER\tTruth\tHypothesis\tInsertions\tDeletions\tSubstitutions\n",
        )
        for r in results:
            fout.write(
                f"{r['wav_path']}\t{r['wer']}\t{r['raw_truth']}\t{r['raw_hypo']}\t"
                f"{r['ins']}\t{r['del']}\t{r['sub']}\n",
            )
        seedtts_line = f"Seed-TTS WER (Avg of WERs): {wer_avg}%"
        weighted_line = f"WER (Weighted): {wer_weighted}%"
        errors_line = (
            f"Errors: {sum(inses)} ins, {sum(deles)} del, {sum(subses)} sub "
            f"/ {word_nums} words"
        )
        fout.write(seedtts_line + "\n")
        fout.write(weighted_line + "\n")
        fout.write(errors_line + "\n")

    print("-" * 50)
    logging.info("Processed %d/%d files.", len(results), total)
    logging.info(seedtts_line)
    logging.info(weighted_line)
    logging.info(errors_line)
    print("-" * 50)


if __name__ == "__main__":
    main()
