"""Multi-GPU generation over OmniVoice-style JSONL test lists.

Reads OmniVoice JSONL test lists (fields: ``id``, ``text``, ``ref_audio``,
``ref_text``, ``language_id``, ...) and writes one wav per row to
``--res_dir/{id}.wav`` at 24 kHz so that
``omnivoice.eval.{speaker_similarity.sim, wer.*, mos.utmos}`` can score them
against the same JSONL — keeping the upstream eval code untouched
(benchmark-faithful). Sharded across all visible CUDA devices.

Defaults reproduce the paper configuration:
``DRF_BLOCK_SIZE=16, NUM_STEPS=10, TEMPERATURE=0.6, TIME_SHIFT_TAU=0.1,
CFG_SCALE=1.0 (zero_text_batch + pmi_cfg + zero_all), CUDA-graphed FlashInfer
(falls back to torch SDPA via ``--backend torch`` or when flashinfer-python
is not installed)``.

Run as a module::

    python -m chatterbox_flash.eval.generate \\
        --ckpt_dir /path/to/chatterbox-flash-ckpt \\
        --test_list /path/to/test.jsonl \\
        --res_dir   /path/to/output \\
        --ref_audio_root /path/to/jsonl-anchor
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from time import perf_counter

import torch
import torch.multiprocessing as mp
import torchaudio as ta
from tqdm import tqdm

from ...chatterbox.models.s3gen import S3GEN_SR

from ..tts import ChatterboxFlashTTS


# ─────────────────────────────────────────────────────────────────────────
# JSONL parsing
# ─────────────────────────────────────────────────────────────────────────


def _resolve_ref_audio(ref: str, root: Path) -> str:
    p = Path(ref).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    return str(p)


def parse_jsonl(
    test_list: str, lang_filter: str, ref_audio_root: str | None,
) -> list[dict]:
    test_path = Path(test_list).expanduser().resolve()
    root = (
        Path(ref_audio_root).expanduser().resolve()
        if ref_audio_root else test_path.parent
    )
    out: list[dict] = []
    with test_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"[parse_jsonl] line {line_no} skipped (bad JSON): {e}",
                    flush=True,
                )
                continue
            uid, text, ref_audio = obj.get("id"), obj.get("text"), obj.get("ref_audio")
            if uid is None or text is None or ref_audio is None:
                continue
            lang = obj.get("language_id")
            if lang_filter != "all" and lang is not None:
                if not str(lang).lower().startswith(lang_filter.lower()):
                    continue
            out.append({
                "id": str(uid),
                "text": text,
                "ref_audio": _resolve_ref_audio(ref_audio, root),
                "ref_text": obj.get("ref_text") or "",
                "language_id": lang,
            })
    return out


# ─────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Chatterbox-Flash generation for OmniVoice eval JSONL.",
    )
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Directory containing the chatterbox-flash safetensors "
                   "(t3_flash.safetensors, s3gen.safetensors, ve.safetensors, "
                   "tokenizer.json).")
    p.add_argument("--test_list", type=str, required=True,
                   help="OmniVoice JSONL test list.")
    p.add_argument("--res_dir", type=str, required=True,
                   help="Output directory; one wav per row written as "
                   "{id}.wav at 24 kHz.")
    p.add_argument("--ref_audio_root", type=str, default=None,
                   help="Root for relative ref_audio paths (default: dir of "
                   "--test_list).")
    p.add_argument("--language_id_filter", type=str, default="en",
                   help="Only process rows whose language_id starts with this "
                   "prefix. Pass 'all' to disable.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip rows whose {res_dir}/{id}.wav already exists.")

    # Inference knobs — defaults match the paper.
    p.add_argument("--dtype", type=str, default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--drf_block_size", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--time_shift_tau", type=float, default=0.1)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--exaggeration", type=float, default=0.5)
    p.add_argument("--no_cuda_graph", action="store_true")
    p.add_argument("--backend", type=str, default="auto",
                   choices=["auto", "flashinfer", "torch"],
                   help="Inference engine. 'auto' picks flashinfer when "
                   "importable, else torch SDPA. Override per-run with the "
                   "CHATTERBOX_FLASH_ENGINE env var.")
    p.add_argument("--no_text_norm", action="store_true",
                   help="Skip en_us_cleaner (use raw JSONL text).")
    p.add_argument("--mp_stagger_sec", type=float, default=3.0,
                   help="Delay rank*staffer seconds between worker startups "
                   "(amortises HF downloads).")
    p.add_argument("--max_speech_tokens", type=int, default=None,
                   help="Optional per-row token budget cap.")
    return p


def _worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    logging.getLogger("filelock").setLevel(logging.WARNING)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    if world_size > 1 and args.mp_stagger_sec > 0:
        time.sleep(float(rank) * float(args.mp_stagger_sec))

    dtype_map = {
        "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
    }
    target_dtype = dtype_map[args.dtype]

    print(f"[GPU {rank}] Loading Chatterbox-Flash from {args.ckpt_dir} …",
          flush=True)
    tts = ChatterboxFlashTTS.from_local(
        args.ckpt_dir, device,
        dtype=target_dtype,
        drf_block_size=args.drf_block_size,
    )

    out_dir = os.path.abspath(args.res_dir)
    os.makedirs(out_dir, exist_ok=True)

    all_records = parse_jsonl(
        args.test_list, args.language_id_filter, args.ref_audio_root,
    )
    if rank == 0:
        print(f"[GPU 0] {len(all_records)} JSONL row(s) after "
              f"lang_filter={args.language_id_filter!r}", flush=True)
    my_records = all_records[rank::world_size]

    bs = max(1, args.batch_size)

    common_gen = dict(
        normalize_text=not args.no_text_norm,
        num_steps=args.num_steps,
        temperature=args.temperature,
        time_shift_tau=args.time_shift_tau,
        cfg_scale=args.cfg_scale,
        use_cuda_graph=not args.no_cuda_graph,
        backend=args.backend,
        max_speech_tokens=args.max_speech_tokens,
    )

    def _row_eligible(row: dict) -> bool:
        if not os.path.isfile(row["ref_audio"]):
            return False
        if args.skip_existing and os.path.isfile(
            os.path.join(out_dir, f"{row['id']}.wav"),
        ):
            return False
        return True

    eligible = [r for r in my_records if _row_eligible(r)]
    n_batches = (len(eligible) + bs - 1) // bs
    pbar = tqdm(total=n_batches, desc=f"[GPU {rank}]", position=rank)
    for i in range(0, len(eligible), bs):
        batch = eligible[i : i + bs]
        try:
            t0 = perf_counter()
            if len(batch) == 1:
                wav = tts.generate(
                    batch[0]["text"],
                    audio_prompt_path=batch[0]["ref_audio"],
                    exaggeration=args.exaggeration,
                    **common_gen,
                )
                wavs = [wav]
            else:
                wavs = tts.generate_batch(
                    [r["text"] for r in batch],
                    audio_prompt_paths=[r["ref_audio"] for r in batch],
                    exaggeration=args.exaggeration,
                    **common_gen,
                )
            dt = perf_counter() - t0
            for r, w in zip(batch, wavs):
                ta.save(
                    os.path.join(out_dir, f"{r['id']}.wav"),
                    w.unsqueeze(0).cpu(), S3GEN_SR,
                )
            if rank == 0:
                tqdm.write(
                    f"[GPU 0] batch={len(batch)}  {dt * 1000:.0f} ms  "
                    f"({1000 * dt / max(1, len(batch)):.0f} ms/row)",
                )
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"[GPU {rank}] batch failed ({len(batch)} rows): {e}")
        pbar.update(1)
    pbar.close()

    if rank == 0:
        print(f"Saved outputs to {out_dir}")


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _build_arg_parser().parse_args()
    world_size = torch.cuda.device_count()
    assert world_size > 0, "No CUDA devices found"
    if world_size == 1:
        _worker(0, 1, args)
    else:
        mp.spawn(_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
