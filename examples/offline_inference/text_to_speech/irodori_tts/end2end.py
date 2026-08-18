# SPDX-License-Identifier: Apache-2.0
"""Generate, benchmark, and compare native Irodori-TTS output.

Examples:
  python end2end.py --text 'こんにちは' --ref-audio reference.wav
  python end2end.py --text 'こんにちは' --warmup 1 --runs 3
  python end2end.py --text 'こんにちは' --baseline upstream.wav
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Aratako/Irodori-TTS-v4-Small")
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--caption", default="")
    parser.add_argument("--ref-audio", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/irodori_tts"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--cfg-refresh-interval", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--baseline", type=Path, help="Reference WAV for fixed-seed parity (one text only).")
    parser.add_argument("--max-relative-rmse", type=float, default=0.1)
    parser.add_argument("--min-correlation", type=float, default=0.99)
    return parser.parse_args()


def load_references(paths: list[str]) -> list[tuple[np.ndarray, int]]:
    from vllm.multimodal.media.audio import load_audio

    references = []
    for path in paths:
        waveform, sample_rate = load_audio(path, sr=None)
        references.append((np.asarray(waveform, dtype=np.float32), int(sample_rate)))
    return references


def extract_audio(output: object) -> tuple[np.ndarray, int]:
    request_output = getattr(output, "request_output", None)
    payload = getattr(output, "multimodal_output", None)
    if payload is None and request_output is not None:
        payload = getattr(request_output, "multimodal_output", None)
    if not isinstance(payload, dict) or "audio" not in payload:
        raise RuntimeError("Irodori produced no audio output.")
    audio = np.asarray(payload["audio"], dtype=np.float32).squeeze()
    sample_rate = int(payload.get("audio_sample_rate", payload.get("sr", 0)))
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise RuntimeError(f"Irodori returned invalid audio shape {audio.shape}.")
    if sample_rate != 48_000:
        raise RuntimeError(f"Expected 48000 Hz audio, got {sample_rate}.")
    return audio, sample_rate


def check_parity(generated: np.ndarray, sample_rate: int, args: argparse.Namespace) -> None:
    baseline, baseline_rate = sf.read(args.baseline, dtype="float32", always_2d=False)
    baseline = np.asarray(baseline, dtype=np.float32).squeeze()
    if baseline_rate != sample_rate or baseline.ndim != 1 or len(baseline) != len(generated):
        raise RuntimeError(
            f"Parity shape mismatch: generated={generated.shape}@{sample_rate}, "
            f"baseline={baseline.shape}@{baseline_rate}."
        )
    relative_rmse = float(np.sqrt(np.mean((generated - baseline) ** 2)) / max(np.sqrt(np.mean(baseline**2)), 1e-8))
    generated_centered = generated - generated.mean()
    baseline_centered = baseline - baseline.mean()
    correlation = float(
        np.dot(generated_centered, baseline_centered)
        / max(np.linalg.norm(generated_centered) * np.linalg.norm(baseline_centered), 1e-8)
    )
    print(f"parity: relative_rmse={relative_rmse:.6g}, correlation={correlation:.8f}")
    if relative_rmse > args.max_relative_rmse or correlation < args.min_correlation:
        raise RuntimeError("Parity thresholds failed.")


def main() -> None:
    args = parse_args()
    if args.runs < 1 or args.warmup < 0 or args.max_num_seqs < 1:
        raise ValueError("--runs and --max-num-seqs must be positive; --warmup cannot be negative.")
    texts = args.text or ["こんにちは。これは音声合成のテストです。"]
    if args.baseline and len(texts) != 1:
        raise ValueError("--baseline parity accepts exactly one --text.")

    prompt_base: dict[str, object] = {"caption": args.caption}
    references = load_references(args.ref_audio)
    if references:
        prompt_base["ref_audio"] = references
    prompts = [{**prompt_base, "input": text} for text in texts]
    extra_args: dict[str, float | int] = {
        "duration_scale": args.duration_scale,
        "cfg_refresh_interval": args.cfg_refresh_interval,
    }
    if args.seconds is not None:
        extra_args["seconds"] = args.seconds
    sampling = OmniDiffusionSamplingParams(
        seed=args.seed,
        num_inference_steps=args.num_steps,
        extra_args=extra_args,
    )

    omni = Omni(
        model=args.model,
        model_class_name="IrodoriTTSPipeline",
        max_num_seqs=args.max_num_seqs,
        diffusion_batch_size=args.max_num_seqs,
        step_execution=True,
    )
    try:
        for _ in range(args.warmup):
            omni.generate(prompts, sampling_params_list=[sampling], use_tqdm=False)
        timings = []
        outputs = None
        for _ in range(args.runs):
            started = time.perf_counter()
            outputs = omni.generate(prompts, sampling_params_list=[sampling], use_tqdm=False)
            timings.append(time.perf_counter() - started)
        if outputs is None or len(outputs) != len(prompts):
            raise RuntimeError(f"Expected {len(prompts)} outputs, got {0 if outputs is None else len(outputs)}.")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        total_audio_seconds = 0.0
        first_audio = None
        first_rate = 0
        for index, output in enumerate(outputs):
            audio, sample_rate = extract_audio(output)
            sf.write(args.output_dir / f"irodori-{index:02d}.wav", audio, sample_rate)
            total_audio_seconds += len(audio) / sample_rate
            if first_audio is None:
                first_audio, first_rate = audio, sample_rate
        median = statistics.median(timings)
        print(
            f"speed: requests={len(prompts)}, runs={args.runs}, median={median:.4f}s, "
            f"audio={total_audio_seconds:.2f}s, realtime_factor={median / total_audio_seconds:.4f}"
        )
        if args.baseline:
            assert first_audio is not None
            check_parity(first_audio, first_rate, args)
    finally:
        omni.close()


if __name__ == "__main__":
    main()
