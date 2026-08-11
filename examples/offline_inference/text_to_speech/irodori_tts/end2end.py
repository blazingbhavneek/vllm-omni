# SPDX-License-Identifier: Apache-2.0
"""Generate final-only 48 kHz mono audio with native Irodori-TTS v4-Small."""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Aratako/Irodori-TTS-v4-Small")
    parser.add_argument("--text", default="こんにちは。これは音声合成のテストです。")
    parser.add_argument("--caption", default="")
    parser.add_argument("--ref-audio", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--output", default="irodori.wav")
    args = parser.parse_args()

    from vllm.multimodal.media.audio import load_audio

    prompt: dict = {"input": args.text, "caption": args.caption}
    if args.ref_audio:
        references = []
        for path in args.ref_audio:
            waveform, sample_rate = load_audio(path, sr=None)
            references.append((np.asarray(waveform, dtype=np.float32), sample_rate))
        prompt["ref_audio"] = references
    extra_args = {"duration_scale": args.duration_scale}
    if args.seconds is not None:
        extra_args["seconds"] = args.seconds
    sampling = OmniDiffusionSamplingParams(
        seed=args.seed,
        num_inference_steps=args.num_steps,
        extra_args=extra_args,
    )
    omni = Omni(model=args.model)
    try:
        outputs = list(omni.generate(prompt, sampling_params_list=[sampling]))
        final = outputs[-1]
        request_output = getattr(final, "request_output", None)
        mm = getattr(final, "multimodal_output", None) or getattr(request_output, "multimodal_output", None)
        if not mm or "audio" not in mm:
            raise RuntimeError("Irodori produced no audio output.")
        waveform = np.asarray(mm["audio"], dtype=np.float32).squeeze()
        sample_rate = int(mm.get("audio_sample_rate", mm.get("sr", 0)))
        if sample_rate != 48000:
            raise RuntimeError(f"Irodori must return 48000 Hz audio, got {sample_rate}.")
        sf.write(args.output, waveform, sample_rate)
        print(f"Saved {args.output} ({len(waveform) / sample_rate:.2f}s, {sample_rate} Hz)")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
