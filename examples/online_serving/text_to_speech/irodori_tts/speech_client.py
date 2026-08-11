"""Send text, caption, and ordered reference clips to Irodori-TTS."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import httpx


def as_data_uri(path: str) -> str:
    suffix = Path(path).suffix.lower()
    media_type = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg"}.get(suffix, "audio/wav")
    return f"data:{media_type};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8091/v1/audio/speech")
    parser.add_argument("--model", default="Aratako/Irodori-TTS-v4-Small")
    parser.add_argument("--text", default="こんにちは。これは音声合成のテストです。")
    parser.add_argument("--instructions", default="")
    parser.add_argument("--ref-audio", action="append", default=[])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--output", default="irodori.wav")
    args = parser.parse_args()
    payload = {
        "model": args.model,
        "input": args.text,
        "instructions": args.instructions,
        "seed": args.seed,
        "response_format": "wav",
        "extra_params": {"num_steps": args.num_steps, "duration_scale": 1.0},
    }
    if args.ref_audio:
        payload["ref_audio"] = [as_data_uri(path) for path in args.ref_audio]
    response = httpx.post(args.url, json=payload, timeout=300.0)
    response.raise_for_status()
    Path(args.output).write_bytes(response.content)


if __name__ == "__main__":
    main()
