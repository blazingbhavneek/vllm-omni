# SPDX-License-Identifier: Apache-2.0
"""Offline native Irodori-TTS inference and step-batching smoke tester.

The defaults mirror the reference-cloning command in the workspace's
``command.txt``. Run it with the new step-execution path:

    uv run --active --no-sync python examples/offline_inference/text_to_speech/irodori_tts/tester.py \
        --seed 0 --output-wav outputs/irodori-vllm.wav

Exercise continuous batching with exact output lengths:

    uv run --active --no-sync python examples/offline_inference/text_to_speech/irodori_tts/tester.py \
        --checkpoint ../Aratako/Irodori-TTS-v4-Small/model.safetensors \
        --text "一つ目のリクエストです。" --text "二つ目のリクエストです。" \
        --seconds 3 --max-num-seqs 2 --num-steps 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

DEFAULT_TEXT = (
    "ねえ、今日の帰りに少しだけ寄り道しない？🤔 新しくできた小さな喫茶店があるらしいんだ。"
    "⏸️ 🫶 急がなくていいから、あたたかいコーヒーを飲みながら、ゆっくり話せたらうれしいな。"
)
DEFAULT_CAPTION = (
    "親しい相手にやさしく語りかける、穏やかで少し照れくさい大人の男性。"
    "自然な会話調で、柔らかく温度感のある声。"
)
WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MODEL_DIR = WORKSPACE_ROOT / "Aratako/Irodori-TTS-v4-Small"
DEFAULT_REFERENCE = WORKSPACE_ROOT / "Aratako/Irodori-TTS-v4-Small/samples/clone_ref1.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline vLLM-Omni Irodori-TTS tester")
    parser.add_argument(
        "--checkpoint",
        "--model",
        dest="model",
        default=str(DEFAULT_MODEL_DIR),
        help="Local model directory, checkpoint .safetensors file, or Hugging Face model ID.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Text to synthesize. Repeat this flag to submit several requests together.",
    )
    parser.add_argument("--caption", default=DEFAULT_CAPTION, help="Optional speaking-style caption.")
    parser.add_argument(
        "--ref-wav",
        "--ref-audio",
        dest="ref_audio",
        action="append",
        default=None,
        help="Reference WAV/audio clip. Replaces the default clone_ref1.wav; repeat to preserve order.",
    )
    parser.add_argument("--no-ref", action="store_true", help="Require text-only synthesis (command.txt compatibility).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Optional explicit output length; omit it to batch predicted durations.",
    )
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--cfg-scale-text", type=float, default=3.0)
    parser.add_argument("--cfg-scale-caption", type=float, default=3.0)
    parser.add_argument("--cfg-scale-speaker", type=float, default=5.0)
    parser.add_argument("--max-num-seqs", type=int, default=1, help="Active step requests; >1 enables batching.")
    parser.add_argument(
        "--request-mode",
        dest="step_execution",
        action="store_false",
        help="Use the legacy full-request loop instead of the default step-execution path.",
    )
    parser.set_defaults(step_execution=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable Irodori CUDA graph capture while keeping step batching enabled.",
    )
    parser.add_argument("--model-device", default="cuda", choices=["cuda"])
    parser.add_argument("--codec-device", default="cuda", choices=["cuda"])
    parser.add_argument("--model-precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--codec-precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--output-wav", default="outputs/irodori-vllm.wav")
    return parser.parse_args()


def load_references(paths: list[str]) -> list[tuple[np.ndarray, int]]:
    if not paths:
        return []
    from vllm.multimodal.media.audio import load_audio

    references: list[tuple[np.ndarray, int]] = []
    for path in paths:
        waveform, sample_rate = load_audio(path, sr=None)
        references.append((np.asarray(waveform, dtype=np.float32), int(sample_rate)))
    return references


def output_paths(output_wav: Path, count: int) -> list[Path]:
    if count == 1:
        return [output_wav]
    suffix = output_wav.suffix or ".wav"
    return [output_wav.with_name(f"{output_wav.stem}-{idx:02d}{suffix}") for idx in range(count)]


def resolve_model_source(model: str) -> str:
    """Return the directory required by Omni when passed an Irodori weight file."""
    model_path = Path(model).expanduser()
    if model_path.is_file():
        return str(model_path.parent.resolve())
    return model


def require_codec_bridge() -> None:
    """Fail before worker startup when Irodori's optional audio codec is absent."""
    try:
        from dacvae import DACVAE
    except ImportError:
        active_python = Path(sys.executable)
        raise SystemExit(
            "Irodori-TTS needs its optional DACVAE codec bridge. From the "
            "vllm-omni checkout, install it into this active environment:\n\n"
            f'  uv pip install --python "{active_python}" -e ".[irodori-tts]"\n\n'
            "Then rerun this tester."
        )
    del DACVAE


def extract_audio(output: object) -> tuple[np.ndarray, int]:
    request_output = getattr(output, "request_output", None)
    multimodal_output = getattr(output, "multimodal_output", None)
    if multimodal_output is None and request_output is not None:
        multimodal_output = getattr(request_output, "multimodal_output", None)
    if not isinstance(multimodal_output, dict) or "audio" not in multimodal_output:
        raise RuntimeError("Irodori produced no audio output.")
    waveform = np.asarray(multimodal_output["audio"], dtype=np.float32).squeeze()
    sample_rate = int(multimodal_output.get("audio_sample_rate", multimodal_output.get("sr", 0)))
    if waveform.ndim != 1 or waveform.size == 0:
        raise RuntimeError(f"Irodori returned invalid audio shape {waveform.shape}.")
    if sample_rate != 48000:
        raise RuntimeError(f"Irodori must return 48000 Hz audio, got {sample_rate}.")
    return waveform, sample_rate


def main() -> None:
    args = parse_args()
    if args.model_precision != args.codec_precision:
        raise ValueError("vLLM-Omni Irodori uses one shared dtype; model and codec precision must match.")
    if args.no_ref and args.ref_audio:
        raise ValueError("--no-ref cannot be combined with --ref-wav/--ref-audio.")
    if args.max_num_seqs < 1:
        raise ValueError("--max-num-seqs must be at least one.")
    require_codec_bridge()
    model_source = resolve_model_source(args.model)
    texts = args.text or [DEFAULT_TEXT]
    reference_paths = [] if args.no_ref else (args.ref_audio or [str(DEFAULT_REFERENCE)])
    references = load_references(reference_paths)
    prompt_base: dict[str, object] = {"input": None, "caption": args.caption}
    if references:
        prompt_base["ref_audio"] = references
    prompts = [{**prompt_base, "input": text} for text in texts]
    extra_args = {
        "duration_scale": args.duration_scale,
        "cfg_scale_text": args.cfg_scale_text,
        "cfg_scale_caption": args.cfg_scale_caption,
        "cfg_scale_speaker": args.cfg_scale_speaker,
    }
    if args.seconds is not None:
        extra_args["seconds"] = args.seconds
    sampling = OmniDiffusionSamplingParams(
        seed=args.seed,
        num_inference_steps=args.num_steps,
        extra_args=extra_args,
    )
    dtype = {"bf16": "bfloat16", "fp32": "float32"}[args.model_precision]
    outputs_path = output_paths(Path(args.output_wav), len(prompts))
    for output_path in outputs_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model           : {model_source}")
    print(f"Requests        : {len(prompts)}")
    print(f"Step execution  : {args.step_execution}")
    print(f"Max active seqs : {args.max_num_seqs}")
    graph_enabled = not args.enforce_eager and not args.disable_cuda_graph
    print(f"CUDA graphs     : {'enabled' if graph_enabled else 'disabled'}")
    print(f"Steps           : {args.num_steps}")
    print(f"Seconds         : {args.seconds if args.seconds is not None else 'predicted'}")
    print(f"References      : {len(references)}")

    os.environ["VLLM_OMNI_IRODORI_CUDA_GRAPH"] = "1" if graph_enabled else "0"
    omni = Omni(
        model=model_source,
        model_class_name="IrodoriTTSPipeline",
        dtype=dtype,
        max_num_seqs=args.max_num_seqs,
        diffusion_batch_size=args.max_num_seqs,
        step_execution=args.step_execution,
        enforce_eager=args.enforce_eager,
    )
    try:
        start = time.perf_counter()
        outputs = omni.generate(prompts, sampling_params_list=[sampling], use_tqdm=False)
        elapsed = time.perf_counter() - start
        if len(outputs) != len(prompts):
            raise RuntimeError(f"Expected {len(prompts)} final outputs, got {len(outputs)}.")
        total_audio_seconds = 0.0
        for idx, (output, output_path) in enumerate(zip(outputs, outputs_path, strict=True)):
            waveform, sample_rate = extract_audio(output)
            sf.write(output_path, waveform, sample_rate)
            duration = len(waveform) / sample_rate
            total_audio_seconds += duration
            print(f"Saved request {idx}: {output_path} ({duration:.2f}s, {sample_rate} Hz)")
        rtf = elapsed / total_audio_seconds if total_audio_seconds else float("inf")
        print(f"Inference wall time: {elapsed:.2f}s | aggregate audio: {total_audio_seconds:.2f}s | RTF: {rtf:.3f}")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
