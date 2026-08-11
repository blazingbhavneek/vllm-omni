# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-stage native Irodori-TTS v4-Small diffusion pipeline."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn as nn
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportAudioOutput, SupportsComponentDiscovery
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .codec import DACVAECodec, patchify_latent, unpatchify_latent
from .config import IrodoriCheckpointConfig, read_irodori_checkpoint_config, resolve_irodori_checkpoint
from .duration import build_duration_features
from .model import TextToLatentRFDiT
from .sampler import sample_euler_rf_cfg
from .text_normalization import normalize_text
from .tokenizer import PretrainedTextTokenizer


def get_irodori_tts_post_process_func(_od_config: OmniDiffusionConfig):
    """Keep tensors on-device unless the request explicitly wants NumPy."""

    def post_process_func(audio: torch.Tensor, output_type: str = "np"):
        if output_type in {"latent", "pt"}:
            return audio
        if output_type != "np":
            raise ValueError(f"Unsupported Irodori output_type={output_type!r}; expected np, pt, or latent.")
        return audio.detach().cpu().float().numpy()

    return post_process_func


class IrodoriTTSPipeline(nn.Module, SupportAudioOutput, SupportsComponentDiscovery):
    """Irodori v4-Small with one complete request per diffusion invocation.

    All transformer, speaker, duration, and DiT parameters live under
    ``model`` because the released checkpoint is monolithic.  Consequently
    offload and distributed parallel modes are deliberately unsupported here.
    """

    supports_request_batch: ClassVar[bool] = False
    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 48000
    _dit_modules: ClassVar[list[str]] = ["model"]
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = ["vae"]
    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {
            "num_steps",
            "seed",
            "seconds",
            "duration_scale",
            "cfg_scale_text",
            "cfg_scale_caption",
            "cfg_scale_speaker",
        }
    )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        del prefix
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)
        if self.dtype is None:
            self.dtype = torch.bfloat16

        checkpoint_path = resolve_irodori_checkpoint(od_config.model, getattr(od_config, "revision", None))
        self.checkpoint_config = read_irodori_checkpoint_config(checkpoint_path)
        self.model = TextToLatentRFDiT(
            self.checkpoint_config.model,
            pretrained_backbone_config=self.checkpoint_config.text_encoder_config,
            load_pretrained_backbone_weights=False,
        ).to(device=self.device, dtype=self.dtype)

        self.tokenizer = self._load_tokenizer(od_config.model, checkpoint_path)
        self.codec = DACVAECodec.load(device=str(self.device), dtype=self.dtype)
        # Register the codec module exactly once. ``self.codec`` is only a
        # lightweight dataclass wrapper around this registered module.
        self.vae = self.codec.model
        if self.codec.latent_dim != self.checkpoint_config.model.latent_dim:
            raise ValueError(
                "Irodori model and DACVAE latent dimensions differ: "
                f"{self.checkpoint_config.model.latent_dim} != {self.codec.latent_dim}."
            )
        if self.codec.sample_rate != self.audio_sample_rate:
            raise ValueError(
                f"Irodori requires {self.audio_sample_rate} Hz DACVAE output; got {self.codec.sample_rate}."
            )
        weight_source = od_config.model
        if os.path.isfile(weight_source):
            weight_source = os.path.dirname(checkpoint_path)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=weight_source,
                subfolder=None,
                revision=getattr(od_config, "revision", None),
                prefix="model.",
                fall_back_to_pt=False,
                allow_patterns_overrides=["model.safetensors"],
            )
        ]

    def _load_tokenizer(self, model_or_path: str, checkpoint_path: str) -> PretrainedTextTokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - core dependency
            raise RuntimeError("transformers is required for Irodori-TTS tokenization.") from exc
        checkpoint_parent = os.path.dirname(checkpoint_path)
        if os.path.isdir(model_or_path) or os.path.isfile(model_or_path):
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(checkpoint_parent, "tokenizer"),
                use_fast=True,
                trust_remote_code=False,
                local_files_only=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_or_path,
                subfolder="tokenizer",
                revision=getattr(self.od_config, "revision", None),
                use_fast=True,
                trust_remote_code=False,
            )
        return PretrainedTextTokenizer(tokenizer, add_bos=self.checkpoint_config.model.text_add_bos)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)

    @staticmethod
    def _positive_int(value: Any, *, name: str, minimum: int, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}], got {value!r}.")
        return value

    @staticmethod
    def _finite_float(value: Any, *, name: str, minimum: float, maximum: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number in [{minimum}, {maximum}], got {value!r}.")
        value = float(value)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be a finite number in [{minimum}, {maximum}], got {value!r}.")
        return value

    def _parse_prompt(self, prompt: Any) -> tuple[str, str, Any | None]:
        if isinstance(prompt, str):
            prompt = {"input": prompt}
        if not isinstance(prompt, dict):
            raise ValueError("Irodori prompt must be text or a mapping.")
        text = prompt.get("input") or prompt.get("text") or prompt.get("prompt")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Irodori input text cannot be empty.")
        caption = prompt.get("caption")
        if caption is None:
            caption = prompt.get("instruct")
        if caption is None:
            caption = ""
        if not isinstance(caption, str):
            raise ValueError("Irodori caption must be a string when provided.")
        ref_audio = prompt.get("ref_audio")
        if ref_audio is None:
            multimedia = prompt.get("multi_modal_data")
            if isinstance(multimedia, dict):
                ref_audio = multimedia.get("audio")
        return text, caption, ref_audio

    @staticmethod
    def _as_ref_list(ref_audio: Any | None) -> list[tuple[Any, int]]:
        if ref_audio is None:
            return []
        if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
            return [(ref_audio[0], ref_audio[1])]
        if isinstance(ref_audio, list):
            if not ref_audio:
                raise ValueError("Irodori reference audio list cannot be empty.")
            result: list[tuple[Any, int]] = []
            for item in ref_audio:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError("Each Irodori reference audio item must be a (samples, sample_rate) tuple.")
                result.append((item[0], item[1]))
            return result
        raise ValueError("Irodori ref_audio must be a (samples, sample_rate) tuple or ordered list of tuples.")

    def _prepare_reference(self, ref_audio: Any | None) -> tuple[torch.Tensor, torch.Tensor, bool]:
        config = self.checkpoint_config.model
        refs = self._as_ref_list(ref_audio)
        if not refs:
            length = max(1, config.speaker_patch_size)
            return (
                torch.zeros((1, length, config.patched_latent_dim), device=self.device, dtype=self.dtype),
                torch.zeros((1, length), device=self.device, dtype=torch.bool),
                False,
            )
        total_seconds = 0.0
        pieces: list[torch.Tensor] = []
        for samples, sample_rate in refs:
            if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
                raise ValueError(f"Irodori reference sample rate must be positive, got {sample_rate!r}.")
            waveform = torch.as_tensor(samples, dtype=torch.float32)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.ndim not in (2, 3) or waveform.shape[-1] <= 0:
                raise ValueError("Irodori reference waveform must have non-empty samples.")
            if not bool(torch.isfinite(waveform).all().item()):
                raise ValueError("Irodori reference waveform contains non-finite samples.")
            total_seconds += float(waveform.shape[-1]) / sample_rate
            if total_seconds > self.checkpoint_config.max_ref_seconds:
                raise ValueError(
                    f"Combined Irodori reference audio must be at most {self.checkpoint_config.max_ref_seconds:g} seconds."
                )
            pieces.append(self.codec.encode_waveform(waveform, sample_rate))
        latent = torch.cat(pieces, dim=1)
        latent = patchify_latent(latent, config.latent_patch_size).to(device=self.device, dtype=self.dtype)
        if latent.shape[1] == 0:
            raise ValueError("Irodori reference audio is too short after latent patchification.")
        return latent, torch.ones(latent.shape[:2], device=self.device, dtype=torch.bool), True

    def _sampling_options(self, sampling: Any) -> dict[str, Any]:
        extra = dict(getattr(sampling, "extra_args", {}) or {})
        unknown = set(extra) - self.EXTRA_BODY_PARAMS
        if unknown:
            raise ValueError(f"Unsupported Irodori sampling options: {sorted(unknown)}")
        num_steps = getattr(sampling, "num_inference_steps", None)
        if num_steps is None:
            num_steps = extra.get("num_steps", 40)
        seed = getattr(sampling, "seed", None)
        if seed is None:
            seed = extra.get("seed", 0)
        generator = getattr(sampling, "generator", None)
        if isinstance(generator, list):
            raise ValueError("Irodori does not support multiple request generators.")
        return {
            "num_steps": self._positive_int(num_steps, name="num_steps", minimum=1, maximum=100),
            "seed": self._positive_int(seed, name="seed", minimum=0, maximum=2**63 - 1),
            "generator": generator,
            "seconds": (
                None
                if extra.get("seconds") is None
                else self._finite_float(extra["seconds"], name="seconds", minimum=0.5, maximum=30.0)
            ),
            "duration_scale": self._finite_float(
                extra.get("duration_scale", 1.0), name="duration_scale", minimum=0.25, maximum=4.0
            ),
            "cfg_scale_text": self._finite_float(
                extra.get("cfg_scale_text", 3.0), name="cfg_scale_text", minimum=0.0, maximum=10.0
            ),
            "cfg_scale_caption": self._finite_float(
                extra.get("cfg_scale_caption", 3.0), name="cfg_scale_caption", minimum=0.0, maximum=10.0
            ),
            "cfg_scale_speaker": self._finite_float(
                extra.get("cfg_scale_speaker", 5.0), name="cfg_scale_speaker", minimum=0.0, maximum=10.0
            ),
            "latents": getattr(sampling, "latents", None),
            "output_type": getattr(sampling, "output_type", None) or "np",
        }

    def _duration_steps(
        self,
        *,
        text: str,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
        ref_latent: torch.Tensor,
        ref_mask: torch.Tensor,
        has_reference: bool,
        options: dict[str, Any],
    ) -> tuple[int, int]:
        if options["seconds"] is not None:
            target_samples = round(options["seconds"] * self.codec.sample_rate)
            return math.ceil(target_samples / self.codec.hop_length), target_samples
        features = build_duration_features(
            [text], token_counts=text_mask.sum(dim=1), max_text_len=self.checkpoint_config.max_text_len,
            has_speaker=[has_reference],
        ).to(self.device)
        text_state, text_condition_mask, speaker_state, speaker_mask, caption_state, caption_condition_mask = (
            self.model.encode_conditions(
                text_input_ids=text_ids, text_mask=text_mask, ref_latent=ref_latent, ref_mask=ref_mask,
                caption_input_ids=caption_ids, caption_mask=caption_mask,
            )
        )
        prediction = self.model.predict_duration_log_frames(
            text_state=text_state, text_mask=text_condition_mask, speaker_state=speaker_state,
            speaker_mask=speaker_mask, caption_state=caption_state, caption_mask=caption_condition_mask,
            duration_features=features, has_speaker=torch.tensor([has_reference], device=self.device),
            has_caption=torch.tensor([bool(caption_condition_mask.any().item())], device=self.device),
        )
        latent_steps = round(float(torch.expm1(prediction).mean().item()) * options["duration_scale"])
        minimum = math.ceil(0.5 * self.codec.sample_rate / self.codec.hop_length)
        maximum = math.ceil(30.0 * self.codec.sample_rate / self.codec.hop_length)
        latent_steps = min(max(latent_steps, minimum), maximum)
        return latent_steps, latent_steps * self.codec.hop_length

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if req.num_reqs != 1:
            raise ValueError("Irodori-TTS supports one request per diffusion invocation.")
        text, caption, ref_audio = self._parse_prompt(req.prompts[0])
        text = normalize_text(text).strip()
        if not text:
            raise ValueError("Irodori input text is empty after normalization.")
        options = self._sampling_options(req.sampling_params)
        if options["output_type"] not in {"np", "pt", "latent"}:
            raise ValueError("Irodori output_type must be one of: np, pt, latent.")
        text_ids, text_mask = self.tokenizer.batch_encode([text], max_length=self.checkpoint_config.max_text_len)
        # A blank caption stays a real tensor with an all-false mask, matching
        # upstream unconditional caption conditioning rather than omitting it.
        caption_ids, caption_mask = self.tokenizer.batch_encode(
            [caption], max_length=self.checkpoint_config.max_caption_len
        )
        if not caption.strip():
            caption_mask.zero_()
        text_ids, text_mask = text_ids.to(self.device), text_mask.to(self.device)
        caption_ids, caption_mask = caption_ids.to(self.device), caption_mask.to(self.device)
        ref_latent, ref_mask, has_reference = self._prepare_reference(ref_audio)
        latent_steps, target_samples = self._duration_steps(
            text=text, text_ids=text_ids, text_mask=text_mask, caption_ids=caption_ids, caption_mask=caption_mask,
            ref_latent=ref_latent, ref_mask=ref_mask, has_reference=has_reference, options=options,
        )
        patched_steps = math.ceil(latent_steps / self.checkpoint_config.model.latent_patch_size)
        latent = sample_euler_rf_cfg(
            self.model, text_ids, text_mask, ref_latent, ref_mask, patched_steps,
            caption_input_ids=caption_ids, caption_mask=caption_mask, num_steps=options["num_steps"],
            cfg_scale_text=options["cfg_scale_text"], cfg_scale_caption=options["cfg_scale_caption"],
            cfg_scale_speaker=options["cfg_scale_speaker"], cfg_guidance_mode="independent", cfg_min_t=0.5,
            cfg_max_t=1.0, seed=options["seed"], generator=options["generator"], initial_latents=options["latents"],
            use_context_kv_cache=True, t_schedule_mode="linear",
        )
        latent = unpatchify_latent(
            latent, self.checkpoint_config.model.latent_patch_size, self.checkpoint_config.model.latent_dim
        )[:, :latent_steps]
        if options["output_type"] == "latent":
            return DiffusionOutput(output=latent)
        waveform = self.codec.decode_latent(latent)[:, :, :target_samples]
        return DiffusionOutput(output=waveform)
