# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-stage native Irodori-TTS v4-Small diffusion pipeline."""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import (
    SupportAudioOutput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

from .batching import (
    IrodoriBatchingConfig,
    IrodoriContextBucketPolicy,
    IrodoriDenoiseBatch,
    IrodoriExecutionKey,
    IrodoriLatentBucketPolicy,
    IrodoriLengthState,
)
from .codec import DACVAECodec, patchify_latent, unpatchify_latent
from .config import read_irodori_checkpoint_config, resolve_irodori_checkpoint
from .cudagraph import IrodoriCUDAGraphRunner
from .duration import build_duration_features
from .model import TextToLatentRFDiT
from .sampler import (
    IrodoriConditionState,
    IrodoriSamplingState,
    apply_euler_rf_cfg_step,
    encode_irodori_conditions,
    predict_euler_rf_cfg_batch,
    predict_euler_rf_cfg_step,
    prepare_euler_rf_cfg,
    run_packed_varlen_euler_rf_cfg_step,
    run_packed_euler_rf_cfg_step,
    supports_packed_euler_rf_cfg_batch,
)
from .text_normalization import normalize_text
from .tokenizer import PretrainedTextTokenizer

logger = init_logger(__name__)


def get_irodori_tts_post_process_func(_od_config: OmniDiffusionConfig):
    """Keep tensors on-device unless the request explicitly wants NumPy."""

    def post_process_func(audio: torch.Tensor, output_type: str = "np"):
        if output_type in {"latent", "pt"}:
            return audio
        if output_type != "np":
            raise ValueError(f"Unsupported Irodori output_type={output_type!r}; expected np, pt, or latent.")
        return audio.detach().cpu().float().numpy()

    return post_process_func


@dataclass(frozen=True)
class IrodoriStepBatchKey:
    """Request-wide properties that must match in the scheduler active set."""

    num_steps: int
    cfg_branches: tuple[str, ...]
    cfg_scale_text: float
    cfg_scale_caption: float
    cfg_scale_speaker: float
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    cfg_schedule: str = "linear"
    output_type: str = "np"


def get_irodori_tts_step_batch_key_func(od_config: OmniDiffusionConfig):
    """Build scheduler-owned compatibility keys without loading model weights."""
    del od_config

    def step_batch_key(request: Any) -> IrodoriStepBatchKey:
        sampling = request.sampling_params
        extra = dict(getattr(sampling, "extra_args", {}) or {})
        raw_num_steps = getattr(sampling, "num_inference_steps", None)
        if raw_num_steps is None:
            raw_num_steps = extra.get("num_steps", 40)
        num_steps = IrodoriTTSPipeline._positive_int(
            raw_num_steps,
            name="num_steps",
            minimum=1,
            maximum=100,
        )
        cfg_scale_text = IrodoriTTSPipeline._finite_float(
            extra.get("cfg_scale_text", 3.0),
            name="cfg_scale_text",
            minimum=0.0,
            maximum=10.0,
        )
        cfg_scale_caption = IrodoriTTSPipeline._finite_float(
            extra.get("cfg_scale_caption", 3.0),
            name="cfg_scale_caption",
            minimum=0.0,
            maximum=10.0,
        )
        cfg_scale_speaker = IrodoriTTSPipeline._finite_float(
            extra.get("cfg_scale_speaker", 5.0),
            name="cfg_scale_speaker",
            minimum=0.0,
            maximum=10.0,
        )
        prompt = getattr(request, "prompt", None)
        caption = ""
        if isinstance(prompt, dict):
            caption = prompt.get("caption") or prompt.get("instruct") or ""
        if not isinstance(caption, str):
            raise ValueError("Irodori caption must be a string when provided.")
        cfg_branches = tuple(
            name
            for name, enabled in (
                ("text", cfg_scale_text > 0),
                ("speaker", cfg_scale_speaker > 0),
                ("caption", cfg_scale_caption > 0 and bool(caption.strip())),
            )
            if enabled
        )
        output_type = getattr(sampling, "output_type", None) or "np"
        if output_type not in {"np", "pt", "latent"}:
            raise ValueError("Irodori output_type must be one of: np, pt, latent.")
        return IrodoriStepBatchKey(
            num_steps=num_steps,
            cfg_branches=cfg_branches,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_caption=cfg_scale_caption,
            cfg_scale_speaker=cfg_scale_speaker,
            output_type=output_type,
        )

    return step_batch_key


@dataclass
class _IrodoriPreparedRequest:
    sampling_state: IrodoriSamplingState
    lengths: IrodoriLengthState
    output_type: str
    cfg_refresh_interval: int = 1


class IrodoriTTSPipeline(
    nn.Module,
    SupportAudioOutput,
    SupportsComponentDiscovery,
):
    """Irodori v4-Small with bucketed continuous step batching.

    All transformer, speaker, duration, and DiT parameters live under
    ``model`` because the released checkpoint is monolithic.  Consequently
    offload and distributed parallel modes are deliberately unsupported here.
    """

    supports_request_batch: ClassVar[bool] = False
    supports_step_execution: ClassVar[bool] = True
    supports_step_execution_partition: ClassVar[bool] = True
    supports_fused_step_execution: ClassVar[bool] = True
    supports_ragged_step_execution: ClassVar[bool] = True
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
            "cfg_refresh_interval",
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
        self.batching_config = IrodoriBatchingConfig.from_od_config(od_config)
        # Per-stage precision policy: TF32 for the DiT/codec matmuls, BF16 for
        # joint attention, strict IEEE FP32 for the condition encoders and the
        # duration predictor.  See precision.py for why the split exists.
        self.precision_policy = self.batching_config.precision_policy
        self.model.set_precision_policy(self.precision_policy)
        self.codec.precision_policy = self.precision_policy
        self.packed_varlen_enabled = self.model.supports_packed_varlen_attention()
        logger.info(
            "Irodori precision profile %r: dit=%s codec=%s condition=%s attention=%s",
            self.precision_policy.name,
            self.precision_policy.dit_matmul,
            self.precision_policy.codec_matmul,
            self.precision_policy.condition_matmul,
            self.precision_policy.attention_dtype,
        )
        logger.info(
            "Irodori packed varlen DiT batching: %s",
            "enabled" if self.packed_varlen_enabled else "unavailable; using latent buckets",
        )
        self.latent_bucket_policy = IrodoriLatentBucketPolicy(
            sample_rate=self.codec.sample_rate,
            hop_length=self.codec.hop_length,
            latent_patch_size=self.checkpoint_config.model.latent_patch_size,
            bucket_seconds=self.batching_config.latent_bucket_seconds,
            overflow_bucket_seconds=self.batching_config.overflow_bucket_seconds,
        )
        self.context_bucket_policy = IrodoriContextBucketPolicy(
            self.batching_config.context_bucket_tokens
        )
        graph_enabled = self.batching_config.enable_cuda_graph and not bool(
            getattr(od_config, "enforce_eager", False)
        )
        # Keep manual graph capture separate from real torch.compile output.
        # Irodori's repeated DiT blocks use regional compilation, so the
        # low-value manual graph path is disabled when that compile is active.
        compile_granularity = getattr(
            od_config,
            "diffusion_compile_granularity",
            "regional",
        )
        regional_compile_is_active = bool(getattr(self.model, "_repeated_blocks", None))
        if compile_granularity == "full" or regional_compile_is_active:
            graph_enabled = False
        self.cuda_graph_runner = IrodoriCUDAGraphRunner(
            enabled=graph_enabled,
            batch_sizes=self.batching_config.cuda_graph_batch_sizes,
            max_entries=self.batching_config.cuda_graph_max_entries,
            max_dynamic_entries=self.batching_config.cuda_graph_max_dynamic_entries,
            min_hits=self.batching_config.cuda_graph_min_hits,
        )
        self._denoise_batches: OrderedDict[IrodoriExecutionKey, IrodoriDenoiseBatch] = (
            OrderedDict()
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
        loaded = AutoWeightsLoader(self).load_weights(weights)
        if self.batching_config.fuse_linear_projections:
            if getattr(self.od_config, "quantization_config", None) is not None:
                logger.warning(
                    "Irodori linear projection fusion is disabled for quantized weights."
                )
            elif getattr(self.od_config, "lora_path", None):
                logger.warning(
                    "Irodori linear projection fusion is disabled for the configured LoRA; "
                    "the adapter targets the original projection module names."
                )
            else:
                attention_count, swiglu_count = (
                    self.model.fuse_linear_projections_for_inference()
                )
                logger.info(
                    "Irodori fused same-input projections in %d attention and %d SwiGLU modules.",
                    attention_count,
                    swiglu_count,
                )
        return loaded

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
                    "Combined Irodori reference audio must be at most "
                    f"{self.checkpoint_config.max_ref_seconds:g} seconds."
                )
            pieces.append(self.codec.encode_waveform(waveform, sample_rate))
        latent = torch.cat(pieces, dim=1)
        latent = patchify_latent(latent, config.latent_patch_size).to(device=self.device, dtype=self.dtype)
        if latent.shape[1] == 0:
            raise ValueError("Irodori reference audio is too short after latent patchification.")
        return latent, torch.ones(latent.shape[:2], device=self.device, dtype=torch.bool), True

    def _sampling_options(self, sampling: Any) -> dict[str, Any]:
        extra = dict(getattr(sampling, "extra_args", {}) or {})
        # DiffusionEngine's generic startup warmup disables image/text CFG with
        # these two Bagel-shaped defaults. They have no Irodori meaning; drop
        # only the exact pair so real unsupported user options still fail.
        warmup_cfg_defaults = {"cfg_text_scale": 1.0, "cfg_img_scale": 1.0}
        if all(extra.get(name) == value for name, value in warmup_cfg_defaults.items()):
            for name in warmup_cfg_defaults:
                extra.pop(name)
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
            # Per-request so fidelity can be compared without a server restart.
            "cfg_refresh_interval": self._positive_int(
                extra.get(
                    "cfg_refresh_interval",
                    self.batching_config.cfg_refresh_interval,
                ),
                name="cfg_refresh_interval",
                minimum=1,
                maximum=100,
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

    def _duration_lengths(
        self,
        *,
        text: str,
        condition: IrodoriConditionState,
        has_reference: bool,
        options: dict[str, Any],
    ) -> IrodoriLengthState:
        if options["seconds"] is not None:
            target_samples = round(options["seconds"] * self.codec.sample_rate)
            return self.latent_bucket_policy.lengths_for_samples(target_samples)
        features = build_duration_features(
            [text],
            token_counts=condition.text_mask.sum(dim=1),
            max_text_len=self.checkpoint_config.max_text_len,
            has_speaker=[has_reference],
        ).to(self.device)
        prediction = self.model.predict_duration_log_frames(
            text_state=condition.text_state,
            text_mask=condition.text_mask,
            speaker_state=condition.speaker_state,
            speaker_mask=condition.speaker_mask,
            caption_state=condition.caption_state,
            caption_mask=condition.caption_mask,
            duration_features=features, has_speaker=torch.tensor([has_reference], device=self.device),
            has_caption=torch.tensor(
                [condition.caption_mask is not None and bool(condition.caption_mask.any().item())],
                device=self.device,
            ),
        )
        valid_codec_frames = round(
            float(torch.expm1(prediction).mean().item()) * options["duration_scale"]
        )
        minimum = math.ceil(0.5 * self.codec.sample_rate / self.codec.hop_length)
        maximum = math.ceil(30.0 * self.codec.sample_rate / self.codec.hop_length)
        valid_codec_frames = min(max(valid_codec_frames, minimum), maximum)
        return self.latent_bucket_policy.lengths_for_predicted_frames(valid_codec_frames)

    def _prepare_request(self, prompt: Any, sampling: Any) -> _IrodoriPreparedRequest:
        text, caption, ref_audio = self._parse_prompt(prompt)
        text = normalize_text(text).strip()
        if not text:
            raise ValueError("Irodori input text is empty after normalization.")
        options = self._sampling_options(sampling)
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
        condition = encode_irodori_conditions(
            self.model,
            text_ids,
            text_mask,
            ref_latent,
            ref_mask,
            caption_input_ids=caption_ids,
            caption_mask=caption_mask,
        )
        lengths = self._duration_lengths(
            text=text,
            condition=condition,
            has_reference=has_reference,
            options=options,
        )
        return _IrodoriPreparedRequest(
            sampling_state=prepare_euler_rf_cfg(
                self.model,
                text_ids,
                text_mask,
                ref_latent,
                ref_mask,
                lengths.valid_latent_len,
                caption_input_ids=caption_ids,
                caption_mask=caption_mask,
                num_steps=options["num_steps"],
                cfg_scale_text=options["cfg_scale_text"],
                cfg_scale_caption=options["cfg_scale_caption"],
                cfg_scale_speaker=options["cfg_scale_speaker"],
                cfg_guidance_mode="independent",
                cfg_min_t=0.5,
                cfg_max_t=1.0,
                seed=options["seed"],
                generator=options["generator"],
                initial_latents=options["latents"],
                use_context_kv_cache=True,
                t_schedule_mode="linear",
                condition_state=condition,
                bucket_sequence_length=lengths.bucket_latent_len,
            ),
            lengths=lengths,
            output_type=options["output_type"],
            cfg_refresh_interval=options["cfg_refresh_interval"],
        )

    def _decode_prepared_request(
        self,
        prepared: _IrodoriPreparedRequest,
        *,
        output_type: str | None = None,
    ) -> DiffusionOutput:
        latent = unpatchify_latent(
            prepared.sampling_state.latents,
            self.checkpoint_config.model.latent_patch_size,
            self.checkpoint_config.model.latent_dim,
        )[:, : prepared.lengths.valid_codec_frames]
        if (output_type or prepared.output_type) == "latent":
            return DiffusionOutput(output=latent)
        waveform = self.codec.decode_latent(latent)[:, :, : prepared.lengths.target_samples]
        return DiffusionOutput(output=waveform)

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if req.num_reqs != 1:
            raise ValueError("Irodori-TTS supports one request per diffusion invocation.")
        prepared = self._prepare_request(req.prompts[0], req.sampling_params)
        while prepared.sampling_state.step_index < prepared.sampling_state.total_steps:
            prediction = predict_euler_rf_cfg_step(self.model, prepared.sampling_state)
            apply_euler_rf_cfg_step(prepared.sampling_state, prediction)
        return self._decode_prepared_request(prepared)

    def prepare_encode(self, state: StepRequestState, **_: Any) -> StepRequestState:
        """Prepare one request's conditions, noise, schedule, and static K/V."""
        prepared = self._prepare_request(state.prompt, state.sampling)
        state.latents = prepared.sampling_state.latents
        state.timesteps = prepared.sampling_state.t_schedule[:-1]
        state.step_index = 0
        state.extra["irodori"] = prepared
        return state

    def get_step_execution_key(self, state: StepRequestState) -> IrodoriExecutionKey:
        """Return the post-duration physical microbatch key for one request."""
        prepared = state.extra.get("irodori")
        if not isinstance(prepared, _IrodoriPreparedRequest):
            raise ValueError(f"Missing Irodori step state for request {state.request_id}.")
        sampling_state = prepared.sampling_state
        cfg_active = sampling_state.cfg_active[sampling_state.step_index]
        device = sampling_state.latents.device
        return IrodoriExecutionKey(
            bucket_latent_len=(
                None
                if supports_packed_euler_rf_cfg_batch(self.model, [sampling_state])
                else prepared.lengths.bucket_latent_len
            ),
            dtype=sampling_state.latents.dtype,
            device_type=device.type,
            device_index=device.index,
            cfg_guidance_mode=sampling_state.cfg_guidance_mode,
            cfg_layout=sampling_state.independent_names if cfg_active else ("cond",),
            cfg_refresh=self._needs_cfg_refresh(
                sampling_state,
                cfg_active,
                prepared.cfg_refresh_interval,
            ),
        )

    @staticmethod
    def _needs_cfg_refresh(
        sampling_state: IrodoriSamplingState,
        cfg_active: bool,
        interval: int,
    ) -> bool:
        """Whether this step must recompute the unconditional CFG branches.

        Refreshing and reusing requests cannot share a physical forward, so
        this feeds the execution key and splits them into separate
        microbatches.
        """
        if not cfg_active:
            return True
        if interval <= 1:
            return True
        # A request without a carried-over correction has nothing to reuse.
        if sampling_state.cfg_correction is None:
            return True
        return sampling_state.step_index % interval == 0

    def denoise_step(
        self,
        input_batch: InputBatch,
        *,
        states: list[StepRequestState] | None = None,
        **_: Any,
    ) -> torch.Tensor | None:
        """Run one fused DiT call per compatible active Irodori subgroup."""
        active_states = states or list(input_batch.states)
        grouped: dict[
            IrodoriExecutionKey,
            list[tuple[StepRequestState, _IrodoriPreparedRequest]],
        ] = {}
        for state in active_states:
            prepared = state.extra.get("irodori")
            if not isinstance(prepared, _IrodoriPreparedRequest):
                raise ValueError(f"Missing Irodori step state for request {state.request_id}.")
            if state.step_index != prepared.sampling_state.step_index:
                raise ValueError(f"Irodori step index diverged for request {state.request_id}.")
            grouped.setdefault(self.get_step_execution_key(state), []).append(
                (state, prepared)
            )

        predictions: dict[str, torch.Tensor] = {}
        for group in grouped.values():
            sampling_states = [prepared.sampling_state for _, prepared in group]
            for state, prediction in zip(
                (state for state, _ in group),
                predict_euler_rf_cfg_batch(self.model, sampling_states),
                strict=True,
            ):
                predictions[state.request_id] = prediction
        return torch.cat([predictions[state.request_id] for state in active_states], dim=0)

    def denoise_and_step(
        self,
        input_batch: InputBatch,
        *,
        states: list[StepRequestState],
        **_: Any,
    ) -> None:
        """Run a packed eager/graph step and advance every request exactly once."""
        if tuple(input_batch.request_ids) != tuple(state.request_id for state in states):
            raise ValueError("Irodori InputBatch/request ordering diverged.")

        prepared_requests: list[_IrodoriPreparedRequest] = []
        for state in states:
            prepared = state.extra.get("irodori")
            if not isinstance(prepared, _IrodoriPreparedRequest):
                raise ValueError(f"Missing Irodori step state for request {state.request_id}.")
            if state.step_index != prepared.sampling_state.step_index:
                raise ValueError(f"Irodori step index diverged for request {state.request_id}.")
            prepared_requests.append(prepared)

        sampling_states = [prepared.sampling_state for prepared in prepared_requests]
        if any(
            sampling_state.cfg_guidance_mode != "independent"
            or sampling_state.rescale_k is not None
            or sampling_state.rescale_sigma is not None
            or sampling_state.speaker_kv_active
            for sampling_state in sampling_states
        ):
            predictions = predict_euler_rf_cfg_batch(self.model, sampling_states)
            for state, sampling_state, prediction in zip(
                states,
                sampling_states,
                predictions,
                strict=True,
            ):
                apply_euler_rf_cfg_step(sampling_state, prediction)
                state.latents = sampling_state.latents
                state.step_index += 1
            return

        if supports_packed_euler_rf_cfg_batch(self.model, sampling_states):
            execution_key = self.get_step_execution_key(states[0])
            if any(self.get_step_execution_key(state) != execution_key for state in states[1:]):
                raise ValueError("Irodori packed step received a heterogeneous execution group.")
            next_latents = run_packed_varlen_euler_rf_cfg_step(
                self.model,
                sampling_states,
                cfg_refresh=execution_key.cfg_refresh,
            )
            for state, sampling_state, request_latents in zip(
                states,
                sampling_states,
                next_latents,
                strict=True,
            ):
                sampling_state.latents = request_latents
                sampling_state.step_index += 1
                state.latents = request_latents
                state.step_index += 1
            return

        execution_key = self.get_step_execution_key(states[0])
        if any(self.get_step_execution_key(state) != execution_key for state in states[1:]):
            raise ValueError("Irodori fused step received a heterogeneous execution group.")
        cached_batch = self._denoise_batches.get(execution_key)
        denoise_batch = IrodoriDenoiseBatch.make(
            [state.request_id for state in states],
            sampling_states,
            context_policy=self.context_bucket_policy,
            is_dynamic_latent_bucket=any(
                prepared.lengths.is_dynamic_bucket for prepared in prepared_requests
            ),
            cached_batch=cached_batch,
            cfg_refresh=execution_key.cfg_refresh,
        )
        self._denoise_batches[execution_key] = denoise_batch
        self._denoise_batches.move_to_end(execution_key)
        max_packed_batches = max(2, int(getattr(self.od_config, "max_num_seqs", 1)) * 2)
        while len(self._denoise_batches) > max_packed_batches:
            self._denoise_batches.popitem(last=False)

        next_latents = self.cuda_graph_runner(
            self.model,
            denoise_batch,
            run_packed_euler_rf_cfg_step,
        )
        refreshed_correction = (
            denoise_batch.cfg_correction if execution_key.cfg_refresh else None
        )
        for index, (state, sampling_state) in enumerate(
            zip(states, sampling_states, strict=True)
        ):
            # Graph outputs are shared fixed buffers, so each request owns a copy.
            request_latents = next_latents[index : index + 1].clone()
            sampling_state.latents = request_latents
            if refreshed_correction is not None:
                sampling_state.cfg_correction = refreshed_correction[index : index + 1].clone()
            sampling_state.step_index += 1
            state.latents = request_latents
            state.step_index += 1

    def clear_cuda_graphs(self) -> None:
        self.cuda_graph_runner.clear()
        self._denoise_batches.clear()

    def get_cuda_graph_stats(self) -> dict[str, int]:
        return self.cuda_graph_runner.stats()

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **_: Any) -> None:
        """Apply one exact Euler update to one request-local latent."""
        prepared = state.extra.get("irodori")
        if not isinstance(prepared, _IrodoriPreparedRequest):
            raise ValueError(f"Missing Irodori step state for request {state.request_id}.")
        if state.step_index != prepared.sampling_state.step_index:
            raise ValueError(f"Irodori step index diverged for request {state.request_id}.")
        apply_euler_rf_cfg_step(prepared.sampling_state, noise_pred)
        state.latents = prepared.sampling_state.latents
        state.step_index += 1

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        """Return the final latent or waveform for a completed Irodori request."""
        prepared = state.extra.get("irodori")
        if not isinstance(prepared, _IrodoriPreparedRequest):
            raise ValueError(f"Missing Irodori step state for request {state.request_id}.")
        return self._decode_prepared_request(prepared, output_type=kwargs.get("output_type"))
