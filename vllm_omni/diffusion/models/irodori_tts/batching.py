# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Length buckets and fixed-shape denoise batches for Irodori-TTS."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .sampler import ConditionBundle, ContextKVCache, IrodoriSamplingState


DEFAULT_LATENT_BUCKET_SECONDS = (2.0, 4.0, 8.0, 16.0, 32.0)
DEFAULT_CONTEXT_BUCKET_TOKENS = (8, 16, 32, 64, 128, 256, 512, 1024)
DEFAULT_CUDA_GRAPH_BATCH_SIZES = (1, 2, 4, 8)
IRODORI_CUDA_GRAPH_ENV = "VLLM_OMNI_IRODORI_CUDA_GRAPH"


def _environment_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}."
    )


def _positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return result


def _sequence(value: Any, *, name: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a non-empty sequence.")
    return list(value)


def _deduplicate_increasing(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} values must be positive, got {value}.")
        if result and value == result[-1]:
            continue
        if result and value < result[-1]:
            raise ValueError(f"{name} must be increasing after runtime conversion.")
        result.append(value)
    if not result:
        raise ValueError(f"{name} produced no usable buckets.")
    return tuple(result)


def _round_up(value: int, granularity: int) -> int:
    return math.ceil(value / granularity) * granularity


@dataclass(frozen=True)
class IrodoriLengthState:
    """Logical and physical audio lengths for one request."""

    valid_codec_frames: int
    valid_latent_len: int
    bucket_latent_len: int
    target_samples: int
    is_dynamic_bucket: bool


@dataclass(frozen=True)
class IrodoriExecutionKey:
    """Worker-side key for one physically homogeneous denoise microbatch."""

    bucket_latent_len: int
    dtype: torch.dtype
    device_type: str
    device_index: int | None
    cfg_guidance_mode: str
    cfg_layout: tuple[str, ...]


@dataclass(frozen=True)
class IrodoriGraphKey:
    """Every shape and control-flow choice fixed by a CUDA graph."""

    device_type: str
    device_index: int | None
    dtype: torch.dtype
    request_batch_size: int
    bucket_latent_len: int
    text_context_bucket: int
    speaker_context_bucket: int
    caption_context_bucket: int
    cfg_active: bool
    cfg_layout: tuple[str, ...]


@dataclass(frozen=True)
class IrodoriBatchingConfig:
    latent_bucket_seconds: tuple[float, ...]
    overflow_bucket_seconds: float
    context_bucket_tokens: tuple[int, ...]
    cuda_graph_batch_sizes: tuple[int, ...]
    cuda_graph_max_entries: int
    cuda_graph_max_dynamic_entries: int
    cuda_graph_min_hits: int
    enable_cuda_graph: bool

    @classmethod
    def from_od_config(cls, od_config: Any) -> IrodoriBatchingConfig:
        extras = dict(getattr(od_config, "extras", {}) or {})
        latent_seconds = tuple(
            _positive_float(value, name="irodori_latent_bucket_seconds")
            for value in _sequence(
                extras.get("irodori_latent_bucket_seconds", DEFAULT_LATENT_BUCKET_SECONDS),
                name="irodori_latent_bucket_seconds",
            )
        )
        if any(right <= left for left, right in zip(latent_seconds, latent_seconds[1:])):
            raise ValueError("irodori_latent_bucket_seconds must be strictly increasing.")

        context_tokens = tuple(
            _positive_int(value, name="irodori_context_bucket_tokens")
            for value in _sequence(
                extras.get("irodori_context_bucket_tokens", DEFAULT_CONTEXT_BUCKET_TOKENS),
                name="irodori_context_bucket_tokens",
            )
        )
        if any(right <= left for left, right in zip(context_tokens, context_tokens[1:])):
            raise ValueError("irodori_context_bucket_tokens must be strictly increasing.")

        graph_batch_sizes = tuple(
            _positive_int(value, name="irodori_cuda_graph_batch_sizes")
            for value in _sequence(
                extras.get("irodori_cuda_graph_batch_sizes", DEFAULT_CUDA_GRAPH_BATCH_SIZES),
                name="irodori_cuda_graph_batch_sizes",
            )
        )
        if len(set(graph_batch_sizes)) != len(graph_batch_sizes):
            raise ValueError("irodori_cuda_graph_batch_sizes must not contain duplicates.")

        if "irodori_enable_cuda_graph" in extras:
            enable_graph = extras["irodori_enable_cuda_graph"]
        else:
            enable_graph = _environment_bool(IRODORI_CUDA_GRAPH_ENV, default=True)
        if not isinstance(enable_graph, bool):
            raise ValueError("irodori_enable_cuda_graph must be a boolean.")

        return cls(
            latent_bucket_seconds=latent_seconds,
            overflow_bucket_seconds=_positive_float(
                extras.get("irodori_overflow_bucket_seconds", 8.0),
                name="irodori_overflow_bucket_seconds",
            ),
            context_bucket_tokens=context_tokens,
            cuda_graph_batch_sizes=graph_batch_sizes,
            cuda_graph_max_entries=_positive_int(
                extras.get("irodori_cuda_graph_max_entries", 8),
                name="irodori_cuda_graph_max_entries",
            ),
            cuda_graph_max_dynamic_entries=_positive_int(
                extras.get("irodori_cuda_graph_max_dynamic_entries", 0),
                name="irodori_cuda_graph_max_dynamic_entries",
                minimum=0,
            ),
            cuda_graph_min_hits=_positive_int(
                extras.get("irodori_cuda_graph_min_hits", 2),
                name="irodori_cuda_graph_min_hits",
            ),
            enable_cuda_graph=enable_graph,
        )


class IrodoriLatentBucketPolicy:
    """Convert exact sample lengths into runtime-derived DiT token buckets."""

    def __init__(
        self,
        *,
        sample_rate: int,
        hop_length: int,
        latent_patch_size: int,
        bucket_seconds: Sequence[float],
        overflow_bucket_seconds: float,
        max_output_seconds: float = 30.0,
    ) -> None:
        self.sample_rate = _positive_int(sample_rate, name="sample_rate")
        self.hop_length = _positive_int(hop_length, name="hop_length")
        self.latent_patch_size = _positive_int(latent_patch_size, name="latent_patch_size")
        self.max_output_seconds = _positive_float(max_output_seconds, name="max_output_seconds")
        converted = [self.seconds_to_latent_len(value) for value in bucket_seconds]
        self.standard_buckets = _deduplicate_increasing(
            converted,
            name="irodori_latent_bucket_seconds",
        )
        self.overflow_granularity = self.seconds_to_latent_len(overflow_bucket_seconds)

    def seconds_to_latent_len(self, seconds: float) -> int:
        seconds = _positive_float(seconds, name="bucket seconds")
        samples = math.ceil(seconds * self.sample_rate)
        codec_frames = math.ceil(samples / self.hop_length)
        return math.ceil(codec_frames / self.latent_patch_size)

    def lengths_for_samples(self, target_samples: int) -> IrodoriLengthState:
        target_samples = _positive_int(target_samples, name="target_samples")
        maximum_samples = math.ceil(self.max_output_seconds * self.sample_rate)
        if target_samples > maximum_samples:
            raise ValueError(
                f"Irodori output exceeds the {self.max_output_seconds:g}-second limit: {target_samples} samples."
            )
        valid_codec_frames = math.ceil(target_samples / self.hop_length)
        valid_latent_len = math.ceil(valid_codec_frames / self.latent_patch_size)
        bucket_latent_len, dynamic = self.select_bucket(valid_latent_len)
        return IrodoriLengthState(
            valid_codec_frames=valid_codec_frames,
            valid_latent_len=valid_latent_len,
            bucket_latent_len=bucket_latent_len,
            target_samples=target_samples,
            is_dynamic_bucket=dynamic,
        )

    def lengths_for_predicted_frames(self, valid_codec_frames: int) -> IrodoriLengthState:
        valid_codec_frames = _positive_int(valid_codec_frames, name="valid_codec_frames")
        maximum_frames = math.ceil(self.max_output_seconds * self.sample_rate / self.hop_length)
        if valid_codec_frames > maximum_frames:
            raise ValueError(
                f"Irodori predicted output exceeds the {self.max_output_seconds:g}-second limit: "
                f"{valid_codec_frames} codec frames."
            )
        valid_latent_len = math.ceil(valid_codec_frames / self.latent_patch_size)
        bucket_latent_len, dynamic = self.select_bucket(valid_latent_len)
        return IrodoriLengthState(
            valid_codec_frames=valid_codec_frames,
            valid_latent_len=valid_latent_len,
            bucket_latent_len=bucket_latent_len,
            target_samples=valid_codec_frames * self.hop_length,
            is_dynamic_bucket=dynamic,
        )

    def select_bucket(self, valid_latent_len: int) -> tuple[int, bool]:
        valid_latent_len = _positive_int(valid_latent_len, name="valid_latent_len")
        for bucket in self.standard_buckets:
            if valid_latent_len <= bucket:
                return bucket, False
        return _round_up(valid_latent_len, self.overflow_granularity), True


class IrodoriContextBucketPolicy:
    """Select fixed token buckets for text, speaker, and caption contexts."""

    def __init__(self, buckets: Sequence[int]) -> None:
        self.standard_buckets = _deduplicate_increasing(
            [_positive_int(value, name="context bucket") for value in buckets],
            name="irodori_context_bucket_tokens",
        )

    def select_bucket(self, valid_length: int) -> tuple[int, bool]:
        valid_length = max(1, int(valid_length))
        for bucket in self.standard_buckets:
            if valid_length <= bucket:
                return bucket, False
        return _round_up(valid_length, self.standard_buckets[-1]), True


def _logical_prefix_len(mask: torch.Tensor | None, fallback: int) -> int:
    if mask is None:
        return fallback
    active = torch.nonzero(mask.any(dim=0), as_tuple=False)
    if active.numel() == 0:
        return 1
    return int(active[-1, 0].item()) + 1


def _pad_sequence(value: torch.Tensor, target: int) -> torch.Tensor:
    if value.shape[1] < target:
        shape = list(value.shape)
        shape[1] = target - value.shape[1]
        value = torch.cat((value, value.new_zeros(shape)), dim=1)
    return value[:, :target]


def _pack_required(values: Sequence[torch.Tensor], target: int) -> torch.Tensor:
    return torch.cat([_pad_sequence(value, target) for value in values], dim=0).contiguous()


def _pack_optional(values: Sequence[torch.Tensor | None], target: int) -> torch.Tensor | None:
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise ValueError("Cannot pack mixed optional Irodori condition tensors.")
    return _pack_required([value for value in values if value is not None], target)


def _source_bucket(
    bundles: Sequence[ConditionBundle],
    *,
    state_index: int,
    mask_index: int,
    policy: IrodoriContextBucketPolicy,
) -> tuple[int, bool]:
    tensors = [bundle[state_index] for bundle in bundles]
    masks = [bundle[mask_index] for bundle in bundles]
    present_tensors = [tensor for tensor in tensors if tensor is not None]
    if not present_tensors:
        return 1, False
    valid_length = max(
        _logical_prefix_len(mask, tensor.shape[1])
        for tensor, mask in zip(tensors, masks, strict=True)
        if tensor is not None
    )
    return policy.select_bucket(valid_length)


def _pack_bundles(
    bundles: Sequence[ConditionBundle],
    context_buckets: tuple[int, int, int],
) -> ConditionBundle:
    text_bucket, speaker_bucket, caption_bucket = context_buckets
    return (
        _pack_required([bundle[0] for bundle in bundles], text_bucket),
        _pack_required([bundle[1] for bundle in bundles], text_bucket),
        _pack_optional([bundle[2] for bundle in bundles], speaker_bucket),
        _pack_optional([bundle[3] for bundle in bundles], speaker_bucket),
        _pack_optional([bundle[4] for bundle in bundles], caption_bucket),
        _pack_optional([bundle[5] for bundle in bundles], caption_bucket),
    )


def _pack_context_kv(
    states: Sequence[IrodoriSamplingState],
    caches: Sequence[ContextKVCache | None],
    context_buckets: tuple[int, int, int],
) -> ContextKVCache | None:
    if not any(cache is not None for cache in caches):
        return None
    if not all(cache is not None for cache in caches):
        raise ValueError("Cannot pack mixed Irodori context K/V cache modes.")
    present = [cache for cache in caches if cache is not None]
    layer_count = len(present[0])
    if any(len(cache) != layer_count for cache in present):
        raise ValueError("Irodori context K/V layer counts differ within a batch.")

    has_speaker = states[0].condition.speaker_state is not None
    has_caption = states[0].condition.caption_state is not None
    source_buckets = [context_buckets[0], context_buckets[0]]
    if has_speaker:
        source_buckets.extend([context_buckets[1], context_buckets[1]])
    if has_caption:
        source_buckets.extend([context_buckets[2], context_buckets[2]])

    result: list[tuple[torch.Tensor, ...]] = []
    for layer_index in range(layer_count):
        width = len(present[0][layer_index])
        if width != len(source_buckets):
            raise ValueError("Irodori context K/V layout does not match enabled condition sources.")
        if any(len(cache[layer_index]) != width for cache in present):
            raise ValueError("Irodori context K/V layouts differ within a batch.")
        result.append(
            tuple(
                _pack_required(
                    [cache[layer_index][value_index] for cache in present],
                    source_buckets[value_index],
                )
                for value_index in range(width)
            )
        )
    return result


def _copy_optional_tensor(destination: torch.Tensor | None, source: torch.Tensor | None) -> None:
    if destination is None and source is None:
        return
    if destination is None or source is None or destination.shape != source.shape:
        raise ValueError("Irodori packed optional tensor layouts differ.")
    destination.copy_(source)


@dataclass
class IrodoriDenoiseBatch:
    """Reusable fixed-shape inputs for one Irodori denoise-and-update call."""

    request_ids: tuple[str, ...]
    cfg_active: bool
    cfg_layout: tuple[str, ...]
    latents: torch.Tensor
    latent_mask: torch.Tensor
    timesteps: torch.Tensor
    dt: torch.Tensor
    cfg_scales: torch.Tensor
    bundle: ConditionBundle
    context_kv_cache: ContextKVCache | None
    context_buckets: tuple[int, int, int]
    dynamic_context_buckets: tuple[bool, bool, bool]
    is_dynamic_latent_bucket: bool

    @property
    def graph_key(self) -> IrodoriGraphKey:
        device = self.latents.device
        return IrodoriGraphKey(
            device_type=device.type,
            device_index=device.index,
            dtype=self.latents.dtype,
            request_batch_size=len(self.request_ids),
            bucket_latent_len=int(self.latents.shape[1]),
            text_context_bucket=self.context_buckets[0],
            speaker_context_bucket=self.context_buckets[1],
            caption_context_bucket=self.context_buckets[2],
            cfg_active=self.cfg_active,
            cfg_layout=self.cfg_layout,
        )

    @property
    def is_dynamic_graph_shape(self) -> bool:
        return self.is_dynamic_latent_bucket or any(self.dynamic_context_buckets)

    @property
    def composition_key(self) -> tuple[str, ...]:
        return self.request_ids

    @classmethod
    def make(
        cls,
        request_ids: Sequence[str],
        states: Sequence[IrodoriSamplingState],
        *,
        context_policy: IrodoriContextBucketPolicy,
        is_dynamic_latent_bucket: bool,
        cached_batch: IrodoriDenoiseBatch | None = None,
    ) -> IrodoriDenoiseBatch:
        if not states or len(request_ids) != len(states):
            raise ValueError("Irodori packed batch requires matching non-empty request/state lists.")
        cfg_active = states[0].cfg_active[states[0].step_index]
        cfg_layout = states[0].independent_names if cfg_active else ("cond",)
        if any(state.cfg_active[state.step_index] != cfg_active for state in states[1:]):
            raise ValueError("Irodori packed batch has mixed CFG activity.")
        if cfg_active and any(state.independent_names != cfg_layout for state in states[1:]):
            raise ValueError("Irodori packed batch has mixed CFG layouts.")

        bundles = [state.independent_bundle if cfg_active else state.cond_bundle for state in states]
        caches = [state.context_kv_cfg if cfg_active else state.context_kv_cond for state in states]
        text_bucket = _source_bucket(
            bundles,
            state_index=0,
            mask_index=1,
            policy=context_policy,
        )
        speaker_bucket = _source_bucket(
            bundles,
            state_index=2,
            mask_index=3,
            policy=context_policy,
        )
        caption_bucket = _source_bucket(
            bundles,
            state_index=4,
            mask_index=5,
            policy=context_policy,
        )
        context_buckets = (text_bucket[0], speaker_bucket[0], caption_bucket[0])
        dynamic_context = (text_bucket[1], speaker_bucket[1], caption_bucket[1])

        latents = torch.cat([state.latents for state in states], dim=0)
        masks = [
            state.latent_mask
            if state.latent_mask is not None
            else torch.ones(state.latents.shape[:2], dtype=torch.bool, device=state.latents.device)
            for state in states
        ]
        latent_mask = torch.cat(masks, dim=0)
        timesteps = torch.stack([state.current_timestep for state in states]).to(
            device=latents.device,
            dtype=latents.dtype,
        )
        next_timesteps = torch.stack([state.t_schedule[state.step_index + 1] for state in states]).to(
            device=latents.device, dtype=latents.dtype
        )
        dt = next_timesteps - timesteps
        scale_names = cfg_layout[1:] if cfg_active else ()
        cfg_scales = latents.new_tensor([[state.cfg_scales[name] for name in scale_names] for state in states]).reshape(
            len(states), len(scale_names)
        )

        can_reuse = (
            cached_batch is not None
            and cached_batch.request_ids == tuple(request_ids)
            and cached_batch.cfg_active == cfg_active
            and cached_batch.cfg_layout == cfg_layout
            and cached_batch.context_buckets == context_buckets
            and cached_batch.latents.shape == latents.shape
        )
        if can_reuse:
            assert cached_batch is not None
            cached_batch.latents.copy_(latents)
            cached_batch.latent_mask.copy_(latent_mask)
            cached_batch.timesteps.copy_(timesteps)
            cached_batch.dt.copy_(dt)
            cached_batch.cfg_scales.copy_(cfg_scales)
            return cached_batch

        return cls(
            request_ids=tuple(request_ids),
            cfg_active=cfg_active,
            cfg_layout=cfg_layout,
            latents=latents.contiguous(),
            latent_mask=latent_mask.contiguous(),
            timesteps=timesteps.contiguous(),
            dt=dt.contiguous(),
            cfg_scales=cfg_scales.contiguous(),
            bundle=_pack_bundles(bundles, context_buckets),
            context_kv_cache=_pack_context_kv(states, caches, context_buckets),
            context_buckets=context_buckets,
            dynamic_context_buckets=dynamic_context,
            is_dynamic_latent_bucket=bool(is_dynamic_latent_bucket),
        )

    def clone(self) -> IrodoriDenoiseBatch:
        return IrodoriDenoiseBatch(
            request_ids=self.request_ids,
            cfg_active=self.cfg_active,
            cfg_layout=self.cfg_layout,
            latents=self.latents.clone(),
            latent_mask=self.latent_mask.clone(),
            timesteps=self.timesteps.clone(),
            dt=self.dt.clone(),
            cfg_scales=self.cfg_scales.clone(),
            bundle=tuple(None if value is None else value.clone() for value in self.bundle),
            context_kv_cache=(
                None
                if self.context_kv_cache is None
                else [tuple(value.clone() for value in layer) for layer in self.context_kv_cache]
            ),
            context_buckets=self.context_buckets,
            dynamic_context_buckets=self.dynamic_context_buckets,
            is_dynamic_latent_bucket=self.is_dynamic_latent_bucket,
        )

    def copy_dynamic_from(self, source: IrodoriDenoiseBatch) -> None:
        self.latents.copy_(source.latents)
        self.latent_mask.copy_(source.latent_mask)
        self.timesteps.copy_(source.timesteps)
        self.dt.copy_(source.dt)
        self.cfg_scales.copy_(source.cfg_scales)

    def copy_static_from(self, source: IrodoriDenoiseBatch) -> None:
        for destination, value in zip(self.bundle, source.bundle, strict=True):
            _copy_optional_tensor(destination, value)
        if self.context_kv_cache is None and source.context_kv_cache is None:
            return
        if self.context_kv_cache is None or source.context_kv_cache is None:
            raise ValueError("Irodori packed context K/V cache layouts differ.")
        for destination_layer, source_layer in zip(
            self.context_kv_cache,
            source.context_kv_cache,
            strict=True,
        ):
            for destination, value in zip(destination_layer, source_layer, strict=True):
                destination.copy_(value)

    def tensor_bytes(self) -> int:
        tensors = [
            self.latents,
            self.latent_mask,
            self.timesteps,
            self.dt,
            self.cfg_scales,
            *(value for value in self.bundle if value is not None),
        ]
        if self.context_kv_cache is not None:
            tensors.extend(value for layer in self.context_kv_cache for value in layer)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)
