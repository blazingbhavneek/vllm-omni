# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions of this file are derived from Irodori-TTS (MIT),
# Copyright (c) 2026 Aratako. See Irodori-TTS/LICENSE in the upstream project.
# Upstream revision: 9f19d9a9048099a4b978a762d0509228fe624e3f.
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .batching import IrodoriDenoiseBatch
from .model import TextToLatentRFDiT
from .speaker_inversion import SPEAKER_INVERSION_UNCOND_MODES


def _make_rng(seed: int, device: torch.device) -> tuple[torch.Generator, torch.device]:
    # MPS generators are not available on some PyTorch builds; use CPU generator as fallback.
    try:
        return torch.Generator(device=device).manual_seed(seed), device
    except RuntimeError:
        return torch.Generator(device="cpu").manual_seed(seed), torch.device("cpu")


def sample_logit_normal_t(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    t_min: float = 1e-3,
    t_max: float = 0.999,
) -> torch.Tensor:
    z = torch.randn(batch_size, device=device) * std + mean
    t = torch.sigmoid(z)
    return t.clamp(min=t_min, max=t_max)


def sample_stratified_logit_normal_t(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    t_min: float = 1e-3,
    t_max: float = 0.999,
) -> torch.Tensor:
    """
    Stratified sampling for logit-normal timesteps.

    u ~ stratified U(0, 1), z = mean + std * Phi^{-1}(u), t = sigmoid(z)
    """
    if batch_size <= 0:
        return torch.empty((0,), device=device)
    u = (
        torch.arange(batch_size, device=device, dtype=torch.float32)
        + torch.rand(batch_size, device=device)
    ) / float(batch_size)
    u = u.clamp(1e-6, 1.0 - 1e-6)
    # Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1)
    z = torch.erfinv(2.0 * u - 1.0) * (2.0**0.5)
    z = z * std + mean
    t = torch.sigmoid(z)
    # Randomize assignment order so dataset ordering does not correlate with t bins.
    t = t[torch.randperm(batch_size, device=device)]
    return t.clamp(min=t_min, max=t_max)


def rf_interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # Straight line interpolation: x_t = (1-t) x0 + t z.
    return (1.0 - t[:, None, None]) * x0 + t[:, None, None] * noise


def rf_velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    # For x_t = (1-t) x0 + t z, velocity is d/dt x_t = z - x0.
    return noise - x0


def temporal_score_rescale(
    v_pred: torch.Tensor,
    x_t: torch.Tensor,
    t: float | torch.Tensor,
    rescale_k: float,
    rescale_sigma: float,
) -> torch.Tensor:
    """
    Temporal score rescaling from https://arxiv.org/pdf/2510.01184.
    """
    t_value = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
    if t_value >= 1.0:
        return v_pred
    one_minus_t = 1.0 - t_value
    snr = (one_minus_t * one_minus_t) / (t_value * t_value)
    sigma_sq = float(rescale_sigma) * float(rescale_sigma)
    ratio = (snr * sigma_sq + 1.0) / (snr * sigma_sq / float(rescale_k) + 1.0)
    return (ratio * (one_minus_t * v_pred + x_t) - x_t) / one_minus_t


def scale_speaker_kv_cache(
    context_kv_cache: list[tuple[torch.Tensor, ...]],
    scale: float,
    max_layers: int | None = None,
) -> None:
    """
    In-place scaling of speaker K/V tensors in precomputed context cache.
    """
    if max_layers is None:
        n_layers = len(context_kv_cache)
    else:
        n_layers = max(0, min(int(max_layers), len(context_kv_cache)))
    for i in range(n_layers):
        layer_kv = context_kv_cache[i]
        if len(layer_kv) < 4:
            raise ValueError(
                f"Expected at least 4 tensors in context KV cache entry, got {len(layer_kv)}"
            )
        k_speaker = layer_kv[2]
        v_speaker = layer_kv[3]
        k_speaker.mul_(scale)
        v_speaker.mul_(scale)


ConditionBundle = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]
ContextKVCache = list[tuple[torch.Tensor, ...]]


@dataclass(frozen=True)
class IrodoriConditionState:
    """Encoded request conditions shared by duration prediction and sampling."""

    text_state: torch.Tensor
    text_mask: torch.Tensor
    speaker_state: torch.Tensor | None
    speaker_mask: torch.Tensor | None
    caption_state: torch.Tensor | None
    caption_mask: torch.Tensor | None


@dataclass
class IrodoriSamplingState:
    """Mutable request-local state for rectified-flow Euler sampling."""

    latents: torch.Tensor
    t_schedule: torch.Tensor
    cfg_active: tuple[bool, ...]
    condition: IrodoriConditionState
    cond_bundle: ConditionBundle
    independent_bundle: ConditionBundle
    independent_names: tuple[str, ...]
    cfg_scales: dict[str, float]
    cfg_guidance_mode: str
    enabled_cfg_names: tuple[str, ...]
    joint_uncond_bundle: ConditionBundle
    alternating_bundles: dict[str, ConditionBundle]
    context_kv_cond: ContextKVCache | None
    context_kv_cfg: ContextKVCache | None
    context_kv_joint_uncond: ContextKVCache | None
    context_kv_alternating: dict[str, ContextKVCache]
    rescale_k: float | None
    rescale_sigma: float | None
    speaker_kv_scale: float | None
    speaker_kv_max_layers: int | None
    speaker_kv_min_t: float | None
    speaker_kv_active: bool
    latent_mask: torch.Tensor | None = None
    step_index: int = 0

    @property
    def total_steps(self) -> int:
        return int(self.t_schedule.shape[0] - 1)

    @property
    def current_timestep(self) -> torch.Tensor:
        if self.step_index >= self.total_steps:
            raise ValueError("Sampling has already completed.")
        return self.t_schedule[self.step_index]

    @property
    def cfg_rows(self) -> int:
        return len(self.independent_names)


def encode_irodori_conditions(
    model: TextToLatentRFDiT,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    ref_latent: torch.Tensor | None,
    ref_mask: torch.Tensor | None,
    *,
    caption_input_ids: torch.Tensor | None = None,
    caption_mask: torch.Tensor | None = None,
    speaker_state_override: torch.Tensor | None = None,
    speaker_mask_override: torch.Tensor | None = None,
    speaker_uncond_mode: str = "mask",
) -> IrodoriConditionState:
    """Encode text, speaker, and caption conditions once for one request."""
    (
        text_state,
        condition_text_mask,
        speaker_state,
        speaker_mask,
        caption_state,
        condition_caption_mask,
    ) = model.encode_conditions(
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        ref_latent=ref_latent,
        ref_mask=ref_mask,
        caption_input_ids=caption_input_ids,
        caption_mask=caption_mask,
        speaker_state_override=speaker_state_override,
        speaker_mask_override=speaker_mask_override,
        speaker_uncond_mode=speaker_uncond_mode,
    )
    return IrodoriConditionState(
        text_state=text_state,
        text_mask=condition_text_mask,
        speaker_state=speaker_state,
        speaker_mask=speaker_mask,
        caption_state=caption_state,
        caption_mask=condition_caption_mask,
    )


def _bundle(
    *,
    text_state: torch.Tensor,
    text_mask: torch.Tensor,
    speaker_state: torch.Tensor | None,
    speaker_mask: torch.Tensor | None,
    caption_state: torch.Tensor | None,
    caption_mask: torch.Tensor | None,
) -> ConditionBundle:
    return (
        text_state,
        text_mask,
        speaker_state,
        speaker_mask,
        caption_state,
        caption_mask,
    )


def _cat_optional_tensors(values: list[torch.Tensor | None]) -> torch.Tensor | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError("Cannot concatenate optional condition tensors with mixed presence.")
    return torch.cat(present, dim=0)


@torch.inference_mode()
def prepare_euler_rf_cfg(
    model: TextToLatentRFDiT,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    ref_latent: torch.Tensor | None,
    ref_mask: torch.Tensor | None,
    sequence_length: int,
    caption_input_ids: torch.Tensor | None = None,
    caption_mask: torch.Tensor | None = None,
    speaker_state_override: torch.Tensor | None = None,
    speaker_mask_override: torch.Tensor | None = None,
    speaker_uncond_mode: str = "mask",
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_caption: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_guidance_mode: str = "independent",
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    seed: int = 0,
    cfg_scale: float | None = None,
    truncation_factor: float | None = None,
    rescale_k: float | None = None,
    rescale_sigma: float | None = None,
    use_context_kv_cache: bool = True,
    speaker_kv_scale: float | None = None,
    speaker_kv_max_layers: int | None = None,
    speaker_kv_min_t: float | None = None,
    t_schedule_mode: str = "linear",
    sway_coeff: float = -1.0,
    generator: torch.Generator | None = None,
    initial_latents: torch.Tensor | None = None,
    condition_state: IrodoriConditionState | None = None,
    latent_mask: torch.Tensor | None = None,
    bucket_sequence_length: int | None = None,
) -> IrodoriSamplingState:
    """Prepare request-local state without running a DiT denoise step."""
    device = model.device
    dtype = model.dtype
    batch_size = text_input_ids.shape[0]
    latent_dim = model.cfg.patched_latent_dim
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length <= 0:
        raise ValueError(f"sequence_length must be a positive integer, got {sequence_length!r}.")
    if bucket_sequence_length is None:
        bucket_sequence_length = sequence_length
    if (
        isinstance(bucket_sequence_length, bool)
        or not isinstance(bucket_sequence_length, int)
        or bucket_sequence_length < sequence_length
    ):
        raise ValueError(
            "bucket_sequence_length must be an integer greater than or equal to "
            f"sequence_length, got {bucket_sequence_length!r}."
        )
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or not 1 <= num_steps <= 100:
        raise ValueError(f"num_steps must be an integer in [1, 100], got {num_steps!r}.")
    for name, value in (
        ("cfg_scale_text", cfg_scale_text),
        ("cfg_scale_caption", cfg_scale_caption),
        ("cfg_scale_speaker", cfg_scale_speaker),
        ("cfg_min_t", cfg_min_t),
        ("cfg_max_t", cfg_max_t),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}.")
    if not 0.0 <= float(cfg_min_t) <= float(cfg_max_t) <= 1.0:
        raise ValueError("cfg_min_t and cfg_max_t must satisfy 0 <= min <= max <= 1.")

    if generator is None:
        rng, rng_device = _make_rng(seed=seed, device=device)
    else:
        rng = generator
        rng_device = torch.device(generator.device)
    expected_shape = (batch_size, sequence_length, latent_dim)
    if initial_latents is None:
        x_t = torch.randn(expected_shape, device=rng_device, dtype=dtype, generator=rng)
        if rng_device != device:
            x_t = x_t.to(device=device)
    else:
        if tuple(initial_latents.shape) != expected_shape:
            raise ValueError(
                "initial_latents shape mismatch: "
                f"expected {expected_shape}, got {tuple(initial_latents.shape)}."
            )
        if initial_latents.device != device:
            raise ValueError(
                "initial_latents device mismatch: "
                f"expected {device}, got {initial_latents.device}."
            )
        if initial_latents.dtype != dtype:
            raise ValueError(
                "initial_latents dtype mismatch: "
                f"expected {dtype}, got {initial_latents.dtype}."
            )
        x_t = initial_latents
    if truncation_factor is not None:
        x_t = x_t * float(truncation_factor)

    if latent_mask is not None:
        if tuple(latent_mask.shape) != (batch_size, sequence_length):
            raise ValueError(
                "latent_mask shape mismatch: "
                f"expected {(batch_size, sequence_length)}, got {tuple(latent_mask.shape)}."
            )
        latent_mask = latent_mask.to(device=device, dtype=torch.bool)
    if bucket_sequence_length > sequence_length:
        if latent_mask is None:
            latent_mask = torch.ones(
                (batch_size, sequence_length),
                dtype=torch.bool,
                device=device,
            )
        latent_tail = x_t.new_zeros(
            (batch_size, bucket_sequence_length - sequence_length, latent_dim)
        )
        mask_tail = torch.zeros(
            (batch_size, bucket_sequence_length - sequence_length),
            dtype=torch.bool,
            device=device,
        )
        x_t = torch.cat((x_t, latent_tail), dim=1)
        latent_mask = torch.cat((latent_mask, mask_tail), dim=1)

    if cfg_scale is not None:
        # Backward compatibility for old single-scale caller.
        cfg_scale_text = float(cfg_scale)
        cfg_scale_caption = float(cfg_scale)
        cfg_scale_speaker = float(cfg_scale)
    if not model.cfg.use_speaker_condition_resolved:
        cfg_scale_speaker = 0.0
        speaker_kv_scale = None
    speaker_uncond_mode = str(speaker_uncond_mode).strip().lower()
    if speaker_uncond_mode not in SPEAKER_INVERSION_UNCOND_MODES:
        raise ValueError(
            f"speaker_uncond_mode must be one of {sorted(SPEAKER_INVERSION_UNCOND_MODES)}, "
            f"got {speaker_uncond_mode!r}"
        )

    cfg_guidance_mode = str(cfg_guidance_mode).strip().lower()
    if cfg_guidance_mode not in {"independent", "joint", "alternating"}:
        raise ValueError(
            f"Unsupported cfg_guidance_mode={cfg_guidance_mode!r}. "
            "Expected one of: independent, joint, alternating."
        )

    init_scale = 0.999
    t_schedule_mode_norm = str(t_schedule_mode).strip().lower()
    sway_coeff_value = float(sway_coeff)
    if not math.isfinite(sway_coeff_value):
        raise ValueError(f"sway_coeff must be finite, got {sway_coeff!r}.")
    if t_schedule_mode_norm == "linear":
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    elif t_schedule_mode_norm == "sway":
        # F5-TTS-style Sway Sampling. Negative sway_coeff densifies the noise
        # side of the schedule (early steps); positive densifies the data side.
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        u = u + sway_coeff_value * (torch.cos(0.5 * math.pi * u) + u - 1.0)
        u = u.clamp(0.0, 1.0)
    else:
        raise ValueError(
            f"Unsupported t_schedule_mode={t_schedule_mode!r}. Expected 'linear' or 'sway'."
        )
    t_schedule = (1.0 - u) * init_scale
    if not bool(torch.all(t_schedule[:-1] > t_schedule[1:]).item()):
        raise ValueError("t_schedule must be strictly decreasing; adjust num_steps or sway_coeff.")
    use_independent_cfg = cfg_guidance_mode == "independent"
    use_joint_cfg = cfg_guidance_mode == "joint"
    use_alternating_cfg = cfg_guidance_mode == "alternating"

    condition = condition_state or encode_irodori_conditions(
        model,
        text_input_ids,
        text_mask,
        ref_latent,
        ref_mask,
        caption_input_ids=caption_input_ids,
        caption_mask=caption_mask,
        speaker_state_override=speaker_state_override,
        speaker_mask_override=speaker_mask_override,
        speaker_uncond_mode=speaker_uncond_mode,
    )
    text_state_cond = condition.text_state
    text_mask_cond = condition.text_mask
    speaker_state_cond = condition.speaker_state
    speaker_mask_cond = condition.speaker_mask
    caption_state_cond = condition.caption_state
    caption_mask_cond = condition.caption_mask
    text_state_uncond = torch.zeros_like(text_state_cond)
    text_mask_uncond = torch.zeros_like(text_mask_cond)
    speaker_state_uncond = None
    speaker_mask_uncond = None
    if model.cfg.use_speaker_condition_resolved:
        if speaker_state_cond is None or speaker_mask_cond is None:
            raise RuntimeError(
                "Speaker conditioning is enabled but encoded speaker state is missing."
            )
        if speaker_uncond_mode == "noise":
            speaker_noise = torch.randn(
                speaker_state_cond.shape,
                device=rng_device,
                dtype=speaker_state_cond.dtype,
                generator=rng,
            )
            if rng_device != device:
                speaker_noise = speaker_noise.to(device=device)
            speaker_state_uncond = speaker_noise * speaker_state_cond.std().clamp_min(1e-6)
            speaker_mask_uncond = torch.ones_like(speaker_mask_cond)
        else:
            speaker_state_uncond = torch.zeros_like(speaker_state_cond)
            speaker_mask_uncond = torch.zeros_like(speaker_mask_cond)
    caption_state_uncond = None
    caption_mask_uncond = None
    if model.cfg.use_caption_condition:
        if caption_state_cond is None or caption_mask_cond is None:
            raise RuntimeError(
                "Caption conditioning is enabled but encoded caption state is missing."
            )
        caption_state_uncond = torch.zeros_like(caption_state_cond)
        caption_mask_uncond = torch.zeros_like(caption_mask_cond)

    has_text_cfg = cfg_scale_text > 0
    has_caption_cfg = (
        model.cfg.use_caption_condition
        and cfg_scale_caption > 0
        and caption_mask_cond is not None
        and bool(caption_mask_cond.any().item())
    )
    has_speaker_cfg = cfg_scale_speaker > 0

    cond_bundle = _bundle(
        text_state=text_state_cond,
        text_mask=text_mask_cond,
        speaker_state=speaker_state_cond,
        speaker_mask=speaker_mask_cond,
        caption_state=caption_state_cond,
        caption_mask=caption_mask_cond,
    )
    enabled_cfg_names: list[str] = []
    cfg_scales: dict[str, float] = {}
    if has_text_cfg:
        enabled_cfg_names.append("text")
        cfg_scales["text"] = float(cfg_scale_text)
    if has_speaker_cfg:
        enabled_cfg_names.append("speaker")
        cfg_scales["speaker"] = float(cfg_scale_speaker)
    if has_caption_cfg:
        enabled_cfg_names.append("caption")
        cfg_scales["caption"] = float(cfg_scale_caption)

    independent_bundles = [cond_bundle]
    independent_names = ["cond"]
    if use_independent_cfg:
        for name in enabled_cfg_names:
            independent_names.append(name)
            independent_bundles.append(
                _bundle(
                    text_state=text_state_uncond if name == "text" else text_state_cond,
                    text_mask=text_mask_uncond if name == "text" else text_mask_cond,
                    speaker_state=(
                        speaker_state_uncond if name == "speaker" else speaker_state_cond
                    ),
                    speaker_mask=(
                        speaker_mask_uncond if name == "speaker" else speaker_mask_cond
                    ),
                    caption_state=(
                        caption_state_uncond if name == "caption" else caption_state_cond
                    ),
                    caption_mask=(
                        caption_mask_uncond if name == "caption" else caption_mask_cond
                    ),
                )
            )
    cfg_batch_mult = len(independent_bundles)

    independent_text_state = torch.cat([bundle[0] for bundle in independent_bundles], dim=0)
    independent_text_mask = torch.cat([bundle[1] for bundle in independent_bundles], dim=0)
    independent_speaker_state = _cat_optional_tensors([bundle[2] for bundle in independent_bundles])
    independent_speaker_mask = _cat_optional_tensors([bundle[3] for bundle in independent_bundles])
    independent_caption_state = _cat_optional_tensors([bundle[4] for bundle in independent_bundles])
    independent_caption_mask = _cat_optional_tensors([bundle[5] for bundle in independent_bundles])

    joint_uncond_bundle = _bundle(
        text_state=text_state_uncond,
        text_mask=text_mask_uncond,
        speaker_state=speaker_state_uncond,
        speaker_mask=speaker_mask_uncond,
        caption_state=caption_state_uncond,
        caption_mask=caption_mask_uncond,
    )

    alternating_bundles: dict[
        str,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
    ] = {
        "text": _bundle(
            text_state=text_state_uncond,
            text_mask=text_mask_uncond,
            speaker_state=speaker_state_cond,
            speaker_mask=speaker_mask_cond,
            caption_state=caption_state_cond,
            caption_mask=caption_mask_cond,
        ),
        "caption": _bundle(
            text_state=text_state_cond,
            text_mask=text_mask_cond,
            speaker_state=speaker_state_cond,
            speaker_mask=speaker_mask_cond,
            caption_state=caption_state_uncond,
            caption_mask=caption_mask_uncond,
        ),
    }
    if has_speaker_cfg:
        alternating_bundles["speaker"] = _bundle(
            text_state=text_state_cond,
            text_mask=text_mask_cond,
            speaker_state=speaker_state_uncond,
            speaker_mask=speaker_mask_uncond,
            caption_state=caption_state_cond,
            caption_mask=caption_mask_cond,
        )

    # Force-speaker scaling operates on projected speaker K/V, so it requires context KV caches.
    effective_use_context_kv_cache = bool(use_context_kv_cache or (speaker_kv_scale is not None))

    context_kv_cond = None
    context_kv_cfg = None
    context_kv_joint_uncond = None
    context_kv_alternating: dict[str, list[tuple[torch.Tensor, ...]]] = {}
    if effective_use_context_kv_cache:
        context_kv_cond = model.build_context_kv_cache(
            text_state=text_state_cond,
            speaker_state=speaker_state_cond,
            caption_state=caption_state_cond,
        )
        if use_independent_cfg and cfg_batch_mult > 1:
            context_kv_cfg = model.build_context_kv_cache(
                text_state=independent_text_state,
                speaker_state=independent_speaker_state,
                caption_state=independent_caption_state,
            )
        elif use_joint_cfg:
            if enabled_cfg_names:
                context_kv_joint_uncond = model.build_context_kv_cache(
                    text_state=joint_uncond_bundle[0],
                    speaker_state=joint_uncond_bundle[2],
                    caption_state=joint_uncond_bundle[4],
                )
        elif use_alternating_cfg:
            for name in enabled_cfg_names:
                bundle = alternating_bundles[name]
                context_kv_alternating[name] = model.build_context_kv_cache(
                    text_state=bundle[0],
                    speaker_state=bundle[2],
                    caption_state=bundle[4],
                )
    if speaker_kv_scale is not None:
        scale_speaker_kv_cache(
            context_kv_cache=context_kv_cond,
            scale=float(speaker_kv_scale),
            max_layers=speaker_kv_max_layers,
        )
        if context_kv_cfg is not None:
            scale_speaker_kv_cache(
                context_kv_cache=context_kv_cfg,
                scale=float(speaker_kv_scale),
                max_layers=speaker_kv_max_layers,
            )
        for cache in context_kv_alternating.values():
            scale_speaker_kv_cache(
                context_kv_cache=cache,
                scale=float(speaker_kv_scale),
                max_layers=speaker_kv_max_layers,
            )
    cfg_active_values = t_schedule[:-1].detach().float().cpu().tolist()
    return IrodoriSamplingState(
        latents=x_t,
        t_schedule=t_schedule,
        cfg_active=tuple(
            bool(enabled_cfg_names) and cfg_min_t <= float(t) <= cfg_max_t
            for t in cfg_active_values
        ),
        condition=condition,
        cond_bundle=cond_bundle,
        independent_bundle=(
            independent_text_state,
            independent_text_mask,
            independent_speaker_state,
            independent_speaker_mask,
            independent_caption_state,
            independent_caption_mask,
        ),
        independent_names=tuple(independent_names),
        cfg_scales=cfg_scales,
        cfg_guidance_mode=cfg_guidance_mode,
        enabled_cfg_names=tuple(enabled_cfg_names),
        joint_uncond_bundle=joint_uncond_bundle,
        alternating_bundles=alternating_bundles,
        context_kv_cond=context_kv_cond,
        context_kv_cfg=context_kv_cfg,
        context_kv_joint_uncond=context_kv_joint_uncond,
        context_kv_alternating=context_kv_alternating,
        rescale_k=rescale_k,
        rescale_sigma=rescale_sigma,
        speaker_kv_scale=speaker_kv_scale,
        speaker_kv_max_layers=speaker_kv_max_layers,
        speaker_kv_min_t=speaker_kv_min_t,
        speaker_kv_active=speaker_kv_scale is not None,
        latent_mask=latent_mask,
    )


def _forward_with_bundle(
    model: TextToLatentRFDiT,
    *,
    x_t: torch.Tensor,
    t: torch.Tensor,
    bundle: ConditionBundle,
    latent_mask: torch.Tensor | None,
    context_kv_cache: ContextKVCache | None,
) -> torch.Tensor:
    return model.forward_with_encoded_conditions(
        x_t=x_t,
        t=t,
        text_state=bundle[0],
        text_mask=bundle[1],
        speaker_state=bundle[2],
        speaker_mask=bundle[3],
        caption_state=bundle[4],
        caption_mask=bundle[5],
        latent_mask=latent_mask,
        context_kv_cache=context_kv_cache,
    )


@torch.inference_mode()
def predict_euler_rf_cfg_step(
    model: TextToLatentRFDiT,
    state: IrodoriSamplingState,
) -> torch.Tensor:
    """Return one request's velocity prediction without mutating its latent."""
    t = state.current_timestep
    batch_size = state.latents.shape[0]
    tt = torch.full((batch_size,), t, device=model.device, dtype=model.dtype)

    if state.cfg_active[state.step_index]:
        if state.cfg_guidance_mode == "independent":
            cfg_rows = state.cfg_rows
            prediction = _forward_with_bundle(
                model,
                x_t=torch.cat([state.latents] * cfg_rows, dim=0).to(model.dtype),
                t=tt.repeat(cfg_rows),
                bundle=state.independent_bundle,
                latent_mask=(
                    None
                    if state.latent_mask is None
                    else torch.cat([state.latent_mask] * cfg_rows, dim=0)
                ),
                context_kv_cache=state.context_kv_cfg,
            )
            chunks = prediction.chunk(cfg_rows, dim=0)
            velocity = chunks[0]
            for name, chunk in zip(state.independent_names[1:], chunks[1:], strict=True):
                velocity = velocity + state.cfg_scales[name] * (chunks[0] - chunk)
            return velocity

        velocity_cond = _forward_with_bundle(
            model,
            x_t=state.latents.to(model.dtype),
            t=tt,
            bundle=state.cond_bundle,
            latent_mask=state.latent_mask,
            context_kv_cache=state.context_kv_cond,
        )
        if state.cfg_guidance_mode == "joint":
            if len(state.enabled_cfg_names) > 1:
                joint_scales = [state.cfg_scales[name] for name in state.enabled_cfg_names]
                if max(joint_scales) - min(joint_scales) > 1e-6:
                    raise ValueError(
                        "cfg_guidance_mode='joint' expects equal enabled guidance scales; "
                        "set matching text/speaker/caption scales or use --cfg-scale."
                    )
            joint_scale = state.cfg_scales[state.enabled_cfg_names[0]]
            velocity_uncond = _forward_with_bundle(
                model,
                x_t=state.latents.to(model.dtype),
                t=tt,
                bundle=state.joint_uncond_bundle,
                latent_mask=state.latent_mask,
                context_kv_cache=state.context_kv_joint_uncond,
            )
            return velocity_cond + joint_scale * (velocity_cond - velocity_uncond)
        if state.cfg_guidance_mode == "alternating":
            name = state.enabled_cfg_names[state.step_index % len(state.enabled_cfg_names)]
            velocity_uncond = _forward_with_bundle(
                model,
                x_t=state.latents.to(model.dtype),
                t=tt,
                bundle=state.alternating_bundles[name],
                latent_mask=state.latent_mask,
                context_kv_cache=state.context_kv_alternating.get(name),
            )
            return velocity_cond + state.cfg_scales[name] * (velocity_cond - velocity_uncond)
        raise RuntimeError(f"Unexpected cfg_guidance_mode: {state.cfg_guidance_mode}")

    return _forward_with_bundle(
        model,
        x_t=state.latents.to(model.dtype),
        t=tt,
        bundle=state.cond_bundle,
        latent_mask=state.latent_mask,
        context_kv_cache=state.context_kv_cond,
    )


def _pad_and_cat(values: list[torch.Tensor], *, dim: int = 1) -> torch.Tensor:
    if not values:
        raise ValueError("Cannot concatenate an empty tensor list.")
    if any(value.ndim != values[0].ndim for value in values):
        raise ValueError("Batched tensors must have matching ranks.")
    max_size = max(value.shape[dim] for value in values)
    padded: list[torch.Tensor] = []
    for value in values:
        if value.shape[dim] == max_size:
            padded.append(value)
            continue
        shape = list(value.shape)
        shape[dim] = max_size - value.shape[dim]
        padded.append(torch.cat((value, value.new_zeros(shape)), dim=dim))
    return torch.cat(padded, dim=0)


def _pad_and_cat_optional(values: list[torch.Tensor | None], *, dim: int = 1) -> torch.Tensor | None:
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise ValueError("Cannot batch requests with mixed optional condition tensors.")
    return _pad_and_cat([value for value in values if value is not None], dim=dim)


def _collate_bundles(bundles: list[ConditionBundle]) -> ConditionBundle:
    return (
        _pad_and_cat([bundle[0] for bundle in bundles]),
        _pad_and_cat([bundle[1] for bundle in bundles]),
        _pad_and_cat_optional([bundle[2] for bundle in bundles]),
        _pad_and_cat_optional([bundle[3] for bundle in bundles]),
        _pad_and_cat_optional([bundle[4] for bundle in bundles]),
        _pad_and_cat_optional([bundle[5] for bundle in bundles]),
    )


def _collate_context_kv_caches(caches: list[ContextKVCache | None]) -> ContextKVCache | None:
    if not any(cache is not None for cache in caches):
        return None
    if not all(cache is not None for cache in caches):
        raise ValueError("Cannot batch requests with mixed context K/V cache modes.")
    present = [cache for cache in caches if cache is not None]
    first = present[0]
    if any(len(cache) != len(first) for cache in present[1:]):
        raise ValueError("Context K/V cache layer counts must match within a batch.")
    result: ContextKVCache = []
    for layer_idx, first_layer in enumerate(first):
        if any(len(cache[layer_idx]) != len(first_layer) for cache in present[1:]):
            raise ValueError("Context K/V cache layouts must match within a batch.")
        result.append(
            tuple(
                _pad_and_cat([cache[layer_idx][value_idx] for cache in present])
                for value_idx in range(len(first_layer))
            )
        )
    return result


@torch.inference_mode()
def predict_euler_rf_cfg_batch(
    model: TextToLatentRFDiT,
    states: list[IrodoriSamplingState],
) -> list[torch.Tensor]:
    """Run one fused independent-CFG denoise step for compatible requests."""
    if not states:
        return []
    if len(states) == 1:
        return [predict_euler_rf_cfg_step(model, states[0])]
    if any(state.cfg_guidance_mode != "independent" for state in states):
        raise ValueError("Batched Irodori step execution currently requires independent CFG.")
    if any(state.latents.shape[1:] != states[0].latents.shape[1:] for state in states[1:]):
        raise ValueError("Batched Irodori denoise requests must have matching latent shapes.")
    if any(state.latents.dtype != states[0].latents.dtype for state in states[1:]):
        raise ValueError("Batched Irodori denoise requests must have matching latent dtypes.")

    cfg_active = states[0].cfg_active[states[0].step_index]
    if any(state.cfg_active[state.step_index] != cfg_active for state in states[1:]):
        raise ValueError("Batched Irodori requests must have matching active CFG layouts.")
    if cfg_active and any(state.independent_names != states[0].independent_names for state in states[1:]):
        raise ValueError("Batched Irodori requests must have matching CFG branches.")

    if cfg_active:
        cfg_rows = states[0].cfg_rows
        latents = torch.cat(
            [torch.cat([state.latents] * cfg_rows, dim=0) for state in states], dim=0
        ).to(model.dtype)
        timesteps = torch.cat(
            [
                state.current_timestep.reshape(1).expand(state.latents.shape[0] * cfg_rows)
                for state in states
            ]
        ).to(device=model.device, dtype=model.dtype)
        latent_mask = _pad_and_cat_optional(
            [
                None if state.latent_mask is None else torch.cat([state.latent_mask] * cfg_rows, dim=0)
                for state in states
            ]
        )
        prediction = _forward_with_bundle(
            model,
            x_t=latents,
            t=timesteps,
            bundle=_collate_bundles([state.independent_bundle for state in states]),
            latent_mask=latent_mask,
            context_kv_cache=_collate_context_kv_caches([state.context_kv_cfg for state in states]),
        )
        result: list[torch.Tensor] = []
        offset = 0
        for state in states:
            row_count = state.latents.shape[0] * cfg_rows
            chunks = prediction[offset : offset + row_count].chunk(cfg_rows, dim=0)
            velocity = chunks[0]
            for name, chunk in zip(state.independent_names[1:], chunks[1:], strict=True):
                velocity = velocity + state.cfg_scales[name] * (chunks[0] - chunk)
            result.append(velocity)
            offset += row_count
        return result

    latents = torch.cat([state.latents for state in states], dim=0).to(model.dtype)
    timesteps = torch.cat(
        [state.current_timestep.reshape(1).expand(state.latents.shape[0]) for state in states]
    ).to(device=model.device, dtype=model.dtype)
    prediction = _forward_with_bundle(
        model,
        x_t=latents,
        t=timesteps,
        bundle=_collate_bundles([state.cond_bundle for state in states]),
        latent_mask=_pad_and_cat_optional([state.latent_mask for state in states]),
        context_kv_cache=_collate_context_kv_caches([state.context_kv_cond for state in states]),
    )
    result = []
    offset = 0
    for state in states:
        row_count = state.latents.shape[0]
        result.append(prediction[offset : offset + row_count])
        offset += row_count
    return result


@torch.inference_mode()
def run_packed_euler_rf_cfg_step(
    model: TextToLatentRFDiT,
    batch: IrodoriDenoiseBatch,
) -> torch.Tensor:
    """Run one packed independent-CFG DiT forward and masked Euler update."""
    request_count = len(batch.request_ids)
    if batch.latents.shape[0] != request_count:
        raise ValueError("Packed Irodori execution expects one latent row per request.")

    if batch.cfg_active:
        cfg_rows = len(batch.cfg_layout)
        model_latents = (
            batch.latents[:, None]
            .expand(request_count, cfg_rows, *batch.latents.shape[1:])
            .reshape(request_count * cfg_rows, *batch.latents.shape[1:])
            .to(model.dtype)
        )
        model_timesteps = (
            batch.timesteps[:, None]
            .expand(request_count, cfg_rows)
            .reshape(request_count * cfg_rows)
            .to(device=model.device, dtype=model.dtype)
        )
        model_mask = (
            batch.latent_mask[:, None]
            .expand(request_count, cfg_rows, batch.latent_mask.shape[1])
            .reshape(request_count * cfg_rows, batch.latent_mask.shape[1])
        )
        prediction = _forward_with_bundle(
            model,
            x_t=model_latents,
            t=model_timesteps,
            bundle=batch.bundle,
            latent_mask=model_mask,
            context_kv_cache=batch.context_kv_cache,
        ).reshape(request_count, cfg_rows, *batch.latents.shape[1:])
        conditional = prediction[:, 0]
        if cfg_rows > 1:
            deltas = conditional[:, None] - prediction[:, 1:]
            velocity = conditional + (
                deltas * batch.cfg_scales[:, :, None, None]
            ).sum(dim=1)
        else:
            velocity = conditional
    else:
        velocity = _forward_with_bundle(
            model,
            x_t=batch.latents.to(model.dtype),
            t=batch.timesteps.to(device=model.device, dtype=model.dtype),
            bundle=batch.bundle,
            latent_mask=batch.latent_mask,
            context_kv_cache=batch.context_kv_cache,
        )

    next_latents = batch.latents + velocity * batch.dt[:, None, None]
    return next_latents.masked_fill(~batch.latent_mask[:, :, None], 0)


@torch.inference_mode()
def apply_euler_rf_cfg_step(
    state: IrodoriSamplingState,
    prediction: torch.Tensor,
) -> None:
    """Apply one Euler update and advance a request-local sampling state."""
    t = state.current_timestep
    t_next = state.t_schedule[state.step_index + 1]
    velocity = prediction
    if state.rescale_k is not None and state.rescale_sigma is not None:
        velocity = temporal_score_rescale(
            v_pred=velocity,
            x_t=state.latents,
            t=t,
            rescale_k=float(state.rescale_k),
            rescale_sigma=float(state.rescale_sigma),
        )
    if (
        state.speaker_kv_active
        and state.speaker_kv_min_t is not None
        and (t_next < state.speaker_kv_min_t)
        and (t >= state.speaker_kv_min_t)
    ):
        assert state.speaker_kv_scale is not None
        inv_scale = 1.0 / float(state.speaker_kv_scale)
        scale_speaker_kv_cache(
            context_kv_cache=state.context_kv_cond,
            scale=inv_scale,
            max_layers=state.speaker_kv_max_layers,
        )
        if state.context_kv_cfg is not None:
            scale_speaker_kv_cache(
                context_kv_cache=state.context_kv_cfg,
                scale=inv_scale,
                max_layers=state.speaker_kv_max_layers,
            )
        for cache in state.context_kv_alternating.values():
            scale_speaker_kv_cache(
                context_kv_cache=cache,
                scale=inv_scale,
                max_layers=state.speaker_kv_max_layers,
            )
        state.speaker_kv_active = False
    state.latents = state.latents + velocity * (t_next - t)
    if state.latent_mask is not None:
        state.latents.masked_fill_(~state.latent_mask[:, :, None], 0)
    state.step_index += 1


@torch.inference_mode()
def sample_euler_rf_cfg(
    model: TextToLatentRFDiT,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    ref_latent: torch.Tensor | None,
    ref_mask: torch.Tensor | None,
    sequence_length: int,
    caption_input_ids: torch.Tensor | None = None,
    caption_mask: torch.Tensor | None = None,
    speaker_state_override: torch.Tensor | None = None,
    speaker_mask_override: torch.Tensor | None = None,
    speaker_uncond_mode: str = "mask",
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_caption: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_guidance_mode: str = "independent",
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    seed: int = 0,
    cfg_scale: float | None = None,
    truncation_factor: float | None = None,
    rescale_k: float | None = None,
    rescale_sigma: float | None = None,
    use_context_kv_cache: bool = True,
    speaker_kv_scale: float | None = None,
    speaker_kv_max_layers: int | None = None,
    speaker_kv_min_t: float | None = None,
    t_schedule_mode: str = "linear",
    sway_coeff: float = -1.0,
    generator: torch.Generator | None = None,
    initial_latents: torch.Tensor | None = None,
    condition_state: IrodoriConditionState | None = None,
    latent_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the legacy complete Euler loop through the reusable step helpers."""
    state = prepare_euler_rf_cfg(
        model,
        text_input_ids,
        text_mask,
        ref_latent,
        ref_mask,
        sequence_length,
        caption_input_ids=caption_input_ids,
        caption_mask=caption_mask,
        speaker_state_override=speaker_state_override,
        speaker_mask_override=speaker_mask_override,
        speaker_uncond_mode=speaker_uncond_mode,
        num_steps=num_steps,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_caption=cfg_scale_caption,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_guidance_mode=cfg_guidance_mode,
        cfg_min_t=cfg_min_t,
        cfg_max_t=cfg_max_t,
        seed=seed,
        cfg_scale=cfg_scale,
        truncation_factor=truncation_factor,
        rescale_k=rescale_k,
        rescale_sigma=rescale_sigma,
        use_context_kv_cache=use_context_kv_cache,
        speaker_kv_scale=speaker_kv_scale,
        speaker_kv_max_layers=speaker_kv_max_layers,
        speaker_kv_min_t=speaker_kv_min_t,
        t_schedule_mode=t_schedule_mode,
        sway_coeff=sway_coeff,
        generator=generator,
        initial_latents=initial_latents,
        condition_state=condition_state,
        latent_mask=latent_mask,
    )
    while state.step_index < state.total_steps:
        apply_euler_rf_cfg_step(state, predict_euler_rf_cfg_step(model, state))
    return state.latents
